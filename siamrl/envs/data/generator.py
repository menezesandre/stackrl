"""
Run this file to (re)generate the models in 'generated' folder
"""
import os
import sys

import numpy as np
from scipy import stats
try:
  from trimesh import creation
  from trimesh import transformations
except ImportError:
  raise ImportError("""
  Package 'trimesh' is necessary to run generator module.
  Install using 'pip install trimesh'"""
  )

from siamrl.envs import data

def irregular(
  subdivisions=2,
  radius=0.0625,
  factor=0.1,
  seed=None,
):
  """Generates a mesh from an icosphere, by applying some randomly
    parametrized transformations.

  Args:
    subdivisions: subdivisions parameter of the icosphere
    radius: the mesh always fits within a sphere of this radius.
    factor: ratio of the maximum radius that corresponds to the 
      mode distance from vertice to center. A smaller factor allows
      greater irregularity.
    seed: seed for the random number generator, or generator to be 
      used. 
  """
  random = np.random.default_rng(seed)
  # Create icosphere mesh with max radius and given subdivisions
  mesh = creation.icosphere(
    subdivisions=subdivisions, 
    radius=factor*radius,
  )
  # Apply random vertex displacement in normal direction
  mesh.vertices += random.triangular(
    -factor*radius,
    0, 
    (1-factor)*radius, 
    (len(mesh.vertices),1)
  )*mesh.vertex_normals
  # Get the convex hull
  mesh = mesh.convex_hull
  # pylint: disable=no-member

  # Align the minimum volume bounding box
  mesh.apply_obb()

  # Apply a random scaling in the direction of the smallest extent of 
  # the bounding box. The scalling is only applied if the ratio between
  # smallest and largest extents is greater than factor.
  extents = mesh.bounding_box.extents
  direction = tuple(int(i==np.argmin(extents)) for i in range(3))
  ratio = min(extents)/max(extents)
  if ratio > factor:
    mesh.apply_transform(transformations.scale_matrix(
      factor=random.triangular(
        factor/ratio,
        1., # min(1., max(0.5, factor)/ratio),
        1., 
      ),
      # factor=random.uniform(factor/ratio, 1.),
      direction=direction,
    ))
  return mesh

def box_(
  radius=0.0625, 
  irregularity=0.2,
  extents_ratio=1/3., 
  subdivisions=2,
  beta=True,
  seed=None,
):
  random = np.random.default_rng(seed)
  mesh = creation.icosphere(subdivisions=subdivisions, radius=radius)

  if irregularity > 0:
    if beta:
      # Concentration (alpha+beta) of the Beta distributions
      k = 3/irregularity**2 - 1
      distr = lambda m: random.beta(
        m*(k-2) + 1, # alpha
        (1-m)*(k-2) + 1, # beta
      )
    else:
      distr = lambda m: stats.truncnorm.rvs(
        -m*2/irregularity,
        (1-m)*2/irregularity,
        loc=m, 
        scale=irregularity/2, 
        random_state=random
      )
  else:
    distr = lambda m: m

  if np.isscalar(extents_ratio):
    # Modes of the Beta distribution for the extents
    # (high aspect ratios are more likely)
    m = np.array([
      0,
      max(0, (0.5-extents_ratio)/(1-extents_ratio)),
      1.
    ])
    
    extents = distr(m)

    # Transform from [0, 1] to [extents_ratio, 1]
    extents = extents_ratio + (1-extents_ratio)*extents
  elif len(extents_ratio) == 3:
    # Deterministic extents
    extents = np.array(extents_ratio)/max(extents_ratio)
  else:
    raise ValueError("Invalid value {} for argument extents_ratio".format(extents_ratio))
  # Scale to the size of the inscribed cube.
  extents *= radius*np.sqrt(3)/3

  # Mode of the Beta distribution for the scaling to be aplied to each 
  # point. Corresponds to the values that place each point on the surface
  # of the box with given extents.
  m = np.min(extents/np.abs(mesh.vertices), axis=-1)
  
  scale = distr(m)

  mesh.vertices *= scale[:,np.newaxis]

  return mesh.convex_hull

def box(
  radius=0.0625, 
  irregularity=0.2,
  extents_ratio=1/3., 
  subdivisions=2,
  beta=True,
  seed=None,
):
  random = np.random.default_rng(seed)
  std = irregularity

  mesh = creation.icosphere(subdivisions=subdivisions, radius=radius)

  if np.isscalar(extents_ratio):
    # Mean extents (high aspect ratios are more likely)
    extents = np.array([
      extents_ratio,
      max(extents_ratio, 0.5),
      1.
    ])
    
    # if std:
    #   extents = stats.truncnorm.rvs(
    #       (extents_ratio-extents)*2/std,
    #       (1-extents)*2/std,
    #       loc=extents, 
    #       scale=std, 
    #       random_state=random
    #     )
  elif len(extents_ratio) == 3:
    # Deterministic extents
    extents = np.array(extents_ratio)/max(extents_ratio)
  else:
    raise ValueError("Invalid value {} for argument extents_ratio".format(extents_ratio))
  # Scale to the size of the inscribed cube.
  extents *= radius*np.sqrt(3)/3

  # Scaling factors that place each vertex on the surface
  # of the box with given extents.
  scale = np.min(extents/np.abs(mesh.vertices), axis=-1)
  mesh.vertices *= scale[:,np.newaxis]

  # Aply random noise to vertices
  if irregularity > 0:
    amax = np.argmax(np.abs(mesh.vertex_normals), axis=-1)
    dotvn = np.sum(mesh.vertices*mesh.vertex_normals, axis=-1)

    a = - mesh.vertices[np.arange(len(amax)),amax]/mesh.vertex_normals[np.arange(len(amax)),amax]
    b = 1/2*(-dotvn+np.sqrt(np.maximum(0, dotvn**2-4*(np.linalg.norm(mesh.vertices, axis=-1)**2-radius**2))))
    
    if beta:
      k = 3/irregularity**2 - 1
      m = -a/(b-a)
      n = a + (b-a)*random.beta(
        m*(k-2) + 1, # alpha
        (1-m)*(k-2) + 1, # beta
      )
    else:
      std = irregularity*radius*np.sqrt(3)/3
      n = stats.truncnorm.rvs(
        a/std,
        b/std,
        scale=std,
        random_state=random,
      )

    mesh.vertices += mesh.vertex_normals*n[:,np.newaxis]

  return mesh.convex_hull

methods = {
  'box': box,
  'irregular': irregular,
}

def generate(  
  n,
  method='box',
  align_obb=False,
  density=(2200,2600),
  directory='',
  name=None,
  seed=None,
  show=False,
  start_index=0,
  max_index=None,
  make_log=True,
  **kwargs,
):
  """Generates n pairs of files (.obj and .urdf) from meshes created
    with a given method.
  Args:
    n: number of objects to be generated.
    method: method used to generate the seed. Either a callable or the
      string identifier of one of the implemented methods.
    align_obb: whether to align the mesh's oriented bounding bos with the
      frame axis. If False, the mesh's principal inertia axes are aligned
      instead.
    density: used for the model mass and inertia. Either a scalar
      or a tuple with the limits of a uniform random distribution
    directory: path to the location where to save the models.
    name: name of the models (to be suffixed with an index). If None, only
      the index is used to name the files.
    seed: seed of the random generator.
    show: whether to show a visualization of each model.
    start_index: index of the first generated object. Used to generate
      objects on a directory without overwriting previous ones. 
    max_index: maximum index expected in the directory. Used to correctly
    set the number of leading zeros of the index in the name of the files
    if more files are expected to be generated outside this function call.
    If None, the maximum index generated in this call is used.
  """
  # Set method
  if isinstance(method, str):
    method = methods[method]
  elif not callable(method):
    raise TypeError("method must be callable or a string.")

  # Load the template urdf
  with data.open('template.urdf') as f:
      urdf = f.read()
  # Create directory if needed
  if not os.path.isdir(directory):
    os.makedirs(directory)

  # Set logging
  if make_log:      
    log_name = os.path.join(
      directory, 
      name+'.csv' if name else 'log.csv'
    )
    if start_index and os.path.isfile(log_name):
      logf = open(log_name, 'a')
    else:
      logf = open(log_name, 'w')
      logf.write('Name,Volume,Rectangularity,AspectRatio\n')
  else:
    logf = None

  max_index = max(max_index or n+start_index-1,1)
  name_format = '{:0'+str(int(np.log10(max_index))+1)+'}'
  if isinstance(name, str):
    name_format = '{}_{}'.format(name, name_format)
  name = lambda i: name_format.format(i)

  # Create random number generator.
  random = np.random.default_rng(seed)

  for i in range(start_index, start_index+n):
    # Create mesh
    mesh = method(seed=random, **kwargs)
    # Align the principal axes with (Z,Y,X).
    if align_obb:
      mesh.apply_obb()
    else:
      mesh.apply_transform(mesh.principal_inertia_transform)
    mesh.apply_transform(transformations.rotation_matrix(
      angle=np.pi/2, 
      direction=[0,1,0]
    ))
    # Correct residual translation due to rotation
    mesh.apply_translation(-mesh.center_mass) 

    mesh.process()
    assert mesh.is_watertight

    # Set density
    if np.isscalar(density):
      mesh.density = density
    else:
      mesh.density = random.uniform(density[0], density[1])

    if show:
      mesh.show()

    name_i = name(i)

    # Log shape metrics
    if logf is not None:
      extents = mesh.bounding_box_oriented.extents
      logf.write('{},{},{},{}\n'.format(
        name_i,
        mesh.volume,
        mesh.volume/mesh.bounding_box_oriented.volume,
        max(extents)/min(extents),
      ))

    # Export mesh to .obj file
    fname = os.path.join(directory, name_i)
    with open(fname+'.obj', 'w') as f:
      mesh.export(file_obj=f, file_type='obj')
    # Create .urdf file from template with mesh specs
    with open(fname+'.urdf', 'w') as f:
      f.write(urdf.format(
          name_i, 
          0.6,
          mesh.mass,
          mesh.moment_inertia[0,0],
          mesh.moment_inertia[0,1],
          mesh.moment_inertia[0,2],
          mesh.moment_inertia[1,1],
          mesh.moment_inertia[1,2],
          mesh.moment_inertia[2,2],
          name_i+'.obj',
          name_i+'.obj'))

if __name__ == '__main__':
  # Default args
  n,directory,split,seed,show, verbose, irregularity, align_obb, extents = (
    10000, 
    data.path('generated'), 
    0.05, 
    None, 
    False,
    True,
    np.arange(0.1,0.6,0.05),
    False,
    1/3.,
  )
  # Parse arguments
  argv = sys.argv[:0:-1]
  while argv:
    arg=argv.pop()
    if arg == '-d':
      directory = argv.pop()
    elif arg == '--split':
      split = float(argv.pop())
    elif arg == '--seed':
      seed = int(argv.pop())
    elif arg == '--show':
      show = True
    elif arg == '--param':
      params = argv.pop().split(',')
      if len(params) == 1:
        irregularity = [float(params[0])]
      elif len(params) == 2:
        start = float(params[0])
        stop = float(params[1])
        irregularity = np.arange(start, stop, (start-stop)/10)
      elif len(params) == 3:
        irregularity = np.arange(float(params[0]), float(params[1]), float(params[2]))
      elif len(params) == 4:
        if params[-1] == 'log':
          irregularity = np.logspace(
            np.log10(float(params[0])), 
            np.log10(float(params[1])), 
            num=int(params[2]),
          )
        else:
          irregularity = np.logspace(
            float(params[0]), 
            float(params[1]), 
            num=int(params[2]),
            base=float(params[3]),
          )
      else:
        raise ValueError('Invalid value for --param')
    elif arg == '--extents':
      extents = [float(i) for i in argv.pop().split(',')]
      if len(extents) == 1:
        extents = extents[0]
    elif arg == '--obb':
      align_obb = True
    elif arg == '-q':
      verbose = False
    else:
      n = int(arg)
  
  n_i = int(n*(1-split)//len(irregularity))
  n_train = n_i*len(irregularity)

  n_test = n - n_train
  seed_test = seed+1 if seed is not None else None

  if verbose:
    from datetime import datetime
    import time

    print('{}: Generating {} objects.'.format(datetime.now(), n))
    itime = time.perf_counter()

  for i, irr in enumerate(irregularity):
    generate(
      n=n_i, 
      name='{}'.format(int(round(100*irr))), 
      align_obb=align_obb, 
      directory=directory, 
      seed=seed, 
      show=show, 
      irregularity=irr,
      extents_ratio=extents,
    )
    if verbose:
      print('{}: {}/{} done.'.format(datetime.now(), (i+1)*n_i, n))

  if n_test:
    generate(
      n=n_test, 
      name='{}'.format(int(100*irr)), 
      align_obb=align_obb, 
      directory=os.path.join(directory, 'test'), 
      seed=seed_test, 
      show=show, 
      irregularity=np.max(irregularity),
      extents_ratio=extents,
    )

  etime = time.perf_counter() - itime
  if verbose:
    print('{}: {}/{} done. Total elapsed time: {} s ({} s/object)'.format(
      datetime.now(),
      n,
      n,
      etime, 
      etime/n,
    ))

# def from_box(
#   n,
#   mode_extents=(0.1,0.075,0.05),
#   deformation=True,
#   density=(2200,2600),
#   directory='.',
#   name='object',
#   seed=None,
#   show=False,
#   start_index=0
# ):
#   """Generates objects by randomly displacing the vertices of a box.
#     Two files (.obj and .urdf) are saved for each model.
#   Args:
#     n: number of objects to be generated.
#     mode_extents: modes of (each dimention of) the triangular
#       distribution from which the extents of the box are randomly
#       taken.
#     deformation: whether to apply random displacements to the 
#       vertices
#     density: used for the model mass and inertia. Either a scalar
#       or a tuple with the limits of a uniform random distribution
#     directory: path to the location where to save the models.
#     name: name of the models (to be suffixed with an index).
#     seed: seed of the random generator.
#     show: whether to show a visualization of each model.
#     start_index: index of the first generated object. Used to generate
#       objects on a directory without overwriting previous ones. 
#   """
#   # Load the template urdf
#   with data.open('template.urdf') as f:
#       urdf = f.read()

#   index_format = '%0'+str(int(np.log10(n-1))+1)+'d'

#   mode_extents = np.array(mode_extents)
#   random.seed(seed)
#   for i in range(start_index, start_index+n):
#     # Take extents from a random triangular distribution with given mode
#     extents = random.triangular(
#       0.6*mode_extents, 
#       mode_extents, 
#       1.2*mode_extents, 
#       3
#     )
#     # Create a box mesh with these extents
#     mesh = creation.box(extents)
#     if deformation:
#       # Random vertex displacement
#       displ = random.triangular(
#         -0.15*extents, 
#         [0,0,0], 
#         0.15*extents, 
#         (mesh.vertices.shape[0], 3)
#       )
#       mesh.vertices += displ
#       # Center the mesh
#       mesh.apply_translation(-mesh.center_mass)

#       mesh.process()
#     assert mesh.is_watertight
#     if show:
#       mesh.show()

#     # Set density
#     if len(density) > 1:
#       mesh.density = float(random.randint(density[0], density[1]))
#     else:
#       mesh.density = density

#     name_i = name+index_format%i
#     # Export mesh to .obj file
#     filename = os.path.join(directory, name_i+'.obj')
#     with open(filename, 'w') as f:
#       mesh.export(file_obj=f, file_type='obj')
#     # Create .urdf file from template with mesh specs
#     filename = os.path.join(directory, name_i+'.urdf')
#     with open(filename, 'w') as f:
#       f.write(urdf.format(
#           name_i, 
#           0.6,
#           mesh.mass,
#           mesh.moment_inertia[0,0],
#           mesh.moment_inertia[0,1],
#           mesh.moment_inertia[0,2],
#           mesh.moment_inertia[1,1],
#           mesh.moment_inertia[1,2],
#           mesh.moment_inertia[2,2],
#           name_i+'.obj',
#           name_i+'.obj'))

# def from_icosphere(
#   n,
#   subdivisions=1,
#   max_radius=0.0625,
#   convex=False,
#   density=(2200,2600),
#   directory='',
#   name='object',
#   seed=None,
#   show=False,
#   start_index=0
# ):
#   """Generates objects from an icosphere, by applying some randomly
#     parametrized transformations (rotation, vertices displacement along 
#     normal direction, directional scaling). Two files (.obj and .urdf) 
#     are saved for each model.
#   Args:
#     n: number of objects to be generated.
#     subdivisions: subdivisions parameter of the icosphere
#     max_radius: all objects fit within a sphere of this radius.
#     convex: whether the meshes must be convex, if true the convex
#       hull is used as the mesh
#     density: used for the model mass and inertia. Either a scalar
#       or a tuple with the limits of a uniform random distribution
#     directory: path to the location where to save the models.
#     name: name of the models (to be suffixed with an index).
#     seed: seed of the random generator.
#     show: whether to show a visualization of each model.
#     start_index: index of the first generated object. Used to generate
#       objects on a directory without overwriting previous ones. 
#   """
#   # Load the template urdf
#   with data.open('template.urdf') as f:
#       urdf = f.read()

#   index_format = '%0'+str(int(np.log10(n-1))+1)+'d'
#   random.seed(seed)
#   for i in range(start_index, start_index+n):
#     # Create icosphere mesh with max radius and given subdivisions
#     mesh = creation.icosphere(subdivisions=subdivisions, radius=max_radius)
#     #Apply random vertex displacement in inwards normal direction
#     displ = random.triangular(
#       -0.25*max_radius, 
#       0, 
#       0, 
#       (mesh.vertex_normals.shape[0],1)
#     )
#     displ = displ*mesh.vertex_normals
#     mesh.vertices += displ
#     #Apply three random scalings
#     for _ in range(3):
#       mesh.apply_transform(random_scale_matrix(
#         min_factor=0.1**(1/3),
#         max_factor=0.75**(1/3)
#       ))
#     # If object must be convex, use the convex hull
#     if convex:
#       mesh = mesh.convex_hull

#     # Center and align mesh
#     mesh.apply_translation(-mesh.center_mass)
#     mesh.apply_transform(mesh.principal_inertia_transform)
#     mesh.apply_transform(transformations.rotation_matrix(
#       angle=np.pi/2, 
#       direction=[0,1,0]
#     ))
#     # Do this again because rotations introduce residual cm translation
#     mesh.apply_translation(-mesh.center_mass)

#     mesh.process()
#     assert mesh.is_watertight
#     if show:
#       mesh.show()

#     # Set density
#     if len(density) > 1:
#       mesh.density = float(random.randint(density[0], density[1]))
#     else:
#       mesh.density = density

#     name_i = name+index_format%i
#     # Export mesh to .obj file
#     filename = os.path.join(directory, name_i+'.obj')
#     with open(filename, 'w') as f:
#       mesh.export(file_obj=f, file_type='obj')
#     # Create .urdf file from template with mesh specs
#     filename = os.path.join(directory, name_i+'.urdf')
#     with open(filename, 'w') as f:
#       f.write(urdf.format(
#           name_i, 
#           0.6,
#           mesh.mass,
#           mesh.moment_inertia[0,0],
#           mesh.moment_inertia[0,1],
#           mesh.moment_inertia[0,2],
#           mesh.moment_inertia[1,1],
#           mesh.moment_inertia[1,2],
#           mesh.moment_inertia[2,2],
#           name_i+'.obj',
#           name_i+'.obj'))

# def random_scale_matrix(min_factor=0.5, max_factor=1):
#   """Compute a scale matrix for a random factor and direction.
#   Args:
#     min_factor: minimum allowed scaling factor.
#     max_factor: maximum allowed scaling factor.
#   Returns:
#     The scale matrix.
#   """
#   # Randomly choose a factor between minimum and maximum.
#   factor = random.rand()*(max_factor-min_factor)+min_factor
#   # Randomly choose a direction (only upper hemisphere considered,
#   # because symmetric is equivalent)
#   theta = random.rand()*2*np.pi
#   phi = random.rand()*np.pi/2
#   direction = np.array([
#     np.cos(phi)*np.cos(theta),
#     np.cos(phi)*np.sin(theta),
#     np.sin(phi)
#   ])
#   return transformations.scale_matrix(factor=factor, direction=direction)