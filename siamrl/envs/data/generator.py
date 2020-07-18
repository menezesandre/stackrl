"""
Run this file to (re)generate the models in 'generated' folder
"""
import os
import sys

import numpy as np
try:
  from trimesh import creation
  from trimesh import transformations
except ImportError:
  raise ImportError("""
  Package 'trimesh' is necessary to run generator module.
  Install using 'pip install trimesh'"""
  )

from siamrl.envs import data

NAME_FROM_RECTANGULARITY = 0
NAME_FROM_ASPECTRATIO = 1
NAME_FROM_SHAPE = 2

def irregular(
  subdivisions=2,
  radius=0.0625,
  factor=0.1,
  random=None
):
  """Generates a mesh from an icosphere, by applying some randomly
    parametrized transformations.

  Args:
    subdivisions: subdivisions parameter of the icosphere
    radius: the mesh always fits within a sphere of this radius.
    factor: ratio of the maximum radius that corresponds to the 
      mode distance from vertice to center. A smaller factor allows
      greater irregularity.
    random: random number generator to be used. If None, np.random
      is used.
  """
  random = random or np.random
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

def box(
  radius=0.0625, 
  irregularity=0., 
  subdivisions=None,
  random=None,
):
  if subdivisions is None:
    subdivisions = 2 if irregularity < 0.5 else 3

  random = random or np.random

  mesh = creation.icosphere(subdivisions=subdivisions, radius=radius)

  extents = random.uniform(1e-1, 1., size=(3,))*radius*np.sqrt(3)/3
  r = np.min(extents/np.abs(mesh.vertices), axis=-1)

  if irregularity > 0.:
    r = random.triangular(
        (1-irregularity)*r,
        r,
        irregularity + (1-irregularity)*r,
     )
  mesh.vertices *= r[:,np.newaxis]

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
  name=NAME_FROM_SHAPE,
  seed=None,
  show=False,
  start_index=0,
  max_index=None,
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
    name: name of the models (to be suffixed with an index). Either a
      string or an int for one of the special cases:
        NAME_FROM_RECTANGULARITY (0); 
        NAME_FROM_ASPECT_RATIO (1);
        NAME_FROM_SHAPE (2).
      In this case, the name of the files is based on a measure of the
      obsect's shape: (3D) rectangularity, aspect ratio (between smallest 
      and largest extents), or both. If None, only the index is used to
      name the files.
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

  # Set file naming
  use_rectangularity = name in [NAME_FROM_RECTANGULARITY, NAME_FROM_SHAPE]
  use_aspectratio = name in [NAME_FROM_ASPECTRATIO, NAME_FROM_SHAPE]

  n = int(n)
  start_index = int(start_index)
  max_index = max(max_index or n+start_index-1,1)
  name_format = '{:0'+str(int(np.log10(max_index))+1)+'}'
  if isinstance(name, str):
    name_format = name+name_format
  name = lambda i: name_format.format(i)

  # Create random number generator.
  random = np.random.RandomState(seed)

  for i in range(start_index, start_index+n):
    # Create mesh
    mesh = method(random=random, **kwargs)
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
    # assert mesh.is_watertight

    # Set density
    if np.isscalar(density):
      mesh.density = density
    else:
      mesh.density = random.uniform(density[0], density[1])

    if show:
      mesh.show()

    name_i = ''
    if use_rectangularity:
      rectangularity = mesh.volume/mesh.bounding_box_oriented.volume
      name_i += '{:03}_'.format(int(rectangularity*100))
    if use_aspectratio:
      extents = mesh.bounding_box.extents
      aspectratio = max(extents)/min(extents)
      name_i += '{:03}_'.format(int(aspectratio*10))

    name_i += name(i)
    # Export mesh to .obj file
    filename = os.path.join(directory, name_i+'.obj')
    with open(filename, 'w') as f:
      mesh.export(file_obj=f, file_type='obj')
    # Create .urdf file from template with mesh specs
    filename = os.path.join(directory, name_i+'.urdf')
    with open(filename, 'w') as f:
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
  n,directory,split,seed,show, verbose = 10000, data.path('generated'), 0.01, None, False,True
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
    elif arg == '-q':
      verbose = False
    else:
      n = int(arg)
  
  irregularity = np.arange(0.,1.,0.05)
  n_i = n*(1-split)//len(irregularity)
  n_train = n_i*len(irregularity)

  n_test = n - n_train
  seed_test = seed+1 if seed is not None else None

  if verbose:
    from datetime import datetime
    import time

    itime = time.time()
    print('{}: Generating {} objects.'.format(datetime.now(), n))

  for i, irr in enumerate(irregularity):
    generate(n=n_i, align_obb=irr<0.5, start_index=i*n_i, max_index=n_train-1, directory=directory, seed=seed, show=show, irregularity=irr)
    if verbose:
      print('{}: {}/{} done.'.format(datetime.now(), (i+1)*n_i, n))

  if n_test:
    generate(n=n_test, directory=os.path.join(directory, 'test'), seed=seed_test, show=show, irregularity=np.max(irregularity))

  if verbose:
    print('{}: {}/{} done. Total elapsed time: {}s'.format(datetime.now(), n, n, time.time()-itime))


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