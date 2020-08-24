"""
Run this file to (re)generate the models in 'generated' folder
"""
import os
import sys

import numpy as np
from scipy import stats
from trimesh import creation
from trimesh import transformations

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

def box(
  radius=0.0625,
  irregularity=0.,
  extents=None,
  subdivisions=3,
  seed=None,
):
  """Generate an irregular mesh from a box.
  Args:
    radius: maximum radius of the object's bounding sphere.
    irregularity: irregularity measure, between 0 and 1.
    extents: ratio between extents. (Adimensional, scaled by radius.)
    subdivisions: number of subdivisions to perform to the box mesh. (A 
      subdivision replaces each face with four smaller faces.)
  """
  random = np.random.default_rng(seed)

  extents = extents or (1,1/2,1/3)
  extents = np.array(extents)*2*radius/np.linalg.norm(extents)
  mesh = creation.box(extents=extents)
  if irregularity > 0:
    # Add noise
    mesh.vertices += stats.truncnorm.rvs(
      -1/irregularity,
      1/irregularity,
      loc=0, 
      scale=irregularity*radius, 
      size=mesh.vertices.shape,
      random_state=random,
    )
  for i in range(subdivisions):
    nv = mesh.vertices.shape[0]
    mesh = mesh.subdivide()
    if irregularity > 0:
      # Add noise to the new vertices
      mesh.vertices[nv:] += stats.truncnorm.rvs(
        -1/irregularity,
        1/irregularity,
        loc=0, 
        scale=irregularity*radius*2**(-(i+1)), 
        size=mesh.vertices[nv:].shape,
        random_state=random,
      )
    
  mesh = mesh.convex_hull
  mesh.apply_translation(-mesh.center_mass)  # pylint: disable=no-member
  factor = 2*radius/mesh.bounding_sphere.extents[0]  # pylint: disable=no-member
  if factor < 1:
    mesh.apply_transform(transformations.scale_matrix(factor))  # pylint: disable=no-member
  return mesh

methods = {
  'box': box,
  'irregular': irregular,
}

def generate(
  n,
  method=None,
  align_pai=False,
  density=(2200,2600),
  directory='.',
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
    align_pia: whether to align the mesh's principal axes of inertia with 
      the frame axes. If False, the mesh's oriented bounding box is 
      aligned instead.
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
  if method is None:
    method = box
  if isinstance(method, str):
    method = methods[method]
  elif not callable(method):
    raise TypeError("method must be callable or a string.")

  # Load the template urdf
  with open(os.path.join(os.path.dirname(__file__), 'template.urdf')) as f:
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
      logf.write('Name,Volume,Rectangularity,AspectRatio,NumVertices\n')
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
    if align_pai:
      mesh.apply_transform(mesh.principal_inertia_transform)
    else:
      mesh.apply_obb()
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
      grayscale = 0.5
    else:
      mesh.density = random.uniform(density[0], density[1])
      # Set color according to density
      grayscale = 0.6-0.2*(mesh.density-density[0])/(density[1]-density[0])

    if show:
      mesh.show()

    name_i = name(i)

    # Log shape metrics
    if logf is not None:
      extents = mesh.bounding_box_oriented.extents
      logf.write('{},{},{},{},{}\n'.format(
        name_i,
        mesh.volume,
        mesh.volume/mesh.bounding_box_oriented.volume,
        max(extents)/min(extents),
        len(mesh.vertices),
      ))

    # Export mesh to .obj file
    fname = os.path.join(directory, name_i)
    with open(fname+'.obj', 'w') as f:
      mesh.export(file_obj=f, file_type='obj')
    # Create .urdf file from template with mesh specs
    with open(fname+'.urdf', 'w') as f:
      f.write(urdf.format(
        name=name_i,
        friction=0.6,
        mass=mesh.mass,
        ixx=mesh.moment_inertia[0,0],
        ixy=mesh.moment_inertia[0,1],
        ixz=mesh.moment_inertia[0,2],
        iyy=mesh.moment_inertia[1,1],
        iyz=mesh.moment_inertia[1,2],
        izz=mesh.moment_inertia[2,2],
        mesh=name_i+'.obj',
        r=grayscale, 
        g=grayscale, 
        b=grayscale, 
        a=1.,
      ))
