"""
Run this file to (re)generate the models in 'generated' folder
"""
import os
import sys

import numpy as np
from numpy import random
try:
  from trimesh import creation
  from trimesh import transformations
except ImportError:
  raise ImportError("""
  Package 'trimesh' is necessary to run generator module.
  Install using 'pip install trimesh'"""
  )

from siamrl.envs import data

def create(
  n,
  subdivisions=2,
  max_radius=0.0625,
  factor=0.2,
  density=(2200,2600),
  directory='',
  name='object',
  seed=None,
  show=False,
  start_index=0
):
  """Generates objects from an icosphere, by applying some randomly
    parametrized transformations (rotation, vertices displacement along 
    normal direction, directional scaling). Two files (.obj and .urdf) 
    are saved for each model.
  Args:
    n: number of objects to be generated.
    subdivisions: subdivisions parameter of the icosphere
    max_radius: all objects fit within a sphere of this radius.
    convex: whether the meshes must be convex, if true the convex
      hull is used as the mesh
    density: used for the model mass and inertia. Either a scalar
      or a tuple with the limits of a uniform random distribution
    directory: path to the location where to save the models.
    name: name of the models (to be suffixed with an index).
    seed: seed of the random generator.
    show: whether to show a visualization of each model.
    start_index: index of the first generated object. Used to generate
      objects on a directory without overwriting previous ones. 
  """
  # Load the template urdf
  with data.open('template.urdf') as f:
      urdf = f.read()

  index_format = '%0'+str(int(np.log10(n-1))+1)+'d'

  random.seed(seed)
  for i in range(start_index, start_index+n):
    # Create icosphere mesh with max radius and given subdivisions
    mesh = creation.icosphere(
      subdivisions=subdivisions, 
      radius=factor*max_radius,
    )
    # Apply random vertex displacement in normal direction
    mesh.vertices += np.random.triangular(
      -factor*max_radius,
      0, 
      (1-factor)*max_radius, 
      (len(mesh.vertices),1)
    )*mesh.vertex_normals
    # Get the convex hull
    mesh = mesh.convex_hull
    # pylint: disable=no-member
    bb = mesh.bounding_box_oriented

    # Center mesh and align the oriented bounding box
    mesh.apply_translation(-mesh.center_mass)
    mesh.apply_transform(bb.principal_inertia_transform)

    # Apply a random scaling in direction of the smaller extent of the 
    # bounding box. Assert the scaling makes the ratio between the larger
    # and smaller extents less or equal to 0.5, and no less than factor.
    max_f = min(1., 0.5*max(bb.extents)/min(bb.extents))
    min_f = min(1., factor*max(bb.extents)/min(bb.extents))
    if min_f == max_f:
      f = min_f
    else:
      f = np.random.triangular(
        min_f,
        max(min(1-factor, max_f), min_f),
        max_f, 
      )

    mesh.apply_transform(transformations.scale_matrix(
      factor=f,
      direction=(1,0,0),
    ))

    # Realign and rotate to make z the principal inertia axis.
    mesh.apply_translation(-mesh.center_mass)
    mesh.apply_transform(mesh.principal_inertia_transform)
    mesh.apply_transform(transformations.rotation_matrix(
      angle=np.pi/2, 
      direction=[0,1,0]
    ))
    # Repeat to correct residual translation due to rotations
    mesh.apply_translation(-mesh.center_mass) 

    mesh.process()
    assert mesh.is_watertight
    if show:
      mesh.show()

    # Set density
    if len(density) > 1:
      mesh.density = float(random.randint(density[0], density[1]))
    else:
      mesh.density = density

    name_i = name+index_format%i
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

def from_box(
  n,
  mode_extents=(0.1,0.075,0.05),
  deformation=True,
  density=(2200,2600),
  directory='.',
  name='object',
  seed=None,
  show=False,
  start_index=0
):
  """Generates objects by randomly displacing the vertices of a box.
    Two files (.obj and .urdf) are saved for each model.
  Args:
    n: number of objects to be generated.
    mode_extents: modes of (each dimention of) the triangular
      distribution from which the extents of the box are randomly
      taken.
    deformation: whether to apply random displacements to the 
      vertices
    density: used for the model mass and inertia. Either a scalar
      or a tuple with the limits of a uniform random distribution
    directory: path to the location where to save the models.
    name: name of the models (to be suffixed with an index).
    seed: seed of the random generator.
    show: whether to show a visualization of each model.
    start_index: index of the first generated object. Used to generate
      objects on a directory without overwriting previous ones. 
  """
  # Load the template urdf
  with data.open('template.urdf') as f:
      urdf = f.read()

  index_format = '%0'+str(int(np.log10(n-1))+1)+'d'

  mode_extents = np.array(mode_extents)
  random.seed(seed)
  for i in range(start_index, start_index+n):
    # Take extents from a random triangular distribution with given mode
    extents = random.triangular(
      0.6*mode_extents, 
      mode_extents, 
      1.2*mode_extents, 
      3
    )
    # Create a box mesh with these extents
    mesh = creation.box(extents)
    if deformation:
      # Random vertex displacement
      displ = random.triangular(
        -0.15*extents, 
        [0,0,0], 
        0.15*extents, 
        (mesh.vertices.shape[0], 3)
      )
      mesh.vertices += displ
      # Center the mesh
      mesh.apply_translation(-mesh.center_mass)

      mesh.process()
    assert mesh.is_watertight
    if show:
      mesh.show()

    # Set density
    if len(density) > 1:
      mesh.density = float(random.randint(density[0], density[1]))
    else:
      mesh.density = density

    name_i = name+index_format%i
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

def from_icosphere(
  n,
  subdivisions=1,
  max_radius=0.0625,
  convex=False,
  density=(2200,2600),
  directory='',
  name='object',
  seed=None,
  show=False,
  start_index=0
):
  """Generates objects from an icosphere, by applying some randomly
    parametrized transformations (rotation, vertices displacement along 
    normal direction, directional scaling). Two files (.obj and .urdf) 
    are saved for each model.
  Args:
    n: number of objects to be generated.
    subdivisions: subdivisions parameter of the icosphere
    max_radius: all objects fit within a sphere of this radius.
    convex: whether the meshes must be convex, if true the convex
      hull is used as the mesh
    density: used for the model mass and inertia. Either a scalar
      or a tuple with the limits of a uniform random distribution
    directory: path to the location where to save the models.
    name: name of the models (to be suffixed with an index).
    seed: seed of the random generator.
    show: whether to show a visualization of each model.
    start_index: index of the first generated object. Used to generate
      objects on a directory without overwriting previous ones. 
  """
  # Load the template urdf
  with data.open('template.urdf') as f:
      urdf = f.read()

  index_format = '%0'+str(int(np.log10(n-1))+1)+'d'
  random.seed(seed)
  for i in range(start_index, start_index+n):
    # Create icosphere mesh with max radius and given subdivisions
    mesh = creation.icosphere(subdivisions=subdivisions, radius=max_radius)
    #Apply random vertex displacement in inwards normal direction
    displ = random.triangular(
      -0.25*max_radius, 
      0, 
      0, 
      (mesh.vertex_normals.shape[0],1)
    )
    displ = displ*mesh.vertex_normals
    mesh.vertices += displ
    #Apply three random scalings
    for _ in range(3):
      mesh.apply_transform(random_scale_matrix(
        min_factor=0.1**(1/3),
        max_factor=0.75**(1/3)
      ))
    # If object must be convex, use the convex hull
    if convex:
      mesh = mesh.convex_hull

    # Center and align mesh
    mesh.apply_translation(-mesh.center_mass)
    mesh.apply_transform(mesh.principal_inertia_transform)
    mesh.apply_transform(transformations.rotation_matrix(
      angle=np.pi/2, 
      direction=[0,1,0]
    ))
    # Do this again because rotations introduce residual cm translation
    mesh.apply_translation(-mesh.center_mass)

    mesh.process()
    assert mesh.is_watertight
    if show:
      mesh.show()

    # Set density
    if len(density) > 1:
      mesh.density = float(random.randint(density[0], density[1]))
    else:
      mesh.density = density

    name_i = name+index_format%i
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

def random_scale_matrix(min_factor=0.5, max_factor=1):
  """Compute a scale matrix for a random factor and direction.
  Args:
    min_factor: minimum allowed scaling factor.
    max_factor: maximum allowed scaling factor.
  Returns:
    The scale matrix.
  """
  # Randomly choose a factor between minimum and maximum.
  factor = random.rand()*(max_factor-min_factor)+min_factor
  # Randomly choose a direction (only upper hemisphere considered,
  # because symmetric is equivalent)
  theta = random.rand()*2*np.pi
  phi = random.rand()*np.pi/2
  direction = np.array([
    np.cos(phi)*np.cos(theta),
    np.cos(phi)*np.sin(theta),
    np.sin(phi)
  ])
  return transformations.scale_matrix(factor=factor, direction=direction)

if __name__ == '__main__':
  # Default args
  n,directory,split,seed = 5000, data.path('generated'), 0.1, None
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
    else:
      n = int(arg)
  
  n_test = int(n*split)
  n_train = n - n_test
  seed_test = seed+1 if seed is not None else None
  create(n=n_train, directory=directory, name='train', seed=seed)
  create(n=n_test, directory=directory, name='test', seed=seed_test)
  