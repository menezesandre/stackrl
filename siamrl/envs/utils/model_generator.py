import sys
import os

import trimesh
from trimesh.creation import icosphere, box
from trimesh.transformations import random_rotation_matrix, scale_matrix, rotation_matrix

import numpy as np
from numpy import random

import siamrl as s

def fromBox(n=5,
            mode_extents=(0.1,0.075,0.05),
            deformation=True,
            density=(2200,2600),
            directory='',
            name='object',
            seed=None,
            show=False):
  """
  Generates objects by randomly displacing the vertices of a box.
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
    show: whether to show a visualization of each model
  """
  # Load the template urdf
  filename = os.path.join(s.envs.getDataPath(), 'template.urdf')
  with open(filename, 'r') as f:
      urdf = f.read()

  index_format = '%0'+str(int(np.log10(n-1))+1)+'d'

  mode_extents = np.array(mode_extents)
  random.seed(seed)
  for i in range(n):
    # Take extents from a random triangular distribution with given mode
    extents = random.triangular(0.6*mode_extents, mode_extents, 1.2*mode_extents, 3)
    # Create a box mesh with these extents
    mesh = box(extents)
    if deformation:
      # Random vertex displacement
      displ = random.triangular(-0.15*extents, [0,0,0], 0.15*extents, (mesh.vertices.shape[0],3))
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

def fromIcosphere(n=5,
                  subdivisions=1,
                  mode_radius=0.05,
                  convex=False,
                  density=(2200,2600),
                  directory='',
                  name='object',
                  seed=None,
                  show=False):
  """
  Generates objects from an icosphere, by applying some randomly
    parametrized transformations (rotation, vertices displacement
    along normal direction, directional scaling).
  Two files (.obj and .urdf) are saved for each model.
  
  Args:
    n: number of objects to be generated.
    subdivisions: subdivisions parameter of the icosphere
    mode_radius: mode of the triangular random distribution from
      which the radius of the icosphere is taken.
    convex: whether the meshes must be convex, if true the convex
      hull is used as the mesh
    density: used for the model mass and inertia. Either a scalar
      or a tuple with the limits of a uniform random distribution
    directory: path to the location where to save the models.
    name: name of the models (to be suffixed with an index).
    seed: seed of the random generator.
    show: whether to show a visualization of each model
  """
  # Load the template urdf
  filename = os.path.join(s.envs.getDataPath(), 'template.urdf')
  with open(filename, 'r') as f:
      urdf = f.read()

  index_format = '%0'+str(int(np.log10(n-1))+1)+'d'

  random.seed(seed)
  for i in range(n):
    # Take radius from random triangular distribution with given mode
    radius = random.triangular(0.4*mode_radius, mode_radius, 1.2*mode_radius)
    # Create icosphere mesh with this radius and given subdivisions
    mesh = icosphere(subdivisions=subdivisions, radius=radius)
    # Apply random rotation
    mesh.apply_transform(random_rotation_matrix())
    #Apply random vertex displacement in normal direction
    displ = random.triangular(-0.2*radius, 0, 0.2*radius, (mesh.vertex_normals.shape[0],1))
    displ = displ*mesh.vertex_normals
    mesh.vertices += displ
    #Apply two random scalings
    for _ in range(2):
      mesh.apply_transform(random_scale_matrix())
    # If object must be convex, use the convex hull
    if convex:
      mesh = mesh.convex_hull

    # Center and align mesh
    mesh.apply_translation(-mesh.center_mass)
    mesh.apply_transform(mesh.principal_inertia_transform)
    mesh.apply_transform(rotation_matrix(angle=np.pi/2, direction=[0,1,0]))
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

def random_scale_matrix(min_factor=0.5):
  """
  Returns the matrix of a scaling by a random factor on a random
    direction

  Args:
    min_factor: minimum allowed scaling factor

  Returns:
    The scale matrix
  """
  assert min_factor < 1
  # Randomly choose a factor between minimum and 1
  factor = random.rand()*(1-min_factor)+min_factor
  # Randomly choose a direction (only upper hemisphere considered,
  # because symmetric is equivalent)
  theta = random.rand()*2*np.pi
  phi = random.rand()*np.pi/2
  direction = np.array([
    np.cos(phi)*np.cos(theta),
    np.cos(phi)*np.sin(theta),
    np.sin(phi)])
  return scale_matrix(factor=factor, direction=direction)
