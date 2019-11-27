import os

import trimesh
from trimesh.creation import icosphere
from trimesh.transformations import random_rotation_matrix, scale_matrix, rotation_matrix

import numpy as np
from numpy import random

import siamrl as s

def generateRocks(N=5, subdivisions=1, mode_radius=0.05, convex=False, directory='', seed=None):
  filename = os.path.join(s.envs.getDataPath(), 'urdftemplate.txt')
  with open(filename, 'r') as f:
      urdf = f.read()

  random.seed(seed)
  for i in range(N):
    radius = random.triangular(0.4*mode_radius, mode_radius, 1.2*mode_radius)
    mesh = icosphere(subdivisions=subdivisions, radius=radius)
    #Random rotation
    mesh.apply_transform(random_rotation_matrix())
    #Random vertex displacement in normal direction
    displ = random.triangular(-0.2*radius, 0, 0.2*radius, (mesh.vertex_normals.shape[0],1))
    displ = displ*mesh.vertex_normals
    mesh.vertices += displ
    #Random scalings in random directions
    for _ in range(2):
      mesh.apply_transform(random_scale_matrix())
    #If object must be convex, use the convex hull
    if convex:
      mesh = mesh.convex_hull

    assert mesh.is_watertight
    mesh.apply_translation(-mesh.center_mass)
    mesh.apply_transform(mesh.principal_inertia_transform)
    mesh.apply_transform(rotation_matrix(angle=np.pi/2, direction=[0,1,0]))
    mesh.process()
    mesh.density = float(random.randint(2200, 2600))

    mesh.apply_translation(-mesh.center_mass)
    filename = os.path.join(directory, 'rock%d.obj'%i)
    with open(filename, 'w') as f:
      mesh.export(file_obj=f, file_type='obj')

    filename = os.path.join(directory, 'rock%d.urdf'%i)
    with open(filename, 'w') as f:
      f.write(urdf.format(
          'rock%d'%i, 
          0.6,
          mesh.mass,
          mesh.moment_inertia[0,0],
          mesh.moment_inertia[0,1],
          mesh.moment_inertia[0,2],
          mesh.moment_inertia[1,1],
          mesh.moment_inertia[1,2],
          mesh.moment_inertia[2,2],
          'rock%d.obj'%i,
          'rock%d.obj'%i))

def random_scale_matrix(min_factor=0.5):
  assert min_factor <= 1
  factor = random.triangular(min_factor, min_factor, 1)
  theta = random.rand()*2*np.pi
  phi = random.triangular(0, np.pi/2, np.pi/2)
  direction = np.array([
    np.cos(phi)*np.cos(theta),
    np.cos(phi)*np.sin(theta),
    np.sin(phi)])
  return scale_matrix(factor=factor, direction=direction)
