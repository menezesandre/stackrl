"""
Import this file to (re)generate the models in 'generated' folder
"""
from siamrl.envs.utils import generate
from siamrl.envs import getDataPath
import os

path = os.path.join(getDataPath(),'generated')

# Catch the exception if trimesh is not instaled
try:
  generate.fromBox(n=200, deformation=False, directory=path, name='bp')
  generate.fromBox(n=400, directory=path, name='b')
  generate.fromIcosphere(n=400, directory=path, name='i')
  generate.fromIcosphere(n=4000, convex=True, directory=path, name='ic')
except ImportError:
  print('Package installed without trimesh, no models generated')
