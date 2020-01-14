"""
Import this file to (re)generate the models in 'generated' folder
"""
from siamrl.envs.utils import generate
from siamrl.envs import getDataPath
import os

path = os.path.join(getDataPath(),'generated')

# Catch the exception if trimesh is not instaled
try:
  generate.fromBox(n=500, directory=path, name='b')
  generate.fromIcosphere(n=1000, directory=path, name='i')
  generate.fromIcosphere(n=1000, convex=True, directory=path, name='ic')
except ModuleNotFoundError:
  print('Package installed without trimesh, no models generated')
