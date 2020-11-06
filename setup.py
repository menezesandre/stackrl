from setuptools import setup
import os

path = lambda *args: os.path.join(os.path.dirname(__file__), *args)

# Read version number from package __init__.py
version = {}
with open(path('stackrl', '__init__.py')) as f:
  for line in f:
    if '=' in line:
      line = line.split('=')
      if '_VERSION' in line[0]:
        version[line[0].strip()] = int(line[1])
version = '{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}'.format(**version)

REQUIRES = [
  'numpy>=1.19.0',
  'scipy',
  'gym', 
  'pybullet',
  'gin-config',
]

# Only add this requirement if tensorflow is not installed.
# This is necessary because it may be installed under a different
# project name (e.g. tf-nightly).
try:
  import tensorflow as tf
  assert int(tf.__version__.split('.')[0]) >= 2 # pylint: disable=no-member
except (ImportError, AssertionError):
  REQUIRES.insert(0, 'tensorflow>=2.0.0')

with open(path('README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(
  name='stackrl',
  version=version,
  description='Learning to dry stack with irregular blocks using Reinforcement Learning.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/menezesandre/stackrl',
  author='André Menezes',
  author_email='andre.menezes@tecnico.ulisboa.pt',
  install_requires=REQUIRES,
  extra_requires={
    'all': ['trimesh', 'opencv-python', 'matplotlib'],
    'generator': ['trimesh'],
    'baselines': ['opencv-python'],
    'plot': ['matplotlib'],
  },
)
