from setuptools import setup
import os

# Read version number from package __init__.py
version = {}
with open(os.path.join(os.path.dirname(__file__),'siamrl', '__init__.py')) as f:
  for line in f:
    if '=' in line:
      line = line.split('=')
      if '_VERSION' in line[0]:
        version[line[0].strip()] = int(line[1])
version = '{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}'.format(**version)

REQUIRES = [
  'numpy',
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

with open('README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(
  name='siamrl',
  version=version,
  description='Reinforcement learning with (pseudo) siamese networks.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/atmenezes96/Siam-RL',
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

# import glob
# import sys

# # Install apps
# for fname in glob.glob('apps/*.py'):
#   # Interpreter is the program executing this instalation
#   shebang = '#!'+sys.executable+'\n'
#   with open(fname,'r') as f:
#     lines = f.readlines()
#     # Check if shebang is correct
#     write = lines[0] != shebang
#   # Overwrite file if necessary
#   if write:
#     if lines[0].startswith('#!'):
#       lines[0] = shebang
#     else:
#       lines.insert(0,shebang)
#     with open(fname, 'w') as f:
#       for line in lines:
#         f.write(line)
#   # Add execute permission to those who have read permission
#   mode = os.stat(fname).st_mode
#   mode |= (mode & 0o444) >> 2
#   os.chmod(fname, mode)
