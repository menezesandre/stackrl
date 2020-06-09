from setuptools import setup
import os
import sys
import glob

MAJOR_VERSION = 1
MINOR_VERSION = 0
PATCH_VERSION = 0

version = '{}.{}.{}'.format(MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)

# Set __version__ as a package attribute.
init_file = './siamrl/__init__.py'
with open(init_file) as f:
  lines = f.readlines()
if lines[-1].startswith('__version__'):
  lines[-1] = '__version__ = "{}"\n'.format(version)
  with open(init_file, 'w') as f:
    for line in lines:
      f.write(line)
else:
  with open(init_file, 'a') as f:
    f.write('__version__ = "{}"\n'.format(version))

REQUIRES = [
  'numpy', 
  'gym', 
  'pybullet',
  'gin-config',
]

# Only add this requirement if tensorflow is not installed.
# This is necessary because it may be installed under a different
# project name (e.g. tf-nightly-gpu).
try:
  import tensorflow as tf
  assert int(tf.__version__.split('.')[0]) >= 2 # pylint: disable=no-member
except (ImportError, AssertionError):
  REQUIRES.insert(-2, 'tensorflow>=2.0.0')

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
    'compat': ['tf-agents'],
  },
)

# Install apps
for fname in glob.glob('apps/*.py'):
  # Interpreter is the program executing this instalation
  shebang = '#!'+sys.executable+'\n'
  with open(fname,'r') as f:
    lines = f.readlines()
    # Check if shebang is correct
    write = lines[0] != shebang
  # Overwrite file if necessary
  if write:
    if lines[0].startswith('#!'):
      lines[0] = shebang
    else:
      lines.insert(0,shebang)
    with open(fname, 'w') as f:
      for line in lines:
        f.write(line)
  # Add execute permission to those who have read permission
  mode = os.stat(fname).st_mode
  mode |= (mode & 0o444) >> 2
  os.chmod(fname, mode)

