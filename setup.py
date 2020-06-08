from setuptools import setup
import os
import sys
import glob

from siamrl.version import __version__ as version

with open('README.md', encoding='utf-8') as f:
  long_description = f.read()

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

