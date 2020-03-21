from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

REQUIRES = [
  'numpy', 
  'gym', 
  'pybullet',
  'gin-config',
  'tensorflow-probability==0.8.0',
  'tf-agents==0.3.0']


# Only add this requirement if tensorflow is not installed.
# This is necessary because it may be installed under a different
# project name (e.g. tf-nightly-gpu).
try:
  import tensorflow as tf
  assert eval(tf.__version__[:3]) == 2.0
except:
  REQUIRES.insert(-1, 'tensorflow==2.0.0')

setup(
  name='siamrl',
  version='1.3.21',
  description='', #TODO
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/atmenezes96/Siam-RL',
  author='André Menezes',
  author_email='andre.menezes@tecnico.ulisboa.pt',
  install_requires=REQUIRES,
  extra_requires={'all': ['trimesh', 'opencv-python', 'matplotlib'],
                  'generator': ['trimesh'],
                  'baseline': ['opencv-python'],
                  'plot': ['matplotlib']}
)
