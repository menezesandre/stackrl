from setuptools import setup
import os, sys, stat, glob

with open('README.md', encoding='utf-8') as f:
  long_description = f.read()

REQUIRES = [
  'numpy', 
  'gym', 
  'pybullet',
  'tensorflow-probability==0.8.0',
  'tf-agents==0.3.0',
  'gin-config==0.1.3'
]

# Only add this requirement if tensorflow is not installed.
# This is necessary because it may be installed under a different
# project name (e.g. tf-nightly-gpu).
try:
  import tensorflow as tf
  assert eval(tf.__version__[:3]) == 2.0
except:
  REQUIRES.insert(-2, 'tensorflow==2.0.0')

setup(
  name='siamrl',
  version='2.0.dev0412',
  description='', #TODO
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/atmenezes96/Siam-RL',
  author='André Menezes',
  author_email='andre.menezes@tecnico.ulisboa.pt',
  install_requires=REQUIRES,
  extra_requires={'all': ['trimesh', 'opencv-python', 'matplotlib'],
                  'generator': ['trimesh'],
                  'baselines': ['opencv-python'],
                  'plot': ['matplotlib']}
)

# Install apps
for fname in glob.glob('apps/*.py'):
  # Add shebang if necessary
  with open(fname,'r+') as f:
    lines = f.readlines()
    # Interpreter is the program executing this instalation
    interpreter = '#!'+sys.executable+'\n'
    write_interpreter = True
    if lines[0] == interpreter:
      write_interpreter = False
    elif lines[0].startswith('#!'):
      lines[0] = '#!'+sys.executable+'\n'
    else:
      lines.insert(0,'#!'+sys.executable+'\n')
    if write_interpreter:
      f.seek(0)
      for line in lines:
        f.write(line)
  # Add execute permission to those who have read permission
  mode = os.stat(fname).st_mode
  mode |= (mode & 0o444) >> 2
  os.chmod(fname, mode)

