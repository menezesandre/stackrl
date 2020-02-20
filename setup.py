from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
 
setup(name='siamrl',
      version='0.3',
      description='', #TODO
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/atmenezes96/Siam-RL',
      author='André Menezes',
      author_email='andre.menezes@tecnico.ulisboa.pt',
      install_requires=['numpy', 
                        'tf-nightly', 
                        'gym', 
                        'pybullet', 
                        'tf-agents-nightly'],
      extra_requires={'all': ['trimesh', 'opencv-python', 'matplotlib'],
                      'generator': ['trimesh'],
                      'baseline': ['opencv-python'],
                      'plot': ['matplotlib']}
)
