from os import path as _path
from glob import glob as _glob

def path(path=None):
  """
  Args:
    path: relative path from this directory
      ('siamrl/envs/data').
  Return:
    The absolute path to 'siamrl/envs/data/path' or,
    if no path is given, to this directory.
  Raises:
    FileNotFoundError: if path doesn't exist in this
      directory.
  """
  dirname = _path.dirname(__file__)
  if path:
    path = _path.join(dirname,path)
    if _path.exists(path):
      return path
    else:
      raise FileNotFoundError(
        'No such file or directory: {}'.format(path)
      )
  else:
    return dirname

_open = open
def open(file, **kwargs):
  """Wrapper of the built-in function open() that prepends 
    the absolute path to 'siamrl/envs/data' to the file path."""
  return _open(_path.join(path(), file), **kwargs)

def files(pattern):
  """
  Args:
    pattern: format of the names of the required files.
  Return:
    A list of the files from 'siamrl/envs/data' directory that
    match the pattern.
  """
  return _glob(_path.join(path(), pattern))

def generated(name='i'):
  """
  Args:
    name: common name of required urdf's

  Return:
    A list of the urdf's that start with given name, from the 
    'generated' directory
  """
  return files(_path.join('generated',name+'*.urdf'))
