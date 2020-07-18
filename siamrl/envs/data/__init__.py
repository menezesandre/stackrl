import glob
import os

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
  dirname = os.path.dirname(__file__)
  if path:
    path = os.path.join(dirname,path)
    if os.path.exists(path):
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
  return _open(os.path.join(path(), file), **kwargs)

def files(pattern):
  """
  Args:
    pattern: format of the names of the required files.
  Return:
    A list of the files from 'siamrl/envs/data' directory that
    match the pattern.
  """
  return glob.glob(os.path.join(path(), pattern))

def generated(
  name=None, 
  rectangularity=None, 
  aspectratio=None, 
  test=False,
):
  """
  Args:
    name: common name of required urdf's

  Return:
    A list of the urdf's that start with given name, from the 
    'generated' directory
  """
  if name:
    return files(os.path.join('generated',name+'*.urdf'))
  else:
    flist = files(os.path.join(
      os.path.join('generated', 'test') if test else 'generated',
      '[0-9][0-9][0-9]_[0-9][0-9][0-9]_*.urdf',
    ))
    if rectangularity:
      if hasattr(rectangularity, '__len__'):
        if len(rectangularity) > 1:
          min_rectangularity, max_rectangularity = rectangularity[:2]
        else:
          min_rectangularity, max_rectangularity = rectangularity[0], 1.
      else:
          min_rectangularity, max_rectangularity = rectangularity, 1.


      # Filter objects with rectangularity outside bounds
      min_rectangularity = int(min_rectangularity*100)
      max_rectangularity = int(max_rectangularity*100)
      flist = [
        fname for fname in flist if 
        int(os.path.split(fname)[-1][:3]) >= min_rectangularity and
        int(os.path.split(fname)[-1][:3]) <= max_rectangularity
      ]
    if aspectratio:
      if hasattr(aspectratio, '__len__'):
        if len(aspectratio) > 1:
          min_aspectratio, max_aspectratio = aspectratio[:2]
        else:
          min_aspectratio, max_aspectratio = aspectratio[0], 99.9
      else:
          min_aspectratio, max_aspectratio = aspectratio, 99.9
      # Filter objects with aspect ratio outside bounds
      min_aspectratio = int(min_aspectratio*10)
      max_aspectratio = int(max_aspectratio*10)
      flist = [
        fname for fname in flist if 
        int(os.path.split(fname)[-1][4:7]) >= min_aspectratio and
        int(os.path.split(fname)[-1][4:7]) <= max_aspectratio
      ]
    return flist