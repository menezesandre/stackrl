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
def open(file, *args, **kwargs):
  """Wrapper of the built-in function open() that prepends 
    the absolute path to 'siamrl/envs/data' to the file path."""
  return _open(os.path.join(path(), file), *args, **kwargs)

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
  test=False,
  volume=None,
  rectangularity=None, 
  aspectratio=None, 
):
  """ Returns a list of the urdf names from the 'generated' directory.

  Args:
    name: common prefix of the required urdf's.
    test: whether to fetch the file names from the test set directory.
    volume: if provided, returned files are filtered by object 
      volume. Either a scalar for minimum value or a tuple with 
      (min,max) values.
    rectangularity: if provided, returned files are filtered by object 
      rectangularity. Either a scalar for minimum value or a tuple with 
      (min,max) values.
    aspectratio: if provided, returned files are filtered by object 
      aspect ratio. Same type as rectangularity.
  """
  directory = 'generated'
  if test:
    directory = os.path.join(directory, 'test')
  flist = files(os.path.join( 
    directory,
    name+'_*.urdf' if name else '*.urdf',
  ))
  
  # Compatibility with old names
  if not flist:
    flist = files(os.path.join( 
      directory,
      name+'*.urdf',
    ))


  if volume or rectangularity or aspectratio:
    raise NotImplementedError('Filtering not supported yet.')
  
  return flist

  # flist = files(os.path.join(
  #   os.path.join('generated', 'test') if test else 'generated',
  #   '[0-9][0-9][0-9]_[0-9][0-9][0-9]_*.urdf',
  # ))
  # if rectangularity:
  #   if hasattr(rectangularity, '__len__'):
  #     if len(rectangularity) > 1:
  #       min_rectangularity, max_rectangularity = rectangularity[:2]
  #     else:
  #       min_rectangularity, max_rectangularity = rectangularity[0], 1.
  #   else:
  #       min_rectangularity, max_rectangularity = rectangularity, 1.


  #   # Filter objects with rectangularity outside bounds
  #   min_rectangularity = int(min_rectangularity*100)
  #   max_rectangularity = int(max_rectangularity*100)
  #   flist = [
  #     fname for fname in flist if 
  #     int(os.path.split(fname)[-1][:3]) >= min_rectangularity and
  #     int(os.path.split(fname)[-1][:3]) <= max_rectangularity
  #   ]
  # if aspectratio:
  #   if hasattr(aspectratio, '__len__'):
  #     if len(aspectratio) > 1:
  #       min_aspectratio, max_aspectratio = aspectratio[:2]
  #     else:
  #       min_aspectratio, max_aspectratio = aspectratio[0], 99.9
  #   else:
  #       min_aspectratio, max_aspectratio = aspectratio, 99.9
  #   # Filter objects with aspect ratio outside bounds
  #   min_aspectratio = int(min_aspectratio*10)
  #   max_aspectratio = int(max_aspectratio*10)
  #   flist = [
  #     fname for fname in flist if 
  #     int(os.path.split(fname)[-1][4:7]) >= min_aspectratio and
  #     int(os.path.split(fname)[-1][4:7]) <= max_aspectratio
  #   ]
