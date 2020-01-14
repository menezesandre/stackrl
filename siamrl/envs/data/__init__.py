import os
import glob

def getDataPath():
  """
  Return:
    The path to the 'data' directory (this directory).
  """
  resdir = os.path.join(os.path.dirname(__file__))
  return resdir

def getGeneratedFiles(pattern='i*.urdf'):
  """
  Args:
    pattern: format of the names of the required files

  Return:
    A list of the files from 'generated' directory that match
    the format
    
  """
  resdir = os.path.join(os.path.dirname(__file__))
  pattern = os.path.join(resdir, 'generated', pattern)
  return glob.glob(pattern)

def getGeneratedURDF(name='i'):
  """
  Args:
    name: common name of required urdf's

  Return:
    A list of the urdf's that start with given name
  """
  return getGeneratedFiles(name+'*.urdf')
