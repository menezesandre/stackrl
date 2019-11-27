import os
import glob

def getDataPath():
  resdir = os.path.join(os.path.dirname(__file__))
  return resdir

def getRockFiles(mode='all'):
  ret = []
  resdir = os.path.join(os.path.dirname(__file__))
  if mode == 'convex' or mode == 'all':
    ret += glob.glob(os.path.join(resdir, 'rocks/convex/rock*.urdf'))
  if mode == 'non_convex' or mode == 'all':
    ret += glob.glob(os.path.join(resdir, 'rocks/rock*.urdf'))
  return ret
