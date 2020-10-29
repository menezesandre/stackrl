import os as _os

from stackrl import agents
from stackrl import baselines
from stackrl import envs
from stackrl import external_configurables
from stackrl import nets
from stackrl import test
from stackrl import train

from stackrl.baselines import Baseline
from stackrl.train import Training

MAJOR_VERSION = 1
MINOR_VERSION = 0
PATCH_VERSION = 0

__version__ = '{}.{}.{}'.format(MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION)

def datapath(*args):
  """Returns the full path given by args from the Siam-RL/data directory."""
  return _os.path.join(
    _os.path.dirname(_os.path.dirname(__file__)),
    'data',
    *args,
  )
