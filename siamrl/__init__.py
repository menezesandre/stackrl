import os as _os

from siamrl import agents
from siamrl import baselines
from siamrl import envs
from siamrl import external_configurables
from siamrl import nets
from siamrl import test
from siamrl import train

from siamrl.baselines import Baseline
from siamrl.train import Training

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
