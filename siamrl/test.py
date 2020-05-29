import os
import gin
from siamrl.nets import PseudoSiamFCN
from siamrl.agents.policies import GreedyPolicy, PyWrapper

def load_policy(observation_spec, path='.', config_file=None, py_format=False):
  if config_file:
    try:
      gin.parse_config_file(config_file)
    except OSError:
      gin.parse_config_file(os.path.join(path, config_file))

  net = PseudoSiamFCN(observation_spec)
  if os.path.isdir(path):
    path = os.path.join(path,'weights')
  net.load_weights(path)
  policy = GreedyPolicy(net)
  if py_format:
    policy = PyWrapper(policy)
  
  return policy
