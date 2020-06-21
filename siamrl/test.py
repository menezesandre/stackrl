import os

import gin
import gym
import numpy as np

from siamrl.nets import PseudoSiamFCN
from siamrl.agents.policies import GreedyPolicy, PyWrapper
from siamrl.envs.utils import get_space_spec

def load_policy(
  observation_spec, 
  path='.', 
  iters=None, 
  config_file=None, 
  debug=False
):
  # Parse config file
  if not config_file:
    try:
      gin.parse_config_file(os.path.join(path, 'config.gin'))
    except OSError:
      pass
  elif os.path.isfile(config_file):
    gin.parse_config_file(config_file)
  elif os.path.isfile(os.path.join(path, config_file)):
    gin.parse_config_file(os.path.join(path, config_file))
  else:
    raise FileNotFoundError("Couldn't find '{}'".format(config_file))
  # Set observation spec
  if isinstance(observation_spec, gym.Space):
    observation_spec = get_space_spec(observation_spec)
    py = True

  net = PseudoSiamFCN(observation_spec)
  if os.path.isdir(os.path.join(path,'saved_weights')):
    if iters is not None:
      path = os.path.join(path,'saved_weights', str(iters))
    elif os.path.isfile(os.path.join(path,'eval.csv')):
      # Use best evaluated policy
      iters, reward = np.loadtxt(
        os.path.join(path,'eval.csv'),
        delimiter=',',
        skiprows=2,
        unpack=True,
      )
      i = np.argmax(reward)
      iters = int(iters[i])
      # print('Iters: {}'.format(iters))
      path = os.path.join(path,'saved_weights', str(iters))

  if not os.path.isdir(path):
    raise FileNotFoundError("Couldn't find '{}'".format(path))
  net.load_weights(os.path.join(path,'weights'))
  policy = GreedyPolicy(net, debug=debug)
  if py:
    policy = PyWrapper(policy)
  
  return policy
