import os

import gin
import gym
import numpy as np

from siamrl import agents
from siamrl import envs
from siamrl import nets

def load_policy(
  observation_spec, 
  path='.', 
  iters=None, 
  config_file=None, 
  value=False,
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
    batch = len(observation_spec[0].shape) == 4
    observation_spec = envs.utils.get_space_spec(observation_spec)
    py = True
  else:
    batch = False
    py = False

  if os.path.isdir(os.path.join(path,'saved_weights')):
    if iters is not None:
      if not isinstance(iters, list):
        iters = [iters]  
      paths = [os.path.join(path,'saved_weights', str(i)) for i in iters]
    elif os.path.isfile(os.path.join(path,'eval.csv')):
      # Use best evaluated policy
      iters, reward = np.loadtxt(
        os.path.join(path,'eval.csv'),
        delimiter=',',
        skiprows=2,
        unpack=True,
      )[:2]
      i = np.argmax(reward)
      iters = int(iters[i])
      # print('Iters: {}'.format(iters))
      paths = [os.path.join(path,'saved_weights', str(iters))]
    else:
      paths = [path]
  else:
    paths = [path]

  policies = []
  for path in paths:
    if not os.path.isdir(path):
      raise FileNotFoundError("Couldn't find '{}'".format(path))
    net = nets.PseudoSiamFCN(observation_spec)
    net.load_weights(os.path.join(path,'weights'))
    policy = agents.policies.Greedy(net, value=value, batchwise=batch)
    if py:
      policy = agents.policies.PyWrapper(policy, batched=batch)
    policies.append(policy)

  if len(policies) == 1:
    return policies[0]
  else:
    return policies
