import os

import gin
import numpy as np
from tensorflow import errors

import siamrl
from siamrl import agents
from siamrl import envs
from siamrl import nets

def load(
  observation_spec, 
  path='.', 
  iters=None, 
  config_file='config.gin',
  value=False,
  verbose=False,
):
  """Load a policy saved from training.

  Args:
    observation_spec: environment observation spec to use as input spec 
      for the network. Either a nest of tf.TensorSpec or a gym.Space. In
      the later case, the policy is wrapped to receive and return numpy 
      arrays.
    path: path to the train directory or to the folder where the network
      weights are stored. If weights aren't found in either format, path
      is seen as relative from Siam-RL/data/train/ directory.
    iters: train iteration from which to load the saved weights. Only used
      if path is a train directory. If None, best evaluated policy is 
      used (taken from path/eval.csv).
    config_file: configuration file with network specifications. Seen as 
      a relative path from given path, and if not present as an absolute
      path / relative to current working directory. If None, no 
      configuration file is parsed.
    value: whether to return a policy that returns actions and estimated 
      values (opposed to only actions).
    verbose: whether to print complete path from which weights are being
      loaded.
  """
  try:
    # Set observation spec
    if envs.isspace(observation_spec):
      batchwise = len(observation_spec[0].shape) == 4
      _observation_spec = envs.get_space_spec(observation_spec)
      py = True
    else:
      batchwise = False
      py = False

    if os.path.isdir(os.path.join(path,'saved_weights')):
      if iters is not None:
        wpath = os.path.join(
          path,
          'saved_weights',
          str(iters),
        )
      elif os.path.isfile(os.path.join(path,'eval.csv')):
        # Use best evaluated policy
        eiters, returns = np.loadtxt(
          os.path.join(path,'eval.csv'),
          delimiter=',',
          skiprows=2,
          unpack=True,
        )[:2]
        eiters = np.atleast_1d(eiters)
        returns = np.atleast_1d(returns)
        wpath = os.path.join(
          path,
          'saved_weights', 
          str(int(eiters[np.argmax(returns)])),
        )
    else:
      wpath = path

    if not os.path.isdir(wpath):
      raise FileNotFoundError("Couldn't find '{}'".format(wpath))
    if config_file is not None:
      # Store current config state
      _config = gin.config._CONFIG.copy()
      # Parse config file
      if os.path.isfile(os.path.join(path, config_file)):
        gin.parse_config_file(os.path.join(path, config_file))
      elif os.path.isfile(config_file):
        gin.parse_config_file(config_file)
      else:
        raise FileNotFoundError("Failed to find config file '{}'.".format(config_file))
      # Instantiate net with binded parameters
      with gin.config_scope('load'):
        net = nets.PseudoSiamFCN(_observation_spec)
      # Restore config state
      gin.config._CONFIG = _config
    else:
      net = nets.PseudoSiamFCN(_observation_spec)

    net.load_weights(os.path.join(wpath,'weights'))
    if verbose:
      print('Weights loaded from {}'.format(wpath))
    policy = agents.policies.Greedy(net, value=value, batchwise=batchwise)
    if py:
      policy = agents.policies.PyWrapper(policy, batched=batchwise)

    return policy
  except (FileNotFoundError, errors.NotFoundError) as e:
    if not path.startswith(siamrl.datapath('train')):
      return load(
        observation_spec,
        path=os.path.join(siamrl.datapath('train'), path),
        iters=iters,
        config_file=config_file,
        value=value,
        verbose=verbose,
      )
    else:
      raise e