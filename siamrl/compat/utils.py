import os
import gin
from tf_agents.policies import greedy_policy
from tf_agents.policies import q_policy
from siamrl.compat import networks

def load_policy(time_step_spec, action_spec, path='.', config_file=None):
  """
  Args:
    time_step_spec: specification of the TimeStep expected by the policy.
    action_spec: specificaiton of the action returned by the policy.
    path: path to the weights' files. Either to the directory where the files
      are stored under the name 'weights' or the complete file path.
    config_file: path to the config file to be parsed (providing network's
    configuration).
  Returns:
    Instance of the greedy Q policy with loaded Q net weights.
  Raises:
    OSError: if provided config_file doesn't exist.
    tensorflow.python.framework.errors_impl.NotFoundError: if no matching
      files are found to be loaded.
  """
  if config_file:
    gin.parse_config_file(config_file)

  net = networks.SiamQNetwork(
    time_step_spec.observation,
    action_spec
  )
  if os.path.isdir(path):
    net.load_weights(os.path.join(path,'weights'))
  else:
    net.load_weights(path)

  return greedy_policy.GreedyPolicy(q_policy.QPolicy(
    time_step_spec,
    action_spec,
    net
  ))
