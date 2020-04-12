import tensorflow as tf
from tf_agents.policies import greedy_policy
from tf_agents.policies import q_policy
import gin

from siamrl import networks

def load_policy(env, filepath, config_file=None):
  """Returns the greedy q policy for env with loaded net
    weights from filepath."""
  if config_file:
    gin.parse_config_file(config_file)

  net = networks.SiamQNetwork(
    env.observation_spec(),
    env.action_spec())
  net.load_weights(filepath)

  return greedy_policy.GreedyPolicy(q_policy.QPolicy(
    env.time_step_spec(),
    env.action_spec(),
    net
  ))
