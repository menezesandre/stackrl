import tensorflow as tf
import siamrl, tf_agents

from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.agents import DqnAgent
from tf_agents.utils import common

from siamrl.networks import SiamQNetwork

if __name__ == '__main__':

  train_env = tf_py_environment.TFPyEnvironment(suite_gym.load('BrickStack-v0'))

  q_net = SiamQNetwork(train_env.observation_spec(), train_env.action_spec())

  optimizer = tf.keras.optimizers.Adam()

  train_step_counter = tf.Variable(0)

  agent = DqnAgent(
      train_env.time_step_spec(),
      train_env.action_spec(),
      q_network=q_net,
      optimizer=optimizer,
      td_errors_loss_fn=common.element_wise_squared_loss,
      train_step_counter=train_step_counter)

  agent.initialize()

  print(agent)
