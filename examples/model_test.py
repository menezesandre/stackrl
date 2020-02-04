import tensorflow as tf
import siamrl
import tf_agents

from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics

from siamrl.networks import SiamQNetwork

if __name__=='__main__':
  env_id = siamrl.utils.register_stack_env(goal=True,gui=True)
  env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_id))
  print('batch: ',env.batch_size)
  q_net = SiamQNetwork(env.observation_spec(), 
      env.action_spec())


  policy = tf.saved_model.load('./avg_occ/5000')

  metric = tf_metrics.AverageReturnMetric()
  driver = dynamic_step_driver.DynamicStepDriver(env, policy, 
    observers=[metric], num_steps=150)

  initial_time_step = env.reset()
  final_time_step, _ = driver.run(initial_time_step)
  print(policy.__class__.__name__)
  print('- Average Return: ', metric.result())
  

