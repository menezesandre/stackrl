import sys

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics

from siamrl.utils import load_policy
from siamrl.envs import stack

import numpy as np

if __name__=='__main__':
  if len(sys.argv) > 1:
    model_dir = sys.argv[1]
  else:
    model_dir = None
  
  env_id = stack.register(
    episode_length=8,
    urdfs='test',
    observable_size_ratio=4,
    goal_size_ratio=0.375,
    sim_time_step=1/60,
    max_z=0.5,
    use_gui=True, 
    positions_weight=1,
    dtype='uint8'
  )
  env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_id))
  if model_dir is not None:
    try:
      policy = load_policy(env.time_step_spec(), env.action_spec(), model_dir, 'config.gin')
    except OSError:
      policy = load_policy(env.time_step_spec(), env.action_spec(), model_dir)
  else:
    policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())
  metric = tf_metrics.AverageReturnMetric()
  driver = dynamic_step_driver.DynamicStepDriver(
    env, 
    policy, 
    observers=[metric], 
    num_steps=1024
  )

  initial_time_step = env.reset()
  final_time_step, _ = driver.run(initial_time_step)

  print('Average Return: ', metric.result().numpy())
  

