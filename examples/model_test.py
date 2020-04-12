import sys, os

import tensorflow as tf
import tf_agents

from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics

from siamrl.envs import stack
from siamrl.networks import SiamQNetwork

import numpy as np

if __name__=='__main__':
  if len(sys.argv) > 1:
    model_dir = sys.argv[1]
  else:
    model_dir = '.'
  
  env_id = stack.register(
    episode_length=32,
    urdfs='test',
    observable_size_ratio=4,
    goal_size_ratio=1/3,
    sim_time_step=1/60,
    use_gui=True, 
    positions_weight=1,
    dtype='uint8'
  )
  env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_id))
  policy = tf.saved_model.load(model_dir)
  metric = tf_metrics.AverageReturnMetric()
  try:
    driver = dynamic_step_driver.DynamicStepDriver(
      env, 
      policy, 
      observers=[metric], 
      num_steps=1024
    )

    initial_time_step = env.reset()
    final_time_step, _ = driver.run(initial_time_step)
  except ValueError:
    net = SiamQNetwork(
      env.observation_spec(),
      env.action_spec()
    )
    for vn,vm in zip(net.variables, policy.model_variables):
      vn.assign(vm)
    policy = greedy_policy.GreedyPolicy(q_policy.QPolicy(
      env.time_step_spec(),
      env.action_spec(),
      net
    ))

    driver = dynamic_step_driver.DynamicStepDriver(
      env, 
      policy, 
      observers=[metric], 
      num_steps=1024
    )

    initial_time_step = env.reset()
    final_time_step, _ = driver.run(initial_time_step)


  print(policy.__class__.__name__)
  print('- Average Return: ', metric.result().numpy())
  

