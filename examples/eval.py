import sys,os

import tensorflow as tf
tf.random.set_seed(0)
import tf_agents

from tf_agents.agents.dqn import dqn_agent
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics

from siamrl.envs import stack
from siamrl.networks import SiamQNetwork

if __name__=='__main__':
  env_id_s = stack.register(
    episode_length=32,
    observable_size_ratio=[4,6],
    max_z=0.5,
    goal_size_ratio=1/3.,
    occupation_ratio_weight=1,
    seed=0
  )
  env_id_ns = stack.register(
    episode_length=32,
    observable_size_ratio=[4,6],
    max_z=0.5,
    goal_size_ratio=1/3.,
    occupation_ratio_weight=1,
    seed=0,
    smooth_placing=False
  )
  if not os.path.isfile('eval_s.csv'):
    with open('eval_s.csv', 'w') as f:
      f.write('Iter,Reward\n')
  if not os.path.isfile('eval_ns.csv'):
    with open('eval_ns.csv', 'w') as f:
      f.write('Iter,Reward\n')


  metric = tf_metrics.AverageReturnMetric(buffer_size=20)
  net = None

  # env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_id_ns))
  # policy = random_tf_policy.RandomTFPolicy(
  #   env.time_step_spec(),
  #   env.action_spec()
  # )
  # driver = dynamic_step_driver.DynamicStepDriver(
  #   env, 
  #   policy, 
  #   observers=[metric], 
  #   num_steps=640,
  # )
  # metric.reset()
  # initial_time_step = env.reset()
  # final_time_step, _ = driver.run(initial_time_step)
  # print(metric.result().numpy())

  for i in range(5,165):
    model_dir = '%d0000'%i
    env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_id_s))

    loaded = tf.saved_model.load(model_dir)

    if net is None:
      net = SiamQNetwork(
        env.observation_spec(),
        env.action_spec()
      )
      policy = greedy_policy.GreedyPolicy(q_policy.QPolicy(
        env.time_step_spec(),
        env.action_spec(),
        net
      ))


    for vn,vm in zip(net.variables, loaded.model_variables):
      vn.assign(vm)
    del(loaded)

    
    driver = dynamic_step_driver.DynamicStepDriver(
      env, 
      policy, 
      observers=[metric], 
      num_steps=640,
    )
    metric.reset()
    initial_time_step = env.reset()
    final_time_step, _ = driver.run(initial_time_step)
    result_s = metric.result().numpy()
    del(env, driver)
    with open('eval_s.csv','a') as f:
      f.write('%d0000,%f\n'%(i,result_s))

    env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_id_ns))

    driver = dynamic_step_driver.DynamicStepDriver(
      env, 
      policy, 
      observers=[metric], 
      num_steps=640,
    )
    metric.reset()
    initial_time_step = env.reset()
    final_time_step, _ = driver.run(initial_time_step)
    result_ns = metric.result().numpy()
    del(env, driver)
    with open('eval_ns.csv','a') as f:
      f.write('%d0000,%f\n'%(i,result_ns))

    print(i, result_s, result_ns)
    