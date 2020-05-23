import sys, time
from datetime import datetime

import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics

from siamrl.compat.utils import load_policy
from siamrl.envs.stack import register

import numpy as np

gui = True
config_file = None
N = 96
if __name__=='__main__':
  argv = sys.argv[:0:-1]
  model_dirs=[]
  while argv:
    arg=argv.pop()
    if arg == '-c':
      config_file = argv.pop()
    elif arg == '-d':
      gui = False
    elif arg == '-n':
      N = int(argv.pop())
    else:
      model_dirs.append(arg)

  env_id = register(
    episode_length=32,
    urdfs='test',
    observable_size_ratio=4,
    goal_size_ratio=0.375,
    sim_time_step=1/60,
    max_z=0.5,
    use_gui=gui, 
    occupation_ratio_weight=1.,
    seed=11,
    dtype='uint8'
  )
  env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_id))

  for model_dir in model_dirs:
    if config_file:
      policy = load_policy(env.time_step_spec(), env.action_spec(), model_dir, 'config.gin')
    else:
      policy = load_policy(env.time_step_spec(), env.action_spec(), model_dir)
        
    metric = tf_metrics.AverageReturnMetric()
    driver = dynamic_step_driver.DynamicStepDriver(
      env, 
      policy, 
      observers=[metric], 
      num_steps=1
    )

    time_step = env.reset()
    if gui:
      import pybullet as pb
      pb.resetDebugVisualizerCamera(1., 90, -45, [0.25,0.25,0])
      # time.sleep(5.)
    time.sleep(1-(datetime.now().microsecond/1e6))
    now = datetime.now()
    for i in range(N):
      n = datetime.now() - now
      n = (i+1)*0.5 - (n.seconds + n.microseconds/1e6)
      if n > 0:
        time.sleep(n)

      time_step, _ = driver.run(time_step)
    del(policy)
    print('Average Return ({}): {}'.format(model_dir, metric.result().numpy()))
  
