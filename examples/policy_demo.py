import sys
import os
import time
from datetime import datetime

import gym
import numpy as np

from siamrl import baselines, load_policy
from siamrl.envs.stack import register

config_file = None
iters = None
path = None
ns = 96
sleep_time = 1.
gui = False

if __name__=='__main__':
  """Raises KeyError if provided argument isn't one of the implemented 
  baseline policies."""
  argv = sys.argv[:0:-1]
  while argv:
    arg=argv.pop()
    if arg == '-c':
      config_file = argv.pop()
    elif arg == '-g':
      gui = True
    elif arg == '-n':
      ns = int(argv.pop())
    elif arg == '-i':
      iters = int(argv.pop())
    elif arg == '-f':
      sleep_time = 1/float(argv.pop())
    else:
      path = arg
  path = path or '.'

  # if iters is None:
  #   iters, reward = np.loadtxt(os.path.join(path, 'eval.csv'), delimiter=',', skiprows=2, unpack=True)
  #   # Use the best evaluated policy
  #   iters = int(iters[np.argmax(reward)])
  #   del(reward)

  kwargs = {
    'episode_length':16,
    # urdfs:'train',
    # object_max_dimension:0.125,
    # sim_time_step:1/60.,
    # gravity:9.8,
    # num_sim_steps:None,
    # velocity_threshold:0.01,
    # smooth_placing : True,
    # observable_size_ratio:4,
    # resolution_factor:5,
    'max_z':0.5,
    'goal_size_ratio':.25,
    'occupation_ratio_weight':10.,
    # occupation_ratio_param:False,
    # positions_weight:0.,
    # positions_param:0.,
    # n_steps_weight:0.,
    # n_steps_param:0.,
    # contact_points_weight:0.,
    # contact_points_param:0.,
    # differential:True,
    # flat_action:True,
    'seed':11,
    'dtype':'uint8',
  }
  if gui:
    kwargs['use_gui'] = True

  env_id = register(**kwargs)
  env = gym.make(env_id)

  policy = load_policy(
    env.observation_space, 
    path=path, 
    iters=iters, 
    config_file=config_file, 
  )
  
  results={}
    
  tr = 0.
  ne = 0
  o = env.reset()

  if gui:
    import pybullet as pb
    pb.resetDebugVisualizerCamera(1., 90, -30, [0.25,0.25,0])
    time.sleep(3.)
  
  for _ in range(ns):
    if gui:
      time.sleep(sleep_time-(datetime.now().microsecond/1e6)%sleep_time)
    o,r,d,_=env.step(policy(o))
    tr+=r
    if d:
      ne+=1
      print('  Current average ({}): {}'.format(
        ne,
        tr/ne
      ))
      o=env.reset()

  print('Final average: {}'.format(tr/ne))
