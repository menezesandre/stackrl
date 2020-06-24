import sys
import os
import time
from datetime import datetime
from glob import glob

import gym
import matplotlib.pyplot as plt
import numpy as np

from siamrl import load_policy
from siamrl.envs.stack import register

config_file = None
iters = None
path = None
recompute = False
ns = 96
show = None
sleep_time = 1.
if __name__=='__main__':
  argv = sys.argv[:0:-1]
  while argv:
    arg=argv.pop()
    if arg == '-c':
      config_file = argv.pop()
    elif arg == '-r':
      recompute = True
    elif arg == '-s':
      show = True
    elif arg == '--disable-show':
      show = False
    elif arg == '-n':
      ns = int(argv.pop())
    elif arg == '-i':
      iters = argv.pop().split(',')
    elif arg == '-t':
      sleep_time = float(argv.pop())
    else:
      path = arg
  path = path or '.'

  if not iters:
    if show is None:
      show = False
    iters, reward = np.loadtxt(os.path.join(path, 'eval.csv'), delimiter=',', skiprows=2, unpack=True)
    best_iters = iters[np.argsort(reward)]
    del(reward)
    if os.path.isfile(os.path.join(path, 'q_values.csv')) and not recompute:
      data = np.loadtxt(os.path.join(path, 'q_values.csv'), delimiter=',', skiprows=1, unpack=True)
      iters = np.setdiff1d(iters, data[0])
    # Use the actions from the best evaluated policy
    for i in best_iters:
      if i in iters:
        b, = np.nonzero(iters == i)
        b = int(b)
        break
    iters = np.int64(iters)
  else:
    # Use the actions from the first given policy
    b = 0

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
    'occupation_ratio_weight':100.,
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
  env_id = register(**kwargs)
  env = gym.make(env_id)

  policies = [
    load_policy(env.observation_space, path=path, iters=i, config_file=config_file, debug=True)
    for i in iters
  ]
  
  o = env.reset()
  values = [[] for _ in policies]
  
  if len(policies) <= 4 and show != False: 
    show = True
    fig, axs = plt.subplots(
      2,1+len(policies), 
      gridspec_kw={'height_ratios':[4, 1]}
    )
    v_shape = (o[0].shape[0]-o[1].shape[0]+1,o[0].shape[1]-o[1].shape[1]+1)
  else:
    fig = None

  for n in range(ns):
    actions = [p(o) for p in policies]

    if fig is not None:
      o0, o1 = env.render(mode='rgb_array')
      axs[0][0].cla()
      axs[0][0].imshow(o0)
      axs[1][0].cla()
      axs[1][0].imshow(o1)

    for i, (_,v) in enumerate(actions):
      values[i].append(v)
      if fig is not None:
        mean_ = np.mean(v)
        v = v.reshape(v_shape)

        axs[0][i+1].cla()
        axs[0][i+1].imshow(v)
        axs[1][i+1].cla()
        axs[1][i+1].imshow(np.where(v>mean_, v, mean_))

    if fig is not None:
      fig.show()
      plt.pause(sleep_time)

    o,_,d,_=env.step(actions[b][0])
    if d:
      o=env.reset()  

  values = [np.concatenate(v) for v in values]
  v_mean = [np.mean(v) for v in values]
  v_std = [np.std(v) for v in values]
  v_min = [np.min(v) for v in values]
  v_max = [np.max(v) for v in values]

  file_name = os.path.join(path, 'q_values.csv')
  if os.path.isfile(file_name):
    with open(file_name, 'a') as f:
      for i,a,d,l,h in zip(iters, v_mean, v_std, v_min, v_max):
        f.write('{},{},{},{},{}\n'.format(i,a,d,l,h))
    iters, v_mean, v_std, v_min, v_max = np.loadtxt(
      file_name,
      delimiter=',',
      skiprows=1,
      unpack=True,      
    )
    iters, idx = np.unique(iters, return_index=True)
    v_mean = v_mean[idx]
    v_std = v_std[idx]
    v_min = v_min[idx]
    v_max = v_max[idx]
  else:
    with open(file_name, 'w') as f:
      f.write('Iters,Mean,StdDev,Min,Max\n')
      for i,a,d,l,h in zip(iters, v_mean, v_std, v_min, v_max):
        f.write('{},{},{},{},{}\n'.format(i,a,d,l,h))

  v_std_l = [m-d for m,d in zip(v_mean, v_std)]
  v_std_h = [m+d for m,d in zip(v_mean, v_std)]


  if fig is not None:
    plt.close()
  plt.plot(
    iters, v_mean, '-b',
    iters, v_std_h, '--b',
    iters, v_max, ':b',
    iters, v_std_l, '--b',
    iters, v_min, ':b',
  )
  plt.legend(['mean', '+/- std dev', 'max/min'])
  plt.xlabel('Iters')
  plt.ylabel('Q value')
  if show:
    plt.show()
  else:
    path, name = os.path.split(path)
    if name != '.':
      name += '_q_value'
    else:
      name = 'q_value'
    for ext in ['png','pdf']:
      plt.savefig(os.path.join(path,'plots',ext,'{}.{}'.format(name,ext)))
    plt.close()