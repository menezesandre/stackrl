from datetime import datetime
import os
import sys
import time

import gin
import gym
import numpy as np

from siamrl import envs
from siamrl import load_policy

def test(
  env_id, 
  path='.',
  iters=None,
  num_steps=1024,
  verbose=True,
  visualize=False,
  gui=False, 
  sleep=0.5, 
  seed=11
):
  sleep = 1. if sleep > 1 else 0. if sleep < 0 else sleep

  env = gym.make(env_id, use_gui=gui)
  policies = load_policy(env.observation_space, path=path, iters=iters, value=True)
  if not isinstance(policies, list):
    policies = [policies]
    iters = [iters]

  if visualize:
    import matplotlib.pyplot as plt
    
    plot_all = len(policies) > 1 and len(policies) < 5
    if plot_all:
      fig, axs = plt.subplots(
        2,1+len(policies), 
        gridspec_kw={'height_ratios':[4, 1]}
      )
    else:
      fig, axs = plt.subplots(
        2,2, 
        gridspec_kw={'height_ratios':[4, 1]}
      )
    v_shape = (
      env.observation_space[0].shape[0]-env.observation_space[1].shape[0]+1,
      env.observation_space[0].shape[1]-env.observation_space[1].shape[1]+1
    )

  for i in range(len(policies)):
    if verbose and iters[i]:
      print('__ {} __'.format(iters[i]))

    tr = 0.
    tv = 0.
    ne = 0
    env.seed(seed)
    o = env.reset()

    if gui:
      import pybullet as pb
      pb.resetDebugVisualizerCamera(1., 90, -45, [0.25,0.25,0])
      time.sleep(5*sleep)
  
    for n in range(num_steps):
      if visualize and plot_all:
        values = [p(o) for p in policies]
        a,v = values[i]
        values = [value for action,value in values]
      else:
        a,v = policies[i](o)

      if visualize:
        o0, o1 = env.render(mode='rgb_array')
        axs[0][0].cla()
        axs[0][0].imshow(o0)
        axs[1][0].cla()
        axs[1][0].imshow(o1)
        if plot_all:
          for j,value in enumerate(values):
            h_value = np.mean(value)
            h_value += np.std(value)
            value = value.reshape(v_shape)

            axs[0][j+1].cla()
            axs[0][j+1].imshow(value)
            axs[1][j+1].cla()
            axs[1][j+1].imshow(np.where(value>h_value, value, h_value))
            if j == i:
              axs[1][j+1].set_xlabel('Current policy')
        else:
          value = v
          h_value = np.mean(value)
          h_value += np.std(value)
          value = v.reshape(v_shape)

          axs[0][1].cla()
          axs[0][1].imshow(value)
          axs[1][1].cla()
          axs[1][1].imshow(np.where(value>h_value, value, h_value))
          
      if visualize:
        fig.show()
        plt.pause(sleep-(datetime.now().microsecond/1e6)%sleep)
      elif gui:
        time.sleep(sleep-(datetime.now().microsecond/1e6)%sleep)

      o,r,d,_ = env.step(a)
      tr += r
      tv += v[a]
      if d:
        ne+=1
        if verbose:
          print('  Current average ({}): Reward {}, Value {}'.format(
            ne,
            tr/ne,
            tv/n
          ))
        o=env.reset()

    if verbose:
      print('Final average: Reward {}, Value {}'.format(tr/ne, tv/ne))
  
  if visualize:
    plt.close()

if __name__ == '__main__':
  # Default args
  path,config_file,env = '.','config.gin',None
  # Parse arguments
  argv = sys.argv[:0:-1]
  kwargs = {}
  while argv:
    arg=argv.pop()
    if arg == '--config':
      config_file = argv.pop()
    elif arg == '-e':
      env = argv.pop()
    elif arg == '-i':
      kwargs['iters'] = argv.pop().split(',')
    elif arg == '-q':
      kwargs['verbose'] = False
    elif arg == '-v':
      kwargs['visualize'] = True
    elif arg == '--gui':
      kwargs['gui'] = True
    elif arg == '-t':
      kwargs['sleep'] = float(argv.pop())
    elif arg == '--seed':
      kwargs['seed'] = int(argv.pop())
    else:
      path = arg
  # If env is not provided, register it with args binded from config_file
  if not env:
    if config_file:
      try:
        gin.parse_config_file(os.path.join(path, config_file))
      except OSError:
        gin.parse_config_file(config_file)
    env = envs.stack.register()

  test(env, path, **kwargs)