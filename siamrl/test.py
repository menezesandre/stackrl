from datetime import datetime
import os
import sys
import time

import gin
import gym
import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb

from siamrl import baselines
from siamrl import envs
from siamrl import load_policy

def test(
  env_id, 
  path='.',
  iters=None,
  num_steps=1024,
  compare_with=None,
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

  n_policies = len(policies)

  if compare_with:

    if not isinstance(compare_with, list):
      compare_with = [compare_with]

    hmean_overlap = tuple(tuple([] for _ in policies) for _ in compare_with)
    hstd_overlap = tuple(tuple([] for _ in policies) for _ in compare_with)
    distance = tuple(tuple([] for _ in policies) for _ in compare_with)
    correlation = tuple(tuple([] for _ in policies) for _ in compare_with)
      

    for m in compare_with:
      policies.append(baselines.Baseline(m, value=True))
      iters.append(m)

  if visualize:    
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

  for i in range(n_policies):
    if verbose and iters[i]:
      print('{}'.format(iters[i]))

    tr = 0.
    tv = 0.
    ne = 0
    env.seed(seed)
    o = env.reset()

    if gui:
      pb.removeAllUserDebugItems()
      # pb.configureDebugVisualizer(pb.COV_ENABLE_WIREFRAME,0)
      pb.resetDebugVisualizerCamera(1., 90, -45, [0.25,0.25,0])
      time.sleep(5*sleep)
  
    for n in range(num_steps):
      if visualize and plot_all or compare_with:
        actions_values = [p(o) for p in policies]
        a,v = actions_values[i]
        
        actions = [np.unravel_index(action, v_shape) for action,_ in actions_values]
        values = [(value.reshape(v_shape) - np.mean(value))/np.std(value) for _,value in actions_values]
      else:
        a,v = policies[i](o)

      if compare_with:
        for j, (baction,bvalue) in enumerate(zip(
          actions[-len(compare_with):], 
          values[-len(compare_with):], 
        )):
          for k, (action,value) in enumerate(zip(
            actions[:-len(compare_with)], 
            values[:-len(compare_with)], 
          )):
            # Overlap between values above mean
            hmean_overlap[j][k].append(
              np.count_nonzero(np.logical_and(bvalue>0,value>0)) /
              np.count_nonzero(np.logical_or(bvalue>0,value>0))
            )
            # Overlap between values one standard deviation above mean
            hstd_overlap[j][k].append(
              np.count_nonzero(np.logical_and(bvalue>1,value>1)) /
              np.count_nonzero(np.logical_or(bvalue>1,value>1))
            )
            # Euclidian distance (in pixels) between actions to compare
            distance[j][k].append(np.linalg.norm(np.subtract(baction,action)))
            # Correlation coefficient between values
            correlation[j][k].append(np.corrcoef(bvalue.ravel(), value.ravel())[0,1])

      if visualize:
        o0, o1 = env.render(mode='rgb_array')
        axs[0][0].cla()
        axs[0][0].imshow(o0)
        axs[1][0].cla()
        axs[1][0].imshow(o1)
        if plot_all:
          for j,value in enumerate(values):
            axs[0][j+1].cla()
            axs[0][j+1].imshow(value)
            axs[1][j+1].cla()
            axs[1][j+1].imshow(np.where(value>1, value, 1))
            if j == i:
              axs[1][j+1].set_xlabel('Current policy')
        else:
          threshold = np.mean(v) + np.std(v)
          value = v.reshape(v_shape)

          axs[0][1].cla()
          axs[0][1].imshow(value)
          axs[1][1].cla()
          axs[1][1].imshow(np.where(value>threshold, value, threshold))
          
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
          if compare_with:
            for j in range(len(compare_with)):
              print('  Compare with {}: Overlap {} {}, Distance {}, Correlation {}'.format(
                iters[n_policies+j],
                np.mean(hmean_overlap[j][i]),
                np.mean(hstd_overlap[j][i]),
                np.mean(distance[j][i]),
                np.mean(correlation[j][i]),
              ))
        o=env.reset()

    if verbose:
      print('Final average: Reward {}, Value {}'.format(tr/ne, tv/ne))
  
  if visualize:
    plt.close()
  
  if compare_with:

    for j in range(len(policies)-len(compare_with)):
      for i in range(len(compare_with)):
        print('{} with {}'.format(iters[j], iters[n_policies+i]))
        print('  Overlap between regions with value higher than mean: avg {} (std {})'.format(
          np.mean(hmean_overlap[i][j]),
          np.std(hmean_overlap[i][j]),
        ))
        print('  Overlap between regions with value at least 1 std dev above mean: avg {} (std {})'.format(
          np.mean(hstd_overlap[i][j]),
          np.std(hstd_overlap[i][j]),
        ))
        print('  Correlation coefficients between value maps: avg {} (std {})'.format(
          np.mean(correlation[i][j]),
          np.std(correlation[i][j]),
        ))
        print('  Distance between actions: avg {} (std {})'.format(
          np.mean(distance[i][j]),
          np.std(distance[i][j]),
        ))
        plt.hist(distance[i][j], bins=16)
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.show()


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
    elif arg == '--iters':
      kwargs['iters'] = argv.pop().split(',')
    elif arg == '-n':
      kwargs['num_steps'] = int(argv.pop())
    elif arg == '--compare':
      kwargs['compare_with'] = argv.pop().split(',')
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