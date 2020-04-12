import sys, os

import gym
from siamrl.envs import stack
import numpy as np
import time

if __name__=='__main__':
  try:
    N = int(sys.argv[1])
  except:
    N = 2**10
  try:
    L = int(sys.argv[2])
  except:
    L = 2**5

  #freqs = [60, 120, 240]
  freqs = [30]
  names = ['tn', 'tc']

  np.random.seed(0)
  actions = np.random.randint(15617, size=N*L, dtype='uint16')

  rewards = {}

  for freq in freqs:
    for name in names:
      env_id = stack.register(
        episode_length=L,  
        urdfs=name,
        sim_time_step=1/freq,
        positions_weight=1,
        seed=0
      )
      key='{}{}'.format(name[-1], freq)
      rewards[key] = []

      t = np.zeros(L+1)
      n = np.zeros(L+1)
      avg_rew = 0
      num_eps = 0
      with gym.make(env_id) as env:
        env.reset()
        i = 0
        reward = 0
        with open('.poses_'+key+'.csv','w') as f:
          f.write('x,y,z,qx,qy,qz,qw\n')

        for a in actions:
          i += 1
          t[i] -= time.process_time()
          _,r,d,_ = env.step(a)
          t[i] += time.process_time()
          n_steps = env.unwrapped._sim.n_steps
          n[i] += n_steps[0]+n_steps[1]
          reward += r
          if d:
            rewards[key].append(reward)
            avg_rew += reward
            reward = 0
            i = 0
            num_eps += 1

            with open('.poses_'+key+'.csv','a') as f:
              for p, o in env.unwrapped._sim.poses:
                f.write('{},{},{},{},{},{},{}\n'.format(*p,*o))

            t[i] -= time.process_time()
            env.reset()
            t[i] += time.process_time()

      avg_rew /= num_eps
      t /= num_eps
      n /= num_eps
      tpn = t/np.where(n!=0, n, 1)

      with open('.times_'+key+'.csv', 'w') as f:
        f.write('t,n,tpn\n')
        for i in zip(t,n,tpn):
          f.write('{},{},{}\n'.format(*i))

      with open('{}_{}.log'.format(L,N), 'a') as f:
        f.write('{}: reset {}; step {}, {}, {}; reward {}\n'.format(
          key, 
          t[0], 
          np.mean(t[1:]), 
          np.mean(n[1:]), 
          np.mean(tpn[1:]), 
          avg_rew))

  if os.path.isfile('.rewards.csv'):
    with open('.rewards.csv', 'r+') as f:
      lines = f.readlines()
      f.seek(0)
      f.write(','.join([lines[0][:-1],*rewards.keys()])+'\n')
      for line, values in zip(lines, *rewards.values()):
        f.write(','.join([line[:-1]]+[str(i) for i in line])+'\n')
  else:
    with open('.rewards.csv', 'w') as f:
      f.write(','.join(rewards.keys())+'\n')
      for line in zip(*rewards.values()):
        f.write(','.join([str(i) for i in line])+'\n')
