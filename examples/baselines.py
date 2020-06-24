import sys
import time
from datetime import datetime

import gym

from siamrl import baselines
from siamrl.envs.stack import register

# policies = {
#   'random': baselines.random, # 0.18017877265810966
#   'ccoeff': baselines.ccoeff, # 0.2157407896593213
#   'gradcorr': baselines.gradcorr #0.2028804305009544
# }

policy = None
ns = 1024
gui = False
methods = []

if __name__=='__main__':
  """Raises KeyError if provided argument isn't one of the implemented 
  baseline policies."""
  argv = sys.argv[:0:-1]
  while argv:
    arg=argv.pop()
    if arg == '-n':
      ns = int(argv.pop())
    elif arg == '-g':
      gui = True
    else:
      methods.append(arg)

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
  
  if methods:
    policies = {m: baselines.Baseline(method=m) for m in methods}
  else:
    policies = {m:baselines.Baseline(method=m) for m in baselines.methods}
    policies['random (anywhere)'] = lambda o: env.action_space.sample()

  results={}
  for name, policy in policies.items():
    env = gym.make(env_id)
    
    tr = 0.
    ne = 0
    o = env.reset()

    if gui:
      import pybullet as pb
      pb.resetDebugVisualizerCamera(1., 90, -30, [0.25,0.25,0])
      time.sleep(3.)
    
    print('__ {} __'.format(name))
    for _ in range(ns):
      if gui:
        time.sleep(0.5-(datetime.now().microsecond/1e6)%0.5)
      o,r,d,_=env.step(policy(o))
      tr+=r
      if d:
        ne+=1
        print('  Current average ({}): {}'.format(
          ne,
          tr/ne
        ))
        o=env.reset()

    results[name]=tr/ne
    print('Final average: {}'.format(results[name]))
    del(env)

  fname = 'baselines'
  for k,v in kwargs.items():
    if isinstance(v, float):
      fname += '_{}_{:.3}'.format(k,v)
    else:
      fname += '_{}_{}'.format(k,v)
  with open(fname, 'a') as f:
    for k,v in results.items():
      f.write('{}:{}\n'.format(k,v))