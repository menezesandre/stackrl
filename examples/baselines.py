import sys
import time
from datetime import datetime
import gym
from siamrl.baselines import Baseline
from siamrl.envs.stack import register

# policies = {
#   'random': baselines.random, # 0.18017877265810966
#   'ccoeff': baselines.ccoeff, # 0.2157407896593213
#   'gradcorr': baselines.gradcorr #0.2028804305009544
# }

policy = None
ns = 1024
gui = True

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
      policy = Baseline(method=arg)
  
  env_id = register(
    episode_length=32,
    urdfs='test',
    observable_size_ratio=4,
    goal_size_ratio=0.375,
    sim_time_step=1/60,
    max_z=0.5,
    use_gui=gui, 
    occupation_ratio_weight=1,
    seed=11,
    dtype='uint8'
  )
  env = gym.make(env_id)
    
  policy = policy or (lambda o: env.action_space.sample())

  tr = 0.
  ne = 0
  o = env.reset()
  if gui:
    import pybullet as pb
    pb.resetDebugVisualizerCamera(1., 90, -30, [0.25,0.25,0])
    time.sleep(5.)
  for _ in range(ns):
    if gui:
      time.sleep(0.5-(datetime.now().microsecond/1e6)%0.5)
    o,r,d,_=env.step(policy(o))
    tr+=r
    if d:
      ne+=1
      print('Current average ({}): {}'.format(
        ne,
        tr/ne
      ))
      o=env.reset()

  print('Final average: {}'.format(tr/ne))