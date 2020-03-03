from siamrl.utils import register
import gym
from numpy import random
from time import time
import matplotlib.pyplot as plt

def env_timer(N, reg_fn, **kwargs):
  env_id = reg_fn(**kwargs)
  env = gym.make(env_id, info=True)
  env.reset()
  step_count = 0
  step_t = 0
  step_n = 0
  reset_t = 0
  reset_n = 0  
  for i in range(N):
    a = random.randint(env.action_space.n)
    t = time()
    _,r,done,info = env.step(a)
    print(r)
    t = time() - t
    if 'steps' in info.keys():
      step_count += info['steps']
    step_t += t
    step_n += 1
    if done:
      t = time()
      _ = env.reset()
      t = time() - t
      reset_t += t
      reset_n += 1
  return step_count/step_n, step_t/step_n, reset_t/reset_n

N = 1000
n = 32
#kwargs_list = [{'num_objects':i} for i in range(30)]

if __name__=='!__main__':
  x = list(range(1,n+1))
  sc = []
  st = []
  rt = []
  for i in x:
    times = env_timer(N, register.stack_env, num_objects=i)
    sc.append(times[0])
    st.append(times[1])
    rt.append(times[2])
    print(i)
  msc = max(sc)
  sc = [i/msc for i in sc]
  mst = max(st)
  st = [i/mst for i in st]
  mrt = max(rt)
  rt = [i/mrt for i in rt]

  plt.plot(x,sc,x,st,x,rt)
  plt.legend(['Avg step count (x%f)'%msc, 'Avg step time (x%f)'%mst, 'Avg reset time (x%f)'%mrt])
  plt.savefig('reset_res.png')

if __name__=='__main__':
  _,st,rt = env_timer(N, register.goal_env, gui=True, num_objects=50)
  print(st, rt)