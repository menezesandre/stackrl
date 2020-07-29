"""
Baseline policies with no learning, using 
  OpenCV's Template Matching
"""
from datetime import datetime
import time
import os

import gym
import numpy as np

from siamrl import envs
from siamrl.agents import policies

def _goal_overlap(observation):
  g = observation[0][:,:,1]
  t = observation[1][:,:,0]
  shape = np.subtract(g.shape, t.shape) + 1
  overlap = np.zeros(shape)

  for i in range(shape[0]):
    for j in range(shape[1]):
      overlap[i,j] = np.sum(
        g[i:i+t.shape[0],j:j+t.shape[1]]*t
      )

  return overlap/np.max(overlap)
  
def random(observation, *args, **kwargs):
  return np.random.rand(
    observation[0].shape[0]-observation[1].shape[0]+1,
    observation[0].shape[1]-observation[1].shape[1]+1
  )

def lowest(observation, *args, **kwargs):
  x = observation[0][:,:,0]
  w = observation[1][:,:,0]
  shape = np.subtract(x.shape, w.shape) + 1
  h = np.zeros(shape)
  for i in range(shape[0]):
    for j in range(shape[1]):
      h[i,j] = np.max(np.where(
        w > 0.,
        x[i:i+w.shape[0], j:j+w.shape[0]] + w,
        0.,
      ))
  return -h

def closest(observation, *args, **kwargs):
  x = observation[0][:,:,0]
  w = observation[1][:,:,0]
  shape = np.subtract(x.shape, w.shape) + 1
  d = np.zeros(shape)
  for i in range(shape[0]):
    for j in range(shape[1]):
      h = np.where(
        w > 0.,
        x[i:i+w.shape[0], j:j+w.shape[0]] + w,
        0.,
      )
      h -= np.where(
        h != 0.,
        np.max(h),
        0.,
      )
      d[i,j] = np.sum(h)/np.count_nonzero(h)
  return d

try:
  import cv2 as cv

  def ccoeff(observation, **kwargs):
    normed = kwargs.get('normed', True)

    img = observation[0][:,:,0]
    tmp = observation[1][:,:,0]
    return -cv.matchTemplate(  # pylint: disable=no-member
      img, 
      tmp, 
      cv.TM_CCOEFF_NORMED if normed else cv.TM_CCOEFF  # pylint: disable=no-member
    )

  def gradcorr(observation, **kwargs):
    normed = kwargs.get('normed', True)

    img = observation[0][:,:,0]
    tmp = observation[1][:,:,0]

    img_x = cv.Sobel(img, cv.CV_32F, 1, 0)  # pylint: disable=no-member
    img_y = cv.Sobel(img, cv.CV_32F, 0, 1)  # pylint: disable=no-member
    tmp_x = cv.Sobel(tmp, cv.CV_32F, 1, 0)  # pylint: disable=no-member
    tmp_y = cv.Sobel(tmp, cv.CV_32F, 0, 1)  # pylint: disable=no-member

    img = cv.merge([img_x, img_y])  # pylint: disable=no-member
    tmp = cv.merge([tmp_x, tmp_y])  # pylint: disable=no-member
    
    return -cv.matchTemplate(  # pylint: disable=no-member
      img, 
      tmp, 
      cv.TM_CCORR_NORMED if normed else cv.TM_CCORR  # pylint: disable=no-member
    )
  
except ImportError:
  ccoeff = None
  gradcorr = None

methods = {
  'random': random,
  'lowest': lowest,
  'closest': closest,
  'ccoeff': ccoeff,
  'gradcorr': gradcorr
}

class Baseline(policies.PyGreedy):
  def __init__(
    self, 
    method='random', 
    goal=True, 
    value=False, 
    unravel=False,
    batched=False,
    batchwise=False,
    **kwargs,
  ):
    if isinstance(method, str):
      method_list = method.lower().split('-')
      for i,m in enumerate(method_list):
        if m in methods:
          if methods[m] is not None:
            method_list[i] = methods[m]
          else:
            raise ImportError(
              "opencv-python must be installed to use {} method.".format(m)
            )
        else:
          raise ValueError(
            "Invalid value {} for argument method. Must be in {}".format(method, methods)
          )
    elif not callable(method):
      raise TypeError(
        "Invalid type {} for argument method.".format(type(method))
      )

    if len(method_list) > 1:
      def method(inputs, **kwargs):  # pylint: disable=function-redefined
        values = [m(inputs,**kwargs) for m in method_list]
        value = np.ones_like(values[0])
        for v in values:
          value *= v-np.min(v)
        return value
    else:
      method = method_list[0]

    def model(inputs):
      values = method(inputs, **kwargs)
      if goal:
        min_values = np.min(values) - 1e-12 # slightly smaller than the minimum value
        values = min_values + (values-min_values)*_goal_overlap(inputs)
      return values
    
    super(Baseline, self).__init__(
      model, 
      value=value, 
      unravel=unravel,
      batched=batched,
      batchwise=batchwise,
    )

def test(env_id, method=None, num_steps=1024, verbose=False, visualize=False, gui=False, save_results=None,sleep=0.5, seed=11):
  if save_results is None:
    save_results = method is None

  env = gym.make(env_id, use_gui=gui, seed=seed)
  batched = len(env.observation_space[0].shape) == 4

  if method:
    if isinstance(method, str):
      method = method.split(',')
    elif not hasattr(method, '__len__'):
      method = (method,)
    policies = {str(m):Baseline(method=m, batched=batched, batchwise=batched) for m in method}
  else:
    policies = {m:Baseline(method=m, batched=batched, batchwise=batched) for m in methods}
    policies['random (anywhere)'] = lambda o: env.action_space.sample()

  results = {} if save_results else None

  sleep = 1. if sleep > 1 else sleep
  sleep = 0. if sleep < 0 else sleep

  for name, policy in policies.items():
    if not env:
      env = gym.make(env_id, use_gui=gui, seed=seed)
    
    tr = 0.
    ne = 0
    o = env.reset()

    if gui:
      import pybullet as pb
      pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
      pb.resetDebugVisualizerCamera(.5, 90, -75, (0.25,0.25,0))
      time.sleep(5*sleep)
    
    if verbose:
      print(name.capitalize())
    for _ in range(num_steps):
      if gui:
        time.sleep(sleep-(datetime.now().microsecond/1e6)%sleep)
      o,r,d,_=env.step(policy(o))
      print(r)
      tr+=r
      if d:
        ne+=1
        if verbose:
          print('  Current average ({}): {}'.format(
          ne,
          tr/ne
        ))
        o=env.reset()

    if results is not None:
      results[name]=tr/ne
    if verbose:
      print('Final average: {}'.format(tr/ne))
    del(env)
    env = None

  if results:
    fpath = os.path.join(
      os.path.dirname(__file__),
      '..',
      'data',
      'baselines',
      envs.utils.as_path(env_id),
    )
    fname = os.path.join(fpath, 'results')
    if not os.path.isdir(fpath):
      os.makedirs(fpath)
    with open(fname, 'w') as f:
      for k,v in results.items():
        f.write('{}:{}\n'.format(k,v))
    
    return results