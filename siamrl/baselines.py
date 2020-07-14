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

def _trimmed_goal(observation):
  shape = observation[1].shape
  top = shape[0]//2 + 1
  bottom = shape[0] - top - 1
  left = shape[1]//2 + 1
  right = shape[1] - left - 1
  return observation[0][top:-bottom, left:-right,1]

def random(observation):
  return np.random.rand(
    observation[0].shape[0]-observation[1].shape[0]+1,
    observation[0].shape[1]-observation[1].shape[1]+1
  )

try:
  import cv2 as cv

  def ccoeff(observation, normed=True):
    img = observation[0][:,:,0]
    tmp = observation[1][:,:,0]
    return -cv.matchTemplate(  # pylint: disable=no-member
      img, 
      tmp, 
      cv.TM_CCOEFF_NORMED if normed else cv.TM_CCOEFF  # pylint: disable=no-member
    )

  def gradcorr(observation, normed=True):
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
      method = method.lower()
      if method in methods:
        method = methods[method]
        if method is None:
          raise ImportError(
            "opencv-python must be installed to use {} method.".format(
              method
            )
          )
      else:
        raise ValueError(
          "Invalid value {} for argument method. Must be in {}".format(method, methods)
        )
    elif not callable(method):
      raise TypeError(
        "Invalid type {} for argument method.".format(type(method))
      )

    def model(inputs):
      values = method(inputs, **kwargs)
      if goal:
        values = np.where(
          _trimmed_goal(inputs) > 0.,
          values,
          np.min(values) - 1e-12, # slightly smaller than the minimum value
        )
      return values
    
    super(Baseline, self).__init__(
      model, 
      value=value, 
      unravel=unravel,
      batched=batched,
      batchwise=batchwise,
    )

def test(env_id, method=None, num_steps=1024, verbose=False, gui=False, sleep=0.5, seed=11):
  env = gym.make(env_id, use_gui=gui, seed=seed)
  if len(env.observation_space[0].shape) == 4:
    batched, batchwise = True,True
  else:
    batched, batchwise = False,False

  if method:
    policies = {method:Baseline(method=method, batched=batched, batchwise=batchwise)}
    results=None
  else:
    policies = {m:Baseline(method=m, batched=batched, batchwise=batchwise) for m in methods}
    policies['random (anywhere)'] = lambda o: env.action_space.sample()
    results={}

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
      pb.resetDebugVisualizerCamera(1., 90, -30, [0.25,0.25,0])
      time.sleep(5*sleep)
    
    if verbose:
      print('__ {} __'.format(name))
    for _ in range(num_steps):
      if gui:
        time.sleep(sleep-(datetime.now().microsecond/1e6)%sleep)
      o,r,d,_=env.step(policy(o))
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
      'results',
    )
    fname = os.path.join(fpath, 'results')
    if not os.path.isdir(fpath):
      os.makedirs(fpath)
    with open(fname, 'w') as f:
      for k,v in results.items():
        f.write('{}:{}\n'.format(k,v))
    
    return results