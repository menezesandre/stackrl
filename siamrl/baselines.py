"""
Baseline policies with no learning, using 
  heuristics and/or OpenCV's Template Matching
"""
from datetime import datetime
import time
import os

import gym
import numpy as np
from scipy import signal

# from siamrl import envs
from siamrl.agents import policies

def _apply_limit_and_exponent(inputs, **kwargs):
  limit = kwargs.get('limit', 0)
  exponent = kwargs.get('exponent', 1)

  if isinstance(limit, str):
    if limit=='mean':
      limit = inputs.mean()
    elif limit=='std':
      limit = inputs.mean() + inputs.std()
    else:
      raise ValueError('Invalid value {} for argument limit'.format(limit))



  if limit > 0:
    if limit < 1:
      inputs = np.maximum((inputs-limit)/(1-limit), 0)
    else:
      return np.where(inputs==1, 1., 0.)

  if exponent != 1:
    inputs **= exponent

  return inputs

def _apply_scale(inputs, **kwargs):
  mask = kwargs.get('mask', None)
  if mask is not None:
    inmax = np.max(np.where(mask, inputs, -np.inf))
    inmin = np.min(np.where(mask, inputs, np.inf))
    inputs = np.where(mask, inputs, inmin)
  else:
    inmax = inputs.max()
    inmin = inputs.min()
  
  if inmax > inmin:
    return (inputs - inmin)/(inmax-inmin)
  else:
    return np.ones_like(inputs)
    
def goal_overlap(observation, previous=None, **kwargs):
  """Overlap between object and goal.

  Returned values are computed as
    max((overlap-limit)/(1-limit), 0)**exponent

  """
  g = np.where(observation[0][:,:,1]>0, 1, 0)
  t = np.where(observation[1][:,:,0]>0, 1, 0)
  overlap = signal.convolve2d(g, t, mode='valid')/np.count_nonzero(t)
  
  overlap = _apply_limit_and_exponent(overlap, **kwargs)

  if previous is not None:
    overlap *= previous
  return overlap
  
def random(observation, previous=None, **kwargs):
  return np.random.rand(
    observation[0].shape[0]-observation[1].shape[0]+1,
    observation[0].shape[1]-observation[1].shape[1]+1
  )*(previous if previous is not None else 1)

def lowest(observation, previous=None, limit=0, exponent=1, **kwargs):
  x = observation[0][:,:,0]
  w = observation[1][:,:,0]

  shape = np.subtract(x.shape, w.shape) + 1
  h = np.zeros(shape)
  wbin = w > 0.

  if previous is not None:
    # Use previous as a mask to compute only values that will be used
    if previous.shape != tuple(shape):
      previous.reshape(shape)
    mask = previous > 0
    irange = range(
      mask.argmax(),
      mask.size-np.argmax(np.flip(mask)),
    )
  else:
    irange = range(h.size)
    mask = None

  for idx in irange:
    i,j = np.unravel_index(idx, shape)  # pylint: disable=unbalanced-tuple-unpacking
    if mask is None or mask[i,j]:
      h[i,j] = np.max(np.where(
        wbin,
        x[i:i+w.shape[0], j:j+w.shape[1]] + w,
        0,
      ))

  h = _apply_scale(-h, mask=mask, **kwargs)
  h = _apply_limit_and_exponent(h, **kwargs)
  if previous is not None:
    h *= previous

  return h

def closest(observation, previous=None, **kwargs):
  x = observation[0][:,:,0]
  w = observation[1][:,:,0]
  shape = np.subtract(x.shape, w.shape) + 1
  d = np.zeros(shape)

  wbin = w > 0.
  wnz = np.count_nonzero(w)

  if previous is not None:
    if previous.shape != tuple(shape):
      previous.reshape(shape)
    mask = previous > 0
    irange = range(
      mask.argmax(),
      mask.size-np.argmax(np.flip(mask)),
    )
  else:
    irange = range(d.size)
    mask = None

  for idx in irange:
    i,j = np.unravel_index(idx, shape)  # pylint: disable=unbalanced-tuple-unpacking
    if mask is None or mask[i,j]:
      h = np.where(
        wbin,
        x[i:i+w.shape[0], j:j+w.shape[1]] + w,
        0.,
      )
      h -= np.where(
        wbin,
        h.max(),
        0.,
      )
      d[i,j] = h.sum()/wnz

  d = _apply_scale(d, mask=mask, **kwargs)
  d = _apply_limit_and_exponent(d, **kwargs)

  if previous is not None:
    d *= previous

  return d

try:
  import cv2 as cv

  def ccoeff(observation, previous=None, **kwargs):
    normed = kwargs.get('normed', True)

    img = observation[0][:,:,0]
    tmp = observation[1][:,:,0]
    c = cv.matchTemplate(  # pylint: disable=no-member
      img, 
      tmp, 
      cv.TM_CCOEFF_NORMED if normed else cv.TM_CCOEFF  # pylint: disable=no-member
    )
    c = _apply_scale(
      -c, 
      mask = (previous > 0 if previous is not None else None), 
      **kwargs
    )
    c = _apply_limit_and_exponent(c, **kwargs)

    if previous is not None:
      c *= previous
    
    return c

  def gradcorr(observation, previous=None, **kwargs):
    normed = kwargs.get('normed', True)

    img = observation[0][:,:,0]
    tmp = observation[1][:,:,0]

    img_x = cv.Sobel(img, cv.CV_32F, 1, 0)  # pylint: disable=no-member
    img_y = cv.Sobel(img, cv.CV_32F, 0, 1)  # pylint: disable=no-member
    tmp_x = cv.Sobel(tmp, cv.CV_32F, 1, 0)  # pylint: disable=no-member
    tmp_y = cv.Sobel(tmp, cv.CV_32F, 0, 1)  # pylint: disable=no-member

    img = cv.merge([img_x, img_y])  # pylint: disable=no-member
    tmp = cv.merge([tmp_x, tmp_y])  # pylint: disable=no-member
    
    c = cv.matchTemplate(  # pylint: disable=no-member
      img, 
      tmp, 
      cv.TM_CCORR_NORMED if normed else cv.TM_CCORR  # pylint: disable=no-member
    )

    c = _apply_scale(
      -c, 
      mask = (previous > 0 if previous is not None else None), 
      **kwargs
    )
    c = _apply_limit_and_exponent(c, **kwargs)
    if previous is not None:
      c *= previous
    
    return c
 
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
    elif callable(method):
      method_list = [method]
    else:
      raise TypeError(
        "Invalid type {} for argument method.".format(type(method))
      )

    def model(inputs):
      if goal:
        values = goal_overlap(inputs, **kwargs)
      else:
        values = None
      for m in method_list:
        values = m(inputs, previous=values, **kwargs)
      return values
    
    super(Baseline, self).__init__(
      model, 
      value=value, 
      unravel=unravel,
      batched=batched,
      batchwise=batchwise,
    )

# def test(env_id, method=None, num_steps=1024, verbose=False, visualize=False, gui=False, save_results=None,sleep=0.5, seed=11):
#   if save_results is None:
#     save_results = method is None

#   env = gym.make(env_id, use_gui=gui, seed=seed)
#   batched = len(env.observation_space[0].shape) == 4

#   if method:
#     if isinstance(method, str):
#       method = method.split(',')
#     elif not hasattr(method, '__len__'):
#       method = (method,)
#     policies = {str(m):Baseline(method=m, batched=batched, batchwise=batched) for m in method}
#   else:
#     policies = {m:Baseline(method=m, batched=batched, batchwise=batched) for m in methods}
#     policies['random (anywhere)'] = lambda o: env.action_space.sample()

#   results = {} if save_results else None

#   sleep = 1. if sleep > 1 else sleep
#   sleep = 0. if sleep < 0 else sleep

#   for name, policy in policies.items():
#     if not env:
#       env = gym.make(env_id, use_gui=gui, seed=seed)
    
#     tr = 0.
#     ne = 0
#     o = env.reset()

#     if gui:
#       import pybullet as pb
#       pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
#       pb.resetDebugVisualizerCamera(.5, 90, -75, (0.25,0.25,0))
#       time.sleep(5*sleep)
    
#     if verbose:
#       print(name.capitalize())
#     for _ in range(num_steps):
#       if gui:
#         time.sleep(sleep-(datetime.now().microsecond/1e6)%sleep)
#       o,r,d,_=env.step(policy(o))
#       print(r)
#       tr+=r
#       if d:
#         ne+=1
#         if verbose:
#           print('  Current average ({}): {}'.format(
#           ne,
#           tr/ne
#         ))
#         o=env.reset()

#     if results is not None:
#       results[name]=tr/ne
#     if verbose:
#       print('Final average: {}'.format(tr/ne))
#     del(env)
#     env = None

#   if results:
#     fpath = os.path.join(
#       os.path.dirname(__file__),
#       '..',
#       'data',
#       'baselines',
#       envs.utils.as_path(env_id),
#     )
#     fname = os.path.join(fpath, 'results')
#     if not os.path.isdir(fpath):
#       os.makedirs(fpath)
#     with open(fname, 'w') as f:
#       for k,v in results.items():
#         f.write('{}:{}\n'.format(k,v))
    
#     return results