"""
Baseline policies with no learning, using heuristics
"""
from datetime import datetime
import time
import os

import gin
import gym
import numpy as np
from scipy import signal
from scipy import ndimage

# from siamrl import envs
from siamrl import agents

def get_inputs(inputs, mask=None):
  """Extract and normalize the arrays from inputs"""
  gmax = inputs[0][:,:,1].max()
  o = inputs[0][:,:,0]/gmax
  n = inputs[1][:,:,0]/gmax
  f = np.zeros(np.subtract(o.shape, n.shape)+1)

  if mask is not None:
    if not isinstance(mask, np.ndarray):
      mask = np.array(mask, dtype='bool')
    if mask.size != f.size:
      raise ValueError("Mask doesn't match the output shape")
    elif mask.shape != f.shape:
      mask = mask.reshape(f.shape)

  return o,n,f,mask

def height(inputs, mask=None, add_gradient=False, **kwargs):
  """Height based heuristic."""
  if add_gradient:
    mask = None
  o,n,f,mask = get_inputs(inputs, mask)

  n_where = n > 0

  for i in range(f.shape[0]):
    for j in range(f.shape[1]):
      if mask is None or mask[i,j]:
        f[i,j] = np.max(np.where(
          n_where,
          o[i:i+n.shape[0], j:j+n.shape[0]] + n,
          0,
        ))

  if add_gradient:
    dx,dy = np.gradient(f)
    f += np.sqrt(dx**2 + dy**2)*float(add_gradient)

  return f

def difference(inputs, mask=None, difference_exponent=2, weights_exponent=0, return_height=False, **kwargs):
  """Difference based heuristic."""
  o,n,f,mask = get_inputs(inputs, mask)

  height = np.zeros_like(f) if return_height else None

  n_where = n > 0

  if weights_exponent > 0:
    _wi = (np.arange(n.shape[0], dtype='float') - n.shape[0]/2)**2
    _wj = (np.arange(n.shape[1], dtype='float') - n.shape[1]/2)**2
    w = (_wi[:,np.newaxis] + _wj[np.newaxis,:])**(weights_exponent/2)
    w = np.where(n_where, w, 0)
    w /= w.sum()
  else:
    w = n_where.astype('float')
    w /= w.sum()
 
  for i in range(f.shape[0]):
    for j in range(f.shape[1]):
      if mask is None or mask[i,j]:
        h = o[i:i+n.shape[0], j:j+n.shape[0]] + n
        h0 = np.max(np.where(n_where, h, 0))
        f[i,j] = np.sum(w*np.abs(h0 - h)**difference_exponent)

        if height is not None:
          height[i,j] = h0

  if height is not None:
    f = f, height

  return f

def corrcoef(inputs, mask=None, **kwargs):
  """Correlation coefficient based heuristic."""
  o,n,f,mask = get_inputs(inputs, mask)

  n_where = n > 0
  n_count = np.count_nonzero(n_where)
  n -= np.sum(np.where(n_where, n, 0))/n_count
  n_var = np.sum(np.where(n_where, n**2, 0))

  if n_var == 0:
    # No need to calculate, everything will be zero
    return f

  for i in range(f.shape[0]):
    for j in range(f.shape[1]):
      if mask is None or mask[i,j]:
        o_ = o[i:i+n.shape[0], j:j+n.shape[0]] - np.sum(np.where(
          n_where,
          o[i:i+n.shape[0], j:j+n.shape[0]],
          0,
        ))/n_count
        o_var = np.sum(np.where(n_where, o_**2, 0))

        if o_var != 0:
          f[i,j] = np.sum(np.where(n_where, n*o_, 0))/np.sqrt(n_var*o_var)

  return f

def gradcorr(inputs, mask=None, **kwargs):
  """Correlation coefficient based heuristic."""
  o,n,f,mask = get_inputs(inputs, mask)

  n_dx, n_dy = np.gradient(n)
  n_varx = np.sum(n_dx**2)
  n_vary = np.sum(n_dy**2)

  o_dx, o_dy = np.gradient(o)

  if n_varx == 0 and n_vary == 0:
    # No need to calculate, everything will be zero
    return f

  for i in range(f.shape[0]):
    for j in range(f.shape[1]):
      if mask is None or mask[i,j]:

        o_varx = np.sum(o_dx[i:i+n.shape[0], j:j+n.shape[0]]**2)
        o_vary = np.sum(o_dy[i:i+n.shape[0], j:j+n.shape[0]]**2)
        
        if o_varx != 0 and n_varx != 0:
          f[i,j] += np.sum(o_dx[i:i+n.shape[0], j:j+n.shape[0]]*n_dx)/(2*np.sqrt(o_varx*n_varx))
        if o_vary != 0 and n_vary != 0:
          f[i,j] += np.sum(o_dy[i:i+n.shape[0], j:j+n.shape[0]]*n_dy)/(2*np.sqrt(o_vary*n_vary))

  return f

def filtered(inputs, mask=None, add_gradient=False, **kwargs):
  if add_gradient:
    mask = None

  o,n,f,mask = get_inputs(inputs, mask)
  
  f = signal.convolve2d(o,n,mode='valid')/n.sum()

  # n /= n.sum()
  # for i in range(f.shape[0]):
  #   for j in range(f.shape[1]):
  #     if mask is None or mask[i,j]:
  #       f[i,j] = np.sum(o[i:i+n.shape[0], j:j+n.shape[0]]*n)

  if add_gradient:
    dx,dy = np.gradient(f)
    f += np.sqrt(dx**2 + dy**2)*float(add_gradient)

  return f

def random(inputs, seed=None, **kwargs):
  """Returns random values in the same shape as the heuristics."""
  rng = np.random.default_rng(seed)
  return rng.rand(
    *(np.subtract(inputs[0].shape, inputs[1].shape)[:-1]+1)
  )

def goal_overlap(inputs, threshold=2/3, **kwargs):
  b = (inputs[0][:,:,0] < inputs[0][:,:,1]).astype('int')
  n = (inputs[1][:,:,0] > 0).astype('int')
  f = signal.convolve2d(b, n, mode='valid')
  return f > threshold*f.max()

methods = {
  'random': random,
  'height': height,
  'difference': difference,
  'corrcoef':corrcoef,
  'gradcorr':gradcorr,
  'filtered':filtered,
}

@gin.configurable(module='siamrl')
class Baseline(agents.PyGreedy):
  def __init__(
    self, 
    method='random', 
    goal=True,
    minorder=0,
    value=False, 
    unravel=False,
    batched=False,
    batchwise=False,
    **kwargs,
  ):
    if isinstance(method, str):
      if method in methods:
        method = methods[method]
      else:
        raise ValueError(
          "Invalid value {} for argument method. Must be in {}".format(method, methods)
        )
    elif not callable(method):
      raise TypeError(
        "Invalid type {} for argument method.".format(type(method))
      )

    self.model = method
    self.goal = goal
    self.kwargs = kwargs
    self.minorder = minorder    
    self.value = value
    self.unravel = unravel
    self.batched = batched
    self.batchwise = batchwise

  def call(self, inputs):
    if self.goal:
      mask = goal_overlap(inputs, **self.kwargs)
      
      if self.minorder:
        values = self.model(inputs, **self.kwargs)

        minima = np.logical_and(
          mask,
          ndimage.minimum_filter(values, size=1+2*self.minorder, mode='constant') == values,
        )

        if np.any(minima):
          m = np.argmin(values[minima])
          # print('local minima', np.count_nonzero(minima))
          return np.argmin(np.where(minima, values, np.inf)), -values
      else:
        values = self.model(inputs, mask=mask, **self.kwargs)

      return np.argmin(np.where(mask, values, np.inf)), -values
    else:
      values = self.model(inputs, **self.kwargs)
      return np.argmin(values), -values
      

      






if False:

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

  def _goal_overlap(observation, previous=None, **kwargs):
    """Overlap between object and goal.

    Returned values are computed as
      max((overlap-limit)/(1-limit), 0)**exponent

    """
    g = np.where(observation[0][:,:,1]>0, 1, 0)
    t = np.where(observation[1][:,:,0]>0, 1, 0)
    overlap = signal.convolve2d(g, t, mode='valid')/np.count_nonzero(t)
    
    # Use different defaults for the goal overlap
    limit = kwargs.get('limit', 0.5)
    exponent = kwargs.get('exponent', 0.1)

    overlap = _apply_limit_and_exponent(overlap, limit=limit, exponent=exponent)

    if previous is not None:
      overlap *= previous
    return overlap
    
  def _random(observation, previous=None, **kwargs):
    return np.random.rand(
      observation[0].shape[0]-observation[1].shape[0]+1,
      observation[0].shape[1]-observation[1].shape[1]+1
    )*(previous if previous is not None else 1)

  def lowest(observation, previous=None, limit=0, exponent=1, **kwargs):
    x = observation[0][:,:,0]
    w = observation[1][:,:,0]

    shape = np.subtract(x.shape, w.shape) + 1
    h = np.zeros(shape)
    wbin = w > 0

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

  def wclosest(observation, previous=None, **kwargs):
    x = np.float32(observation[0][:,:,0])
    w = np.float32(observation[1][:,:,0])
    shape = np.subtract(x.shape, w.shape) + 1
    d = np.zeros(shape)

    wbin = w > 0
    wnz = np.count_nonzero(wbin)

    weighted = kwargs.get('weighted', True)
    if weighted:
      wi = np.arange(w.shape[0])
      wi *= wi[::-1]
      wj = np.arange(w.shape[1])
      wj *= wj[::-1]
      weights = np.where(
        wbin,
        np.expand_dims(wi, axis=1)*np.expand_dims(wj, axis=0),
        0,
      )
      weights = np.where(
        wbin,
        weights.max()-weights,
        0,
      )
      weights = weights/(weights.sum() or 1)
    else:
      weights = np.where(
        wbin,
        1/wnz,
        0.,
      )

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
        h = np.where(
          wbin,
          h.max() - h,
          0.,
        )
        h *= weights

        d[i,j] = np.sum(h)

    d = _apply_scale(-d, mask=mask, **kwargs)
    d = _apply_limit_and_exponent(d, **kwargs)

    if previous is not None:
      d *= previous

    return d

  def closest(observation, previous=None, **kwargs):
    x = np.float32(observation[0][:,:,0])
    w = np.float32(observation[1][:,:,0])
    shape = np.subtract(x.shape, w.shape) + 1
    d = np.zeros(shape)

    wbin = w > 0
    wnz = np.count_nonzero(wbin)

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
        h = np.where(
          wbin,
          h.max() - h,
          0.,
        )

        d[i,j] = np.sum(h)/wnz

    d = _apply_scale(-d, mask=mask, **kwargs)
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

    def gradcorr_(observation, previous=None, **kwargs):
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
  

    def goal_overlap_(observation, previous=None, **kwargs):
      """Overlap between object and goal.

      Returned values are computed as
        max((overlap-limit)/(1-limit), 0)**exponent

      """
      g = np.float32(observation[0][:,:,1])
      x = np.float32(observation[0][:,:,0])
      # g = np.where(g > 0, g, 0)
      t = np.float32(observation[1][:,:,0])
      c = cv.matchTemplate(  # pylint: disable=no-member
        g, 
        t, 
        cv.TM_CCORR,  # pylint: disable=no-member
      )
      c -= cv.matchTemplate(  # pylint: disable=no-member
        x, 
        t, 
        cv.TM_CCORR,  # pylint: disable=no-member
      )
      # c = _apply_scale(
      #   c, 
      #   mask = (previous > 0 if previous is not None else None), 
      #   **kwargs
      # )
      # Use different defaults for the goal overlap
      # limit = kwargs.get('limit', 0.5)
      # exponent = kwargs.get('exponent', 0.1)

      # c = _apply_limit_and_exponent(c, limit=limit, exponent=exponent)

      if previous is not None:
        c *= previous
      return c

  except ImportError:
    ccoeff = None
    gradcorr = None

  _methods = {
    'random': random,
    'lowest': lowest,
    'closest': closest,
    'wclosest':wclosest,
    'ccoeff': ccoeff,
    'corrcoef':corrcoef,
    'gradcorr': gradcorr
  }

  # @gin.configurable(module='siamrl')
  class _Baseline(agents.PyGreedy):
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
        if method=='none':
          method_list=[]
        else:
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
