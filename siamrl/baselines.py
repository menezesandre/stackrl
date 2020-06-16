"""
Baseline policies with no learning, using 
  OpenCV's Template Matching
"""
import numpy as np

def _trimmed_goal(observation):
  shape = observation[1].shape
  top = shape[0]//2 + 1
  bottom = shape[0] - top - 1
  left = shape[1]//2 + 1
  right = shape[1] - left - 1
  return observation[0][top:-bottom, left:-right,1]

def random(observation, flat=True):
  return np.random.rand(
    observation[0].shape[0]-observation[1].shape[0]+1,
    observation[0].shape[1]-observation[1].shape[1]+1
  )

try:
  import cv2 as cv

  def ccoeff(observation, normed=True):
    img = observation[0][:,:,0]
    tmp = observation[1][:,:,0]
    return -cv.matchTemplate(
      img, 
      tmp, 
      cv.TM_CCOEFF_NORMED if normed else cv.TM_CCOEFF
    )

  def gradcorr(observation, normed=True):
    img = observation[0][:,:,0]
    tmp = observation[1][:,:,0]

    img_x = cv.Sobel(img, cv.CV_32F, 1, 0)
    img_y = cv.Sobel(img, cv.CV_32F, 0, 1)
    tmp_x = cv.Sobel(tmp, cv.CV_32F, 1, 0)
    tmp_y = cv.Sobel(tmp, cv.CV_32F, 0, 1)

    img = cv.merge([img_x, img_y])
    tmp = cv.merge([tmp_x, tmp_y])
    
    return -cv.matchTemplate(
      img, 
      tmp, 
      cv.TM_CCORR_NORMED if normed else cv.TM_CCORR
    )
  
except ImportError:
  ccoeff = None
  gradcorr = None

methods = {
  'random': random,
  'ccoeff': ccoeff,
  'gradcorr': gradcorr
}

class Baseline(object):
  def __init__(self, method='random', flat=True, goal=True, **kwargs):
    if isinstance(method, str):
      method = method.lower()
      if method in methods:
        self._value = methods[method]
        if self._value is None:
          raise ImportError(
            "opencv-python must be installed to use {} method.".format(
              method
            )
          )
      else:
        raise ValueError(
          "Invalid value {} for argument method.".format(method)
        )
    elif callable(method):
      self._value = method
    else:
      raise TypeError(
        "Invalid type {} for argument method.".format(type(method))
      )
    self._flat = bool(flat)
    self._goal = bool(goal)
    self._kwargs = kwargs

  def __call__(self, inputs):
    value = self._value(inputs, **self._kwargs)
    if self._goal:
      goal = _trimmed_goal(inputs)
      value = np.where(
        goal > 0.,
        value,
        -np.inf
      )
    best = np.argmax(value)
    if not self._flat:
      best = np.array(np.unravel_index(best, value.shape))
    return best
