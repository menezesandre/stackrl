"""
Baseline policies with no learning, using 
  OpenCV's Template Matching
"""
import tf_agents
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step

import numpy as np
import cv2 as cv

__all__=['CCoeffPolicy', 'GradCorrPolicy']

class Base(py_policy.Base):
  """
  Base class that implements the common initialization
  """
  def __init__(self, time_step_spec, action_spec,
      normed=True):
    assert len(time_step_spec.observation)==2
    assert time_step_spec.observation[0].shape[2] == time_step_spec.observation[1].shape[2]
    assert action_spec.shape in [(), (2,)]
    super(Base, self).__init__(time_step_spec, action_spec)
    self._method = self.methods[normed]
    self._unravel = self.action_spec.shape==(2,)

class CCoeffPolicy(Base):
  methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED]

  def _action(self, time_step, policy_state):
    img = time_step.observation[0]
    tmp = time_step.observation[1]
    # this may be necessary because the CCOEFF takes the "empty" 
    # pixels into acount when mean centering
    """
    not_empty_count = np.sum(tmp!=0)
    if not_empty_count != 0:
      mean = np.sum(tmp)/not_empty_count
      tmp -= np.where(tmp!=0,mean,0)
    """
    c = cv.matchTemplate(img, tmp, self._method)
    action = np.argmin(c)
    if self._unravel:
      action = np.array(np.unravel_index(action, c.shape))
    
    return policy_step.PolicyStep(action=action,state=policy_state)

class GradCorrPolicy(Base):
  methods = [cv.TM_CCORR, cv.TM_CCORR_NORMED]

  def _action(self, time_step, policy_state):
    img = time_step.observation[0]
    tmp = -time_step.observation[1]
    depth = cv.CV_32F if img.dtype=='float32' else cv.CV_64F

    img_x = cv.Sobel(img, depth, 1, 0)
    img_y = cv.Sobel(img, depth, 0, 1)
    tmp_x = cv.Sobel(tmp, depth, 1, 0)
    tmp_y = cv.Sobel(tmp, depth, 0, 1)

    img = cv.merge([img_x, img_y])
    tmp = cv.merge([tmp_x, tmp_y])
    
    c = cv.matchTemplate(img, tmp, self._method)
    action = np.argmax(c)
    if self._unravel:
      action = np.array(np.unravel_index(action, c.shape))
    
    return policy_step.PolicyStep(action=action,state=policy_state)
