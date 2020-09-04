import numpy as np
from gym.utils import seeding
from siamrl.envs.stack.simulator import Simulator
from siamrl.envs.stack.observer import Observer

class Rewarder(object):
  metrics = ['IoU', 'OR', 'DIoU', 'DOR', 'all', 'eval']

  EVAL = 5
  ALL = 4
  IOU = 0
  OR = 1
  DIOU = 2
  DOR = 3

  margin_factor = 8
  def __init__(
    self,
    simulator,
    observer,
    metric=None,
    goal_size_ratio=0.5,
    n_objects=76,    
    scale=1.,
    params=None,
    seed=None,
  ):
    """
    Args:
      simulator: instance of Simulator, from which to get the simulator 
        state.
      observer: instance of Observer, from which to get the observed state
        as well as spatial information to define the goal.
      metric: metric used to compute the rewards. One of 'IoU', 'OR', 
        'DIoU', 'DOR' or 'all' (case insensitive), or an integer index 
        from 0 to 4. If 'all', a dictionary with the four metrics is 
        returned. None defaults to 'DOR'.
      goal_size_ratio: size of the goal, given in fractions of the 
        observed space. Either None for completely random dimensions, a 
        scalar for constant area or a tuple with (height fraction, width 
        fraction) for fixed dimensions. In the later case, goal orientation 
        is randomly choosen to be horizontal or vertical.
      n_objects: number of objects in each episode, used for the discounted
        metrics.
      scale: scalar to be multiplied by the computed reward before return.
      params: non negative exponents of the dicount for translation 
        and rotation distance of an object from its original pose. 
        Either a tuple with exponents for each mode (translation and 
        rotation), a scalar to be used in both, or None to use no penalty.
      seed: seed for the random number generator used to define the goal.
    """
    if isinstance(simulator, Simulator):
      self._sim = simulator
    else:
      raise TypeError(
        "Invalid type {} for argument simulator.".format(type(simulator))
      )
    if isinstance(observer, Observer):
      self._obs = observer
    else:
      raise TypeError(
        "Invalid type {} for argument observer.".format(type(simulator))
      )

    self._shape, (self._goal_min_h,self._goal_min_w) = self._obs.shape
    self._goal_max_h, self._goal_max_w = self._shape
    self._goal_z = self._obs.max_z

    # Set target size
    if not goal_size_ratio:
      self._goal_size = None
    elif (
      np.isscalar(goal_size_ratio) and
      goal_size_ratio > 0 and 
      goal_size_ratio <= 1
    ):
      self._goal_size = int(goal_size_ratio*self._shape[0]*self._shape[1])
      # Update minimum and maximum height to get goal area within width range
      self._goal_min_h = max(self._goal_min_h, self._goal_size//self._goal_max_w)
      self._goal_max_h = min(self._goal_max_h, self._goal_size//self._goal_min_w)
    elif (
      len(goal_size_ratio) == 2 and
      all([s > 0 and s <= 1 for s in goal_size_ratio])
    ):
      self._goal_size = tuple(
        [int(g*s) for g,s in zip(goal_size_ratio, self._shape)]
      )
    else:
      raise ValueError('Invalid value {} for argument goal_size_ratio'.format(goal_size_ratio))

    # Initialize goal
    self._goal = np.zeros(self._shape, dtype='float32')
    self._goal_lims = ((0,0),(0,0))
    self._goal_volume = 0.

    # Set reward scale
    self.scale = float(scale) if scale is not None else float(n_objects)

    # Memory to store previous rewards
    self._memory = np.zeros(self.ALL)

    # Set the random number generator
    self._random, seed = seeding.np_random(seed)

    self._t = int(n_objects)

    # Set metric
    if isinstance(metric, str):
      metric = metric.lower()
      for i,name in enumerate(self.metrics):
        if metric == name.lower():
          metric = i
    elif metric is None:
      metric = self.IOU # default to intersection over union

    try:
      _=self.metrics[metric]
    except TypeError:
      raise TypeError('Invalid type {} for argument metric.'.format(type(metric)))
    except IndexError:
      raise ValueError('Invalid value {} for argument metric.'.format(metric))

    self.metric = metric

    # Set params of discounted occupation
    self._pmax = max(self._obs.pixel_to_xy(self._obs.shape[1]))
    self._omax = np.pi

    if params is None:
      self._pexp, self._oexp = None, None
    elif np.isscalar(params):
      if params < 0:
        raise ValueError("Invalid value {} for argument params. Must be non negative.".format(params))
      self._pexp, self._oexp = params, params
    elif len(params) == 1:
      if params[0] < 0:
        raise ValueError("Invalid value {} for argument params. Must be non negative.".format(params))
      self._pexp, self._oexp = params*2
    elif len(params) >= 2:
      if params[0] < 0 or params[1] < 0:
        raise ValueError("Invalid value {} for argument params. Must be non negative.".format(params))
      self._pexp, self._oexp = params[:2]

  def __call__(self):
    """Returns the scaled reward"""
    
    if self.metric == self.EVAL:
      d,n = self._discounted(intersection=False)
      
      r = d/n - self._memory[-1]
      self._memory[-1] = d/n

      return {
        self.metrics[self.IOU]: self.call(self.IOU),
        'AD': r,
      }

    elif self.metric == self.ALL:
      return {k:self.call(i) for i,k in enumerate(self.metrics[:-2])}
    else:
      return self.call()

  def call(self, metric=None):
    metric = metric if metric is not None else self.metric
    if metric == self.DOR:
      reward,_ = self._discounted()
      reward /= self._t
    elif metric == self.DIOU:
      reward,nout = self._discounted()
      reward /= self._t + nout
    elif metric == self.OR:
      reward = self._intersection()/self._goal_volume
    elif metric == self.IOU:
      reward = self._intersection()/self._union()
    else:
      raise ValueError('Invalid value {} for argument metric'.format(metric))
    output = reward - self._memory[metric]
    self._memory[metric] = reward

    return output*self.scale

  @property
  def goal(self):
    """Current goal map"""
    return self._goal

  @property
  def goal_bin(self):
    """Binary goal map"""
    return self._goal_bin

  def reset(self):
    """Reset reward memory and goal"""
    self._memory[:] = 0.
    self._reset_goal()

  def seed(self, seed=None):
    """Set the seed for the random number generator"""
    seed = seeding.create_seed(seed)
    self._random.seed(seed)
    return [seed]

  def visualize(self):
    """Visualize the target as a green rectangle in the
    simulator"""
    size = self._obs.pixel_to_xy(np.subtract(self._goal_lims[1], self._goal_lims[0]))
    offset = self._obs.pixel_to_xy(self._goal_lims[0])
    return self._sim.draw_rectangle(size, offset, (0,1,0)) # Green rectangle

  def _reset_goal(self):
    """Create new goal"""
    # Target dimensions (hihg aspect ratios are more likely)
    if not self._goal_size:
      # Beta distribution parameters are 1 and 3 randomly swapped
      b = 1 + self._random.randint(2)*2
      h = int(
        self._goal_min_h + 
        self._random.beta(b, 4-b)*(self._goal_min_h-self._goal_min_h)
      )
      w = int(
        self._goal_min_w + 
        self._random.beta(4-b, b)*(self._goal_max_w-self._goal_min_w)
      )
    elif np.isscalar(self._goal_size):
      # Beta distribution parameters are 1 and 3 randomly swapped
      b = 1 + self._random.randint(2)*2
      h = int(
        self._goal_min_h + 
        self._random.beta(b, 4-b)*(self._goal_max_h-self._goal_min_h)
      )
      w = min(max(
        self._goal_min_w,
        self._goal_size//h,
      ),
        self._goal_max_w,
      )
    else:
      i = self._random.randint(2)
      h = min(self._goal_size[i], self._goal_max_h)
      w = min(self._goal_size[1-i], self._goal_max_w)

    # Target offset
    u_max = self._shape[0] - h
    u = self._random.randint(
      u_max//self.margin_factor,
      (self.margin_factor-1)*u_max//self.margin_factor+1
    )
    v_max = self._shape[1] - w
    v = self._random.randint(
      v_max//self.margin_factor,
      (self.margin_factor-1)*v_max//self.margin_factor+1
    )

    self._goal = np.zeros(self._shape, dtype='float32')
    self._goal[u:u+h,v:v+w] = self._goal_z
    self._goal_lims = ((u,v),(u+h,v+w))
    self._goal_volume = np.sum(self._goal)
    self._goal_bin = self._goal != 0

  def _discount(self, perr, oerr):
    r = 1.
    if self._pexp is not None:
      r *= max(0.,
        ( 1 - (perr/self._pmax)**self._pexp) )
    if self._oexp is not None:
      r *= max(0.,
        ( 1 - (oerr/self._omax)**self._oexp) )
    return r

  def _discounted(self, intersection=True):
    """Returns the discounted intersection metric."""
    reward = 0.
    n = 0
    for p, (perr,oerr) in zip(self._sim.positions, self._sim.distances_from_place):
      if intersection:
        # Get position in pixels
        u,v = self._obs.xy_to_pixel(p[:2])
        
        # Check if it's inside goal
        if (
          u >= self._goal_lims[0][0] and 
          v >= self._goal_lims[0][1] and 
          u < self._goal_lims[1][0] and
          v < self._goal_lims[1][1]
        ):
          reward += self._discount(perr, oerr)
        else:
          n += 1
      else:
        reward += self._discount(perr, oerr)
        n += 1


    return reward, n

  def _intersection(self):
    return np.sum(np.minimum(
      self._obs.state[0][self._goal_bin],
      self._goal_z,
    ))

  def _union(self):
    return np.sum(np.maximum(
      self._obs.state[0],
      self._goal,
    ))

if False:
  class PoseRewarder(Rewarder):
    def __init__(
      self,
      *args,
      params=None,
      **kwargs,
    ):
      """
      Args:
        params: non negative exponents of the dicount for translation 
          and rotation distance of an object from its original pose. 
          Either a tuple with exponents for each mode (translation and 
          rotation), a scalar to be used in both, or None to use no penalty.
      """
      super(PoseRewarder, self).__init__(*args, **kwargs)

      # Maximum translation distance from original pose (corresponds to the 
      # length of the diagonal of the object image).
      # self._pmax = np.linalg.norm(self._obs.pixel_to_xy(self._obs.shape[1]))
      self._pmax = max(self._obs.pixel_to_xy(self._obs.shape[1]))
      # Maximum rotation distance from original pose
      self._omax = np.pi

      if params is None:
        self._pexp, self._oexp = None, None
      elif np.isscalar(params):
        if params < 0:
          raise ValueError("Invalid value {} for argument params. Must be non negative.".format(params))
        self._pexp, self._oexp = params, params
      elif len(params) == 1:
        if params[0] < 0:
          raise ValueError("Invalid value {} for argument params. Must be non negative.".format(params))
        self._pexp, self._oexp = params*2
      elif len(params) >= 2:
        if params[0] < 0 or params[1] < 0:
          raise ValueError("Invalid value {} for argument params. Must be non negative.".format(params))
        self._pexp, self._oexp = params[:2]

    def call(self):
      """Returns the reward computed from the current poses."""

      reward = 0.
      for p, (perr,oerr) in zip(self._sim.positions, self._sim.distances_from_place):
        # Get position in pixels
        u,v = self._obs.xy_to_pixel(p[:2])
        # Check if it's inside goal
        if (
          u >= self._goal_lims[0][0] and 
          v >= self._goal_lims[0][1] and 
          u < self._goal_lims[1][0] and
          v < self._goal_lims[1][1]
        ):
          r = 1.
          if self._pexp is not None:
            r *= max(0.,
              ( 1 - (perr/self._pmax)**self._pexp) )
          if self._oexp is not None:
            r *= max(0.,
              ( 1 - (oerr/self._omax)**self._oexp) )
          reward += r


      prev_reward = self._memory
      self._memory = reward


      return reward - prev_reward

  class OccupationRatioRewarder(Rewarder):
    def __init__(
      self,
      *args,
      params=False,
      **kwargs,
    ):
      """
      Args:
        params: if True, the reward is based on the ratio of the current
          volume that is inside the target, rather than the ratio of the 
          target that is occupied.
      """
      super(OccupationRatioRewarder, self).__init__(*args, **kwargs)
      if not np.isscalar(params):
        params = params[0]
      self._mode = bool(params)

    def call(self):
      """Returns the reward computed from the occupation ratio."""
      reward = np.sum(np.minimum(
        self._obs.state[0][self._goal_bin],
        self._goal_z
      ))
      if self._mode:
        reward /= self._obs.state[0].sum()
      else:
        reward /= self._goal_volume

      prev_reward = self._memory
      self._memory = reward
      return reward - prev_reward

    def reset(self):
      """Reset reward memory and goal"""
      # If using the ratio of the structure that is in the target,
      # start with memory as 1. 
      self._memory = 1. if self._mode else 0.
      self._reset_goal()
