import numpy as np
from gym.utils import seeding
from siamrl.envs.stack.simulator import Simulator
from siamrl.envs.stack.observer import Observer

class Rewarder(object):
  margin_factor = 8
  def __init__(
    self,
    simulator,
    observer,
    goal_size_ratio=0.5,
    scale=1.,
    seed=None,
  ):
    """
    Args:
      simulator: instance of Simulator, from which to get the simulator 
        state.
      observer: instance of Observer, from which to get the observed state
        as well as spatial information to define the goal.
      goal_size_ratio: size of the goal, given in fractions of the 
        observed space. Either None for completely random dimensions, a 
        scalar for constant area or a tuple with (height fraction, width 
        fraction) for fixed dimensions. In the later case, goal orientation 
        is randomly choosen to be horizontal or vertical.
      scale: scalar to be multiplied by the computed reward before return.
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
    self.scale = float(scale)

    # Memory to store previous rewards
    self._memory = 0.

    # Set the random number generator
    self._random, seed = seeding.np_random(seed)

  def __call__(self):
    """Returns the scaled reward"""
    return self.call()*self.scale

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
    self._memory = 0.
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

  def call(self):
    """Reward computation to be implemented by subclasses."""
    raise NotImplementedError("Subclasses must implement call method.")

class PoseRewarder(Rewarder):
  def __init__(
    self,
    *args,
    p=None,
    o=None,
    **kwargs,
  ):
    """
    Args:
      p: tolerance parameter in the interval (0,1] for the translation 
        distance of an object from its original pose. Corresponds to the
        fraction of the maximum distance at which the discount is 1%. If
        None or 0, no discount is aplied.
      p: tolerance parameter in the interval (0,1] for the rotation 
        distance of an object from its original pose. Corresponds to the
        fraction of the maximum distance at which the discount is 1%. If
        None or 0, no discount is aplied.
    """
    super(PoseRewarder, self).__init__(*args, **kwargs)

    if p is not None:
      # Compute position penalty exponent from parameter p.
      # this parameter gives the fraction of the maximum distance
      # from original position that corresponds to 1% penalty on 
      # the reward.
      p = float(p)
      if p <= 0:
        p = None
      elif p >= 1:
        p = np.inf
      else:
        p = -2/np.log10(p)
    self._pexp = p
    # Maximum translation distance from original pose (corresponds to the 
    # length of the diagonal of the object image).
    # self._pmax = np.linalg.norm(self._obs.pixel_to_xy(self._obs.shape[1]))
    self._pmax = max(self._obs.pixel_to_xy(self._obs.shape[1]))

    if o is not None:
      # Compute orientation penalty exponent from parameter o.
      # this parameter gives the fraction of the maximum distance
      # from original orientation that corresponds to 1% penalty on 
      # the reward.
      o = float(o)
      if o <= 0:
        o = None
      elif o >= 1:
        o = np.inf
      else:
        o = -2/np.log10(o)
    self._oexp = o
    # Maximum rotation distance from original pose
    self._omax = np.pi

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
            (1 - (perr/self._pmax)**self._pexp) )
        if self._oexp is not None:
          r *= max(0.,
            (1 - (oerr/self._omax)**self._oexp) )
        reward += r


    prev_reward = self._memory
    self._memory = reward


    return reward - prev_reward

class OccupationRatioRewarder(Rewarder):
  def __init__(
    self,
    *args,
    negative=False,
    o=None,
    **kwargs,
  ):
    """
    Args:
      negative: if True, the returned reward is the (negative) difference
        between the current occupation ratio and 1 (full target volume). 
    """
    super(OccupationRatioRewarder, self).__init__(*args, **kwargs)
    self._negative = bool(negative)

  def call(self):
    """Returns the reward computed from the occupation ratio."""
    reward = np.sum(np.minimum(
      self._obs.state[0][self._goal_bin],
      self._goal_z
    ))/self._goal_volume

    if self._negative:
      reward -= 1
      # Previous computation is not discounted on the negative case.
      prev_reward = 0
    else:
      prev_reward = self._memory
      self._memory = reward

    return reward - prev_reward
