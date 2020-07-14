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
    occupation_ratio_weight=0.,
    occupation_ratio_param=False,
    positions_weight=0.,
    positions_param=0.,
    n_steps_weight=0.,
    n_steps_param=0.,
    contact_points_weight=0.,
    contact_points_param=0.,
    differential=True,
    seed=None
  ):
    """
    Args:
      simulator: instance of Simulator, from wich to get the simulator 
        state.
      observer: instance of Observer, from which to get the observed state
        as well as spatial information to define the goal.
      goal_size_ratio: size of the goal, given in fractions of the 
        observed space. Either None for completely random dimensions, a 
        scalar for constant area or a tuple with (height fraction, width 
        fraction) for fixed dimensions. In the later case, goal orientation 
        is randomly choosen to be horizontal or vertical.
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
    self._goal_params = ((0,0),(0,0))
    self._goal_v = 0.

    # Set reward weights and parameters
    self._or = 0.
    self._or_w = max(occupation_ratio_weight,0.)
    self._or_p = 1 if occupation_ratio_param else 0
    self._pr = 0.
    self._pr_w = max(positions_weight, 0.)
    self._pr_p = max(positions_param, 0.)
    self._sp_w = max(n_steps_weight, 0.)
    self._sp_p = max(n_steps_param, 0.)
    self._cp_w = max(contact_points_weight, 0.)
    self._cp_p = contact_points_param
    self._diff = differential

    # Set the random number generator
    self._random, seed = seeding.np_random(seed)

  def __call__(self):
    """Returns the reward computed from the current state"""
    reward = 0.
    # Ocupation ratio reward
    if self._or_w != 0:
      if self._diff:
        reward -= self._or_w*self._or
      self._or = np.sum(np.minimum(
        self._obs.state[0][self._goal_b],
        self._goal_z
      ))/self._goal_v - self._or_p
      reward += self._or_w*self._or
    # Positions reward
    if self._pr_w != 0.:
      if self._diff:
        reward -= self._pr_w*self._pr
      self._pr = 0.
      for p in self._sim.positions:
        u,v = self._obs.xy_to_pixel(p[:2])
        if u >= self._goal_params[1][0] and v >= self._goal_params[1][1] \
          and u < self._goal_params[1][0]+self._goal_params[0][0] \
          and v < self._goal_params[1][1]+self._goal_params[0][1] \
        :
          self._pr += 1.
        else:
          self._pr -= self._pr_p
      reward += self._pr_w*self._pr

    if self._sp_w != 0:
      reward += self._sp_w*(2**(-self._sim.n_steps[1]/self._sp_p)-1)

    if self._cp_w != 0:
      raise NotImplementedError('Contact points based reward not implemented.')

    return reward

  @property
  def goal(self):
    """Current goal map"""
    return self._goal

  @property
  def boolean_goal(self):
    return self._goal_b

  def reset(self):
    """Reset the reward memory and the goal"""
    self._or = 0.
    self._pr = 0.
    self._reset_goal()

  def seed(self, seed=None):
    """Set the seed for the random number
      generator"""
    seed = seeding.create_seed(seed)
    self._random.seed(seed)
    return [seed]

  def visualize(self):
    """Visualize the target as a green rectangle in the
    simulator"""
    size = self._obs.pixel_to_xy(self._goal_params[0])
    offset = self._obs.pixel_to_xy(self._goal_params[1])
    return self._sim.draw_rectangle(size, offset, (0,1,0))    

  def _reset_goal(self):
    """Create new goal"""
    # Target dimensions
    if not self._goal_size:
      h = self._random.randint(self._goal_min_h, self._shape[0]+1)
      w = self._random.randint(self._goal_min_w, self._shape[1]+1)
    elif np.isscalar(self._goal_size):
      h_min = max(self._goal_min_h, self._goal_size//self._shape[1])
      h_max = min(self._shape[0]+1, self._goal_size//self._goal_min_w)
      # Make high aspect ratios more likely
      h = int(self._random.triangular(
        h_min,
        h_min if self._random.randint(2) else h_max,
        h_max,
      ))
      w = min(max(
        self._goal_min_w,
        self._goal_size//h,
      ),
        self._shape[1],
      )
    else:
      i = self._random.randint(2)
      h = min(self._goal_size[i], self._shape[0])
      w = min(self._goal_size[1-i], self._shape[1])

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
    self._goal_params = [[h,w],[u,v]]
    self._goal_v = np.sum(self._goal)
    self._goal_b = self._goal != 0
