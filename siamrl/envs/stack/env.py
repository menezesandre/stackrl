import warnings

import gym
from gym import spaces
from gym.envs import registry
from gym.utils import seeding
import numpy as np

from siamrl.envs import data
from siamrl.envs.stack.simulator import Simulator, TestSimulator
from siamrl.envs.stack.observer import Observer
from siamrl.envs.stack.rewarder import Rewarder
# from siamrl.baselines import Baseline

try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None

DEFAULT_EPISODE_LENGTH = 40

class StackEnv(gym.Env):
  metadata = {
    'dtypes': ['uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64'],
    'render.modes': ['human', 'rgb_array'],
  }

  def __init__(
    self,
    episode_length=DEFAULT_EPISODE_LENGTH,
    urdfs=None,
    object_max_dimension=0.125,
    use_gui=False,
    simulator=None,
    sim_time_step=1/100.,
    gravity=9.8,
    num_sim_steps=None,
    velocity_threshold=0.01,
    smooth_placing = True,
    observer=None,
    observable_size_ratio=4,
    resolution_factor=5,
    max_z=0.375,
    rewarder=None,
    goal_size_ratio=.25,
    reward_scale=1.,
    reward_params=None,
    flat_action=True,
    dtype='float32',
    seed=None,
  ):
    """
    Args:
      episode_length: Number of steps per episode (corresponds to the 
      number of objects used on each episode).
      urdfs: list of files (urdf format) that describe the objects to be 
        used on the environment. A name string can be provided to use 
        objects from the 'siamrl/envs/data/generated' directory. On each 
        episode, a fixed number of files is randomly choosen from this 
        list.
      object_max_dimension: maximum dimension of all objects in the list.
        All objects should be completely visible within a square with this
        value as side length.
      use_gui: whether to use physics engine gafical interface.
      simulator: constructor for the environment's simulator. If None,
        Simulator class is used.
      sim_time_step, gravity, num_sim_steps, velocity_threshold, 
        smooth_placing: see Simulator.step.
      observer: constructor for the environment's observations collector. 
        If None, Observer class is used.
      observable_size_ratio: size of the observable space as a multiple of
        object_max_dimension. Either a scalar for square space or a list 
        with [height, width] (as seen in the observation).
      resolution_factor: resolution is such that the number of pixels 
        along object_max_dimensions is two to the power of this factor.
      max_z: See Observer.
      rewarder: constructor for the environment's reward calculator. If None,
        Rewarder class is used.
      goal_size_ratio, reward_scale, reward_params: see Rewarder.
      reward_scale: factor to be multiplied by the computed reward before step
        return.
      reward_params: to be passed to rewarder.
      flat_action: whether to receive action as a flat index or a pair of
        indexes [h, w].
      dtype: data type of the returned observation. Must be one of 
        'uint8', 'uint16', 'uint32', 'float16', 'float32' or 'float64'. 
        Internaly, float32 is used.
      seed: Seed for the env's random number generator.
    """
    self._length = episode_length
    # Set the files list
    if urdfs is None:
      urdfs = {}
    if isinstance(urdfs, (str,int)):
      self._list = data.generated(name=urdfs)
    elif isinstance(urdfs, float):
      self._list = data.generated(rectangularity=urdfs)
    elif isinstance(urdfs, dict):
      self._list = data.generated(**urdfs)
    else:
      self._list = list(urdfs)
    # Assert the list is not empty
    assert self._list, 'List of object descriptor files is empty.'
    # If the list is smaller than the episode length, files have to be
    # sampled with replacement.
    self._replace = len(self._list) < self._length

    self._random, seed = seeding.np_random(seed)

    # Used only when render is called in human mode.
    self._fig = None

    # Set simulator
    self._gui = use_gui
    simulator = simulator or Simulator
    self._sim = simulator(
      use_gui=use_gui, 
      time_step=sim_time_step, 
      gravity=gravity, 
      spawn_position=[0,0,max_z + object_max_dimension], 
      spawn_orientation=[0,0,0,1],
      num_steps=num_sim_steps,
      velocity_threshold=velocity_threshold
    )
    self._smooth_placing = smooth_placing

    # Set observer
    object_resolution = 2**resolution_factor
    if np.isscalar(observable_size_ratio):
      overhead_resolution = object_resolution*observable_size_ratio
    else:
      overhead_resolution = [
        object_resolution*observable_size_ratio[0],
        object_resolution*observable_size_ratio[1]
      ]
    pixel_size = object_max_dimension/object_resolution

    observer = observer or Observer
    self._obs = observer(
      self._sim,
      overhead_resolution=overhead_resolution,
      object_resolution=object_resolution,
      pixel_size=pixel_size,
      max_z=max_z
    )

    # Compatibility
    if rewarder == 'position':
      rewarder = 'dor'
      warnings.warn(
        "Using 'position' as rewarder is drprecated. Use 'discounted_occupation' (or 'do') instead.", DeprecationWarning)
    elif rewarder == 'occupation':
      rewarder = 'or'
      warnings.warn(
        "Using 'occupation' as rewarder is drprecated. Use 'target_ratio' (or 'tr') instead.", DeprecationWarning)
    # Set the rewarder.
    self._rew = Rewarder(
      simulator=self._sim, 
      observer=self._obs,
      metric=rewarder,
      goal_size_ratio=goal_size_ratio,
      num_objects=episode_length,
      scale=reward_scale,
      seed=self._random.randint(2**32),
      params=reward_params,
    )

    # Set the return wrapper for the observations.
    if dtype not in self.metadata['dtypes']:
      raise ValueError('Invalid value {} for argument dtype.'.format(dtype))
    if dtype == 'uint8':
      self._return = lambda x: np.array(x*(2**8-1)/max(max_z, object_max_dimension), dtype=dtype)
    elif dtype == 'uint16':
      self._return = lambda x: np.array(x*(2**16-1)/max(max_z, object_max_dimension), dtype=dtype)
    elif dtype == 'uint32':
      self._return = lambda x: np.array(x*(2**32-1)/max(max_z, object_max_dimension), dtype=dtype)
    elif dtype == 'uint64':
      self._return = lambda x: np.array(x*(2**64-1)/max(max_z, object_max_dimension), dtype=dtype)
    else:
      self._return = lambda x: np.array(x, dtype=dtype)

    obs_shape = self._obs.shape
    # Set observation space
    self.observation_space = spaces.Tuple((
      spaces.Box(
        low=0, 
        high=max_z if dtype!='uint8' else 255, 
        dtype=np.dtype(dtype),
        shape=(
          obs_shape[0][0],
          obs_shape[0][1],
          2
        )
      ),
      spaces.Box(
        low=0, 
        high=object_max_dimension if dtype!='uint8' else 255, 
        dtype=np.dtype(dtype), 
        shape=(
          obs_shape[1][0],
          obs_shape[1][1],
          1
        )
      )
    ))
    # Set action space
    if flat_action:
      self._action_width = obs_shape[0][1] - obs_shape[1][1] + 1
      self.action_space = spaces.Discrete(
        (obs_shape[0][0] - obs_shape[1][0] + 1)*self._action_width
      )
    else:
      self._action_width = None
      self.action_space = spaces.MultiDiscrete([
        obs_shape[0][0] - obs_shape[1][0] + 1, 
        obs_shape[0][1] - obs_shape[1][1] + 1
      ])

    # Set the flag to true to ensure environment is reset before step.
    self._done = True

  def __del__(self):
    self.close()

  @property
  def observation(self):
    """Agent's observation of the environment state."""
    m,n = self._obs.state
    g = self._rew.goal
    return self._return(np.stack([m,g],axis=-1)), \
      self._return(n[:,:,np.newaxis])

  def step(self, action):
    # Reset if necessary.
    if self._done:
      return self.reset(), 0., False, {}
    # Assert action is valid
    assert self.action_space.contains(action), 'Invalid action.'
    # Unflatten action if necessary.
    if self._action_width:
      action = [action//self._action_width, action%self._action_width]
    # Pop next urdf if list isn't empty, otherwise episode is done.
    if self._episode_list:
      urdf = self._episode_list.pop()
    else:
      urdf = None
      self._done = True

    # Run simulation with given action.
    self._sim(
      **self._obs.pose(action), 
      urdf=urdf,
      smooth_placing=self._smooth_placing
    )
    # Get observation of the new state.
    self._obs()
    # Compute reward.
    reward = self._rew()
    if not np.isscalar(reward):
      info = reward
      reward = None
    else:
      info = {}
    return self.observation, reward, self._done, info

  def reset(self):
    # Get episode's list of urdfs
    self._episode_list = list(self._random.choice(
      self._list, 
      size=self._length,
      replace=self._replace
    ))
    # Reset simulator.
    self._sim.reset(self._episode_list.pop())
    if self._gui:
      self._sim.resetDebugVisualizerCamera(
        self._obs.size[2], 
        90, 
        -75, 
        (self._obs.size[0]/2,self._obs.size[1]/2,0),
      )
    # Reset rewarder.
    self._rew.reset()
    # Get first observation.
    self._obs()

    # If using GUI, mark observable space and goal.
    if self._gui:
      self._obs.visualize()
      self._rew.visualize()

    self._done = False
    return self.observation

  def render(self, mode='human'):
    if mode not in self.metadata['render.modes']:
      return super(StackEnv, self).render(mode=mode)

    m, n = self._obs.state
    _max = np.max(m)
    r = m/_max if _max!=0 else m
    b = 1 - r
    g = np.ones(r.shape)*0.5
    g[self._rew.goal_bin] += 0.1
    rgb0 = np.stack([r,g,b], axis=-1)

    _max = np.max(n)
    r = n/_max if _max!=0 else n
    b = 1 - r
    g = np.ones(r.shape)*0.5
    rgb1 = np.stack([r,g,b], axis=-1)
    
    if mode == 'human':
      if plt is None:
        raise ImportError("'render' requires matplotlib.pyplot to run in 'human' mode.")
        

      if not (self._fig and plt.fignum_exists(self._fig.number)):
        width_ratio = rgb0.shape[1]//rgb1.shape[1]
        self._fig, self._axs = plt.subplots(
          1, 2, 
          gridspec_kw={'width_ratios':[width_ratio, 1]}
        )

      self._axs[0].cla()
      self._axs[0].imshow(rgb0)
      self._axs[1].cla()
      self._axs[1].imshow(rgb1)
      self._fig.show()
      
    elif mode == 'rgb_array':
      return rgb0, rgb1

  def close(self):
    if plt and self._fig and plt.fignum_exists(self._fig.number):
      plt.close(self._fig)
      self._fig = None
    self._sim.disconnect()
    self._done = True

  def seed(self, seed=None):
    """Set the seed for the env's random number
      generator"""
    seed = seeding.create_seed(seed)
    self._random.seed(seed)
    return [seed]+self._rew.seed(self._random.randint(2**32))

class StartedStackEnv(StackEnv):
  def __init__(
    self,
    episode_length=DEFAULT_EPISODE_LENGTH//2,
    min_episode_length=None,
    n_objects=DEFAULT_EPISODE_LENGTH,
    start_policy=None,
    flat_action=True,
    **kwargs
  ):
    """
    Args:
      episode_length: number of steps in an episode.
      min_episode_length: if not None, each episode has a random length
        between min_episode_length and episode_length.
      n_objects: number of objects used in each episode. An episode 
        starts from a set of already placed objects, such that the number
        of remaining objects makes the episode length.
      start_policy: policy used to place the initial number of objects at
        the beggining of an episode. Either a string identifying one of 
        the policies implemented in siamrl.baselines or a callable 
        implementing the policy. If None, defaults to 'ccoeff' baseline
        if opencv-python is installed, 'random' otherwise.
      flat_action: whether to receive action as a flat index or a pair of
        indexes [h, w].
      kwargs: see super.
    """
    if n_objects < episode_length:
      raise ValueError(
        "n_objects can't be less than episode_length. Got {} objects for {} steps long episodes.".format(n_objects, episode_length)
      )
    super(StartedStackEnv, self).__init__(
      episode_length=n_objects, 
      flat_action=flat_action,
      **kwargs
    )
    if min_episode_length and min_episode_length < episode_length:
      upper = n_objects - min_episode_length
      lower = n_objects - episode_length
      self._n_start_steps = lambda: self._random.randint(lower, upper+1)
    else:
      self._n_start_steps = n_objects - episode_length

    if start_policy is None:
      # Lowest position inside goal
      def start_policy(inputs):
        x = inputs[0][:,:,0]
        g = inputs[0][:,:,1]
        w = inputs[1][:,:,0]

        wmax = w.max()
        wcount = np.count_nonzero(w)

        v = np.ones(np.subtract(x.shape,w.shape) + 1)*np.inf
        for i in v.shape[0]:
          for j in v.shape[1]:
            if (
              np.any(g[i:i+w.shape[0],j:j+w.shape[1]]) and 
              np.count_nonzero(w*g[i:i+w.shape[0],j:j+w.shape[1]]) == wcount
            ):
              v[i,j] = np.max(w+x[i:i+w.shape[0],j:j+w.shape[1]])
              if v[i,j] == wmax:
                # No need to calculate the remaining values as it can't 
                # be lower than this
                return i*v.shape[0]+j if flat_action else np.array((i,j))                

        if flat_action:
          return np.argmin(v)
        else:
          return np.array(np.unravel_index(np.argmin(v), v.shape))

      self._start_policy = start_policy
    elif callable(start_policy):
      self._start_policy = start_policy
    else:
      raise TypeError(
        "Invalid type {} for argument start_policy. Must be callable.".format(type(start_policy))
      )
    assert self.action_space.contains(
      self._start_policy(self.observation_space.sample())
    ), "Invalid start_policy."

  @property
  def n_start_steps(self):
    if callable(self._n_start_steps):
      return self._n_start_steps()
    else:
      return self._n_start_steps

  def reset(self):
    o = super(StartedStackEnv, self).reset()
    for _ in range(self.n_start_steps):
      o,_,_,_=self.step(self._start_policy(o))
    return o

class TestStackEnv(StackEnv):
  def __init__(
    self,
    ordering_freedom=False,
    orientation_freedom=3,
    **kwargs
  ):
    """
    Args:
      ordering_freedom: If True, all objects are presented at the begining
        of the episode and the action includes the choice of the next object 
        to be placed.
      orientation_freedom: defines the number of possible orientaitons for 
        the object (see Observer). If greater than 0, the action includes the
        choice of which orientation to use in the placing pose.
    """
    super(TestStackEnv, self).__init__(
      simulator=TestSimulator if ordering_freedom else Simulator, 
      observer=lambda *a,**k: Observer(*a, **k, orientation_freedom=orientation_freedom),
      **kwargs,
    )
    self._ordering_freedom = ordering_freedom
    self._obs_high = (
      self.observation_space[0].high.ravel()[0],
      self.observation_space[0].high.ravel()[0]
    )
    self.action_space = spaces.Tuple((spaces.Discrete(0),self.action_space))
    self._update_spaces()

  @property
  def observation(self):
    """Agent's observation of the environment state."""
    m,n = self._obs.state
    g = self._rew.goal
    return (
      self._return(np.array([np.stack([m,g],axis=-1)]*len(n))),
      self._return(np.array(n).reshape(self.observation_space[1].shape))
    )

  def step(self, action):
    # Reset if necessary.
    if self._done:
      return self.reset(), 0., False, {}
    # Assert action is valid
    assert self.action_space.contains(action), 'Invalid action.'
    # Split index and position
    index, action = action
    # Unflatten action if necessary.
    if self._action_width:
      action = [action//self._action_width, action%self._action_width]
    # Run simulation with given action.
    if not self._ordering_freedom:
      if self._episode_list:
        urdf = self._episode_list.pop()
      else:
        urdf = None
        self._done = True
    else: 
      urdf = None
    self._sim(
      **self._obs.pose(action, index=index), 
      smooth_placing=self._smooth_placing,
      urdf=urdf,
    )
    # Get observation of the new state.
    self._obs()
    
    if self._obs.num_objects == 0:
      self._done = True
    self._update_spaces()
    # Compute reward.
    reward = self._rew()
    return self.observation, reward, self._done, {}

  def reset(self):
    # Get episode's list of urdfs
    episode_list = list(self._random.choice(
      self._list, 
      size=self._length,
      replace=self._replace
    ))
    # Reset simulator.
    if self._ordering_freedom:
      self._sim.reset(episode_list)
    else:
      self._sim.reset(episode_list.pop())
      self._episode_list = episode_list
    # Reset rewarder.
    self._rew.reset()
    # Get first observation.
    self._obs()

    # If using GUI, mark observable space and goal.
    if self._gui:
      self._obs.visualize()
      self._rew.visualize()

    self._done = False
    self._update_spaces()
    return self.observation

  def render(self, mode='human'):
    if mode not in self.metadata['render.modes']:
      return super(TestStackEnv, self).render(mode=mode)

    m, n = self._obs.state
    n = n[0]
    _max = np.max(m)
    r = m/_max if _max!=0 else m
    b = 1 - r
    g = np.ones(r.shape)*0.5
    g[self._rew.goal_bin] += 0.1
    rgb0 = np.stack([r,g,b], axis=-1)

    _max = np.max(n)
    r = n/_max if _max!=0 else n
    b = 1 - r
    g = np.ones(r.shape)*0.5
    rgb1 = np.stack([r,g,b], axis=-1)
    
    if mode == 'human':
      if plt is None:
        raise ImportError("'render' requires matplotlib.pyplot to run in 'human' mode.")
        

      if not (self._fig and plt.fignum_exists(self._fig.number)):
        width_ratio = rgb0.shape[1]//rgb1.shape[1]
        self._fig, self._axs = plt.subplots(
          1, 2, 
          gridspec_kw={'width_ratios':[width_ratio, 1]}
        )

      self._axs[0].cla()
      self._axs[0].imshow(rgb0)
      self._axs[1].cla()
      self._axs[1].imshow(rgb1)
      self._fig.show()
      
    elif mode == 'rgb_array':
      return rgb0, rgb1

  def _update_spaces(self):
    n = self._obs.num_objects
    for obs, high in zip(self.observation_space, self._obs_high):
      # Tricky way of safely changing the space shape
      obs.shape = (n,) + obs.shape[-3:]
      obs.low = np.full(obs.shape, 0.)
      obs.high = np.full(obs.shape, high)
      obs.bounded_below = -np.inf < obs.low
      obs.bounded_above = np.inf > obs.high

    self.action_space[0].n = n

def register(env_id=None, entry_point=None, **kwargs):
  """Register a StackEnv in the gym registry.
  Args:
    env_id: id with which the environment will be registered. If None,
      id is of the form 'Stack-v%d', using the lowest unregistered
      version number.
    entry_point: class name or string identifier of the environment 
      to be registered. If None, defaults to StackEnv.
    kwargs: key word arguments for StackEnv.
  Returns:
    Registered environment's id.
  Raises:
    gym.error.Error: if provided env_id is already registered.  
  """
  if not env_id:
    ids = [key for key in registry.env_specs if 'Stack-v' in key]
    i = 0
    while 'Stack-v%d'%i in ids:
      i+=1
    env_id = 'Stack-v%d'%i
  if isinstance(entry_point, str):
    if entry_point.lower() == 'started':
      entry_point = StartedStackEnv
    elif entry_point.lower() == 'test':
      entry_point = TestStackEnv
    else:
      raise ValueError("Invalid value {} for argument entry_point".format(entry_point))
  elif not callable(entry_point):
    entry_point = StackEnv

  gym.register(
    id=env_id,
    entry_point=entry_point,
    # max_episode_steps = 2**64-1,
    kwargs = kwargs,
  )

  return env_id
