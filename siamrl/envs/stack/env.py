import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs import registry

import gin

from siamrl.envs import data
from siamrl.envs.stack.simulator import Simulator
from siamrl.envs.stack.observer import Observer
from siamrl.envs.stack.rewarder import Rewarder
from siamrl.baselines import Baseline

try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None

DEFAULT_EPISODE_LENGTH = 24

class StackEnv(gym.Env):
  metadata = {
    'dtypes': ['uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float32', 'float64'],
    'render.modes': ['human', 'rgb_array']
  }

  def __init__(self,
    episode_length=DEFAULT_EPISODE_LENGTH,
    urdfs='train',
    object_max_dimension=0.125,
    use_gui=False,
    sim_time_step=1/60.,
    gravity=9.8,
    num_sim_steps=None,
    velocity_threshold=0.01,
    smooth_placing = True,
    observable_size_ratio=4,
    resolution_factor=5,
    max_z=1,
    goal_size_ratio=.375,
    occupation_ratio_weight=0.,
    occupation_ratio_param=False,
    positions_weight=0.,
    positions_param=0.,
    n_steps_weight=0.,
    n_steps_param=0.,
    contact_points_weight=0.,
    contact_points_param=0.,
    differential=True,
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
      sim_time_step, gravity, num_sim_steps, velocity_threshold, 
        smooth_placing: see Simulator.step.
      observable_size_ratio: size of the observable space as a multiple of
        object_max_dimension. Either a scalar for square space or a list 
        with [height, width] (as seen in the observation).
      resolution_factor: resolution is such that the number of pixels 
        along object_max_dimensions is two to the power of this factor.
      goal_size_ratio, occupation_ratio_weight, occupation_ratio_param,
        positions_weight, positions_param, n_steps_weight, n_steps_param,
        contact_points_weight, contact_points_param, differential: see 
        Rewarder.
      flat_action: whether to receive action as a flat index or a pair of
        indexes [h, w].
      dtype: data type of the returned observation. Must be one of 
        'uint8', 'uint16', 'uint32', 'float16', 'float32' or 'float64'. 
        Internaly, float32 is used.
      seed: Seed for the env's random number generator.
      allow_same_seed: if false, seed used in each instantiation is 
        stored in a class level list (_seeds). Seed used in this instance
        is incremented until it doesn't match any of the previously used 
        seeds. This is useful when using a seed while instantiating 
        parallel environments (so that parallel observations aren't all 
        the same).
    """
    self._length = episode_length
    # Set the files list
    if isinstance(urdfs, str):
      self._list = data.generated(urdfs)
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
    self._sim = Simulator(
      use_gui=use_gui, 
      time_step=sim_time_step, 
      gravity=gravity, 
      spawn_position=[0,0,2*max_z], 
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

    self._obs = Observer(
      self._sim,
      overhead_resolution=overhead_resolution,
      object_resolution=object_resolution,
      pixel_size=pixel_size,
      max_z=max_z
    )

    # Set the rewarder.
    self._rew = Rewarder(
      self._sim, 
      self._obs,
      goal_size_ratio=goal_size_ratio,
      occupation_ratio_weight=occupation_ratio_weight,
      occupation_ratio_param=occupation_ratio_param,
      positions_weight=positions_weight,
      positions_param=positions_param,
      n_steps_weight=n_steps_weight,
      n_steps_param=n_steps_param,
      contact_points_weight=contact_points_weight,
      contact_points_param=contact_points_param,
      differential=differential,
      seed=self._random.randint(2**32)
    )

    # Set the return wrapper for the observations.
    if dtype not in self.metadata['dtypes']:
      raise ValueError('Invalid value {} for argument dtype.'.format(dtype))
    if dtype == 'uint8':
      self._return = lambda x: np.array(x*(2**8-1)/max_z, dtype=dtype)
    elif dtype == 'uint16':
      self._return = lambda x: np.array(x*(2**16-1)/max_z, dtype=dtype)
    elif dtype == 'uint32':
      self._return = lambda x: np.array(x*(2**32-1)/max_z, dtype=dtype)
    elif dtype == 'uint64':
      self._return = lambda x: np.array(x*(2**64-1)/max_z, dtype=dtype)
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
      position=self._obs.position(action), 
      urdf=urdf,
      smooth_placing=self._smooth_placing
    )
    # Get observation of the new state.
    self._obs()
    # Compute reward.
    reward = self._rew()
    return self.observation, reward, self._done, {}

  def reset(self):
    # Get episode's list of urdfs
    self._episode_list = list(self._random.choice(
      self._list, 
      size=self._length,
      replace=self._replace
    ))
    # Reset simulator.
    self._sim.reset(self._episode_list.pop())
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
    g[self._rew.boolean_goal] += 0.1
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
    episode_length=DEFAULT_EPISODE_LENGTH,
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
        of remaining objects makes the episode length. If smaller than
        episode_length, episode_length is used instead.
      start_policy: policy used to place the initial number of objects at
        the beggining of an episode. Either a string identifying one of 
        the policies implemented in siamrl.baselines or a callable 
        implementing the policy. If None, defaults to 'ccoeff' baseline
        if opencv-python is installed, 'random' otherwise.
      flat_action: whether to receive action as a flat index or a pair of
        indexes [h, w].
      kwargs: see super.
    """
    n_objects = max(n_objects, episode_length)
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
      # Try to load ccoeff policy unless opencv-python is not installed.
      # In that case, use random.
      try:
        self._start_policy = Baseline(
          method='ccoeff', 
          flat=flat_action
        )
      except ImportError:
        self._start_policy = Baseline(
          method='random', 
          flat=flat_action
        )
    elif isinstance(start_policy, str):
      self._start_policy = Baseline(
        method=start_policy, 
        flat=flat_action
      )
    elif callable(start_policy):
      self._start_policy = start_policy
    else:
      raise TypeError(
        "Invalid type {} for argument start_policy. Must be a str or callable.".format(type(start_policy))
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

@gin.configurable(module='siamrl.envs.stack')
def register(env_id=None, entry_point=None, **kwargs):
  """Register a StackEnv in the gym registry.
  Args:
    env_id: id with which the environment will be registered. If None,
      id is of the form 'Stack-v%d', using the lowest unregistered
      version number.
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
  elif not callable(entry_point):
    entry_point = StackEnv

  gym.register(
    id=env_id,
    entry_point=entry_point,
    max_episode_steps = 2**64-1,
    kwargs = kwargs
  )

  return env_id

@gin.configurable(module='siamrl.envs.stack')
def curriculum(goals=[], ckwargs={}, **kwargs):
  """Registers a set of environments to form a curriculum.
  Args:
    goals: list of goal reward for each environment.
    ckwargs: changing keyword arguments. Dictionary of lists.
      Each list has the values of the corresponding keyword
      argument for each environment.
    kwargs: constant keyword aguments (for all environments
      registered).
  Returns:
    List of tuples with registered environment ids and
    respective goals.
  """
  ids = []
  # Turn dict of lists to list of dicts
  ckwargs = [dict(zip(ckwargs,values)) for values in zip(*ckwargs.values())]
  for g, a in zip(goals, ckwargs):
    ids.append((register(**a, **kwargs), g))
  return ids
