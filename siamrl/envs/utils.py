import inspect
import multiprocessing as mp
import os

import gin
import gym
import numpy as np
import tensorflow as tf

def get_space_attr(space, attr='shape'):
  """Recursive utility function to get a (possibly nested) attribute of a 
  gym Space"""
  assert isinstance(space, gym.Space)
  if hasattr(space, 'spaces'):
    return tuple(get_space_attr(s, attr=attr) for s in space.spaces)
  else:
    value = getattr(space, attr)
    # If this value is seen as nested (i.e. a tuple with shape), make it
    # an array so that it is seen as a single object by tf.nest
    if tf.nest.is_nested(value):
      value = np.array(value) 
    return value

def get_space_spec(space, remove_first_dim=None):
  """Returns a (possibly nested) TensorSpec with space's shape and dtype.
  Args:
    space: to take the TensorSpec from.
    remove_first_dim: Either a boolean indicating whether the space shape's
      leading dimension should be removed, or an integer indicating the 
      number of leading dimensions to be removed. If None, at most 3 
      dimensions (i.e. height, width, depth) are kept. Use to remove batch 
      dimension if necessary.
  """
  remove_first_dim = -3 if remove_first_dim is None else int(remove_first_dim)
  return tf.nest.map_structure(
    lambda s,d: tf.TensorSpec(shape=s[remove_first_dim:], dtype=d), 
    get_space_attr(space, 'shape'), 
    get_space_attr(space, 'dtype')
  )

def isspace(obj):
  return isinstance(obj, gym.Space)

@gin.configurable(module='siamrl.envs')
def make(
  env='Stack-v0', 
  n_parallel=None, 
  block=None, 
  curriculum=None,
  unwrapped=False,
  as_path=False,
  **kwargs,
):
  """Instantiate an environment wrapped to be compatible with siamrl.Training.
  Args:
    env: Either an instance of a gym.Env or the id of the environment on 
      the gym registry.
    n_parallel: number of environments to run in parallel with 
      multiprocessing. If None, no multiprocessing is used (i.e. only one
      environment running on the current process). Only used if env is not
      an instance of gym.Env.
    block: whether calls to step and reset block by default (only used
      when multiprocessing).
    curriculum: dict of lists with kwargs for each environment in the 
      curriculum, including a key 'goals' with the goal returns. If 
      provided, the return is a generator that yields tuples with an 
      instance of each environment and corresponding goal. 
    unwrapped: if True, return the unwrapped gym Env. 
    as_path: if True, instead of an environment instance, the return is a
      string containing a path generated from the environment's entry point
      and complete set of parameters (including the kwargs passed to make, 
      gym registry kwargs and class defaults).
    kwargs: passed to the environment.
  Returns:
    environment instance or, if curriculum is not None, a generator that 
      yields tuples with environment instances and goal returns. 
  """
  if curriculum:
    return make_curriculum(
      env, 
      n_parallel=n_parallel, 
      block=block, 
      curriculum=curriculum,
      unwrapped=unwrapped,
      as_path=as_path,
      **kwargs,
    )
  else:
    if as_path:
      if isinstance(env, gym.Env):
        spec = env.unwrapped.spec
      elif isinstance(env, str):
        spec = gym.envs.registry.env_specs[env]
      else:
        raise TypeError(
          "Invalid type {} for argument env.".format(type(env))
        )

      entry_point = spec.entry_point
      if not callable(entry_point):
        entry_point = gym.envs.registration.load(entry_point)

      # Get entry point's default parameters
      args = {
        p.name:(p.default if p.default != p.empty else None)
        for p in inspect.signature(entry_point).parameters.values()
      }
      # Update with registered kwargs
      args.update(spec._kwargs)
      # Update with provided kwargs
      args.update(kwargs)
      
      path = []
      # Make a string with arg name and value for each argument
      for k,v in args.items():
        if k != 'seed':
          k = k.split('_')
          # Each name uses at most 4 chars
          if len(k) > 1:
            k = k[0][:1] + k[-1][:3]
          else:
            k = k[0][:4]
          path.append(k + str(v))
      # Join all argument strings with commas, removing spaces and quotation marks
      path = ','.join(path).replace(' ', '').replace("'", '').replace('"','')
      # Add the entry point's name
      return os.path.join(entry_point.__name__, path)
    elif unwrapped:
      if isinstance(env, gym.Env):
        return env
      elif isinstance(env, str):
        return gym.make(env, **kwargs)
      else:
        raise TypeError(
          "Invalid type {} for argument env.".format(type(env))
        )
    else:
      if not n_parallel:
        return Env(env, **kwargs)
      else:
        return ParallelEnv(env, n_parallel=n_parallel, block=block, **kwargs)

def make_curriculum(
  env='Stack-v0', 
  n_parallel=None, 
  block=None, 
  curriculum={},
  unwrapped=False,
  as_path=False,
  **kwargs,
):
  """Generator function that yields tuples with environment instances and goal returns.
  Args:
    see make
  """
  if 'goals' in curriculum:
    goals = curriculum.pop('goals')
  else:
    goals = None
    # raise ValueError("Missing key 'goals' in curriculum.")

  # Turn dict of lists to list of dicts
  curriculum = [dict(zip(curriculum.keys(),values)) for values in zip(*curriculum.values())]
  
  if goals is not None and len(goals) != len(curriculum):
    raise ValueError("length of goals doesn't match number of environments")
  
  for i, cargs in enumerate(curriculum):
    env_instance =  make(
      env, 
      n_parallel=n_parallel, 
      block=block, 
      curriculum=None,
      unwrapped=unwrapped,
      as_path=as_path,
      **cargs, 
      **kwargs,
    )
    if goals is None:
      yield env_instance, None
    else:
      yield env_instance, goals[i]


class Env(object):
  """Wraps a gym Env to receive and return tensors with batch dimension.
  Info (4th return from gym.Env.step) is supressed."""
  def __init__(self, env, **kwargs):
    """
    Args:
      env: Either an instance of a gym Env or the id of the environment on 
        the gym registry.
    """
    if isinstance(env, gym.Env):
      self._env = env
    elif isinstance(env, str):
      self._env = gym.make(env, **kwargs)
    else:
      raise TypeError(
        "Invalid type {} for argument env.".format(type(env))
      )
    self._observation_spec = get_space_spec(self._env.observation_space)
    self._observation_is_nested = tf.nest.is_nested(self.observation_spec)
    self._observation_is_array = tf.nest.map_structure(
      lambda i: isinstance(i, np.ndarray), 
      self._env.observation_space.sample()
    )
    self._action_spec = get_space_spec(self._env.action_space)
    self._action_is_nested = tf.nest.is_nested(self.action_spec)
    self._action_is_array = tf.nest.map_structure(
      lambda i: isinstance(i, np.ndarray), 
      self._env.action_space.sample()
    )
    # Define action conversion functions
    if self._action_is_nested:
      self._action_in = lambda inputs: tf.nest.map_structure(
        lambda i: i.numpy()[0], 
        inputs
      )
      self._action_out = lambda inputs: tf.nest.map_structure(
        lambda i,nd,spec: tf.constant(
          i[np.newaxis] if nd else [i], 
          dtype=spec.dtype
        ),
        inputs, self._action_is_array, self._action_spec
      )
    else:
      self._action_in = lambda inputs: inputs.numpy()[0]
      if self._action_is_array:
        self._action_out = lambda inputs: tf.constant(
          inputs[np.newaxis], 
          dtype=self._action_spec.dtype
        )
      else:
        self._action_out = lambda inputs: tf.constant(
          [inputs],
          dtype=self._action_spec.dtype
        )
    # Define observation conversion functions
    if self._observation_is_nested:
      self._observation_out = lambda inputs: tf.nest.map_structure(
        lambda i,nd,spec: tf.constant(
          i[np.newaxis] if nd else [i],
          dtype=spec.dtype
        ),
        inputs, self._observation_is_array, self._observation_spec
      )
    else:
      if self._observation_is_array:
        self._observation_out = lambda inputs: tf.constant(
          inputs[np.newaxis],
          dtype=self._observation_spec.dtype
        )
      else:
        self._observation_out = lambda inputs: tf.constant(
          [inputs], 
          dtype=self._observation_spec.dtype
        )
    # Seed the action space (for the sampling)
    if 'seed' in kwargs:
      self._env.action_space.seed(kwargs['seed'])

  def __getattr__(self, value):
    return getattr(self._env, value)

  def __call__(self, *args, **kwargs):
    """Calls step"""
    return self.step(*args, **kwargs)

  @property
  def multiprocessing(self):
    return False
  @property
  def batch_size(self):
    return 1
  @property
  def observation_spec(self):
    return self._observation_spec
  @property
  def action_spec(self):
    return self._action_spec

  def step(self, action):
    o,r,t,_ = self._env.step(self._action_in(action))
    return (
      self._observation_out(o),
      tf.constant([r], dtype=tf.float32), 
      tf.constant([t], dtype=tf.bool)
    )

  def reset(self):
    return (
      self._observation_out(self._env.reset()),
      tf.zeros((1,), dtype=tf.float32),
      tf.zeros((1,), dtype=tf.bool)
    )

  def sample(self):
    """Sample an action from the environment's action space"""
    return self._action_out(self._env.action_space.sample())

class ParallelEnv(Env):
  """Implements parallel environments with multiprocessing."""
  EXIT = 0
  STEP = 1
  RESET = 2
  RENDER = 3
  CLOSE = 4
  SEED = 5

  def __init__(
    self,
    env,
    n_parallel=None,
    block=None,
    seed=None,
    **kwargs
  ):
    """
    Args:
      env: id of the environment on the gym registry. If a gym.Env instance
        is provided, the id is taken from its spec. Note that this ignores any
        kwargs used to make that environment.
      n_parallel: number of environments to run in parallel processes. If
        None (or 0), the number of CPUs in the system is used.
      block: whether calls to step and reset block by default. If None, 
        defaults to False.
      seed: seed of the environments' random number generators. Incremented 
        for each of the parallel envs to make it unique.
      kwargs: passed to the environment.
    """
    # If env is a gym.Env instance, get id
    if isinstance(env, gym.Env):
      env = env.unwrapped.spec.id

    super(ParallelEnv, self).__init__(env, seed=seed, **kwargs)
    # Store arguments
    self._env_id = env
    self._block = block
    self._seed = seed
    self._kwargs = kwargs
    self._n_parallel = n_parallel or mp.cpu_count()
    # Define action conversion functions
    if self._action_is_nested:
      self._action_in = lambda inputs: (
        tf.nest.pack_sequence_as(
          self._action_spec, 
          [k.numpy for k in j]
        ) for j in zip(
          *[tf.unstack(i) for i in tf.nest.flatten(inputs)]
        )
      )
      self._action_out = lambda inputs: tf.nest.map_structure(
        lambda nd, spec, *args: tf.constant(
          np.array(args), 
          dtype=spec.dtype) if nd else tf.constant(
            args, 
            dtype=spec.dtype
          ),
        self._action_spec,
        self._action_is_array,
        *inputs
      )
    else:
      self._action_in = lambda i: (j.numpy() for j in tf.unstack(i))
      if self._action_is_array:
        self._action_out = lambda inputs: tf.constant(
          np.array(inputs), 
          dtype=self._action_spec.dtype
        )
      else:
        self._action_out = lambda inputs: tf.constant(
          inputs, 
          dtype=self._action_spec.dtype
        )
    # Define observation conversion functions
    self._step_spec = (self._observation_spec, tf.TensorSpec((),dtype=tf.float32), tf.TensorSpec((),dtype=tf.bool))
    self._step_is_array = (self._observation_is_array, False, False)
    self._step_out = lambda inputs: tf.nest.map_structure(
      lambda nd,spec,*args: tf.constant(
        np.array(args) if nd else args,
        dtype=spec.dtype
      ),
      self._step_is_array,
      self._step_spec,
      *inputs
    )
    if self._observation_is_nested:
      self._observation_out = lambda inputs: tf.nest.map_structure(
        lambda nd,spec,*args: tf.constant(
          np.array(args) if nd else args,
          dtype=spec.dtype
        ),
        self._observation_is_array, self._observation_spec, *inputs
      )
    else:
      if self._observation_is_array:
        self._observation_out = lambda inputs: tf.constant(
          np.array(inputs),
          dtype=self._observation_spec.dtype
        )
      else:
        self._observation_out = lambda inputs: tf.constant(
          inputs, 
          dtype=self._observation_spec.dtype
        )
    # Initialize internal variables
    self._running = False
    self._conns = []
    self._processes = []

    self.start()

  def __del__(self):
    self.terminate()
 
  @property
  def multiprocessing(self):
    return True
  @property
  def batch_size(self):
    return self._n_parallel

  def start(self):
    """Set and start the processes. (Can be used to restart after a call 
      of terminate)
    Raises:
      AssertionError: if processes are already running.
    """
    assert not self._running
    for i in range(self._n_parallel):
      # Set the unique seed.
      self._kwargs['seed'] = (self._seed + i)%2**32 if self._seed is not None else None
      # Create the pipe to comunicate with this process.
      conn1, conn2 = mp.Pipe()
      # Create and start the process to run an environment.
      p = mp.Process(
        target=self._runner, 
        args=(conn2, self._env_id),
        kwargs=self._kwargs,
        daemon=True
      )
      p.start()
      # Store the connection and the process object.
      self._conns.append(conn1)
      self._processes.append(p)

    self._running = True

  def terminate(self):
    """Sends the exit command and joins each process."""
    while self._conns:
      conn = self._conns.pop()
      try:
        conn.send((self.EXIT, ()))
      except BrokenPipeError:
        pass
      conn.close()
    while self._processes:
      p = self._processes.pop()
      p.join(1)
      if p.exitcode is None:
        # Force termination if necessary
        p.terminate()
        p.join()
    self._running = False

  def step(self, action, block=None):
    """Unstacks batched action and sends the step command to each 
      environment's process. 
    Args:
      action: batch of actions (batch size must match the number of 
        parallel environments).
      block: whether to wait for the results.
    Returns:
      If block is True, returns the batched time step (observation, 
      reward, terminal). Otherwise, returns a callable that will return
      the time step once it is ready.
    """
    for conn, a in zip(self._conns, self._action_in(action)):
      conn.send((self.STEP,(a,)))
    block = self._block if block is None else block
    if block:
      return self._recv_step()
    else:
      return self._recv_step

  def reset(self, block=None):
    """Sends the reset command to each environment's proccess.
    Args:
      block: whether to wait for the results.
    Returns:
      If block is True, returns the batched observation. Otherwise, 
      returns a callable that will return the observation once it is 
      ready.
    """
    for conn in self._conns:
      conn.send((self.RESET,()))
    block = self._block if block is None else block
    if block:
      return self._recv_reset()
    else:
      return self._recv_reset
    
  def render(self, mode=None):
    """Sends the render command to each environment's proccess.
    Args:
      mode: rendering mode sent the environment.
    Returns:
      List of the environments' render returns (may be a list of Nones).
    """
    arg = (mode,) if mode is not None else ()
    for conn in self._conns:
      conn.send((self.RENDER,arg))
    return [conn.recv() for conn in self._conns]

  def close(self):
    """Sends the close command to each environment's proccess."""
    for conn in self._conns:
      conn.send((self.CLOSE,()))

  def seed(self, seed):
    """Sends the new seed to each environment's proccess.
    Args:
      seed: new seed. For each environment, it is incremented to make it
        unique.
    Returns:
      List of the environments' seed returns.
    """
    for i, conn in enumerate(self._conns):
      conn.send((self.SEED, ((seed + i) % 2**32,)))
    return [conn.recv() for conn in self._conns]

  def sample(self):
    """Sample a batch of actions from the environment's action space"""
    return self._action_out(
      [self.action_space.sample() for _ in range(self.batch_size)]
    )

  def _recv_step(self):
    """Receives the observations, rewards and terminal states from steping 
      the environments and stacks them on the batch dimension."""
    return self._step_out([conn.recv() for conn in self._conns])

  def _recv_reset(self):
    """Receives the observations from reseting the environments and 
      stacks them on the batch dimension."""
    return (
      self._observation_out([conn.recv() for conn in self._conns]),
      tf.zeros((self.batch_size,), dtype=tf.float32),
      tf.zeros((self.batch_size,), dtype=tf.bool)
    )
    
  def _runner(self, conn, env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    while True:
      try:
        # if not conn.poll(1):
        #   continue
        m, args = conn.recv()
      except EOFError:
        break
      if m == self.EXIT:
        break
      elif m == self.STEP:
        conn.send(env.step(*args)[:-1])
      elif m == self.RESET:
        conn.send(env.reset())
      elif m == self.RENDER:
        conn.send(env.render(*args))
      elif m == self.CLOSE:
        env.close()
      elif m == self.SEED:
        conn.send(env.seed(*args))
    env.close()
    conn.close()

  # About conversion to tensors:
  # If args is a collection of python elements (i.e. float, bool),
  # tf.constant(args) is the fastest way to get a tensor with the 
  # elements stacked along the batch dimension (axis 0). If the
  # elements are numpy arrays, this (as tf.stack) is REALLY slow. The
  # collection of arrays is then converted to a single array before 
  # converting to a tensor (in this case, np.array(args) is 
  # equivalent to np.stack(args, axis=0), but faster). It is assumed 
  # that all elements are of the same type, so only the first one 
  # needs to be checked.