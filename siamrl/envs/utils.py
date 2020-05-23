import gym
import numpy as np
import tensorflow as tf
import multiprocessing as mp

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

def get_space_spec(space):
  """Returns a (possibly nested) TensorSpec with space's shape and dtype."""
  return tf.nest.map_structure(
    lambda s,d: tf.TensorSpec(shape=s, dtype=d), 
    get_space_attr(space, 'shape'), 
    get_space_attr(space, 'dtype')
  )

def make(env, n_parallel=None, block=None, **kwargs):
  """
  Args:
    env: Either an instance of a gym.Env or the id of the environment on 
      the gym registry.
    n_parallel: number of environments to run in parallel with 
      multiprocessing. If None, no multiprocessing is used (i.e. only one
      environment running on the current process). Only used if env is not
      an instance of gym.Env.
    block: whether calls to step and reset block by default (only used
      when multiprocessing).
  """
  if isinstance(env, gym.Env) or not n_parallel:
    return Env(env, **kwargs)
  else:
    return ParallelEnv(env, n_parallel=n_parallel, block=block, **kwargs)

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
    self.observation_spec = get_space_spec(self._env.observation_space)
    self.action_spec = get_space_spec(self._env.action_space)
    self.batch_size = 1
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

  def step(self, action):
    return tf.nest.map_structure(
      lambda i: tf.expand_dims(i, axis=0),
      self._env.step(tf.squeeze(action, axis=0).numpy())[:-1] # discard info
    )

  def reset(self):
    return tf.nest.map_structure(
      lambda i: tf.expand_dims(i, axis=0),
      self._env.reset()
    )

  def sample(self):
    """Sample an action from the environment's action space"""
    action = self._env.action_space.sample()
    return tf.nest.map_structure(
      lambda i: tf.expand_dims(i, axis=0),
      action
    )

class ParallelEnv(object):
  """Implements parallel environments with multiprocessing."""
  EXIT = 0
  STEP = 1
  RESET = 2
  RENDER = 3
  CLOSE = 4
  SEED = 5

  def __init__(
    self,
    env_id,
    n_parallel=None,
    block=None,
    seed=None,
    **kwargs
  ):
    """
    Args:
      env_id: id of the environment on the gym registry.
      n_parallel: number of environments to run in parallel processes. If
        None (or 0), the number of CPUs in the system is used.
      block: whether calls to step and reset block by default. If None, 
        defaults to False.
      seed: seed of the environments' random number generators. Incremented 
        for each of the parallel envs to make it unique.      
    """
    # Assert env is registered
    if not env_id in gym.envs.registry.env_specs:
      raise gym.error.UnregisteredEnv(
        "No registered env with id: {}".format(env_id)
      )
    # Store arguments
    self._env_id = env_id
    self._block = block
    self._seed = seed
    self._kwargs = kwargs
    # Make an env to get observation and action specs
    self._dummy_env = gym.make(env_id, **kwargs)
    self.observation_spec = get_space_spec(self._dummy_env.observation_space)
    self.action_spec = get_space_spec(self._dummy_env.action_space)
    self.batch_size = n_parallel or mp.cpu_count()
    # Seed the action space (for the sampling)
    self._dummy_env.action_space.seed(seed)

    self._running = False
    self._conns = []
    self._processes = []

    self.start()

  def __getattr__(self, value):
    return getattr(self._dummy_env, value)

  def __call__(self, *args, **kwargs):
    """Calls step"""
    return self.step(*args, **kwargs)

  def __del__(self):
    self.terminate()
 
  @property
  def multiprocessing(self):
    return True

  def start(self):
    """Set and start the processes. (Can be used to restart after a call 
      of terminate)
    Raises:
      AssertionError: if processes are already running.
    """
    assert not self._running
    for i in range(self.batch_size):
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
    for conn, a in zip(self._conns, tf.unstack(action)):
      conn.send((self.STEP,(
        tf.nest.map_structure(lambda i: i.numpy(), a),
      )))
    block = self._block if block is None else block
    if block:
      return self._recv_and_stack()
    else:
      return self._recv_and_stack

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
      return self._recv_and_stack()
    else:
      return self._recv_and_stack
    
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
    actions = [self.action_space.sample() for _ in range(self.batch_size)]
    return tf.nest.map_structure(
      lambda *args: tf.constant(np.array(args), dtype=tf.int64) if \
        isinstance(args[0], np.ndarray) else tf.constant(args, dtype=tf.int64),
      *actions
    )

  def _recv_and_stack(self):
    """Receives the (possibly nested) messages from the worker processes
    and returns a (nest of) tensor(s) with the received data stacked on 
    the batch dimension."""
    return tf.nest.map_structure(
      # If args is a collection of python elements (i.e. float, bool),
      # tf.constant(args) is the fastest way to get a tensor with the 
      # elements stacked along the batch dimension (axis 0). If the
      # elements are numpy arrays, this (as tf.stack) is REALLY slow. The
      # collection of arrays is then converted to a single array before 
      # converting to a tensor (in this case, np.array(args) is 
      # equivalent to np.stack(args, axis=0), but faster). It is assumed 
      # that all elements are of the same type, so only the first one 
      # needs to be checked.
      lambda *args: tf.constant(np.array(args)) if \
        isinstance(args[0], np.ndarray) else tf.constant(args),
      *[conn.recv() for conn in self._conns]
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
