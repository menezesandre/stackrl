import time
import os
import gin
import tensorflow as tf
from siamrl.nets import PseudoSiamFCN
from siamrl.agents.policies import GreedyPolicy, PyWrapper

class Timer(object):
  """Utility context manager that accumulates execution time 
    inside with statements. On call, returns the average 
    execution time (accumulated time / number of executions).
    Supports basic operations on the accumulated time."""
  def __init__(self, clock=None):
    """
    Args:
      clock: clock to be used by the timer. Either a callable 
        or a string with the name of a function from time 
        module. If None, time.perf_counter is used.
    Raises:
      TypeError: if clock is neither a callable or a string.
      AttributeError: if clock is a string with a name that doesn't exist
        in the time module.
    """
    if clock is None:
      self.clock = time.perf_counter
    elif isinstance(clock, str):
      self.clock = getattr(time, clock)
    elif callable(clock):
      self.clock = clock
    else:
      raise TypeError(
        "Invalid type {} for argument clock.".format(type(clock))
      )
    self.reset()

  def __call__(self, reset=True):
    """
    Args:
      reset: whether to reset the timer after getting the result.
    Returns:
      The average block execution time since last reset. If 
        not ready (self.ready==False), returns None. This is to
        avoid computing a result before any execution (zero division) or 
        inside the with statement (in which case the result would be 
        meaningless).
    """
    if self.ready:
      result = self.time/self.n
      if reset:
        self.reset()
      return result

  def __enter__(self):
    self.ready = False
    self.time -= self.clock()

  def __exit__(self, type, value, tb):
    self.time += self.clock()
    self.n +=1
    self.ready = True

  def __add__(self, other):
    return self.time + other
  def __radd__(self, other):
    return self.__add__(other)
  def __sub__(self, other):
    return self.time - other
  def __rsub__(self, other):
    return other - self.time
  def __mul__(self, other):
    return self.time * other
  def __rmul__(self, other):
    return self.__mul__(other)
  def __truediv__(self, other):
    return self.time / other
  def __rtruediv__(self, other):
    return other / self.time
  def __floordiv__(self, other):
    return self.time // other
  def __rfloordiv__(self, other):
    return other // self.time
  def __mod__(self, other):
    return self.time % other
  def __rmod__(self, other):
    return other % self.time

  def reset(self):
    self.ready = False
    self.time = 0.
    self.n = 0

class AverageReward(tf.Module):
  def __init__(self, batch_size):
    """
    Args:
      batch_size: of the rewards (and terminal flags) to be received by
      __call__().
    """
    self._total_reward = tf.Variable(0., dtype=tf.float32)
    self._episode_reward = tf.Variable(tf.zeros(
      (batch_size,), 
      dtype=tf.float32
    ))
    self._episode_count = tf.Variable(0, dtype=tf.int32)
  
  def __call__(self, reward, terminal):
    """Adds a step's reward to the metric."""
    # Add this step's reward to the episode reward
    self._episode_reward.assign_add(reward)

    terminal_count = tf.math.count_nonzero(terminal, dtype=tf.int32)
    terminal_indexes = tf.where(terminal)
    # Add the number of finished episodes to episode count
    self._episode_count.assign_add(terminal_count)
    # Add the rewards from finished episodes to the total reward
    self._total_reward.assign_add(tf.reduce_sum(
      self._episode_reward.sparse_read(terminal_indexes)
    ))
    # Reset the reward from finished episodes
    self._episode_reward.scatter_nd_update(
      terminal_indexes, 
      tf.zeros(terminal_count, dtype=tf.float32)
    )
    # Alternative (TODO check if it's faster)
    # self._total_reward.assign_add(tf.reduce_sum(tf.where(
    #   terminal,
    #   self._episode_reward,
    #   0.
    # )))
    # self._episode_reward.assign(tf.where(
    #   terminal,
    #   0.,
    #   self._episode_reward
    # ))
  
  # Comparison operations
  def __lt__(self, other):
    return self.result < other
  def __le__(self, other):
    return self.result <= other
  def __eq__(self, other):
    return self.result == other
  def __ge__(self, other):
    return self.result >= other
  def __gt__(self, other):
    return self.result > other

  @property
  def episode_count(self):
    return self._episode_count
  
  @property
  def result(self):
    return self._total_reward/tf.cast(self._episode_count, dtype=tf.float32)

  def reset(self, full=True):
    """Reset metric. If full is False, the rewards of the ongoing episodes
    are kept."""
    self._total_reward.assign(0.)
    self._episode_count.assign(0)
    if full:
      self._episode_reward.assign(tf.zeros_like(self._episode_reward))

def load_policy(observation_spec, path='.', config_file=None, py_format=False):
  if config_file:
    try:
      gin.parse_config_file(config_file)
    except OSError:
      gin.parse_config_file(os.path.join(path, config_file))

  net = PseudoSiamFCN(observation_spec)
  if os.path.isdir(path):
    path = os.path.join(path,'weights')
  net.load_weights(path)
  policy = GreedyPolicy(net)
  if py_format:
    policy = PyWrapper(policy)
  
  return policy
