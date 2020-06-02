import time
import os
import gin
import tensorflow as tf

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

class AverageMetric(tf.Module):
  def __init__(self, length=100, dtype=tf.float32):
    super(AverageMetric, self).__init__(name='Average')
    self._dtype = tf.dtypes.as_dtype(dtype)
    self._length = length

    self._index = tf.Variable(0, dtype=tf.int32)
    self._values = tf.Variable(tf.zeros((length,), dtype=self._dtype))

  @property
  def result(self):
    return tf.reduce_mean(self._values) if self.full else \
      tf.reduce_sum(self._values)/tf.cast(self._index, dtype=self._dtype)
  @property
  def full(self):
    return self._index >= self._length

  def add(self, value):
    self._values.scatter_update(tf.IndexedSlices(
      value,
      self._index%self._length
    ))
    self._index.assign_add(1)

  def reset(self):
    """Reset metric."""
    self._values.assign(tf.zeros_like(self._values))
    self._index.assign(tf.zeros_like(self._index))

  # Aliases of add.
  def __call__(self, *args, **kwargs):
    """Calling does the same as add"""
    return self.add(*args, **kwargs)
  def __iadd__(self, other):
    """Inplace adition does the same as add"""
    self.add(other)
    return self
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

class AverageReward(AverageMetric):
  def __init__(self, batch_size, length=100, dtype=tf.float32):
    """
    Args:
      batch_size: of the rewards (and terminal flags) to be received by
      __call__().
    """
    super(AverageReward, self).__init__(length=length, dtype=dtype)
    self._episode_reward = tf.Variable(tf.zeros(
      (batch_size,), 
      dtype=self._dtype
    ))
  
  def add(self, step):
    """Adds a step's reward to the metric.
    Args:
      step: collection in which the last two elements are a batch of 
        rewards and terminal flags."""
    reward = step[-2]
    terminal = step[-1]
    # Add this step's reward to the episode reward
    self._episode_reward.assign_add(reward)
    # Add the rewards from the finished episodes to the buffer and reset them
    for i in tf.squeeze(tf.where(terminal), axis=-1):
      super(AverageReward, self).add(self._episode_reward.sparse_read(i))
      self._episode_reward.scatter_update(tf.IndexedSlices(0, i))
  
  def reset(self, full=False):
    """Reset metric. If full is False, the rewards of the ongoing episodes
    are kept."""
    super(AverageReward, self).reset()
    if full:
      self._episode_reward.assign(tf.zeros_like(self._episode_reward))

# class FreezeDependencies(object):
#   """Utility context manager that freezes the dependencies of a tf.Module
#   (i.e. any attribute added within the context won't be tracked)."""
#   def __init__(self, module):
#     if isinstance(module, tf.Module):
#       self.module = module
#     else:
#       raise TypeError(
#         "Invalid type {} for argument module. Must be a tf.Module.".format(
#           type(module)
#         )
#       )

#   def __enter__(self):
#     self._checkpoint_dependencies = self.module._unconditional_checkpoint_dependencies.copy()

#   def __exit__(self, type, value, tb):
#     to_del = []
#     for i in self.module._unconditional_checkpoint_dependencies:
#       if i not in self._checkpoint_dependencies:
#         to_del.append(i)
#     for i in to_del:
#       try:
#         self.module._unconditional_checkpoint_dependencies.remove(i)
#       except ValueError:
#         pass
#       try:
#         del self.module._unconditional_dependency_names[i.name]
#       except KeyError:
#         pass

