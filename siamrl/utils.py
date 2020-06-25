import os
import time

import gin
import numpy as np
import tensorflow as tf

import siamrl

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

try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None

def plot(fname, x_key, y_keys, split=None, baselines=None, show=False, legend='Train', save_as=None):
  if plt is None:
    raise ImportError("matplotlib must be installed to run plot.")

  if isinstance(y_keys, str):
    y_keys = [y_keys]

  with open(fname) as f:
    line = f.readline()
    if line.endswith('\n'):
      line = line[:-1]
    keys = line.split(',')

  # Check if all keys are present in file.
  for k in [x_key]+y_keys:
    if k not in keys:
      raise ValueError(
        "No {} column in {}.".format(k, fname))
    
  data = {
    key:value for key, value in zip(
      keys, 
      np.loadtxt(fname, delimiter=',', skiprows=1, unpack=True)
    )
  }
  x = data[x_key]
  ys = [data[key] for key in y_keys]

  fig, axs = plt.subplots(len(ys),1,sharex=True)
  if len(ys) == 1:
    # To be consistent for any number of targets
    axs = (axs,)

  if split and os.path.isfile(split):
    data = np.loadtxt(
      split,
      delimiter=',',
      skiprows=1,
      unpack=True,
    )
    x_splits = np.atleast_1d(data[0])
    valid = np.atleast_1d(data[1])

    split = [0]
    i = 0
    for x_split, v in zip(x_splits, valid):
      for i in range(i,len(x)):
        if x[i] > x_split:
          if v:
            split.append(i)
          else:
            if x[0] != 0:
              split = [i]            
          break
    split.append(len(x))

    split_x = [x[split[i]:split[i+1]+1] for i in range(len(split)-1)]
    split_ys = [
      [y[split[i]:split[i+1]+1] for i in range(len(split)-1)] 
      for y in ys
    ]

        
    for ax, split_y in zip(axs, split_ys):
      for xi,yi in zip(split_x, split_y):
        ax.plot(xi,yi)
    legend = ['{} part {}'.format(legend, i) for i in range(len(split_x))]
  else:
    for ax, y in zip(axs, ys):
      ax.plot(x, y)
    legend = [legend]

  if baselines:
    for key in baselines:
      axs[-1].plot([x[0], x[-1]],[baselines[key]]*2)
      legend.append(key.capitalize())

  if len(legend) > 1:
    plt.legend(legend, loc='best')

  for ax, key, y in zip(axs, y_keys, ys):
    ax.set_ylabel(key)
    ylim = ax.get_ylim()
    _mean = np.mean(y)
    _std = np.std(y)
    if ylim[0] >= 0:
      ymin = ylim[0]
    else:
      ymin = min(0, max(ylim[0], _mean - 10*_std))
    if ylim[1] <= 0:
      ymax = ylim[1]
    else:
      ymax = max(0,min(ylim[1], _mean + 10*_std))
    ax.set_ylim(ymin,ymax)

  axs[-1].set_xlabel(x_key)

  if save_as is None:
    if not show:
      path, name = os.path.split(fname)
      name = name.split('.')[0]
      if path and path != '.':
        path, pref = os.path.split(path)
        name = '{}_{}'.format(pref, name)
      for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(path,'plots',ext,'{}.{}'.format(name,ext)))
  else:
    plt.savefig(save_as)
  if show:
    plt.show()
  else:
    plt.close()

def plot_train(path, **kwargs):
  return plot(
    fname=os.path.join(path, 'train.csv'),
    x_key='Iters',
    y_keys=['Loss', 'Reward'],
    split=os.path.join(path, 'curriculum.csv'),
    **kwargs,
  )

def plot_eval(path, **kwargs):
  try:
    gin.parse_config_file(os.path.join(path, 'config.gin'))
    env_id = siamrl.envs.stack.register()
  except OSError:
    env_id = None

  if env_id:
    bpath = os.path.join(
      os.path.dirname(__file__),
      '..',
      'data',
      'baselines',
      siamrl.envs.utils.as_path(env_id),
    )
    bfile = os.path.join(
      bpath,
      'results',
    )
    if os.path.isfile(bfile):
      baselines = {}
      with open(bfile) as f:
        for line in f:
          line = line.split(':')
          baselines[line[0]] = float(line[1])
    else:
      baselines = siamrl.baselines.test(env_id)
  else:
    baselines = None 

  return plot(
    fname=os.path.join(path, 'eval.csv'),
    x_key='Iters',
    y_keys=['Reward'],
    split=os.path.join(path, 'curriculum.csv'),
    baselines=baselines,
    **kwargs,
  )

def plot_results(path):
  plot_train(path)
  plot_eval(path)