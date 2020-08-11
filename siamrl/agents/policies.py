import numpy as np
import tensorflow as tf

class Greedy(tf.Module):
  """Greedy policy os a tf value estimator."""

  def __init__(
    self,
    model,
    output_type='int64',
    value=False,
    batchwise=False,
  ):
    if not callable(model) or not isinstance(model, tf.Module):
      raise TypeError("model must be a callable tf.Module.")
    self.model = model
    self.output_type = tf.dtypes.as_dtype(output_type)
    self.value = value
    self.batchwise = batchwise

  def __call__(self, inputs): 
    values = self.model(inputs)
    outputs = tf.math.argmax(
      values,
      axis=-1,
      output_type=self.output_type,
    )
    if self.batchwise:
      max_values = tf.reduce_max(values, axis=-1)
      batchwise_argmax = tf.math.argmax(
        max_values, 
        output_type=self.output_type
      )
      outputs = batchwise_argmax, outputs[batchwise_argmax]
    if self.value:
      outputs = outputs, values
    return outputs

class PyGreedy(object):
  """Greedy policy of a python value function."""
  def __init__(
    self,
    model,
    value=False,
    unravel=False,
    batched=False,
    batchwise=False,
  ):
    if not callable(model):
      raise TypeError("model must be callable.")
    self.model = model
    self.value = value
    self.unravel = unravel
    self.batched = batched
    self.batchwise = batchwise

  def __call__(self, inputs):
    if self.batched:
      argmax_list = []
      values_list = []
      if self.batchwise:
        max_list = []
      for inps in zip(*tf.nest.flatten(inputs)):
        argmax, values = self.call(tf.nest.pack_sequence_as(inputs, inps))
        if self.unravel:
          argmax = np.unravel_index(argmax, values.shape)
        else:
          values = values.ravel()
        
        argmax_list.append(argmax)
        values_list.append(values)
        if self.batchwise:
          max_list.append(values[argmax])

      outputs = np.array(argmax_list)
      values = np.array(values_list)

      if self.batchwise:
        batchwise_argmax = np.argmax(max_list)
        outputs = batchwise_argmax, outputs[batchwise_argmax]
    else:
      outputs, values = self.call(inputs)
      if self.unravel:
        outputs = np.array(np.unravel_index(outputs, values.shape))
      elif self.value:
        values = values.ravel()

    if self.value:
      outputs = outputs, values

    return outputs

  def call(self, inputs):
    values = self.model(inputs)
    argmax = np.argmax(values)
    return argmax, values

class PolicyWrapper(object):
  """Base class for policy wrappers, that aply a conversion to input
  observations and output actions."""
  def __init__(self, policy):
    """
    Args:
      policy: tf policy to be wrapped.
    Raises:
      TypeError: if policy is not callable
    """
    if not callable(policy):
      raise TypeError("policy is not callable.")
    self._policy = policy

  def __getattr__(self, value):
    """Get attribute from policy."""
    return getattr(self._policy, value)

  def __call__(self, inputs):
    return self._output(self._policy(self._input(inputs)))

  def _input(self, value):
    raise NotImplementedError()
  
  def _output(self, value):
    raise NotImplementedError()

class PyWrapper(PolicyWrapper):
  """Wraps a tf policy to receive and return (possibly unbatched) numpy 
  arrays as observations and actions."""
  def __init__(self, policy, batched=False):
    """
    Args:
      policy: tf policy to be wrapped.
      batched: whether observations and actions are batched.
    Raises:
      TypeError: if policy is not callable"""
    super(PyWrapper, self).__init__(policy)
    if batched:
      self._py2tf = lambda i: tf.identity(i)
      self._tf2py = lambda i: i.numpy()
    else:
      self._py2tf = lambda i: tf.expand_dims(i, axis=0)
      self._tf2py = lambda i: tf.squeeze(i).numpy()

  def _input(self, value):
    return tf.nest.map_structure(self._py2tf, value)

  def _output(self, value):
    return tf.nest.map_structure(self._tf2py, value)

class TFWrapper(PolicyWrapper):
  """Wraps a python policy to receive and return batched tf tensors as 
  observations and actions."""
  def __call__(self, inputs):
    return tf.map_fn(
      lambda i: tf.numpy_function(self._policy, [i], tf.int64),
      inputs,
      dtype=tf.int64
    )
