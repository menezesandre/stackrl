import tensorflow as tf

def greedy_policy(model, output_type=tf.int64, graph=True):
  if not isinstance(model, tf.keras.Model):
    raise TypeError(
      "Invalid type {} for argument model. Must be a keras Model.".format(type(model))
    )

  def policy(inputs):
    return tf.math.argmax(
      model(inputs), 
      axis=-1, 
      output_type=output_type,
    )

  if graph:
    input_spec = tf.nest.map_structure(
      lambda x: tf.TensorSpec(
        shape=x.shape, 
        dtype=x.dtype
      ), 
      model.input
    )
    policy = tf.function(policy, input_signature=[input_spec])

  return policy

def _input_like(model):
  """
  Args:
    model: an instance of a keras Model.
  Returns:
    A keras tensor (or nest of tensors) matching model's
      expected input.
  Raises:
    TypeError: if model is not a keras Model.
  """
  if not isinstance(model, tf.keras.Model):
    raise TypeError(
      "Invalid type {} for argument model. Must be a keras Model.".format(type(model))
    )

  return tf.nest.map_structure(
    lambda i: tf.keras.Input(
      i.shape[1:], 
      batch_size=i.shape[0], 
      dtype=i.dtype
    ), 
    model.input
  )

class GreedyPolicy(tf.keras.Model):
  """Greedy policy"""
  def __init__(
    self, 
    model,
    output_type=tf.int64,
    debug=False,
    name='GreedyPolicy',
  ):
    """
    Args:
      model: estimator of the value/probability of actions.
      output_type: dtype of the model's output (int32 or int64).
      debug: if True, the policy model also returns the values,
        as returned by model.
      name: name of the policy model.
    """
    if len(model.outputs) > 1 or len(model.output_shape) != 2:
      raise ValueError(
        "model must have one output of rank 2 (including batch dimension)."
      )
    inputs = _input_like(model)
    values = model(inputs)  
    outputs = tf.math.argmax(
      values,
      axis=-1, 
      output_type=output_type
    )
    if debug:
      outputs = (outputs, values)
    super(GreedyPolicy, self).__init__(
      inputs=inputs, 
      outputs=outputs, 
      name=name
    )  

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
