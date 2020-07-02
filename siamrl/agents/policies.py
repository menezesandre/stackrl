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

import os

import gin
import gym
import numpy as np

from siamrl import envs
from siamrl import nets

def load_policy(
  observation_spec, 
  path='.', 
  iters=None, 
  config_file=None, 
  debug=False
):
  # Parse config file
  if not config_file:
    try:
      gin.parse_config_file(os.path.join(path, 'config.gin'))
    except OSError:
      pass
  elif os.path.isfile(config_file):
    gin.parse_config_file(config_file)
  elif os.path.isfile(os.path.join(path, config_file)):
    gin.parse_config_file(os.path.join(path, config_file))
  else:
    raise FileNotFoundError("Couldn't find '{}'".format(config_file))
  # Set observation spec
  if isinstance(observation_spec, gym.Space):
    observation_spec = envs.utils.get_space_spec(observation_spec)
    py = True

  if os.path.isdir(os.path.join(path,'saved_weights')):
    if iters is not None:
      if not isinstance(iters, list):
        iters = [iters]  
      paths = [os.path.join(path,'saved_weights', str(i)) for i in iters]
    elif os.path.isfile(os.path.join(path,'eval.csv')):
      # Use best evaluated policy
      iters, reward = np.loadtxt(
        os.path.join(path,'eval.csv'),
        delimiter=',',
        skiprows=2,
        unpack=True,
      )[:2]
      i = np.argmax(reward)
      iters = int(iters[i])
      # print('Iters: {}'.format(iters))
      paths = [os.path.join(path,'saved_weights', str(iters))]

  policies = []
  for path in paths:
    if not os.path.isdir(path):
      raise FileNotFoundError("Couldn't find '{}'".format(path))
    net = nets.PseudoSiamFCN(observation_spec)
    net.load_weights(os.path.join(path,'weights'))
    policy = GreedyPolicy(net, debug=debug)
    if py:
      policy = PyWrapper(policy)
    policies.append(policy)

  if len(policies) == 1:
    return policies[0]
  else:
    return policies


# def greedy_policy(model, name='GreedyPolicy'):
#   """Greedy policy
#   Args:
#     model: estimator of the value/probability of actions.
#     name: name of the policy model.
#   Returns:
#     A keras Model whose output is the argmax of model's output.
#   Raises:
#     ValueError: if model has multiple outputs or output is not 
#       rank 2 (including batch dimension).
#   """
#   if len(model.outputs) > 1 or len(model.output_shape) != 2:
#     raise ValueError(
#       "model must have one output of rank 2 (including batch dimension)."
#     )
#   inputs = input_like(model)
#   outputs = tf.math.argmax(model(inputs), axis=-1)
#   return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)  

# def random_policy_like(model, seed=None, name='RandomPolicy'):
#   """Random policy
#   Args:
#     model: to infer the expected input and output.
#     name: name of the policy model.
#   Returns:
#     A keras Model of the random policy.
#   Raises:
#     ValueError: if model has multiple outputs or output is not 
#       rank 2 (including batch dimension).
#   """
#   if len(model.outputs) > 1 or len(model.output_shape) != 2:
#     raise ValueError(
#       "model must have one output of rank 2 (including batch dimension)."
#     )
#   inputs = input_like(model)
#   outputs = tf.random.uniform(
#     tf.shape(tf.nest.flatten(inputs)[0])[:1], 
#     maxval=model.output_shape[1], 
#     dtype=greedy.output.dtype,
#     seed=seed
#   )

# def epsilon_greedy_policy(
#   model, 
#   epsilon=0.1, 
#   return_greedy=False, 
#   seed=None,
#   name='EpsilonGreedyPolicy'
# ):
#   """Epsilon greedy policy
#   Args:
#     model: estimator of the value/probability of actions.
#     epsilon: probability of taking a random action.
#     return_greedy: whether to also return the greedy policy.
#     seed: Local seed (Use in combination with tf.random.set_seed to
#       get reproducible results.
#   Returns:
#     Tuple with keras Models of the epsilon greedy policy and the 
#       greedy policy if return_greedy ir true, otherwise only the
#       epsilon greedy policy.
#   Raises:
#     ValueError: if model's output is not rank 2 (including batch 
#       dimension.)
#   """
#   greedy = greedy_policy(model)

#   shape = tf.shape(model.output, out_type=greedy.output.dtype)
#   batch_shape = shape[:1]

#   random = tf.random.uniform(
#     batch_shape, 
#     maxval=shape[1], 
#     dtype=greedy.output.dtype,
#     seed=seed
#   )
#   cond = tf.random.uniform(batch_shape, seed=seed) > epsilon
#   output = tf.where(cond, greedy.output, random)
#   epsilon_greedy = tf.keras.Model(
#       inputs=model.inputs, 
#       outputs=output, 
#       name=name
#     )
#   if return_greedy:
#     return epsilon_greedy, greedy
#   else:
#     return epsilon_greedy