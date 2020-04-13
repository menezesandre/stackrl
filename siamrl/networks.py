"""
References:
  [1](https://arxiv.org/abs/1606.09549)
"""
import copy
import gin

import tensorflow as tf
from tensorflow.keras import layers
import tf_agents
from tf_agents.networks import network
from tf_agents.specs import tensor_spec

import siamrl

DEFAULT_BRANCH_PARAMS = [
  lambda: layers.Conv2D(
    filters=32, 
    kernel_size=8,
    strides=4, 
    activation='relu', 
    padding='same',
    kernel_initializer='he_uniform'
  ),
  lambda: layers.Conv2D(
    filters=64, 
    kernel_size=4,
    dilation_rate=2, 
    activation='relu', 
    padding='same',
    kernel_initializer='he_uniform'
  ),
  lambda: layers.UpSampling2D( 
    size=4,
    interpolation='bilinear'
  )  
]
DEFAULT_POS_PARAMS = [
  lambda: layers.Conv2D(
    filters=32, 
    kernel_size=5,
    activation='relu', 
    padding='same',
    kernel_initializer='he_uniform'
  ),
  lambda: layers.Conv2D(
    filters=1, 
    kernel_size=1,
    padding='same',
    kernel_initializer='he_uniform'
  ),
  lambda: layers.Flatten()
]

_floats = [tf.float16, tf.float32, tf.float64]
_uints = [tf.uint8, tf.uint16, tf.uint32, tf.uint64]
allowed_dtypes = _floats+_uints

def _validate_input_tensor_spec(input_tensor_spec):
  try:
    assert len(input_tensor_spec) == 2, \
      "Argument input_tensor_spec must have length 2."
  except TypeError:
    raise AssertionError("Argument input_tensor_spec must have length.")
  assert len(input_tensor_spec[0].shape) == 3, \
    "First inputs must have 3 dimensions [height,width,channels]."
  assert len(input_tensor_spec[1].shape) == 3, \
    "Second inputs must have 3 dimensions [height,width,channels]."
  assert input_tensor_spec[0].shape[0] >= input_tensor_spec[1].shape[0], \
    "First input must have larger height."
  assert input_tensor_spec[0].shape[1] >= input_tensor_spec[1].shape[1], \
    "First input must have larger width."
  assert input_tensor_spec[0].dtype == input_tensor_spec[1].dtype, \
    "Both inputs must be of the same data type."
  assert input_tensor_spec[0].dtype in allowed_dtypes, \
    "Input data type must be a float or unsigned integer."
  return input_tensor_spec[0].dtype
 
class Correlation(layers.Layer):
  def __init__(self):
    super(Correlation, self).__init__()

  @staticmethod
  def _correlation(inputs):
    x = tf.nn.conv2d(tf.expand_dims(inputs[0], 0), tf.expand_dims(inputs[1], -1), strides=1, padding='VALID')
    return tf.squeeze(x, axis=0)
    
  def call(self, inputs):
    return tf.map_fn(self._correlation, inputs, dtype=inputs[0].dtype)

@gin.configurable
class SiamQNetwork(network.Network):
  """
  tf_agents Network wrapper of the PseudoSiamFCN model

  A Q Network that takes two inputs and matches them in a fully
    convolutional (pseudo*)siamese architecture[1] to compute a
    map of Q values.
  (* branch weights are not shared)
  """

  def __init__(
    self,
    input_tensor_spec,
    action_spec,
    batch_size=None,
    left_layers=DEFAULT_BRANCH_PARAMS,
    right_layers=None,
    pseudo=True,
    pos_layers=DEFAULT_POS_PARAMS,
    seed=None,
    name='SiamQNetwork'
  ):
    """
    Args:
      input_tensor_spec: See super,
      action_spec: See super,
      left_layers: list of layer constructors for the layers
        on the left feature extractor.
      right_layers: list of layer constructors for the layers
        on the right feature extractor. If None, same as left
        is used.
      pseudo: whether the branches have each its own layers.
        If false, layers (and weights) are shared (true siamese
        network).
      pos_params: list of dictionaries, each giving the 
        parameters for a keras layer after the correlation
      name: name of the model
      **kwargs: passed 
    Raises:
      AssertionError: if input_tensor_spec doesn't match the input 
        requirements of the network.
    """
    dtype = _validate_input_tensor_spec(input_tensor_spec)
    
    super(SiamQNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), 
        name=name)

    if dtype == tf.uint8:
      self._scale = tf.constant(2**8-1, dtype=tf.uint8)
    elif dtype == tf.uint16:
      self._scale = tf.constant(2**16-1, dtype=tf.uint16)
    elif dtype == tf.uint32:
      self._scale = tf.constant(2**32-1, dtype=tf.uint32)
    elif dtype == tf.uint64:
      self._scale = tf.constant(2**64-1, dtype=tf.uint64)
    else:
      self._scale = None

    if right_layers is None:
      right_layers = left_layers

    if seed is not None:
      tf.random.set_seed(seed)

    self.left = []
    for layer in left_layers:
      self.left.append(layer())

    if pseudo:
      self.right = []
      for layer in right_layers:
        self.right.append(layer())
    else:
      self.right = self.left

    self.correlation = Correlation()

    self.pos = []
    for layer in pos_layers:
      self.pos.append(layer())

    # Build
    if batch_size is None:
      batch_size = 1
    self.__call__(tensor_spec.sample_spec_nest(
        self.input_tensor_spec, outer_dims=(batch_size,)))


  def call(
    self, 
    observation, 
    step_type=None, 
    network_state=(), 
    training=False
  ):
    if self._scale is None:
      x = observation[0]
      w = observation[1]
    else:
      x = tf.cast(observation[0]/self._scale, dtype=tf.float32)
      w = tf.cast(observation[1]/self._scale, dtype=tf.float32)

    for layer in self.left:
      x = layer(x)
    for layer in self.right:
      w = layer(w)
    value = self.correlation([x,w])
    for layer in self.pos:
      value = layer(value)

    return value, network_state
    

