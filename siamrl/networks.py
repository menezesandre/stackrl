"""
References:
  [1](https://arxiv.org/abs/1606.09549)
"""

import tensorflow as tf
from tensorflow.keras import layers #import Lambda, Conv2D, Flatten
import tf_agents
from tf_agents.networks import network
from tf_agents.specs import tensor_spec

import siamrl

def validate_input_tensor_spec(input_tensor_spec):
  assert len(input_tensor_spec) == 2
  assert len(input_tensor_spec[0].shape) == 3
  assert len(input_tensor_spec[1].shape) == 3
  assert input_tensor_spec[0].shape[0] >= input_tensor_spec[1].shape[0]
  assert input_tensor_spec[0].shape[1] >= input_tensor_spec[1].shape[1]
 
class Correlation(layers.Layer):
  def __init__(self):
    super(Correlation, self).__init__()

  @staticmethod
  def _correlation(inputs):
    x = tf.nn.conv2d(tf.expand_dims(inputs[0], 0), tf.expand_dims(inputs[1], -1), strides=1, padding='VALID')
    return tf.squeeze(x, axis=0)
    
  def call(self, inputs):
    return tf.map_fn(self._correlation, inputs, dtype=inputs[0].dtype)


class SiamQNetwork(network.Network):
  """
  tf_agents Network wrapper of the PseudoSiamFCN model

  A Q Network that takes two inputs and matches them in a fully
    convolutional (pseudo*)siamese architecture[1] to compute a
    map of Q values.
  (* branch weights are not shared)
  """

  def __init__(self,
               input_tensor_spec,
               action_spec,
               name='SiamQNetwork',
               **kwargs):
    """
    Args:
      input_tensor_spec: See super
      action_spec: See super
      left_params: list of dictionaries, each giving the 
        parameters for a keras Conv2D layer on the left 
        feature extractor.
      right_params: list of dictionaries, each giving the 
        parameters for a keras Conv2D layer on the right 
        feature extractor. If None, same as left is used.
      pseudo: whether the branches have each its own layers.
        If false, layers (and weights) are shared (true siamese
        network).
      name: name of the model
      **kwargs: passed 

    Raises:
      TypeError: if input_tensor_spec is not a list;
      IndexError: if it doesn't have at least two elements.
    """
    validate_input_tensor_spec(input_tensor_spec)
    super(SiamQNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), 
        name=name)
    self.left = []
    self.left.append(layers.Conv2D(32,8,strides=4, activation='relu', padding='same'))
    self.left.append(layers.Conv2D(64,4,dilation_rate=2, activation='relu', padding='same'))
    self.left.append(layers.UpSampling2D(size=4))

    self.right = []
    self.right.append(layers.Conv2D(32,8,strides=4, activation='relu', padding='same'))
    self.right.append(layers.Conv2D(64,4,dilation_rate=2, activation='relu', padding='same'))
    self.right.append(layers.UpSampling2D(size=4))

    self.correlation = Correlation()

    self.pos = []
    self.pos.append(layers.Conv2D(8,3, activation='relu', padding='same'))
    self.pos.append(layers.Conv2D(1,1))
    self.pos.append(layers.Flatten())


    self.__call__(tensor_spec.sample_spec_nest(
        self.input_tensor_spec, outer_dims=(1,)))


  def call(self, observation, step_type=None, network_state=(),
      training=False):
    x = observation[0]
    w = observation[1]
    for layer in self.left:
      x = layer(x)
    for layer in self.right:
      w = layer(w)
    value = self.correlation([x,w])
    for layer in self.pos:
      value = layer(value)

    return value, network_state

  def create_variables(self, **kwargs):
    if not self.built:
      random_input = tensor_spec.sample_spec_nest(
          self.input_tensor_spec, outer_dims=(0,))
      random_state = tensor_spec.sample_spec_nest(
          self.state_spec, outer_dims=(0,))
      step_type = tf.zeros([time_step.StepType.FIRST], dtype=tf.int32)
      self.__call__(
          random_input, step_type=step_type, network_state=random_state,
          **kwargs)

    

