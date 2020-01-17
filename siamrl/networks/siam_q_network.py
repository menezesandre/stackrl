"""
References:
  [1](https://arxiv.org/abs/1606.09549)
"""

import tensorflow as tf
from tensorflow.keras import layers #import Lambda, Conv2D, Flatten
import tf_agents
from tf_agents.networks import network

import siamrl
from siamrl.networks import dirnet

def validate_input_shape(input_shape):
  assert len(input_shape) == 2
  assert len(input_shape[0]) == 3
  assert len(input_shape[1]) == 3
  assert input_shape[0][0] >= input_shape[1][0]
  assert input_shape[0][1] >= input_shape[1][1]
  assert input_shape[0][2] == input_shape[1][2]

def pseudo_siam_fcn(input_shape, 
                    branch_layers=dirnet.basic_layers, 
                    out_conv=True, 
                    flatten=True, 
                    dtype=tf.float32, 
                    name=None):
  """
  Instantiates a tf.keras Model of the PseudoSiamFCN

  Args:
    input_shape: tuple defining input shape
      (hight, width, channels).
    branch_layers: function that defines the feature extracting 
      layers for the branches of the network. Should receive an
      input tensor and an optional name and return an output 
      tensor.
    out_conv: whether to perform a 3x3 convolution on the 
      correlation map.
    flatten: whether the net's output is to be flattened.
    dtype: data type of the input
    name: name of the model
  
  Returns:
    tf.keras Model instance of PseudoSiamFCN

  Raises:
    AssertionError: if input_shape is not valid (see 
    validate_input_shape)
  """
  validate_input_shape(input_shape)

  in1 = layers.Input(input_shape[0], dtype=dtype)
  in2 = layers.Input(input_shape[1], dtype=dtype)

  x = branch_layers(in1, name='left')
  w = branch_layers(in2, name='right')

  def correlation(inputs):
    x = tf.nn.conv2d(tf.expand_dims(inputs[0], 0), tf.expand_dims(inputs[1], -1), 
      strides=1, padding='VALID')
    return tf.squeeze(x, axis=0)

  out = layers.Lambda(lambda inputs: tf.map_fn(correlation, 
      inputs, dtype=dtype), name='correlation')((x, w))

  if out_conv:
    out = layers.Conv2D(1,3, padding='same', name='out_conv')(out)
  if flatten:
    out = layers.Flatten(name='flatten')(out)

  return tf.keras.Model(inputs=(in1, in2), outputs=out, name=name)


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
      name: name of the model
      **kwargs: passed 

    Raises:
      TypeError: if input_tensor_spec is not a list;
      IndexError: if it doesn't have at least two elements.
    """
    net = pseudo_siam_fcn([input_tensor_spec[0].shape, 
        input_tensor_spec[1].shape], dtype=input_tensor_spec[0].dtype,
        name='PseudoSiamFCN', **kwargs)
    super(SiamQNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), 
        name=name)

    self.net = net
    self.built = self.net.built


  def call(self, observation, step_type=None, network_state=(),
      training=False):
    return self.net(observation, training=training), network_state
    

    
