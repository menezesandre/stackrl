import tensorflow as tf
from tensorflow.keras.layers import Lambda, Conv2D, Flatten
import tf_agents
from tf_agents.networks import network

import siamrl
from siamrl.networks.dirnet import dirnet

class SiamQNetwork(network.Network):

  def __init__(self, input_tensor_spec, action_spec, 
      model=dirnet, out_conv=True, name='SiamQNetwork', **kwargs):
    assert len(input_tensor_spec) == 2
    super(SiamQNetwork, self).__init__(input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    self.left = model(input_tensor_spec[0].shape, name='left')
    self.right = model(input_tensor_spec[1].shape, name='right')

    @tf.function
    def correlation(inputs):
      x = tf.nn.conv2d(tf.expand_dims(inputs[0], 0), tf.expand_dims(inputs[1], -1), 
        strides=1, padding='VALID')
      return tf.squeeze(x, axis=0)

    self.correlation = Lambda(lambda inputs: tf.map_fn(correlation, inputs), name='correlation')
    if out_conv:
      self.out_conv = Conv2D(1,3, padding='same', name='out_conv')
    self.flatten = Flatten()


  def call(self, observation, step_type=None, network_state=(),
      training=False):
    x = self.left(observation[0])
    w = self.right(observation[1])
    q = self.correlation((x, w))
    if out_conv:
      q = self.out_conv(q)
    q = self.flatten(q)

    return q, network_state
    

    
