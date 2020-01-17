"""
References:
  [1](https://arxiv.org/abs/1801.04381)
  [2](https://arxiv.org/abs/1709.01507)    
  [3](https://arxiv.org/abs/1905.11946)
  [4](https://arxiv.org/abs/1606.00915)
  [5](https://arxiv.org/abs/1705.09914)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as kl, backend as K

def _name(name):
  """
  Creates the utility function for a given name
  """
  def f(str):
    """
    Utility function to add given prefix to a base name   
    Args:
      str: suffix to be appended to name
    Returns:
      a string with the suffix if the name is not None, 
        otherwise return None
    """
    return None if name is None else name+'_'+str
  return f

def inverted_residual_block(inputs, filters, rate=1, 
    expand_ratio=1, se_ratio=0., activation=tf.nn.relu6, 
    use_bn=False, kernel_initializer='he_uniform', name=None):
  """
  Inverted residual block [1] with optional 
  squeeze-and-excitation unit [2] as used in EfficientNet[3].
  Dilated (atrous) convolutions [4, 5] are used instead of 
  stride, to mantain spatial resolution.
  """
  _n = _name(name)

  channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
  in_channels = inputs.shape[channel_axis]
  filters_exp = expand_ratio*in_channels

  #Expand
  if expand_ratio != 1:
    x = kl.Conv2D(filters_exp, 1, padding='same', 
        use_bias=not use_bn, kernel_initializer=kernel_initializer,
        name=_n('expand'))(inputs)
    if use_bn:
      x = kl.BatchNormalization(axis=channel_axis, 
          name=_n('expand_bn'))(x)
    x = kl.Activation(activation, name=_n('expand_act'))(x)
  else:
    x = inputs

  #Depthwise
  x = kl.DepthwiseConv2D(3, use_bias=not use_bn, padding='same',
      dilation_rate=rate, kernel_initializer=kernel_initializer,
      name=_n('depthwise'))(x)
  if use_bn:
    x = kl.BatchNormalization(axis=channel_axis, 
        name=_n('depthwise_bn'))(x)
  x = kl.Activation(activation, name=_n('depthwise_act'))(x)

  #Squeeze and excitation
  if 0 < se_ratio <=1:
    filters_se = max(1, int(in_channels * se_ratio))
    se = kl.GlobalAveragePooling2D(name=_n('se_squeeze'))(x)
    se = kl.Reshape((1, 1, filters_exp), name=_n('se_reshape'))(se)
    se = kl.Conv2D(filters_se, 1, padding='same', 
        activation=activation, kernel_initializer=kernel_initializer,
        name=_n('se_reduce'))(se)
    se = kl.Conv2D(filters_exp, 1, padding='same', 
        activation='sigmoid', kernel_initializer=kernel_initializer,
        name=name + 'se_expand')(se)
    x = kl.Multiply(name=_n('se_excite'))([x, se])

  #Project
  x = kl.Conv2D(filters, 1, padding='same', 
      use_bias=not use_bn, kernel_initializer=kernel_initializer,
      name=_n('project_conv'))(x)
  if use_bn:
    x = kl.BatchNormalization(axis=channel_axis, 
        name=_n('project_bn'))(x)

  if (filters==in_channels):
    x = kl.Add(name=_n('add'))([x, inputs])

  return x    

def layers(inputs,
           width=64, 
           stem_depth = 1, 
           levels=4, 
           level_depth=2, 
           interpolation='nearest', 
           top_depth=2,
           kernel_initializer='he_uniform',
           name=None):
  """
  Stacks the layers of a dilated inverted residual net with 
  given parameters

  Args:
    inputs: input tensor.
    width: number of channels of the net's body.
    stem_depth: number of additional layers of the stem, that
      gradualy increase the number of channels to reach the 
      body's width.
    levels: number of levels of dilation (each level uses double
     the rate of the previous one).
    level_depth: number of blocks on each level.
    interpolation: upsampling method to recover the input resolution
    top_depth: number of layers after upsampling.
    kernel_initializer: Initializer used on the convolutional
      layers' kernels
    name: prefixed on layers' names

  Returns:
    output tensor

  """
  _n = _name(name)
  # Stem: increases the number of channels
  filters = int(width/2**stem_depth)
  x = kl.Conv2D(filters, 3, strides=2, padding='same', 
      activation=tf.nn.relu6, kernel_initializer=kernel_initializer,
      name=_n('stem%d'%0))(inputs)
  for i in range(1,stem_depth+1):
    filters *= 2
    x = kl.Conv2D(filters, 3, padding='same', 
        activation=tf.nn.relu6, kernel_initializer=kernel_initializer,
        name=_n('stem%d'%i))(x)

  # Body: inverted residual blocks with increasing dilation 
  # rate and expand ratio
  for i in range(levels):
    for j in range(level_depth):
      x = inverted_residual_block(x, filters, rate=2**i, 
          expand_ratio=2**i, kernel_initializer=kernel_initializer,
          name=_n('block%d'%(i*level_depth+j)))
  
  # Upsampling: recover input resolution
  x = kl.UpSampling2D(interpolation=interpolation, 
      name=_n('upsampling'))(x)
  # Top: refine the upsampled output and remove (possible)
  # gridding artifacts introduced by dilation [5]
  for i in range(top_depth):
    x = kl.SeparableConv2D(filters, 3, padding='same', 
        activation=tf.nn.relu6, kernel_initializer=kernel_initializer,
        name=_n('top%d'%i))(x)

  return x

def basic_layers(inputs, name=None):
  return layers(inputs, width=16, stem_depth=0, 
      levels=1, level_depth=2, interpolation='bilinear', 
      top_depth=0, name=name)

def model(input_shape, **kwargs):
  """
  Instantiates a DIRNet model

  Args:
    input_shape: tuple defining input shape
      (hight, width, channels).
    **kwargs: passed to layers
  Return:
    A tf.keras Model instance of DIRNet  
  """
  inputs = tf.keras.Input(input_shape)
  outputs = layers(inputs, **kwargs)
  return keras.Model(inputs=inputs, outputs=outputs)

  """
def dirnet(input_shape, levels=4, level_depth=2, 
    stem_filters=16, out_filters=None, name=None):
  _n = _name(name)
  inputs = Input(input_shape)

  x = Conv2D(stem_filters, 3, strides=2, padding='same', 
      activation=tf.nn.relu6, name=_n('stem'))(inputs)
  
  filters = stem_filters//2
  x = inverted_residual_block(x, filters, name=_n('block0'))

  for i in range(1, levels):
    filters = 2*filters
    x = inverted_residual_block(x, filters, expand_ratio=6,
        name=_n('block'+str(i)+'_0'))
    for j in range(level_depth):
      x = inverted_residual_block(x, filters, rate=2**i, 
          expand_ratio=6, name=_n('block'+str(i)+'_'+str(j+1)))
  
  x = UpSampling2D(interpolation='nearest', name=_n('upsampling'))(x)
  if out_filters is None:
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    out_filters = x.shape[channel_axis]
  x = SeparableConv2D(out_filters, 3, padding='same', name=_n('out'))(x)

  return Model(inputs=inputs, outputs=x)
  """