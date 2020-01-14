import tensorflow as tf
from tensorflow.keras import Model, Input, backend
from tensorflow.keras.layers import Conv2D, SeparableConv2D,  DepthwiseConv2D, BatchNormalization
from tensorflow.keras.layers import Concatenate, Add, Multiply, Reshape
from tensorflow.keras.layers import UpSampling2D, Activation, GlobalAveragePooling2D

def _name(name):
  def f(str):
    return None if name is None else name+'_'+str
  return f

def inverted_residual_block(inputs, filters, rate=1, 
    expand_ratio=1, se_ratio=0.25, activation=tf.nn.relu6, 
    use_bn=False, name=None):
  _n = _name(name)

  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
  in_channels = inputs.shape[channel_axis]
  filters_exp = expand_ratio*in_channels

  #Expand
  if expand_ratio != 1:
    x = Conv2D(filters_exp, 1, padding='same', 
        use_bias=not use_bn, name=_n('expand'))(inputs)
    if use_bn:
      x = BatchNormalization(axis=channel_axis, 
          name=_n('expand_bn'))(x)
    x = Activation(activation, name=_n('expand_act'))(x)
  else:
    x = inputs

  #Depthwise
  x = DepthwiseConv2D(3, use_bias=not use_bn, padding='same',
      dilation_rate=rate, name=_n('depthwise'))(x)
  if use_bn:
    x = BatchNormalization(axis=channel_axis, 
        name=_n('depthwise_bn'))(x)
  x = Activation(activation, name=_n('depthwise_act'))(x)

  #Squeeze and excitation
  if 0 < se_ratio <=1:
    filters_se = max(1, int(in_channels * se_ratio))
    se = GlobalAveragePooling2D(name=_n('se_squeeze'))(x)
    se = Reshape((1, 1, filters_exp), name=_n('se_reshape'))(se)
    se = Conv2D(filters_se, 1, padding='same', 
        activation=activation, name=_n('se_reduce'))(se)
    se = Conv2D(filters_exp, 1, padding='same', 
        activation='sigmoid', name=name + 'se_expand')(se)
    x = Multiply(name=_n('se_excite'))([x, se])

  #Project
  x = Conv2D(filters, 1, padding='same', use_bias=not use_bn,
      name=_n('project_conv'))(x)
  if use_bn:
    x = BatchNormalization(axis=channel_axis, name=_n('project_bn'))(x)

  if (filters==in_channels):
    x = Add(name=_n('add'))([x, inputs])

  return x    

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