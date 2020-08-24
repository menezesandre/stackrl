"""
References:
  [1](https://arxiv.org/abs/1606.09549)
"""
import random

import gin
import tensorflow as tf
from siamrl.nets import layers


@gin.configurable(module='siamrl.nets')
class PseudoSiamFCN(tf.keras.Model):
  def __init__(
    self,
    input_spec,
    left_layers=layers.unet,
    right_layers=None,
    pos_layers=layers.pos_layers,
    inspect=False,
    seed=None,
    name='PseudoSiamFCN',
  ):
    """
    Args:
      input_spec: tuple of TensorSpecs of the two inputs. Can also be a
        tuple of TensorShapes or a tuple of collections with 
        (height,width,channels). In the later cases, input data type is 
        assumed to be float32. A mix of these cases is also supported.
      input_dtype: data type of the expected input (layers' data type is 
        float32 regardless).
      left_layers: callable that aplies the left feature extractor layers 
        to an input tensor.
      right_layers: callable that aplies the right feature extractor layers 
        to an input tensor. If None, same as left is used.
      pos_layers: callable that aplies the layers after the correlation to 
        an input tensor.
      seed: seed of the kernel initializer used in the model layers.
      name: name of the model
    Raises:
      AssertionError: if input_shape doesn't match the expected input 
        of the network.
      TypeError: if input_dtype is invalid.
    """
    # Set input spec
    try:
      assert len(input_spec) == 2, \
        "Argument input_spec must have two elements."
    except TypeError:
      raise TypeError(
        "Invalid type {} for argument input_spec, must have len()".format(
          type(input_spec)
        )
      )
    input_spec = tuple(
      i if isinstance(i, tf.TensorSpec) else tf.TensorSpec(shape=i) \
        for i in input_spec
    )
    # Set input
    inputs = tf.nest.map_structure(
      lambda i: tf.keras.Input(i.shape.with_rank(3), dtype=i.dtype),
      input_spec
    )
    x = tf.nest.map_structure(
      lambda i: i if i.dtype.is_floating else i/i.dtype.max,
      inputs
    )
    # Set seed
    if seed is not None:
      _random = random.Random(seed)
      seed = lambda: _random.randint(0,2**32-1)
    else:
      seed = lambda: None
    
    # Aply the layers before correlation
    right_layers = right_layers or left_layers
    x = tf.nest.map_structure(
      lambda i,l,n,s: l(
        i, 
        name=n,
        seed=s
      ),
      x,
      (left_layers, right_layers),
      ('Left', 'Right'),
      (seed(), seed())
    )

    corr = layers.correlation(*x)
    values = pos_layers(
      corr, 
      seed=seed(),
    )
    outputs = tf.keras.layers.Flatten()(values)

    if inspect:
      outputs = (outputs, *x, corr, values)

    super(PseudoSiamFCN, self).__init__(
      inputs=inputs, 
      outputs=outputs, 
      name=name
    )

@gin.configurable(module='siamrl.nets')
class DeepQSiamFCN(tf.keras.Model):
  def __init__(
    self,
    input_spec,
    left_filters=32,
    left_depth=4,
    right_filters=None,
    right_depth=None,
    corr_channels=None,
    pos_filters=32,
    pos_depth=2,
    dueling=False,
    dueling_avg_pool=True,
    dueling_units=512,
    inspect=False,
    seed=None,
    name='DeepQSiamFCN',
  ):
    # Set input spec
    try:
      assert len(input_spec) == 2, \
        "Argument input_spec must have two elements."
    except TypeError:
      raise TypeError(
        "Invalid type {} for argument input_spec, must have len()".format(
          type(input_spec)
        )
      )
    input_spec = tuple(
      i if isinstance(i, tf.TensorSpec) else tf.TensorSpec(shape=i) \
        for i in input_spec
    )

    # Set input
    inputs = tf.nest.map_structure(
      lambda i: tf.keras.Input(i.shape.with_rank(3), dtype=i.dtype),
      input_spec
    )
    x,w = tf.nest.map_structure(
      lambda i: i if i.dtype.is_floating else i/i.dtype.max,
      inputs
    )
    # Set seed
    if seed is not None:
      _random = random.Random(seed)
      seed = lambda: _random.randint(0,2**32-1)
    else:
      seed = lambda: None

    right_filters = right_filters or left_filters
    right_depth = right_depth or max(1, left_depth-2)

    if right_filters != left_filters and corr_channels is None:
      corr_channels = min(left_filters, right_filters)

    x,x0 = layers.unet(
      x, 
      depth=left_depth, 
      filters=left_filters, 
      out_channels=corr_channels,
      double_endpoint=True,
      seed=seed(),
      name='Left',
    )
    w,w0 = layers.unet(
      w, 
      depth=right_depth, 
      filters=right_filters, 
      out_channels=corr_channels,
      double_endpoint=True,
      seed=seed(),
      name='Right',
    )
    if dueling:
      v = layers.value(x0, avg=dueling_avg_pool, units=dueling_units, seed=seed())

    corr = layers.correlation(x, w)
    values = layers.pos_layers(
      corr,
      filters=pos_filters,
      depth=pos_depth,
      seed=seed(),
    )

    outputs = tf.keras.layers.Flatten()(values)
    if dueling:
      outputs = outputs - tf.reduce_mean(outputs, axis=-1, keepdims=True) + v

    if inspect:
      outputs = (outputs, x, x0, w, w0, corr, values)

    super(DeepQSiamFCN, self).__init__(
      inputs=inputs, 
      outputs=outputs, 
      name=name
    )
