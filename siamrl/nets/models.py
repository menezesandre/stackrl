"""
References:
  [1](https://arxiv.org/abs/1606.09549)
"""
import gin
import tensorflow as tf
from siamrl.nets import layers


@gin.configurable(module='siamrl.nets')
class PseudoSiamFCN(tf.keras.Model):
  def __init__(
    self,
    input_spec,
    left_layers=layers.default_branch_layers,
    right_layers=None,
    pos_layers=layers.default_pos_layers,
    seed=None,
    name='PseudoSiamFCN'
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
    # Set initializer according to seed  
    if seed is not None:
      initializer = tf.keras.initializers.he_uniform(seed=seed)
    else:
      initializer = 'he_uniform'
    # Aply the layers before correlation
    right_layers = right_layers or left_layers
    x = tf.nest.map_structure(
      lambda i,l,n: l(
        i, 
        kernel_initializer=initializer,
        name_scope=n,
      ),
      x,
      (left_layers, right_layers),
      ('Left', 'Right'),
    )

    outputs = layers.correlation(*x)
    outputs = pos_layers(
      outputs, 
      kernel_initializer=initializer,
    )
    
    super(PseudoSiamFCN, self).__init__(
      inputs=inputs, 
      outputs=outputs, 
      name=name
    )