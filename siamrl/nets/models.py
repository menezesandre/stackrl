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
      lambda i,l: l(i, kernel_initializer=initializer),
      x,
      (left_layers, right_layers)
    )

    outputs = layers.correlation(*x)
    outputs = pos_layers(outputs, kernel_initializer=initializer)
    
    super(PseudoSiamFCN, self).__init__(
      inputs=inputs, 
      outputs=outputs, 
      name=name
    )

#------------------
uints = {
  tf.uint8: 2**8-1, 
  tf.uint16: 2**16-1, 
  tf.uint32: 2**32-1, 
  tf.uint64: 2**64-1
}
dtypes = list(uints.keys()) + [tf.float16, tf.float32, tf.float64]

def _assert_input_shape(input_shape):
  """
  Args:
    input_shape: input shape to be validated.
  Raises:
    AssertionError: if input_shape doesn't match what's expected by the 
      siamese fully convolutional netwok.
  """
  try:
    assert len(input_shape) == 2, \
      "Argument input_shape must have length 2."
  except TypeError:
    raise AssertionError("Argument input_shape must have length.")
  assert len(input_shape[0]) == 3, \
    "First input must have 3 dimensions [height,width,channels]."
  assert len(input_shape[1]) == 3, \
    "Second input must have 3 dimensions [height,width,channels]."
  assert input_shape[0][0] >= input_shape[1][0], \
    "First input must have larger height."
  assert input_shape[0][1] >= input_shape[1][1], \
    "First input must have larger width."
  return (tf.TensorShape(input_shape[0]), tf.TensorShape(input_shape[1]))

def pseudo_siam_fcn(
  input_shape, 
  input_dtype=None,
  left_layers=layers.default_branch_layers,
  right_layers=None,
  pos_layers=layers.default_pos_layers,
  seed=None,
  name='PseudoSiamFCN'
):
  """
  Args:
    input_shape: tuple with shapes of the two inputs (height,width,channels).
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
  # Check if input shape matches the expected
  input_shape = _assert_input_shape(input_shape)

  # Set initializer according to seed  
  if seed is not None:
    initializer = tf.keras.initializers.he_uniform(seed=seed)
  else:
    initializer = 'he_uniform'

  # Set data type
  if input_dtype:
    input_dtype = tf.nest.map_structure(
      lambda i: tf.dtypes.as_dtype(i), 
      input_dtype
    )
    for i in tf.nest.flatten(input_dtype):
      if i not in dtypes:
        raise TypeError(
          "Invalid input dtype {}. Must be in {}".format(
            i, 
            dtypes
          )
        )
  else:
    input_dtype = tf.float32

  if tf.nest.is_nested(input_dtype):
    in0 = tf.keras.Input(input_shape[0], dtype=input_dtype[0])
    in1 = tf.keras.Input(input_shape[1], dtype=input_dtype[1])
    if input_dtype[0] in uints:
      scale = tf.constant(uints[input_dtype[0]], dtype=input_dtype[0])
      x = in0/scale
    else:
      x = in0
    if input_dtype[1] in uints:
      scale = tf.constant(uints[input_dtype[1]], dtype=input_dtype[1])
      w = in1/scale
    else:
      w = in1

  else:
    in0 = tf.keras.Input(input_shape[0], dtype=input_dtype)
    in1 = tf.keras.Input(input_shape[1], dtype=input_dtype)
    if input_dtype in uints:
      scale = tf.constant(uints[input_dtype], dtype=input_dtype)
      x = in0/scale
      w = in1/scale
    else:
      x = in0
      w = in1

  right_layers = right_layers or left_layers

  x = left_layers(x, kernel_initializer=initializer)
  w = right_layers(w, kernel_initializer=initializer)
  out = layers.correlation(x,w)
  out = pos_layers(out, kernel_initializer=initializer)
  
  return tf.keras.Model(
    inputs=(in0,in1), 
    outputs=out, 
    name=name
  )
