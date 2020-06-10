import inspect
import collections
import tensorflow as tf
from tensorflow import keras as k
import gin

def correlation(*inputs, dtype=None):
  """Aplies the correlation layer to inputs tensors"""
  assert len(inputs) == 2, "Correlation layer must have two input tensors."

  def fn(inputs):
    x = tf.nn.conv2d(tf.expand_dims(inputs[0], 0), tf.expand_dims(inputs[1], -1), 
      strides=1, padding='VALID')
    return tf.squeeze(x, axis=0)
  
  return k.layers.Lambda(lambda x: tf.map_fn(fn, x, dtype=inputs[0].dtype))(inputs)

@gin.configurable(module='siamrl.nets')
def sequential(
  inputs, 
  layers, 
  kernel_initializer='he_uniform', 
  dtype=None,
  name_scope=None,
):
  """Aplies a sequence of layers
  Args:
    inputs: input tensor(s).
    layers: sequence of layers to be aplied. List of tuples with layer
      constructor and dictionary with kwargs for the constructor.
    kernel_initializer: kernel initializer to be used in all layers to 
      which it aplies. Either a string identifier (in which case the 
      corresponding initializer is intantiated for each layer) or an
      instance of a keras Initializer to be used in all layers. (Use the
      later with a defined seed, in combination with setting the global 
      seed, to get reproducible results.)
    dtype: data type of the layers. If None, default layer's dtype is 
      used.
    name_scope: layers are instantiated with names prefixed with this 
      scope. Note: if provided, layer's names are made unique within 
      this function, but not globaly (i.e. collisions may happen if 
      this scope is used elsewhere).
  Returns:
    The output tensor
  Raises:
    TypeError: if layers is not iterable; if any of layers' keys is not a 
      layer constructor; if any of the kwargs doesn't match the expected 
      layer constructor input.
    ValueError: if any of layers' entries is invalid.
  """
  # Validate layers argument
  for i, l in enumerate(layers):
    # TODO() relax requirement of being a keras Layer
    if isinstance(l, tuple):
      if len(l) >= 2 and issubclass(l[0], k.layers.Layer) and isinstance(l[1], dict):
        continue
      elif len(l) == 1 and issubclass(l[0], k.layers.Layer):
        layers[i] = (l[0], {})
        continue
    else:
      if issubclass(l, k.layers.Layer):
        layers[i] = (l, {})
        continue
    raise ValueError(
      "Invalid value {} for argument layers at index {}.".format(
        l,
        i
      )
    )

  if name_scope:
    name_counts = collections.defaultdict(lambda: 0)

  x = inputs
  for layer,kwargs in layers:
    # Set layer name
    if 'name' in kwargs:
      name = kwargs.pop('name')
    else:
      name = None
    if name_scope:
      name = name or layer.__name__.lower()
      nid = name_counts[name]
      name_counts[name] += 1
      if nid > 0:
        name += '_{}'.format(nid)
      name = '{}/{}'.format(name_scope, name)

    args,_,_,_ = inspect.getargspec(layer)
    if 'kernel_initializer' in args and 'kernel_initializer' not in kwargs:
      x = layer(
        **kwargs, 
        kernel_initializer=kernel_initializer, 
        dtype=dtype,
        name=name,
      )(x)
    else:
      x = layer(
        **kwargs, 
        dtype=dtype,
        name=name,
      )(x)
  return x

def default_branch_layers(
  inputs, 
  kernel_initializer='he_uniform', 
  dtype=None,
  name_scope=None,
):
  """Aplies the default sequence of layers for the branches of the 
  PseudoSiamFCN
  Args:
    See sequential.
  Returns:
    See sequential.
  """
  return sequential(
    inputs=inputs, 
    layers=[
      (k.layers.Conv2D,
        {'filters':32, 'kernel_size':8, 'strides':4, 'activation':'relu', 
        'padding':'same'}),
      (k.layers.Conv2D,
        {'filters':64, 'kernel_size':4, 'dilation_rate':2, 
        'activation':'relu', 'padding':'same'}),
      (k.layers.Conv2D,
        {'filters':64, 'kernel_size':3, 'activation':'relu', 
        'padding':'same'}),
      (k.layers.UpSampling2D, {'size':4, 'interpolation':'bilinear'})
    ],
    kernel_initializer=kernel_initializer,
    dtype=dtype,
    name_scope=name_scope,
  )

def default_pos_layers(
  inputs, 
  kernel_initializer='he_uniform',
  dtype=tf.float32,
  name_scope=None,
):
  """Aplies the default sequence of layers after the correlation for the
  PseudoSiamFCN.
  Args:
    See sequential.
  Returns:
    See sequential.
  """
  return sequential(
    inputs=inputs, 
    layers=[
      (k.layers.Conv2D, 
        {'filters':160, 'kernel_size':13, 'activation':'relu', 
        'padding':'same'}),
      (k.layers.Conv2D,
        {'filters':1, 'kernel_size':1, 'padding':'same'}),
      (k.layers.Flatten, {})
    ],
    kernel_initializer=kernel_initializer,
    dtype=dtype,
    name_scope=name_scope,
  )