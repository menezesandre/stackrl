import collections
import inspect
import random

import gin
import tensorflow as tf
from tensorflow import keras as k

@gin.configurable(module='siamrl.nets')
def correlation(in0, in1, parallel_iterations=None):
  """Aplies the correlation layer to inputs tensors"""

  def fn(inputs):
    x = tf.nn.conv2d(tf.expand_dims(inputs[0], 0), tf.expand_dims(inputs[1], -1), 
      strides=1, padding='VALID')
    return tf.squeeze(x, axis=0)
  
  return k.layers.Lambda(lambda inputs: tf.map_fn(
    fn=fn, 
    elems=inputs, 
    parallel_iterations=parallel_iterations,
    fn_output_signature=in0.dtype,
  )
  )((in0,in1))

@gin.configurable(module='siamrl.nets')
def sequential(
  inputs, 
  layers, 
  kernel_initializer='he_uniform',
  dtype=None,
  name=None,
  seed=None,
):
  """Aplies a sequence of layers
  Args:
    inputs: input tensor(s).
    layers: sequence of layers to be aplied. List of tuples with layer
      constructor and dictionary with kwargs for the constructor.
    kernel_initializer: String identifier of the kernel initializer to
      be used in all layers that have kernels. Any 'kernel_initializer'
      argument provided in the layers' kwargs is not overwriten by this
      argument. If None, no argument is passed to the layers.
      which it aplies. Either a string identifier (in which case the 
      corresponding initializer is intantiated for each layer) or an
      instance of a keras Initializer to be used in all layers. (Use the
      later with a defined seed, in combination with setting the global 
      seed, to get reproducible results.)
    dtype: data type of the layers. If None, default layer's dtype is 
      used.
    name: layers are instantiated with names prefixed with this name
      scope. Note: if provided, layer's names are made unique within 
      this function, but not globaly (i.e. collisions may happen if 
      this scope is used elsewhere).
    seed: seed of the random sequence of integers that serve as seeds 
      for the kernel initializers.
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
    # TODO relax requirement of being a keras Layer
    if isinstance(l, tuple):
      if len(l) >= 2 and issubclass(l[0], k.layers.Layer) and isinstance(l[1], dict):
        continue
      elif len(l) == 1 and issubclass(l[0], k.layers.Layer):
        layers[i] = (l[0], {})
        continue
    elif issubclass(l, k.layers.Layer):
      layers[i] = (l, {})
      continue
    raise ValueError(
      "Invalid value {} for argument layers at index {}.".format(
        l,
        i
      )
    )

  name_scope = name
  if name_scope:
    name_counts = collections.defaultdict(lambda: 0)
  
  if seed is not None and kernel_initializer:
    _random = random.Random(seed)
    seed = lambda: _random.randint(0,2**32-1)
    kernel_initializer = k.initializers.get(kernel_initializer)

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

    if kernel_initializer:
      args,_,_,_ = inspect.getargspec(layer)
      for kw in ['kernel_initializer','depthwise_initializer','pointwise_initializer']:
        if kw in args and kw not in kwargs:
          if seed:
            config = kernel_initializer.get_config()
            config['seed'] = seed()
            kernel_initializer = kernel_initializer.from_config(config)
          kwargs[kw] = kernel_initializer

    x = layer(
      **kwargs, 
      dtype=dtype,
      name=name,
    )(x)
  return x

@gin.configurable(module='siamrl.nets')
def unet(
  inputs, 
  depth=3,
  filters=64,
  out_channels=None,
  kernel_initializer='he_uniform',
  dtype=None,
  name=None,
  seed=None,
):
  """Aplies a sequence of layers
  Args:
    inputs: input tensor(s).
    depth: number of levels.
    filters: number of filters of the first level (each level doubles
      the number of filters of the previous one).
    out_channels: number of channels in the output.
    kernel_initializer: String identifier of the kernel initializer to
      be used in all layers that have kernels. Any 'kernel_initializer'
      argument provided in the layers' kwargs is not overwriten by this
      argument. If None, no argument is passed to the layers.
      which it aplies. Either a string identifier (in which case the 
      corresponding initializer is intantiated for each layer) or an
      instance of a keras Initializer to be used in all layers. (Use the
      later with a defined seed, in combination with setting the global 
      seed, to get reproducible results.)
    dtype: data type of the layers. If None, default layer's dtype is 
      used.
    name: layers are instantiated with names prefixed with this name
      scope. Note: if provided, layer's names are made unique within 
      this function, but not globaly (i.e. collisions may happen if 
      this scope is used elsewhere).
    seed: seed of the random sequence of integers that serve as seeds 
      for the kernel initializers.
  Returns:
    The output tensor
  """

  name_scope = name
  
  if seed is not None and kernel_initializer:
    _random = random.Random(seed)
    seed = lambda: _random.randint(0,2**32-1)
    kernel_initializer = k.initializers.get(kernel_initializer)

  x = inputs
  levels = []
  for i in range(depth):
    for j in range(2):
      name = 'convdw{}{}'.format(i,j)
      if name_scope:
        name = '{}/{}'.format(name_scope, name)

      if seed:
        config = kernel_initializer.get_config()
        config['seed'] = seed()
        kernel_initializer = kernel_initializer.from_config(config)

      x = k.layers.Conv2D(
        filters=filters*2**i, 
        kernel_size=3, 
        padding='same', 
        kernel_initializer=kernel_initializer,
        name=name,
      )(x)
    
    levels.append(x)
    
    name = 'down{}'.format(i)
    if name_scope:
      name = '{}/{}'.format(name_scope, name)
    x = k.layers.MaxPool2D(name=name)(x)

  for i in range(2):
    name = 'conv{}{}'.format(depth,i)
    if name_scope:
      name = '{}/{}'.format(name_scope, name)
    if seed:
      config = kernel_initializer.get_config()
      config['seed'] = seed()
      kernel_initializer = kernel_initializer.from_config(config)

    x = k.layers.Conv2D(
      filters=filters*2**depth,
      kernel_size=3, 
      padding='same', 
      kernel_initializer=kernel_initializer,
      name=name,
    )(x)

  for i in range(depth-1, -1, -1):
    name = 'up{}'.format(i)
    if name_scope:
      name = '{}/{}'.format(name_scope, name)
    if seed:
      config = kernel_initializer.get_config()
      config['seed'] = seed()
      kernel_initializer = kernel_initializer.from_config(config)
    x = k.layers.Conv2DTranspose(
      filters=filters*2**i,
      kernel_size=3,
      strides=2,
      padding='same',
      kernel_initializer=kernel_initializer,
      name=name,
    )(x)

    name = 'concat{}'.format(i)
    if name_scope:
      name = '{}/{}'.format(name_scope, name)

    x = k.layers.Concatenate(name=name)([x, levels.pop()])

    for j in range(2):
      name = 'convuw{}{}'.format(i,j)
      if name_scope:
        name = '{}/{}'.format(name_scope, name)
      if seed:
        config = kernel_initializer.get_config()
        config['seed'] = seed()
        kernel_initializer = kernel_initializer.from_config(config)

      x = k.layers.Conv2D(
        filters=filters*2**i, 
        kernel_size=3, 
        padding='same', 
        kernel_initializer=kernel_initializer,
        name=name,
      )(x)

  name = 'convout'
  if name_scope:
    name = '{}/{}'.format(name_scope, name)
  if seed:
    config = kernel_initializer.get_config()
    config['seed'] = seed()
    kernel_initializer = kernel_initializer.from_config(config)
  
  if out_channels is not None:
    x = k.layers.Conv2D(
      filters=out_channels,
      kernel_size=1, 
      kernel_initializer=kernel_initializer,
      name=name,
    )(x)
  
  return x

@gin.configurable(module='siamrl.nets')
def mobile_unet(
  inputs, 
  depth=3,
  filters=64,
  out_channels=None,
  kernel_initializer='he_uniform',
  dtype=None,
  name=None,
  seed=None,
):
  """Aplies a sequence of layers
  Args:
    inputs: input tensor(s).
    depth: number of levels.
    filters: number of filters of the first level (each level doubles
      the number of filters of the previous one).
    out_channels: number of channels in the output.
    kernel_initializer: String identifier of the kernel initializer to
      be used in all layers that have kernels. Any 'kernel_initializer'
      argument provided in the layers' kwargs is not overwriten by this
      argument. If None, no argument is passed to the layers.
      which it aplies. Either a string identifier (in which case the 
      corresponding initializer is intantiated for each layer) or an
      instance of a keras Initializer to be used in all layers. (Use the
      later with a defined seed, in combination with setting the global 
      seed, to get reproducible results.)
    dtype: data type of the layers. If None, default layer's dtype is 
      used.
    name: layers are instantiated with names prefixed with this name
      scope. Note: if provided, layer's names are made unique within 
      this function, but not globaly (i.e. collisions may happen if 
      this scope is used elsewhere).
    seed: seed of the random sequence of integers that serve as seeds 
      for the kernel initializers.
  Returns:
    The output tensor
  """

  name_scope = name
  
  if seed is not None and kernel_initializer:
    _random = random.Random(seed)
    seed = lambda: _random.randint(0,2**32-1)
    kernel_initializer = k.initializers.get(kernel_initializer)

  name = 'convin'
  if name_scope:
    name = '{}/{}'.format(name_scope, name)
  if seed:
    config = kernel_initializer.get_config()
    config['seed'] = seed()
    kernel_initializer = kernel_initializer.from_config(config)

  x = inputs
  levels = []

  name = 'convdw00'
  if name_scope:
    name = '{}/{}'.format(name_scope, name)
  if seed:
    config = kernel_initializer.get_config()
    config['seed'] = seed()
    kernel_initializer = kernel_initializer.from_config(config)
  x = k.layers.Conv2D(
    filters=filters//2,
    kernel_size=3, 
    padding='same', 
    kernel_initializer=kernel_initializer,
    name=name,
  )(x)
  name = 'convdw01'
  if name_scope:
    name = '{}/{}'.format(name_scope, name)
  if seed:
    config = kernel_initializer.get_config()
    config['seed'] = seed()
    kernel_initializer = kernel_initializer.from_config(config)
  x = k.layers.SeparableConv2D(
    filters=filters, 
    kernel_size=3, 
    padding='same', 
    depthwise_initializer=kernel_initializer,
    pointwise_initializer=kernel_initializer,
    name=name,
  )(x)


  for i in range(1, depth+1):
    levels.append(x)

    name = 'convdw{}0'.format(i)
    if name_scope:
      name = '{}/{}'.format(name_scope, name)
    if seed:
      config = kernel_initializer.get_config()
      config['seed'] = seed()
      kernel_initializer = kernel_initializer.from_config(config)
    x = k.layers.SeparableConv2D(
      filters=filters*2**i, 
      kernel_size=3, 
      strides=2,
      padding='same', 
      depthwise_initializer=kernel_initializer,
      pointwise_initializer=kernel_initializer,
      name=name,
    )(x)

    name = 'convdw{}1'.format(i)
    if name_scope:
      name = '{}/{}'.format(name_scope, name)
    if seed:
      config = kernel_initializer.get_config()
      config['seed'] = seed()
      kernel_initializer = kernel_initializer.from_config(config)
    x = k.layers.SeparableConv2D(
      filters=filters*2**i, 
      kernel_size=3, 
      padding='same', 
      depthwise_initializer=kernel_initializer,
      pointwise_initializer=kernel_initializer,
      name=name,
    )(x)
    
  for i in range(depth-1, -1, -1):
    name = 'up{}0'.format(i)
    if name_scope:
      name = '{}/{}'.format(name_scope, name)

    x = k.layers.UpSampling2D(interpolation='bilinear')(x)

    name = 'up{}1'.format(i)
    if name_scope:
      name = '{}/{}'.format(name_scope, name)
    if seed:
      config = kernel_initializer.get_config()
      config['seed'] = seed()
      kernel_initializer = kernel_initializer.from_config(config)
    x = k.layers.SeparableConv2D(
      filters=filters*2**i,
      kernel_size=3,
      padding='same',
      depthwise_initializer=kernel_initializer,
      pointwise_initializer=kernel_initializer,
      name=name,
    )(x)

    name = 'concat{}'.format(i)
    if name_scope:
      name = '{}/{}'.format(name_scope, name)

    x = k.layers.Concatenate(name=name)([x, levels.pop()])

    for j in range(2):
      name = 'convuw{}{}'.format(i,j)
      if name_scope:
        name = '{}/{}'.format(name_scope, name)
      if seed:
        config = kernel_initializer.get_config()
        config['seed'] = seed()
        kernel_initializer = kernel_initializer.from_config(config)

      x = k.layers.SeparableConv2D(
        filters=filters*2**i, 
        kernel_size=3, 
        padding='same', 
        depthwise_initializer=kernel_initializer,
        pointwise_initializer=kernel_initializer,
        name=name,
      )(x)

  name = 'convout'
  if name_scope:
    name = '{}/{}'.format(name_scope, name)
  if seed:
    config = kernel_initializer.get_config()
    config['seed'] = seed()
    kernel_initializer = kernel_initializer.from_config(config)

  if out_channels:
    x = k.layers.Conv2D(
      filters=out_channels,
      kernel_size=1, 
      kernel_initializer=kernel_initializer,
      name=name,
    )(x)
  
  return x

def default_branch_layers(
  inputs, 
  kernel_initializer='he_uniform', 
  **kwargs
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
    **kwargs,
  )

def default_pos_layers(
  inputs, 
  kernel_initializer='he_uniform',
  **kwargs,
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
        {'filters':32, 'kernel_size':3, 'activation':'relu', 
        'padding':'same'}),
      (k.layers.SeparableConv2D, 
        {'filters':32, 'kernel_size':3, 'activation':'relu', 
        'padding':'same'}),
      (k.layers.SeparableConv2D, 
        {'filters':1, 'kernel_size':3, 
        'padding':'same'}),
      (k.layers.Flatten, {})
    ],
    kernel_initializer=kernel_initializer,
    **kwargs
  )