import collections
import inspect
import random

import gin
import tensorflow as tf
from tensorflow import keras as k

def kernel_initializer_generator(kernel_initializer=None, seed=None):
  """Generates a sequence of kernel initializers from seed. Uses he_normal by default."""
  r = random.Random(seed)
  kernel_initializer = k.initializers.get(
    kernel_initializer or 'he_normal'
  )
  config = kernel_initializer.get_config()
  while True:
    config['seed'] = r.randint(0,2**32-1)
    yield kernel_initializer.from_config(config)

@gin.configurable(module='siamrl.nets')
def correlation(in0, in1, parallel_iterations=None):
  """Aplies the correlation layer to inputs tensors"""
  
  return k.layers.Lambda(lambda inputs: tf.map_fn(
    fn=lambda inps: tf.squeeze(
      tf.nn.conv2d(
        tf.expand_dims(inps[0], 0), 
        tf.expand_dims(inps[1], -1), 
        strides=1, 
        padding='VALID'
      ),
      axis=0,
    ), 
    elems=inputs, 
    parallel_iterations=parallel_iterations,
    fn_output_signature=in0.dtype,
  )
  )((in0,in1))

@gin.configurable(module='siamrl.nets')
def sequential(
  inputs, 
  layers, 
  kernel_initializer=None,
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
      

  kernel_initializer = kernel_initializer_generator(
    kernel_initializer=kernel_initializer,
    seed=seed,
  )

  x = inputs
  for layer,kwargs in layers:
    # Set layer name
    name = kwargs.get('name', None)

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
          kwargs[kw] = next(kernel_initializer)

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
  upsampling_kernel_size=2,
  out_channels=None,
  out_activation=None,
  kernel_initializer=None,
  double_endpoint=False,
  dtype=None,
  name=None,
  seed=None,
):
  """Aplies the layers of U-Net architecture to inputs.
  Args:
    inputs: input tensor.
    depth: number of levels.
    filters: number of filters of the first level (each level doubles
      the number of filters of the previous one).
    upsampling_kernel_size: kernel size of the up convolutions in the
      expanding path. The up convolutions have stride 2, so ideally 
      this parameter should be even to avoid checkerboard artifacts.
    out_channels: number of channels in the output. If provided, a
      1x1 convolution with this number of filters is aplied to the 
      output. If None, number of output channels is the same as filters.
    kernel_initializer: String identifier of the kernel initializer to
      be used in all layers that have kernels. Any 'kernel_initializer'
      argument provided in the layers' kwargs is not overwriten by this
      argument. If None, no argument is passed to the layers.
      which it aplies. Either a string identifier (in which case the 
      corresponding initializer is intantiated for each layer) or an
      instance of a keras Initializer to be used in all layers. (Use the
      later with a defined seed, in combination with setting the global 
      seed, to get reproducible results.)
    double_endpoint: if True, the return is a tuple of tensors with the
      output of the U-Net and the and the output of the last bottom 
      layer (before the first up-conv).
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
  if name_scope:
    name = lambda n: '{}/{}'.format(name_scope, n)
  else:
    name = lambda n: n
  kernel_initializer = kernel_initializer_generator(
    kernel_initializer=kernel_initializer,
    seed=seed,
  )

  x = inputs
  levels = []
  for i in range(depth):
    for j in range(2):
      x = k.layers.Conv2D(
        filters=filters*2**i, 
        kernel_size=3, 
        padding='same',
        activation='relu',
        kernel_initializer=next(kernel_initializer),
        name=name('convdw{}{}'.format(i,j)),
      )(x)
    
    levels.append(x)
    
    x = k.layers.MaxPool2D(name=name('down{}'.format(i)))(x)

  for i in range(2):
    x = k.layers.Conv2D(
      filters=filters*2**depth,
      kernel_size=3, 
      padding='same', 
      activation='relu',
      kernel_initializer=next(kernel_initializer),
      name=name('conv{}{}'.format(depth,i)),
    )(x)

  if double_endpoint:
    x0 = x

  for i in range(depth-1, -1, -1):
    x = k.layers.Conv2DTranspose(
      filters=filters*2**i,
      kernel_size=upsampling_kernel_size,
      strides=2,
      padding='same',
      activation='relu',
      kernel_initializer=next(kernel_initializer),
      name=name('up{}'.format(i)),
    )(x)

    x = k.layers.Concatenate(name=name('concat{}'.format(i)))([x, levels.pop()])

    for j in range(2):
      x = k.layers.Conv2D(
        filters=filters*2**i, 
        kernel_size=3, 
        padding='same', 
        activation='relu',
        kernel_initializer=next(kernel_initializer),
        name=name('convuw{}{}'.format(i,j)),
      )(x)
  
  if out_channels is not None:
    x = k.layers.Conv2D(
      filters=out_channels,
      kernel_size=1, 
      activation=out_activation,
      kernel_initializer=next(kernel_initializer),
      name=name('convout'),
    )(x)

  if double_endpoint:
    x = x,x0

  return x

@gin.configurable(module='siamrl.nets')
def mobile_unet(
  inputs, 
  depth=3,
  filters=64,
  out_channels=None,
  out_activation=None,
  kernel_initializer=None,
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
  if name_scope:
    name = lambda n: '{}/{}'.format(name_scope, n)
  else:
    name = lambda n: n
  kernel_initializer = kernel_initializer_generator(
    kernel_initializer=kernel_initializer,
    seed=seed,
  )

  x = inputs
  levels = []

  x = k.layers.Conv2D(
    filters=filters//2,
    kernel_size=3, 
    padding='same', 
    activation='relu',
    kernel_initializer=next(kernel_initializer),
    name=name('convdw00'),
  )(x)

  x = k.layers.SeparableConv2D(
    filters=filters, 
    kernel_size=3, 
    padding='same', 
    activation='relu',
    depthwise_initializer=next(kernel_initializer),
    pointwise_initializer=next(kernel_initializer),
    name=name('convdw01'),
  )(x)

  for i in range(1, depth+1):
    levels.append(x)

    x = k.layers.SeparableConv2D(
      filters=filters*2**i, 
      kernel_size=3, 
      strides=2,
      padding='same', 
      activation='relu',
      depthwise_initializer=next(kernel_initializer),
      pointwise_initializer=next(kernel_initializer),
      name=name('convdw{}0'.format(i)),
    )(x)

    x = k.layers.SeparableConv2D(
      filters=filters*2**i, 
      kernel_size=3, 
      padding='same', 
      activation='relu',
      depthwise_initializer=next(kernel_initializer),
      pointwise_initializer=next(kernel_initializer),
      name=name('convdw{}1'.format(i)),
    )(x)
    
  for i in range(depth-1, -1, -1):
    x = k.layers.UpSampling2D(interpolation='bilinear', name=name('up{}0'.format(i)))(x)

    x = k.layers.SeparableConv2D(
      filters=filters*2**i,
      kernel_size=3,
      padding='same',
      activation='relu',
      depthwise_initializer=next(kernel_initializer),
      pointwise_initializer=next(kernel_initializer),
      name=name('up{}1'.format(i)),
    )(x)

    x = k.layers.Concatenate(name=name('concat{}'.format(i)))([x, levels.pop()])

    for j in range(2):
      x = k.layers.SeparableConv2D(
        filters=filters*2**i, 
        kernel_size=3, 
        padding='same', 
        activation='relu',
        depthwise_initializer=next(kernel_initializer),
        pointwise_initializer=next(kernel_initializer),
        name=name('convuw{}{}'.format(i,j)),
      )(x)

  if out_channels:
    x = k.layers.Conv2D(
      filters=out_channels,
      kernel_size=1, 
      activation=out_activation,
      kernel_initializer=next(kernel_initializer),
      name=name('convout'),
    )(x)
  
  return x

def default_branch_layers(
  inputs, 
  kernel_initializer=None,
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

def value(inputs, avg=True, units=512, depth=1, kernel_initializer=None, seed=None):
  kernel_initializer = kernel_initializer_generator(
    kernel_initializer=kernel_initializer,
    seed=seed,
  )

  if avg:
    x = k.layers.GlobalAvgPool2D()(inputs)
  else:
    x = k.layers.GlobalMaxPool2D()(inputs)
  for _ in range(depth):
    x = k.layers.Dense(units, activation='relu', kernel_initializer=next(kernel_initializer))(x)
  return k.layers.Dense(1, kernel_initializer=next(kernel_initializer))(x)

@gin.configurable(module='siamrl.nets')
def pos_layers(
  inputs,
  filters=32,
  depth=2,
  kernel_initializer=None,
  compat=False,
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
      (k.layers.Conv2D, {
        'filters':filters,
        'kernel_size':3, 
        'activation':'relu', 
        'padding':'same'
      }),
    ]*depth + [
      (k.layers.Conv2D if not compat else k.layers.SeparableConv2D, {
        'filters':1, 
        'kernel_size':1, 
        'padding':'same'
      }),
    ],
    kernel_initializer=kernel_initializer,
    **kwargs
  )