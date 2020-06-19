"""Provides the necessary set of configurables from external packages."""
from tensorflow import keras as k
from gin import external_configurable as ec

# Layers
ec(k.layers.Activation, module='tf.keras.layers')
ec(k.layers.AvgPool2D, module='tf.keras.layers')
ec(k.layers.BatchNormalization, module='tf.keras.layers')
ec(k.layers.Conv2D, module='tf.keras.layers')
ec(k.layers.Conv2DTranspose, module='tf.keras.layers')
ec(k.layers.ConvLSTM2D, module='tf.keras.layers')
ec(k.layers.DepthwiseConv2D, module='tf.keras.layers')
ec(k.layers.Dropout, module='tf.keras.layers')
ec(k.layers.Flatten, module='tf.keras.layers')
ec(k.layers.LayerNormalization, module='tf.keras.layers')
ec(k.layers.MaxPool2D, module='tf.keras.layers')
ec(k.layers.SeparableConv2D, module='tf.keras.layers')
ec(k.layers.UpSampling2D, module='tf.keras.layers')

# Optimizers
ec(k.optimizers.Adadelta, module='tf.keras.optimizers')
ec(k.optimizers.Adagrad, module='tf.keras.optimizers')
ec(k.optimizers.Adam, module='tf.keras.optimizers')
ec(k.optimizers.Adamax, module='tf.keras.optimizers')
ec(k.optimizers.Ftrl, module='tf.keras.optimizers')
ec(k.optimizers.Nadam, module='tf.keras.optimizers')
ec(k.optimizers.RMSprop, module='tf.keras.optimizers')
ec(k.optimizers.SGD, module='tf.keras.optimizers')

# Learning rate schedules
ec(k.optimizers.schedules.ExponentialDecay, module='tf.keras.optimizers.schedules')
ec(k.optimizers.schedules.InverseTimeDecay, module='tf.keras.optimizers.schedules')
ec(k.optimizers.schedules.PiecewiseConstantDecay, module='tf.keras.optimizers.schedules')
ec(k.optimizers.schedules.PolynomialDecay, module='tf.keras.optimizers.schedules')
