"""Provides the necessary set of configurables from external packages."""
from tensorflow.keras import layers, optimizers
from gin import external_configurable as ec

# Layers
ec(layers.Activation, module='tf.keras.layers')
ec(layers.AvgPool2D, module='tf.keras.layers')
ec(layers.BatchNormalization, module='tf.keras.layers')
ec(layers.Conv2D, module='tf.keras.layers')
ec(layers.Conv2DTranspose, module='tf.keras.layers')
ec(layers.ConvLSTM2D, module='tf.keras.layers')
ec(layers.DepthwiseConv2D, module='tf.keras.layers')
ec(layers.Dropout, module='tf.keras.layers')
ec(layers.Flatten, module='tf.keras.layers')
ec(layers.MaxPool2D, module='tf.keras.layers')
ec(layers.SeparableConv2D, module='tf.keras.layers')
ec(layers.UpSampling2D, module='tf.keras.layers')

# Optimizers
ec(optimizers.Adadelta, module='tf.keras.optimizers')
ec(optimizers.Adagrad, module='tf.keras.optimizers')
ec(optimizers.Adam, module='tf.keras.optimizers')
ec(optimizers.Adamax, module='tf.keras.optimizers')
ec(optimizers.Ftrl, module='tf.keras.optimizers')
ec(optimizers.Nadam, module='tf.keras.optimizers')
ec(optimizers.RMSprop, module='tf.keras.optimizers')
ec(optimizers.SGD, module='tf.keras.optimizers')

# Learning rate schedules
ec(optimizers.schedules.ExponentialDecay, module='tf.keras.optimizers.schedules')
ec(optimizers.schedules.InverseTimeDecay, module='tf.keras.optimizers.schedules')
ec(optimizers.schedules.PiecewiseConstantDecay, module='tf.keras.optimizers.schedules')
ec(optimizers.schedules.PolynomialDecay, module='tf.keras.optimizers.schedules')
