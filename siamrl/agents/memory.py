"""
References:
  [1] The Gumbel-Max Trick for Discrete Distributions
    (https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/)
"""
import tensorflow as tf
from math import inf

class ReplayMemory(tf.Module):
  """Memory for experience replay, supporting prioritized experience 
  replay."""
  # Small constant to prevent sample probability from becoming zero.
  epsilon = 1e-10
  def __init__(
    self,
    state_spec,
    max_length,
    alpha=None,
    beta=None,
    n_steps=None,
    seed=None,
    name='ReplayMemory'
  ):
    """
    Args:
      state_spec: TensorSpec or nest of TensorSpecs specifying state 
        shape (including batch dimension, which is used as the expected 
        number of transitions to be received in add) and data type.
      max_length: maximum number of transitions to be stored on memory 
        (will be truncated to be divisible by state_spec's batch size).
      alpha: exponent that determines how much prioritization is used 
        (0 corresponds to no prioritization). If None, defaults to 0.
      beta: importance sampling weights exponent that determines how much
        of the bias introduced by the non-uniform sample probabilities
        is compensated by the weights. If None, defaults to 1.
      n_steps: number of steps advanced in a transition returned from 
        sample (i.e. next_state is this number of steps ahead of state). 
        If None, defaults to 1.
      seed: to be used as local seed for minibatch sampling.
      name: name of this module.
    """
    super(ReplayMemory, self).__init__(name=name)
    with tf.device('CPU'), self.name_scope:
      # Set number of partitions as the number of transitions expected to
      # be received by add. This is given by state_spec's batch dimension
      self._n_parts = tf.nest.flatten(state_spec)[0].shape[0]
      # Make max_length divisible by num_partitions
      max_length -= max_length % self._n_parts
      # Set maximum partition length
      self._max_length = tf.constant(max_length//self._n_parts, dtype=tf.int32)
      # Set partitions' offsets (shaped to match what is expected by arg
      # indexes of tf.Variable.scatter_nd_update)
      self._offsets = tf.expand_dims(
        tf.range(self._n_parts, dtype=tf.int32)*self._max_length,
        -1
      )
      # Set prioritization and bias compensation
      self._alpha = tf.constant(alpha or 0., dtype=tf.float32)
      self._beta = tf.constant(beta or 1., dtype=tf.float32)
      # Set length of returned transitions
      self._n_steps = tf.constant(n_steps or 1, dtype=tf.int32)
      self._n_steps_range = tf.squeeze(
        tf.range(1,self._n_steps+1, dtype=tf.int32)
      )
      # Assert memory is large enough
      assert self._max_length > self._n_steps
      # Set local seed for random sampling
      self._seed = seed
      # Set storage
      self._state_spec = state_spec
      self._states = [
        tf.Variable(
          tf.zeros((max_length)+i.shape[1:], dtype=i.dtype),
          name='states'
        ) for i in tf.nest.flatten(state_spec)
      ]
      self._rewards = tf.Variable(
        tf.zeros(max_length, dtype=tf.float32), 
        name='rewards'
      )
      self._terminal = tf.Variable(
        tf.ones(max_length, dtype=tf.bool), 
        name='terminal'
      )
      self._actions = tf.Variable(
        tf.zeros(max_length, dtype=tf.int64), 
        name='actions'
      )
      self._logits = tf.Variable(
        -inf*tf.ones(max_length, dtype=tf.float32), 
        name='logits'
      )
      # Set internal variables 
      self._insert_index = tf.Variable(0, dtype=tf.int32, 
        name='insert_index')
      self._max_logit = tf.Variable(0., dtype=tf.float32, 
        name='max_logit')
      self._max_logit_index = tf.Variable(0, dtype=tf.int32, 
        name='max_logit_index') 
      self._min_logit = tf.Variable(0., dtype=tf.float32, 
        name='min_logit')
      self._min_logit_index = tf.Variable(0, dtype=tf.int32, 
        name='min_logit_index') 

  def __len__(self):
    """Returns the number of elements that can be sampled from this 
    memory."""
    return int(tf.math.count_nonzero(tf.math.exp(self._logits)))

  @property
  def max_length(self):
    return self._max_length.numpy()

  # @tf.Module.with_name_scope
  def add(self, state, reward, terminal, action):
    """Stores transitions in the memory."""
    indexes = self._offsets + self._insert_index % self._max_length
    # Update data in storage
    for var, updates in zip(self._states, tf.nest.flatten(state)):
      var.scatter_nd_update(indexes, updates)
    self._rewards.scatter_nd_update(indexes,reward)
    self._terminal.scatter_nd_update(indexes,terminal)
    self._actions.scatter_nd_update(indexes,action)
    # Set this element as unsampleable (until next_state is available)
    self._logits.scatter_nd_update(indexes, [-inf]*self._n_parts)
    # Set element from n steps back as sampleable if there's no episode
    # boundary.
    indexes = self._offsets + (self._insert_index-self._n_steps_range) % self._max_length
    self._logits.scatter_nd_update(
      indexes[:,-1:],
      tf.where(
        tf.reduce_any(
          self._terminal.sparse_read(indexes), 
          axis=-1
        ),
        -inf,
        self._max_logit
      )
    )

    self._insert_index.assign_add(1)

  # @tf.Module.with_name_scope
  def set_terminal(self):
    """Sets the latest transition added as terminal. (To be used when the
      environment is reset after a non terminal state.)"""
    indexes = self._offsets + (self._insert_index-1) % self._max_length
    self._terminal.scatter_nd_update(indexes,[True]*self._n_parts) 

  # @tf.Module.with_name_scope
  def sample(self, minibatch_size, get_weights=False):
    """Samples transitions from memory.
    Args:
      minibatch_size: number of transitions to sample.
      get_weights: whether sample weights should be returned (for use in
        prioritized experience replay.)
    Returns:
      Tuple with transitions: (states, actions, rewards, next_states, terminal).
      If get_weights is true, returns weights, transitions.
    Raises:
      InvalidArgumentError: if minibatch_size is larger than the number 
        of sampleable elements (given by len(self))
    """
    # Sample without replacement using the Gumbel-max trick [1]
    z = -tf.math.log(-tf.math.log(
      tf.random.uniform(tf.shape(self._logits), seed=self._seed)
    ))
    values,indexes = tf.math.top_k(self._logits+z, k=minibatch_size)
    # Assert that no 'unsampleable' values were sampled (this would mean 
    # that the number of sampleable elements is smaller than 
    # minibatch_size)
    tf.debugging.assert_all_finite(
      values, 
      "Not enough elements to sample"
    )
    # Get states
    states = tf.nest.pack_sequence_as(
      self._state_spec,
      [i.sparse_read(indexes) for i in self._states]
    )
    # Get actions
    actions = self._actions.sparse_read(indexes)

    next_indexes = (indexes + self._n_steps) % self._max_length \
      + indexes // self._max_length
    # Get next states
    next_states = tf.nest.pack_sequence_as(
      self._state_spec,
      [i.sparse_read(next_indexes) for i in self._states]
    )
    # Get terminal flags
    terminal = self._terminal.sparse_read(next_indexes)
    # If n_steps is greater than 1, get rewards from the n steps range
    if self._n_steps != self._n_steps_range:
      exp_indexes = tf.expand_dims(indexes,axis=-1)
      next_indexes = (exp_indexes + self._n_steps_range) % self._max_length \
        + exp_indexes // self._max_length
    # Get rewards
    rewards = self._rewards.sparse_read(next_indexes)
    if get_weights:
      weights=tf.math.exp(self._beta*(
        self._min_logit-self._logits.sparse_read(indexes)
      ))
      return indexes, weights, (states, actions, rewards, next_states, terminal)
    else:
      return states, actions, rewards, next_states, terminal

  # @tf.Module.with_name_scope
  def update_priorities(self, indexes, deltas):
    """
    Args:
      indexes: where to update the priorities, as returned by sample.
      deltas: error obtained on each transition on last update.
    """
    logits = tf.math.log(tf.math.pow(deltas+self.epsilon, self._alpha))
    # Update logits
    self._logits.scatter_nd_update(
      tf.expand_dims(indexes,-1), 
      logits      
    )
    # Update max logit if necessary
    argmax = tf.math.argmax(logits)
    max_index = indexes[argmax]
    max_logit = logits[argmax]
    if max_logit>=self._max_logit:
      self._max_logit_index.assign(max_index)
      self._max_logit.assign(max_logit)  
    elif tf.reduce_any(max_index==indexes):
      index = tf.math.argmax(self._logits, output_type=tf.int32)
      self._max_logit_index.assign(index)
      self._max_logit.assign(self._logits[index])
    # Update min logit if necessary
    argmin = tf.math.argmin(logits)
    min_index = indexes[argmin]
    min_logit = logits[argmin]
    if min_logit<=self._min_logit:
      self._min_logit_index.assign(min_index)
      self._min_logit.assign(min_logit)
    elif tf.reduce_any(self._min_logit_index==indexes):
      masked_logits = self._logits*tf.cast(
        tf.logical_not(tf.math.is_inf(self._logits)),
        dtype=tf.float32
      )
      index = tf.math.argmin(masked_logits, output_type=tf.int32)
      value = tf.debugging.assert_all_finite(
        masked_logits[index],
        "No sampleable transition (failed to compute min logit)"
      )
      self._min_logit_index.assign(index)
      self._min_logit.assign(value)

  def dataset(self, minibatch_size, get_weights=False):
    """Returns a tf.data.Dataset instance constructed from a generator 
      that yields the results of sampling minibatch_size transitions from
      the memory.
    """
    sample = tf.function(self.sample).get_concrete_function(
      minibatch_size, 
      get_weights=get_weights
    )
    def generator():
      while True:
        yield sample()
    
    state_types = tf.nest.map_structure(lambda s: s.dtype, self._state_spec)
    state_shapes = tf.nest.map_structure(
      lambda s: (minibatch_size,)+s.shape[1:], 
      self._state_spec
    )
    output_types = (
      state_types, 
      self._actions.dtype, 
      self._rewards.dtype, 
      state_types, 
      self._terminal.dtype
    )
    output_shapes = (
      state_shapes, 
      tf.TensorShape(minibatch_size),
      tf.TensorShape(minibatch_size),
      state_shapes,
      tf.TensorShape(minibatch_size)
    )
    if get_weights:
      output_types = (tf.int32, self._logits.dtype, output_types)
      output_shapes = (
        tf.TensorShape(minibatch_size),
        tf.TensorShape(minibatch_size), 
        output_shapes
      )
    return tf.data.Dataset.from_generator(
      generator, 
      output_types=output_types,
      output_shapes=output_shapes
    )
