"""
References:
  [1] Human-level control through deep reinforcement learning 
    (https://www.nature.com/articles/nature14236)
  [2] Prioritized experience replay 
    (https://arxiv.org/abs/1511.05952)
  [3] Deep Reinforcement Learning with Double Q-learning 
    (https://arxiv.org/abs/1509.06461)
  [4] Rainbow: Combining Improvements in Deep Reinforcement Learning
    (https://arxiv.org/abs/1710.02298)
"""
import gin
import tensorflow as tf
from tensorflow import keras as k

from siamrl.agents.memory import ReplayMemory

@gin.configurable(module='siamrl.agents')
class DQN(tf.Module):
  """DQN agent [1]"""
  # pylint is messing up with tf...
  # pylint: disable=no-member,unexpected-keyword-arg,no-value-for-parameter,invalid-unary-operand-type
  metadata = {
    'exploration_modes': [
      'epsilon-greedy',
      'boltzmann',
    ],
  }

  def __init__(
    self,
    q_net,
    optimizer=None,
    learning_rate=None,
    huber_delta=1.,
    minibatch_size=32,
    replay_memory_size=100000,
    prefetch=None,
    target_update_period=10000,
    reward_scale=None,
    discount_factor=.99,
    collect_batch_size=None,
    exploration_mode=None,
    exploration=None,
    prioritization=None,
    priority_bias_compensation=None,
    double=False,
    n_step=None,
    graph=True,
    seed=None,
    name='DQN'
  ):
    """
    Args:
      q_net: Q-network. Instance of a keras Model. Observation and
        action spec are infered from this model's input and output.
      optimizer: for the q_net training. Either a constructor or an 
        instance of a keras Optimizer. If None, RMSProp with rho=0.95
        and mumentum=0.95 is used.
      learning_rate: only used if optimizer is a constructor. Either
        a scalar or an instance of a keras LearningRateSchedule. If None,
        0.00025 is used.
      loss: value of delta in the Huber loss to be aplied to the 
        temporal difference error. If None, MSE loss is used.
        identifier or an instance of a keras Loss.
      minibatch_size: sample batch size to be used on training.
      replay_memory_size: maximum number of transitions to be stored on
        the replay memory.
      prefetch: maximum number of minibatches to be prefetched from the 
        replay memory. None means no prefetching (each minibatch is sampled
        right before being used). Note: if using prioritized experience 
        replay, a minibatch used in an iteration may be have been sampled 
        with priorities from up to this number of iterations earlier.
      target_update_period: number of iterations between target 
        network updates.
      reward_scale: scaling factor to be aplied to the rewards when fitting
        the Q-network. 
      discount_factor: discount factor for delayed rewards. Scalar between 
        0 and 1.
      collect_batch_size: expected batch size of the observations 
        received in collect (i.e. from parallel environments). If None, 
        defaults to 1.
      exploration_mode: method to be used for exploration. Available 
        policies are 'epsilon-greedy' (0) and 'boltzmann' (1).
      exploration: Exploration parameter. Epsilon for the epsilon-greedy 
        policy (scalar between 0 and 1) or the temperature for the 
        boltzmann policy (float larger than 0). If None, it defaults to
        0.1 for epsilon-greedy and 1 for boltzmann. If final_exploration 
        and final_exploration_frame are not None, this is the initial 
        value of the exploration parameter.
      final_exploration: final value of the exploration parameter.
      final_exploration_iter: number of iterations along witch the 
        exploration parameter is linearly anealed from its initial to its 
        final value.
      prioritization: exponent that determines how much prioritization is
        used on experience replay (alpha in [2]). If None, defaults to 0
        (no priorization).
      priority_bias_compensation: importance sampling weights exponent 
        (beta in [2]) that determines how much the bias introduced by the
        non-uniform sample probabilities is compensated on the network
        updates.
      double: whether to use Double DQN algorithm [3] for target Q values 
        computation.
      n_step: number of steps to use on the multi-step variant of DQN [4]. 
        If None, defaults to 1 (stantard DQN).
      graph: whether collect and train methods should be trace-compiled
        into a graph (with tf.function wrapper).
      name: name of the agent.
    Raises:
      TypeError: if any argument is an invalid type.
      ValueError: if any argument has an invalid value.
    """
    super(DQN, self).__init__(name=name)
    # Set Q network
    if isinstance(q_net, k.Model):
      self._q_net = q_net
      self._target_q_net = k.models.clone_model(q_net)
      self._target_q_net.set_weights(q_net.get_weights())
      # TODO find a cleaner way of creating a cloned model
      # with a different name (to avoid colisions in graph
      # visualization) 
      self._target_q_net._name += '_target'
    else:
      raise TypeError(
        "Invalid type {} for argument q_net. Must be a keras Model."
      )
    # Set optimizer
    if optimizer is None:
      self._optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate or 0.00025, 
        rho=0.95, 
        momentum=0.95
      )
    elif isinstance(optimizer, k.optimizers.Optimizer):
      self._optimizer = optimizer
    elif callable(optimizer):
      self._optimizer = optimizer(learning_rate=learning_rate or 0.00025)
    else:
      raise TypeError(
        "Invalid type {} for argument optimizer. Must be a constructor or instance of a keras Optimizer.".format(type(optimizer))
      )

    # Set exploration
    if exploration_mode is None:
      self._exploration_mode = self.metadata['exploration_modes'][0]
    elif isinstance(exploration_mode, int):
      self._exploration_mode = self.metadata['exploration_modes'][exploration_mode]
    elif isinstance(exploration_mode, str):
      exploration_mode = exploration_mode.lower()
      if exploration_mode in self.metadata['exploration_modes']:
        self._exploration_mode = exploration_mode
      else:
        raise ValueError(
          "Invalid value {} for argument exploration_mode. Must be in {}.".format(exploration_mode, self.metadata['exploration_modes'])
        )
    else:
      raise TypeError(
        "Invalid type {} for argument exploration_mode. Must be int or str.".format(type(exploration_mode))
      )

    # Check if exploration parameter is inside bounds for the given exploration mode
    if self._exploration_mode == 'epsilon-greedy':
      if exploration is None:
        exploration = 0.1
      if callable(exploration):
        dummy = exploration(self.iterations)
      else:
        dummy = exploration
      if dummy < 0 or dummy > 1:
        raise ValueError(
          "Invalid value {} for argument exploration. Must be in [0,1].".format(exploration)
        )
    elif self._exploration_mode == 'boltzmann':
      if exploration is None:
        exploration = 1.
      if callable(exploration):
        dummy = exploration(self.iterations)
      else:
        dummy = exploration
      if dummy <= 0:
        raise ValueError(
          "Invalid value {} for argument exploration. Must be greater than 0.".format(exploration)
        )
    else:
      raise ValueError("Invalid value {} for argument exploration_mode".format(exploration_mode))

    if callable(exploration):
      self._exploration = exploration
    else:
      self._exploration = tf.constant(exploration, dtype=tf.float32)

    # Define loss
    self._huber = huber_delta is not None
    if self._huber:
      self._huber_delta = tf.constant(huber_delta, dtype=tf.float32)
    # Set target update period
    self._target_update_period = target_update_period or 10000
    # Set discount factor (gamma) according to n_step
    n_step = n_step or 1
    self._n_step = n_step > 1
    if self._n_step:
      # Discount for rewards
      self._gamma_r = tf.constant(
        [discount_factor**i for i in range(n_step)], 
        dtype=tf.float32
      )
      # Discount for next step's value
      self._gamma = tf.constant(
        discount_factor**n_step, 
        dtype=tf.float32
      )
    else:
      # Discount for next step's value
      self._gamma = tf.constant(discount_factor, dtype=tf.float32)
    # Set reward scaling
    if reward_scale:
      self._reward_scaling = True
      self._reward_scale_factor = tf.constant(reward_scale, dtype=tf.float32)
    else:
      self._reward_scaling = False
    # Set collect batch size
    collect_batch_size = collect_batch_size or 1
    # Infer state spec and number of actions from the Q net
    state_spec = tf.nest.map_structure(
      lambda x: tf.TensorSpec(
        shape=(collect_batch_size,)+x.shape[1:], 
        dtype=x.dtype
      ), 
      q_net.input
    )
    self._n_actions = q_net.output_shape[-1]
    # Set prioritization
    prioritization = prioritization or 0.
    self._prioritized = prioritization != 0.
    if self._prioritized:
      if priority_bias_compensation is None:
        priority_bias_compensation = 1.
      self._bias_compensation = priority_bias_compensation != 0.

    # Set replay memory
    self._replay_memory = ReplayMemory(
      state_spec,
      replay_memory_size,
      alpha=prioritization,
      beta=priority_bias_compensation,
      iters_counter=self._optimizer.iterations,
      n_steps=n_step,
      seed=seed
    )
    # Get dataset iterator for the replay memory
    dataset = self._replay_memory.dataset(
      minibatch_size, 
      get_weights=self._prioritized
    )
    if prefetch:
      dataset = dataset.prefetch(prefetch)
    self._replay_memory_iter = self._no_dependency(iter(dataset))

    self._double = double

    self._seed = seed

    # Wrap class methods with tf.function
    if graph:
      reward_spec = tf.TensorSpec(
        shape=(collect_batch_size,), 
        dtype=tf.float32
      )
      terminal_spec = tf.TensorSpec(
        shape=(collect_batch_size,), 
        dtype=tf.bool
      )
      action_spec = tf.TensorSpec(
        shape=(collect_batch_size,),
        dtype=tf.int64,
      )
      self.observe = tf.function(
        self.observe,
        input_signature=[
          state_spec,
          reward_spec,
          terminal_spec,
          action_spec
        ]
      )
      self.collect = tf.function(
        self.collect,
        input_signature=[
          state_spec,
          reward_spec,
          terminal_spec
        ]
      )
      self.train = tf.function(self.train, input_signature=[])
      # self.policy = tf.function(self.policy)

  def __del__(self):
    try:
      del(self._replay_memory_iter)
    except:
      pass
    # super(DQN, self).__del__()

  def __call__(self, state, reward, terminal, action=None):
    if action is None:
      return self.collect(state,reward,terminal)
    else:
      return self.observe(state,reward,terminal,action)

  @property
  def epsilon(self):
    if self._exploration_mode == 'epsilon-greedy':
      return self.exploration
    elif self._exploration_mode == 'boltzmann':
      # TODO: Get a better estimate
      return tf.math.exp(-1/self.exploration)
    else:
      raise NotImplementedError()

  @property
  def iterations(self):
    return self._optimizer.iterations
  @property
  def replay_memory_size(self):
    return self._replay_memory.max_length
  @property
  def exploration(self):
    if callable(self._exploration):
      return self._exploration(self.iterations)
    else:
      return self._exploration

  def policy(self, inputs, exploration=False, values=False, output_type=tf.int64):  # pylint: disable=method-hidden
    q_values = self._q_net(inputs)
    if exploration:
      e = self.exploration
      if self._exploration_mode == 'epsilon-greedy':
        batch_size = tf.shape(tf.nest.flatten(inputs)[0])[0]
        actions = tf.where(
          tf.random.uniform(
            (batch_size,),
            seed=self._seed,
          ) > e, 
          tf.math.argmax(q_values, axis=-1, output_type=output_type),
          tf.random.uniform(
            (batch_size,), 
            maxval=self._n_actions, 
            dtype=output_type,
            seed=self._seed,
          ),
        )
      elif self._exploration_mode == 'boltzmann':
        # Gumbel-max trick
        z = -tf.math.log(-tf.math.log(
          tf.random.uniform(tf.shape(q_values), seed=self._seed)
        ))
        actions = tf.math.argmax(
          q_values/e + z, 
          axis=-1, 
          output_type=output_type
        )
        # actions = tf.squeeze(
        #   tf.random.categorical(
        #     q_values/e,
        #     1,
        #     dtype=output_type,
        #     seed=self._seed,
        #   ), 
        #   axis=-1,
        # )
      else:
        raise NotImplementedError()
    else:
      actions = tf.math.argmax(q_values, axis=-1, output_type=output_type)
    if values:
      return actions, q_values
    else:
      return actions

  def save_weights(self, *args, **kwargs):
    """Save weights from the Q net."""
    return self._q_net.save_weights(*args, **kwargs)

  def acknowledge_reset(self):
    """Sets the latest transition in the replay memory as terminal. This
      must be used any time the environment is explicitly reset after a 
      non terminal state."""
    self._replay_memory.set_terminal()

  def observe(self, state, reward, terminal, action):  # pylint: disable=method-hidden
    """Stores time step in the replay memory"""
    self._replay_memory.add(state, reward, terminal, action)

  def collect(self, state, reward, terminal):  # pylint: disable=method-hidden
    """Computes action for given state and stores time step."""
    action = self.policy(state, exploration=True)
    self._replay_memory.add(state, reward, terminal, action)
    return action

  def train(self):  # pylint: disable=method-hidden
    # Sample transitions from replay memory
    if self._prioritized:
      indexes,weights,(states,actions,rewards,next_states,terminal) = \
        next(self._replay_memory_iter)
        # self._replay_memory.sample(self._minibatch_size, get_weights=True)
    else:
      states,actions,rewards,next_states,terminal = \
        next(self._replay_memory_iter)
        # self._replay_memory.sample(self._minibatch_size)

    with tf.GradientTape() as tape:
      # Compute Q values for the given actions
      q_values = self._q_net(states)
      q_values = tf.reduce_sum(
        q_values*tf.one_hot(
          actions, 
          self._n_actions
        ),
        axis=-1,
      )
      if self._reward_scaling:
        rewards *= self._reward_scale_factor
      if self._gamma == 0 and not self._n_step:
        target_q_values = rewards
      else:
        target_q_values = self._target_q_net(next_states)
        if self._double:
          target_q_values = tf.reduce_sum(
            target_q_values*tf.one_hot(
              tf.math.argmax(self._q_net(next_states), axis=-1), 
              self._n_actions
            ),
            axis=-1,
          )
          # target_q_values = tf.map_fn(
          #   lambda i: i[0][i[1]],
          #   (
          #     target_q_values, 
          #     self.policy(next_states)
          #   ),
          #   dtype=q_values.dtype
          # )
        else:
          target_q_values = tf.reduce_max(
            target_q_values,
            axis=-1
          )
        if self._n_step:
          rewards = tf.reduce_sum(
            self._gamma_r*rewards, 
            axis=-1
          )
        target_q_values = rewards + tf.where(
          terminal, 
          0., 
          self._gamma*target_q_values
        )
       
      # Compute temporal difference error
      td = q_values - tf.stop_gradient(target_q_values)
      mtd = tf.reduce_mean(td)
      td = tf.abs(td)
      # Compute the loss
      if self._huber:
        quadratic = tf.minimum(td, self._huber_delta)
        linear = td - quadratic
        loss = 0.5*quadratic**2 + self._huber_delta*linear
      else:
        loss = 0.5*td**2
      if self._prioritized and self._bias_compensation:
        loss = loss*weights
      loss = tf.reduce_mean(loss)
    # Compute and aply gradients on trainable weights
    variables = self._q_net.trainable_weights
    grads = tape.gradient(loss, variables)
    self._optimizer.apply_gradients(zip(grads, variables))
    # If using prioritized experience replay, update priorities
    if self._prioritized:
      self._replay_memory.update_priorities(indexes, td)

    if self.iterations % self._target_update_period == 0:
      # Update target network
      for v,vt in zip(
        self._q_net.trainable_weights, 
        self._target_q_net.trainable_weights
      ):
        vt.assign(v)

    return loss, mtd
