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
from siamrl.agents.policies import GreedyPolicy
from siamrl.agents.memory import ReplayMemory

optimizers = {
  'adadelta': k.optimizers.Adadelta,
  'adagrad': k.optimizers.Adagrad,
  'adam': k.optimizers.Adam,
  'adamax': k.optimizers.Adamax,
  'ftrl': k.optimizers.Ftrl,
  'nadam': k.optimizers.Nadam,
  'rmsprop': k.optimizers.RMSprop,
  'sgd': k.optimizers.SGD
}

@gin.configurable(module='siamrl.agents')
class DQN(tf.Module):
  """DQN agent [1]"""
  def __init__(
    self,
    q_net,
    optimizer='adam',
    learning_rate=None,
    huber_delta=1.,
    minibatch_size=32,
    replay_memory_size=100000,
    prefetch=None,
    target_update_period=10000,
    discount_factor=1.,
    collect_batch_size=None,
    exploration=0.1,
    final_exploration=None,
    final_exploration_iter=None,
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
      q_net: Q network. Instance of a keras Model. Observation and
        action spec are infered from this model's input and output.
      optimizer: for the q_net training. Either a string identifier
        or an instance of a keras Optimizer.
      learning_rate: only used if optimizer is a string identifier. 
        Either a scalar or an instance of a keras LearningRateSchedule. 
        If None, default optimizer's learning rate is used.
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
      gamma: discount factor for delayed rewards. Scalar between 0 and 1.
      collect_batch_size: expected batch size of the observations 
        received in collect (i.e. from parallel environments). If None, 
        defaults to 1.
      exploration: probability of taking a random action for the epsilon 
        greedy policy. Scalar float between 0 and 1. If final_exploration
        and final_exploration_frame are not None, this is the initial 
        exploration.
      final_exploration: final epsilon value for the epsilon greedy
        policy.
      final_exploration_iter: number of iterations along witch epsilon is 
        linearly anealed from its initial to its final value.
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
    else:
      raise TypeError(
        "Invalid type {} for argument q_net. Must be a keras Model."
      )
    # Set optimizer
    if isinstance(optimizer, k.optimizers.Optimizer):
      self._optimizer = optimizer
    elif isinstance(optimizer, str):
      optimizer = optimizer.lower()
      if optimizer in optimizers:
        optimizer = optimizers[optimizer]
        if learning_rate is None:
          self._optimizer = optimizer()
        else:
          self._optimizer = optimizer(learning_rate=learning_rate)
      else:
        raise ValueError(
          "Invalid value {} for argument optimizer. Must be in {}.".format(
            optimizer,
            list(optimizers.keys())
          )
        )
    else:
      raise TypeError(
        "Invalid type {} for argument optimizer. Must be a keras Optimizer or a str."
      )
    # Set exploration (epsilon)
    if exploration < 0 or exploration > 1:
      raise ValueError(
        "Invalid value {} for argument exploration. Must be in [0,1].".format(exploration)
      )
    self._epsilon = tf.constant(exploration, dtype=tf.float32)
    if final_exploration is None or final_exploration_iter is None:
      self._epsilon_anealing = False
    elif final_exploration < 0 or final_exploration > 1:
      raise ValueError(
        "Invalid value {} for argument final_exploration. Must be in [0,1].".format(final_exploration)
      )
    elif final_exploration_iter <= 0:
      raise ValueError(
        "Invalid value {} for argument final_exploration_iter. Must be greater than 0.".format(final_exploration_iter)
      )
    else:
      self._epsilon_anealing = True
      self._delta_epsilon = tf.constant(
        final_exploration - exploration, dtype=tf.float32)
      self._final_epsilon_anealing_iter = tf.constant(
        final_exploration_iter, dtype=tf.float32)

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
    # Set policy
    self.policy = GreedyPolicy(q_net)
    # Set prioritization
    prioritization = prioritization or 0.
    self._prioritized = prioritization > 0.
    if self._prioritized:
      priority_bias_compensation = priority_bias_compensation or 1.
      self._bias_compensation = priority_bias_compensation > 0.
    # Set replay memory
    self._replay_memory = ReplayMemory(
      state_spec,
      replay_memory_size,
      alpha=prioritization,
      beta=priority_bias_compensation,
      n_steps=n_step,
      seed=seed
    )
    # Set minibatch size
    # self._minibatch_size = minibatch_size
    # Get dataset iterator for the replay memory
    dataset = self._replay_memory.dataset(
      minibatch_size, 
      get_weights=self._prioritized
    )
    if prefetch:
      dataset = dataset.prefetch(prefetch)
    # with FreezeDependencies(self):
    #   self._replay_memory_iter = iter(dataset)
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
        dtype=self.policy.output.dtype
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

  def __del__(self):
    del(self._replay_memory_iter)
    super(DQN, self).__del__()

  def __call__(self, state, reward, terminal, action=None):
    if action is None:
      return self.collect(state,reward,terminal)
    else:
      return self.observe(state,reward,terminal,action)

  @property
  def epsilon(self):
    if self._epsilon_anealing:
      return self._epsilon + self._delta_epsilon*tf.minimum(
        1., 
        tf.cast(self.iterations)/self._final_epsilon_anealing_iter
      )
    else:
      return self._epsilon

  @property
  def iterations(self):
    return self._optimizer.iterations

  @property
  def replay_memory_size(self):
    return self._replay_memory.max_length

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
    batch_size = tf.shape(tf.nest.flatten(state)[0])[:1]
    cond = tf.random.uniform(
      batch_size, 
      seed=self._seed
    ) > self.epsilon
    action = tf.where(
      cond, 
      self.policy(state),
      tf.random.uniform(
        batch_size, 
        maxval=self._n_actions, 
        dtype=self.policy.output.dtype,
        seed=self._seed
      )
    )
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
      # Compute Q values
      q_values = self._q_net(states)
      q_values = tf.map_fn(
        lambda i: i[0][i[1]],
        (q_values, actions),
        dtype=q_values.dtype
      )
      # Compute target Q values
      target_q_values = self._target_q_net(next_states)
      if self._double:
        target_q_values = tf.map_fn(
          lambda i: i[0][i[1]],
          (
            target_q_values, 
            self.policy(next_states)
          ),
          dtype=q_values.dtype
        )
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
      # td = tf.abs(q_values - tf.stop_gradient(target_q_values))
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

    tf.cond(
      self.iterations % self._target_update_period == 0,
      true_fn=self._update_target,
      false_fn=lambda: self._target_q_net.trainable_weights
    )      
    return loss, mtd

  def _update_target(self):
    for v,vt in zip(
      self._q_net.trainable_weights, 
      self._target_q_net.trainable_weights
    ):
      vt.assign(v)
    return self._target_q_net.trainable_weights