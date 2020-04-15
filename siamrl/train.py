import sys, os, traceback
from datetime import datetime

import numpy as np

import gin

import tensorflow as tf

for device in tf.config.experimental.list_physical_devices('GPU'):
  try: 
    tf.config.experimental.set_memory_growth(device, True) 
  except: 
    pass

import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.policies import tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from siamrl import networks

@gin.configurable
class Training(object):
  """Implements the DQN training routine"""
  def __init__(
    self,
    env, 
    num_parallel_envs=1,
    eval_env=None,
    net=networks.SiamQNetwork,
    optimizer=tf.keras.optimizers.Adam,
    agent=dqn_agent.DdqnAgent,
    replay_buffer=tf_uniform_replay_buffer.TFUniformReplayBuffer,
    sample_batch_size=32,
    collect_metrics=[],
    collect_metrics_buffer_length=10,
    collect_steps_per_iteration=1,
    eval_metrics=[],
    num_eval_episodes=1,
    directory='.',
    save_evaluated_policies=False,
    log_to_file=True,
    log_interval=100,
    eval_interval=10000,
    checkpoint_interval=10000
  ):
    """
    Args:
      env: Either an instance of TFEnvironment or an id of a registered 
        gym environment.
      num_parallel_envs: number of parallel environments. Only used if env
        is not a constructed instance of TFEnvironment.
      curriculum_goals: reward goals to be zipped with env
      eval_env: as env. Used for policy evaluation. If None, env argument
        is used.
      net: constructor for the Q network. Receives env's observation_spec
        and action_spec as arguments.
      optimizer: constructor for the optimizer to be used by agent.
      agent: constructor for the agent. Receives env's time_step_spec and
        action-spec and instances of net and optimizer as arguments.
      replay_buffer: constructor for the replay buffer. Receives the 
        agent's collect_data_spec end env's batch size as arguments.
      collect_metrics: list of constructors for the metrics for experience 
        collection.
      collect_steps_per_iteration: number of steps to collect from the env
        on each iteration.
      eval_metrics: list of constructors for the metrics for policy
        evaluation.
      num_eval_episodes: number of episodes to run on policy evaluation.
      directory: path to the directory where checkpoints, models and logs
        are to be saved
      save_evaluated_policies: whether to save the current policy as a 
        SavedModel after each evaluation.
      log_to_file: whether verbose is to be printed to a file or to stdout.
      log_interval: number of iterations between logs.
      eval_interval: number of iterations between policy evaluation.
      checkpoint_interval: number of iterations between checkpoints
    """
    # Environment
    if num_parallel_envs > 1:
      constructor = lambda x: tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
          [lambda: suite_gym.load(x)]*num_parallel_envs
        )
      )
    else:
      constructor = lambda x: tf_py_environment.TFPyEnvironment(
        suite_gym.load(x)
      )
    if isinstance(env, tf_environment.TFEnvironment):
      self._env = env
      if eval_env is None:
        self._eval_env = env
    elif isinstance(env, str):
      self._env = constructor(env)
      if eval_env is None:
        self._eval_env = constructor(env)
    else:
      raise ValueError(
        'Invalid type {} for argument env'.format(type(env))
      )
    if eval_env is not None:
      if isinstance(eval_env, tf_environment.TFEnvironment):
        self._eval_env = eval_env
      elif isinstance(env, str):
        self._eval_env = constructor(eval_env)
      else:
        raise ValueError(
          'Invalid type {} for argument eval_env'.format(type(eval_env))
        )
      assert self._eval_env.time_step_spec() == self._env.time_step_spec() \
        and self._eval_env.action_spec() == self._env.action_spec()

    # Agent
    self._net = net(
      self._env.observation_spec(), 
      self._env.action_spec()
    )
    self._agent = agent(
      self._env.time_step_spec(),
      self._env.action_spec(),
      self._net,
      optimizer(),

      train_step_counter=common.create_variable('train_step_counter')
    )
    self._agent.initialize()
    self._agent.train = common.function(self._agent.train)

    # Replay buffer
    self._replay_buffer = replay_buffer(
      self._agent.collect_data_spec,
      self._env.batch_size,
    )
    self._replay_iter = iter(
      self._replay_buffer.as_dataset(
        num_steps=2,
        sample_batch_size=sample_batch_size,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
      ).prefetch(tf.data.experimental.AUTOTUNE)
    )
  
    # Drivers
    self._collect_metrics = []
    for metric in collect_metrics:
      try:
        self._collect_metrics.append(metric(
          batch_size=self._env.batch_size,
          buffer_size=self._env.batch_size*collect_metrics_buffer_length
        ))
      except TypeError:
        self._collect_metrics.append(metric())

    self._collect_driver = dynamic_step_driver.DynamicStepDriver(
      self._env,
      self._agent.collect_policy,
      observers=[self._replay_buffer.add_batch]+self._collect_metrics,
      num_steps=collect_steps_per_iteration
    )

    if not eval_metrics:
      eval_metrics.append(tf_metrics.AverageReturnMetric)
    self._eval_metrics = []
    for metric in eval_metrics:
      try:
        self._eval_metrics.append(metric(
          batch_size=self._eval_env.batch_size,
          buffer_size=self._eval_env.batch_size*num_eval_episodes
        ))
      except TypeError:
        self._eval_metrics.append(metric())

    self._eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      self._eval_env,
      self._agent.policy,
      observers=self._eval_metrics,
      num_episodes=num_eval_episodes
    )

    # Log/Checkpoint
    if not os.path.isdir(directory):
      os.makedirs(directory)

    self._log_file = os.path.join(directory, 'train.log') if \
      log_to_file else None
    self._train_file = os.path.join(directory, 'train.csv')
    self._eval_file = os.path.join(directory, 'eval.csv')

    ckpt_dir = os.path.join(directory, 'checkpoint')
    self._checkpointer = common.Checkpointer(
      ckpt_dir=ckpt_dir, 
      max_to_keep=1,
      #step=self._agent.train_step_counter,
      agent=self._agent,
      #policy=agent.policy,
      replay_buffer=self._replay_buffer
    )

    self._save_weights = save_evaluated_policies
    self._save_filepath = lambda i: os.path.join(
      directory, 
      'saved_weights', 
      str(i), 
      'weights'
    )  

    self._log_interval = log_interval
    self._eval_interval = eval_interval
    self._checkpoint_interval = checkpoint_interval
    # To be used in subclasses
    self._callback_interval = None

    # Internal variables to avoid repeated operations
    self._last_checkpoint_iter = None
    self._last_save_iter = None

    # Flag to assert initialize method is called before run
    self._initialized = False

  def __del__(self):
    del(self._replay_buffer, self._agent)

  @property
  def step_counter(self):
    return self._agent.train_step_counter.numpy()

  @step_counter.setter
  def step_counter(self, value):
    self._agent.train_step_counter.assign(value)

  @gin.configurable(module='siamrl.train.Training')
  def initialize(
    self,
    num_steps=2048, 
    policy=random_tf_policy.RandomTFPolicy
  ):
    """Checks if a checkpoint exists and if it doesn't performs initial
      evaluation and collect.
    Args:
      num_steps: Number of steps for the initial experience collect.
      policy: policy to use on the initial collect. If None, agent's 
        collect policy is used.
    """
    if not self._checkpointer.checkpoint_exists:
      self.log('Starting from scratch.')
      # Reset the step counter
      self.step_counter = 0
      # Evaluate the agent's policy once before training.
      self.eval()
      # Set the initial collect policy
      if policy is None:
        policy = self._agent.collect_policy
      elif isinstance(policy, tf_policy.Base):
        assert policy.time_step_spec == self._env.time_step_spec() \
          and policy.action_spec == self._env.action_spec()
      else:
        try:
          policy = policy(
            self._env.time_step_spec(), self._env.action_spec())
          assert isinstance(policy, tf_policy.Base)
        except:
          raise ValueError(
            'Invalid type {} for argument initial_collect_policy'.format(
              type(policy)
            )
          )
      # Collect transitions
      initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        self._env,
        policy, 
        observers=[self._replay_buffer.add_batch],
        num_steps=num_steps)
      self.log('Running initial collect...')
      initial_collect_driver.run()
      self.log('Done.')
      del(initial_collect_driver, policy)
    else:
      self.log('Starting from checkpoint.')
    
    self._initialized = True

  def run(
    self,
    max_num_iterations=sys.maxsize
  ):
    if not self._initialized:
      self.initialize()
    try:
      for _ in range(max_num_iterations):
        # Colect experience
        self._collect_driver.run()
        # Sample a batch from the replay buffer
        sampled_batch, _ = next(self._replay_iter)
        # Train on the sampled batch
        loss_info = self._agent.train(sampled_batch)

        if self.step_counter % self._log_interval == 0:
          self.log_iteration(loss_info.loss)

        if self.step_counter % self._checkpoint_interval == 0:
          self.checkpoint()

        if self.step_counter % self._eval_interval == 0:
          self.eval()
          if self._save_weights:
            self.save()

        if self._callback_interval and \
          self.step_counter % self._callback_interval == 0 \
        :
          self._callback()
    except:
      self.log_exception()    
    finally:
      self.checkpoint()

  def eval(self):
    """Evaluates the current policy and writes the results."""
    self.log('Running evaluation...')
    # Reset metrics
    for metric in self._eval_metrics:
      metric.reset()
    # Run evaluation
    self._eval_driver.run()
    # If file is to be created, add header
    if not os.path.isfile(self._eval_file):
      line = 'Iter'
      for metric in self._eval_metrics:
        line += ','+metric.name
      line += '\n'
    else:
      line = ''
    # Add iteration number and results
    line += str(self.step_counter)
    for metric in self._eval_metrics:
      line += ','+str(metric.result().numpy())
    line += '\n'
    # Write to file
    with open(self._eval_file, 'a') as f:
      f.write(line)

    self.log('Done.')

  def save(self):
    """Saves the weights of the current Q network"""
    if self.step_counter != self._last_save_iter:
      self.log("Saving Q network's weights...")
      self._net.save_weights(self._save_filepath(self.step_counter))
      self._last_save_iter = self.step_counter
      self.log('Done.')

  def checkpoint(self):
    """Makes a checkpoint of the current training state"""
    if self.step_counter != self._last_checkpoint_iter:
      self.log('Saving checkpoint...')
      self._checkpointer.save(self.step_counter)
      self._last_checkpoint_iter = self.step_counter
      self.log('Done.')

  def log(self, line=None, **kwargs):
    """Logs line with a time stamp. If line is None, logs the kwargs."""
    if line:
      line = str(datetime.now())+': '+line+'\n'
    else:
      line = str(datetime.now())+': '
      for kw in kwargs:
        line += '{} {}\t'.format(
          kw, 
          kwargs[kw]
        )
      line = line[:-1]+'\n'

    if self._log_file is not None:
      with open(self._log_file, 'a') as f:
        f.write(line)
    else:
      sys.stdout.write(line)

  def log_iteration(self, loss):
    """Logs current step's loss and collect metric results."""
    results = {metric.name: metric.result().numpy() 
      for metric in self._collect_metrics}

    # If file doesn't exist, write header
    if not os.path.isfile(self._train_file):
      line = 'Iter,Loss'
      for key in results:
        line += ','+key
      line+='\n'
    else:
      line = ''

    line += '{},{}'.format(self.step_counter, loss)
    for value in results.values():
      line += ',{}'.format(value)
    line+='\n'
    
    with open(self._train_file, 'a') as f:
      f.write(line)

    self.log(Iter=self.step_counter, Loss=loss, **results)

  def log_exception(self):
    """Logs the last exception's traceback with a timestamp"""
    error = str(datetime.now())+': Exception.\n' + \
      traceback.format_exc()
    if self._log_file is not None:
      with open(self._log_file, 'a') as f:
        f.write(error)
    else:
      sys.stderr.write(error)

  def _callback(self):
    """To be implemented in subclasses for costum callbacks within run"""
    raise NotImplementedError('No costum callback implemented.')

@gin.configurable
class CurriculumTraining(Training):
  """Extends Training class to implement a curriculum. A list of 
  environment ids and reward goals is provided. The agent trains on each
  environment until the evaluation average reward reaches the goal. When
  it happens, the environment is automatically replaced by the next one."""

  def __init__(
    self, 
    env_ids,
    curriculum_goals=None,
    eval_env=None,
    collect_metrics=[],
    directory='.',
    check_interval=96,
    **kwargs
  ):
    """
    Args:
      env_ids: forms the curriculum.
        Either a list of ids of registered gym environments,
        or a dict whose keys are the ids and values the reward
        goal for each environment.
      curriculum_goals: reward goals to be zipped with env_ids.
        Ignored if env_ids is a dict.
      eval_env: id of a registered gym environment. If set, all
        parts of the training are evaluated with this environment.
        If None, each part is evaluated with an environment 
        similar to training.
      check_interval: number of iterations between checks of environment
        complete (to move on to the next one on the curriculum). Better 
        if it is a common multiple of all environments' episode lengths
        (number of steps per episode).
    """
    if isinstance(env_ids, dict):
      self._curriculum = [(e, g) 
        for e,g in zip(env_ids.keys(),env_ids.values())
      ][::-1]
    elif isinstance(env_ids, list):
      self._curriculum = [(e, g) 
        for e,g in zip(env_ids, curriculum_goals)
      ][::-1] 
    else:
      raise TypeError(
        'Invalid type {} for argument env_ids'.format(type(env_ids))
      )

    self._curriculum_file = os.path.join(directory, 'curriculum.csv')
    if os.path.isfile(self._curriculum_file):
      end_iter, goal = np.loadtxt(
        self._curriculum_file,
        delimiter=',',
        skiprows=1,
        unpack=True
      )
      for i, g in zip(end_iter, goal):
        env_id, self._current_goal = self._curriculum.pop()
        if self._current_goal != g:
          self._curriculum.append((env_id, self._current_goal))
          break
      if len(self._curriculum) == 0:
        self._complete = True
      else:
        self._complete = False
        env_id, self._current_goal = self._curriculum.pop()
    else:
      self._complete = False
      env_id, self._current_goal = self._curriculum.pop()

    self._replace_eval_env = eval_env is None

    if tf_metrics.AverageReturnMetric in collect_metrics:
      goal_metric_index = collect_metrics.index(
        tf_metrics.AverageReturnMetric
      )
    else:
      collect_metrics.append(tf_metrics.AverageReturnMetric)
      goal_metric_index = -1

    super(CurriculumTraining, self).__init__(
      env_id, 
      eval_env=eval_env,
      collect_metrics=collect_metrics,
      directory=directory, 
      **kwargs
    )

    self._goal_metric = self._collect_metrics[goal_metric_index]
    self._callback_interval = check_interval

  @property
  def _epsilon(self):
    return self._agent.collect_policy._get_epsilon()

  def initialize(self, **kwargs):
    super(CurriculumTraining, self).initialize(**kwargs)
    if not os.path.isfile(self._curriculum_file):
      with open(self._curriculum_file, 'w') as f:
        f.write('EndIter,Goal\n')

  def run(
    self,
    max_num_iterations=sys.maxsize,
    finish_when_complete=False
  ):
    """
    Arg:
      finish_when_complete: Whether to stop training when last goal is
        achieved. If false, training will continue on last environment
        until max_num_iterations is reached.
    """
    self._finish_when_complete = finish_when_complete
    super(CurriculumTraining, self).run(
      max_num_iterations=max_num_iterations
    )
    
  def _update_environment(self):
    """Replaces the environments (and respective drivers) for the next 
    one in the curriculum.
    Raises:
      StopIteration: when curriculum is finished.
    """
    env_id, self._current_goal = self._curriculum.pop()

    self.log('Updating environment...')
    if self._env.batch_size > 1:
      constructor = lambda x: tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
          [lambda: suite_gym.load(x)]*self._env.batch_size
        )
      )
    else:
      constructor = lambda x: tf_py_environment.TFPyEnvironment(
        suite_gym.load(x)
      )
    new_env = constructor(env_id)

    assert new_env.time_step_spec() == self._env.time_step_spec() \
      and new_env.action_spec() == self._env.action_spec(), \
      "All envs in curriculum must have same in and out specs."

    new_driver = dynamic_step_driver.DynamicStepDriver(
      new_env,
      self._agent.collect_policy,
      observers=[self._replay_buffer.add_batch]+self._collect_metrics,
      num_steps=self._collect_driver._num_steps
    )
    del(self._env, self._collect_driver)
    self._env = new_env
    self._collect_driver = new_driver

    if self._replace_eval_env:
      new_env = constructor(env_id)

      assert new_env.time_step_spec() == self._env.time_step_spec() \
        and new_env.action_spec() == self._env.action_spec(), \
        "All envs in curriculum must have same in and out specs."

      new_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        new_env,
        self._agent.policy,
        observers=self._eval_metrics,
        num_episodes=self._eval_driver._num_episodes
      )
      del(self._eval_env, self._eval_driver)
      self._eval_env = new_env
      self._eval_driver = new_driver
    self.log('Done.')

  def _callback(self):
    result = self._goal_metric.result()
    if result >= self._current_goal*(1-self._epsilon):
      if not self._complete:
        self.log('Goal return achieved.')
        with open(self._curriculum_file, 'a') as f:
          f.write('{},{}\n'.format(self.step_counter, self._current_goal))
        try:
          self._update_environment()
        except StopIteration:
          self._complete = True
          if self._finish_when_complete:
            raise StopIteration('Training goal achieved.')
