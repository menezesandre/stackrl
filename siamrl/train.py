import os
import sys
import traceback
import random
from datetime import datetime

import gin
import numpy as np
import tensorflow as tf

from siamrl.nets import PseudoSiamFCN
from siamrl.agents import DQN
from siamrl.envs import make, assert_registered
from siamrl.utils import Timer, AverageMetric, AverageReward

@gin.configurable(module='siamrl')
class Training(object):
  """Implements the DQN training routine"""
  def __init__(
    self,
    env,
    num_parallel_envs=None,
    eval_env=None,
    net=PseudoSiamFCN,
    agent=DQN,
    train_reward_buffer_length=10,
    eval_reward_buffer_length=10,
    directory='.',
    save_evaluated_policies=False,
    log_to_file=True,
    log_interval=100,
    eval_interval=10000,
    checkpoint_interval=10000,
    goal_check_interval=1000,
    memory_growth=True,
    seed=None,
    eval_seed=None,
  ):
    """
    Args:
      env: Training environment. Either an instance of a gym Env or the id 
        of the environment on the gym registry. Can also be a list of 
        tuples with environment ids and reward goals to be used as 
        curriculum.
      num_parallel_envs: number of environments to run in parallel. Only 
        used if env is not an Env instance. None defaults to 1.
      net: constructor for the Q network. Receives a (possibly nested) 
        tuple with input shape as argument.
      agent: constructor for the agent. Receives the Q network as 
        argument.
      train_reward_buffer_length: train reward logged is the average of 
        the rewards from this number of most recent episodes. 
      eval_reward_buffer_length: number of episodes to run on policy 
        evaluation.
      directory: path to the directory where checkpoints, models and logs
        are to be saved
      save_evaluated_policies: whether to save the agent's net weights
        after each evaluation.
      log_to_file: whether verbose is to be printed to a file or to stdout.
      log_interval: number of iterations between logs.
      eval_interval: number of iterations between policy evaluation.
      checkpoint_interval: number of iterations between checkpoints.
      goal_check_interval: number of iterations between checks of 
        goal completion (to move on to the next one on the curriculum).
        Only used if env is a list with the curriculum.
      seed: for the random sequence of integers used to seed all of the 
        components (env, net, agent). (Note: if not provided, None is 
        explicitly passed as seed to the components, overriding any 
        default/configuration.)
      eval_seed: the evaluation environment is seeded with
        this at the beginning of each evaluation. If not provided, a number
        is taken from the random sequence of integers given by seed.
    """
    # Set log directory and file
    if not os.path.isdir(directory):
      os.makedirs(directory)
    self._log_file = os.path.join(directory, 'train.log') if \
      log_to_file else None

    try:
      devices = tf.config.list_physical_devices('GPU')
    except AttributeError:
      # list_physical_devices is under experimental for tensorflow-2.0.0 
      devices = tf.config.experimental.list_physical_devices('GPU')
    for device in devices:
      try: 
        tf.config.experimental.set_memory_growth(device, memory_growth) 
      except RuntimeError: 
        self.log("Couldn't set memory growth to {} for device {}. Already initialized.".format(memory_growth, device))

    # Set seeder.
    _random = random.Random(seed)
    seed = lambda: _random.randint(0,2**32-1)
    # Set global seeds.
    tf.random.set_seed(seed())
    np.random.seed(seed())

    # Set curriculum if provided
    if isinstance(env, list):
      self._curriculum_file = os.path.join(directory, 'curriculum.csv')
      if os.path.isfile(self._curriculum_file):
        _, achieved_goals = np.loadtxt(
          self._curriculum_file,
          delimiter=',',
          skiprows=1,
          unpack=True
        )
        achieved_goals = np.atleast_1d(achieved_goals)
      else:
        achieved_goals = []

      self._curriculum = []
      for eid, goal in env:
        if goal in achieved_goals:
          # If this goal was already achieved, skip this environment
          continue
        assert_registered(eid)
        self._curriculum.insert(0, (eid, goal))
      if self._curriculum:
        env, self._current_goal = self._curriculum.pop()
        self._complete = False
      else:
        # If all goals where previously achieved, continue training on last
        # environment.
        env, self._current_goal = env[-1]
        self._complete = True

      self._replace_eval_env = eval_env is None
      self._goal_check_interval = int(goal_check_interval)
    else:
      self._goal_check_interval = None

    # Set environments
    self._env_seed = seed()
    self._env = make(
      env, 
      n_parallel=num_parallel_envs, 
      seed=self._env_seed,
    )
    self._eval_env = make(
      eval_env or env, 
      n_parallel=num_parallel_envs,
      block=True,
    )
    if eval_seed is None:
      self._eval_seed = seed()
    else:
      self._eval_seed = eval_seed
      # Call the seeder anyway so that the rest of the seeds from the
      # sequence are the same regardless of eval_seed being provided.
      _=seed()

    # Agent
    self._agent = agent(
      net(
        self._env.observation_spec,
        seed=seed()
      ), 
      collect_batch_size=self._env.batch_size,
      seed=seed()
    )

    # Train log
    self._log_interval = int(log_interval)
    self._train_file = os.path.join(directory, 'train.csv')
    # Evaluation log
    self._eval_interval = int(eval_interval)
    self._eval_file = os.path.join(directory, 'eval.csv')    
    
    # Metrics
    self._reward = AverageReward(
      self._env.batch_size,
      length=train_reward_buffer_length)
    self._eval_reward = AverageReward(
      self._eval_env.batch_size,
      length=eval_reward_buffer_length
    )
    self._loss = AverageMetric(length=log_interval)
    self._mean_error = AverageMetric(length=log_interval)
    self._collect_timer = Timer()
    self._train_timer = Timer()

    # Save policy weights
    self._save_weights = save_evaluated_policies
    self._save_filepath = lambda i: os.path.join(
      directory, 
      'saved_weights', 
      str(i), 
      'weights'
    )  

    # Train checkpoints
    self._checkpoint_interval = int(checkpoint_interval)
    self._checkpoint = tf.train.Checkpoint(
      agent=self._agent,
      reward=self._reward
    )
    self._checkpoint_manager = tf.train.CheckpointManager(
      self._checkpoint,
      directory=os.path.join(directory, 'checkpoint'), 
      max_to_keep=1
    )

    # Internal variables to avoid repeated operations
    self._last_checkpoint_iter = None
    self._last_save_iter = None

    # Flag to assert initialize method is called before run
    self._initialized = False

  @property
  def iterations(self):
    return self._agent.iterations.numpy()

  @property
  def reset_env(self):
    """Set self._reset_env to trigger an environment reset on the 
    training loop."""
    if hasattr(self, '_reset_env') and self._reset_env: # pylint: disable=access-member-before-definition
      self._reset_env = False
      return True
    else:
      return False

  @gin.configurable(module='siamrl.Training')
  def initialize(
    self,
    num_steps=None, 
    policy=None
  ):
    """Checks if a checkpoint exists and if it doesn't performs initial
      evaluation and collect.
    Args:
      num_steps: Number of steps for the initial experience collect. 
        If None, the agent's replay memory is filled to its max capacity.
      policy: policy to use on the initial collect. If None, a random 
        collect is run.
    """
    self._checkpoint.restore(self._checkpoint_manager.latest_checkpoint)
    if self._checkpoint_manager.latest_checkpoint:
      self.log('Starting from checkpoint.')
    else:
      self.log('Starting from scratch.')
      # Evaluate the agent's policy once before training.
      self.eval()
      # Set collect policy and number of steps.
      num_steps = num_steps or self._agent.replay_memory_size
      policy = policy or (lambda o: self._env.sample())
      if not callable(policy):
        raise TypeError(
          "Invalid type {} for argument policy. Must be callable.".format(type(policy))
        )
      # Run initial collect
      self.log('Running initial collect...')
      step = self._env.reset()
      for _ in range(num_steps-1):
        if callable(step):
          step = step()
        a = policy(step[0])
        self._agent.observe(*step, a)
        step = self._env.step(a)
      if callable(step):
        o,r,_=step()
      else:
        o,r,_=step
      self._agent.observe(
        o,
        r,
        # Set last step as terminal.
        tf.ones((self._env.batch_size,), dtype=tf.bool),
        # last action is repeated here but it doesn't matter as an 
        # action from a terminal state is never used.
        a 
      ) 
      self.log('Done.')
      
    self._initialized = True

  def run(
    self,
    max_num_iterations=sys.maxsize,
    stop_when_complete=False,
    tensorboard_log=False,
  ):
    """
    Args:
      max_num_iterations: training stops after this number of iterations.
      stop_when_complete: only used if training with curriculum. Whether
        to stop training when last goal is achieved. If false, training 
        will continue on last environment until max_num_iterations is 
        reached.
      tensorboard_log: whether to make logs to be vizualized in tensorboard.
    """
    self._stop_when_complete = stop_when_complete

    if not self._initialized:
      self.initialize()
    if tensorboard_log:
      # Set writer
      logdir = os.path.join(
        os.path.dirname(self._train_file),
        'logdir',
        datetime.now().strftime('%Y%m%d-%H%M%S'),
      )
      writer = tf.summary.create_file_writer(logdir)
      # Set agent's iterations as default step
      tf.summary.experimental.set_step(self._agent.iterations)
      # Log first evaluation
      with writer.as_default():
        tf.summary.scalar('eval', self._eval_reward.result)
      # Check if tf.profiler exists
      profiler = hasattr(tf, 'profiler')
    try:
      step = self._env.reset()
      self._agent.acknowledge_reset()

      for i in range(max_num_iterations):
        # Colect experience
        with self._collect_timer:
          if callable(step):
            step = step() # pylint: disable=not-callable
          self._reward += step
          if tensorboard_log and i == 1:
            profiler_outdir=os.path.join(logdir, 'collect')
            if profiler:
              tf.profiler.experimental.start(profiler_outdir)
            tf.summary.trace_on(graph=True, profiler=not profiler)
          action = self._agent.collect(*step)
          if tensorboard_log and i == 1:
            if profiler:
              tf.profiler.experimental.stop()
              profiler_outdir=None
            with writer.as_default():
              tf.summary.trace_export(
                'collect', 
                profiler_outdir=profiler_outdir,
              )
          step = self._env.step(action)

        # Train on the sampled batch
        with self._train_timer:
          if tensorboard_log and i == 1:
            profiler_outdir = os.path.join(logdir, 'train')
            if profiler:
              tf.profiler.experimental.start(profiler_outdir)
            tf.summary.trace_on(graph=True, profiler=not profiler)
          loss, merr = self._agent.train()
          if tensorboard_log and i == 1:
            if profiler:
              tf.profiler.experimental.stop()
              profiler_outdir=None
            with writer.as_default():
              tf.summary.trace_export(
                'train',
                profiler_outdir=profiler_outdir,
              )

          self._loss += loss
          self._mean_error += merr

        iters = self.iterations

        if iters % self._log_interval == 0:
          if tensorboard_log:
            with writer.as_default():
              tf.summary.scalar('reward', self._reward.result)
              tf.summary.scalar('loss', self._loss.result)
              tf.summary.scalar('mean_error', self._mean_error.result)
          self.log_train()
        if iters % self._eval_interval == 0:
          self.eval()
          if tensorboard_log:
            with writer.as_default():
              tf.summary.scalar('eval', self._eval_reward.result)
          if self._save_weights:
            self.save()
        if self._goal_check_interval and iters % self._goal_check_interval == 0:
          self.check_goal()
        if self.reset_env:
          step = self._env.reset()
          self._agent.acknowledge_reset()
        if iters % self._checkpoint_interval == 0:
          self.checkpoint()
    except:
      self.log_exception()    
    finally:
      self.checkpoint()

  def eval(self):
    """Evaluates the current policy and writes the results."""
    self.log('Running evaluation...')
    # Reset evaluation reward and environment
    self._eval_reward.reset(full=True)
    self._eval_env.seed(self._eval_seed)
    step = self._eval_env.reset()
    values = []
    while not self._eval_reward.full:
      a, value = self._agent.policy(step[0], values=True)
      step = self._eval_env.step(a)
      self._eval_reward += step
      values.append(value)
    
    values = tf.stack(values)
    mean_max_value = tf.reduce_mean(tf.reduce_max(values, axis=-1))
    mean_value = tf.reduce_mean(values)
    std_value = tf.math.reduce_std(values)
    min_value = tf.reduce_min(values)
    max_value = tf.reduce_max(values)

    # If eval file is to be created, add header
    if not os.path.isfile(self._eval_file):
      line = 'Iter,Reward,Value,MeanValue,StdValue,MinValue,MaxValue\n'
    else:
      line = ''
    # Add iteration number and results
    line += '{},{},{},{},{},{},{}\n'.format(
      self.iterations,
      self._eval_reward.result.numpy(),
      mean_max_value.numpy(),
      mean_value.numpy(),
      std_value.numpy(),
      min_value.numpy(),
      max_value.numpy(),
    )
    # Write to file
    with open(self._eval_file, 'a') as f:
      f.write(line)

    self.log('Done.')

  def save(self):
    """Saves the weights of the current Q network"""
    iters = self.iterations
    if iters != self._last_save_iter:
      self.log("Saving Q network's weights...")
      self._agent.save_weights(self._save_filepath(iters))
      self._last_save_iter = iters
      self.log('Done.')

  def checkpoint(self):
    """Makes a checkpoint of the current training state"""
    iters = self.iterations
    if iters != self._last_checkpoint_iter:
      self.log('Saving checkpoint...')
      self._checkpoint_manager.save()
      self._last_checkpoint_iter = iters
      self.log('Done.')

  def log(self, line):
    """Logs line with a time stamp."""
    line = str(datetime.now())+': '+line+'\n'

    if self._log_file is not None:
      with open(self._log_file, 'a') as f:
        f.write(line)
    else:
      sys.stdout.write(line)

  def log_train(self):
    """Logs current step's results."""
    iters = self.iterations

    reward = self._reward.result.numpy()
    loss = self._loss.result.numpy()
    merr = self._mean_error.result.numpy()

    # If file doesn't exist, write header
    if not os.path.isfile(self._train_file):
      line = 'Iter,Reward,Loss,MeanError,CollectTime,TrainTime\n'
    else:
      line = ''
    line += '{},{},{},{},{},{}\n'.format(
      iters,
      reward,
      loss,
      merr,
      self._collect_timer(),
      self._train_timer()
    )
    with open(self._train_file, 'a') as f:
      f.write(line)
    
    self.log('Iter {:8} Reward {:<11.6} Loss {:<11.6}'.format(iters,reward,loss))

  def log_exception(self):
    """Logs the last exception's traceback with a timestamp"""
    error = str(datetime.now())+': Exception.\n' + \
      traceback.format_exc()
    if self._log_file is not None:
      with open(self._log_file, 'a') as f:
        f.write(error)
    else:
      sys.stderr.write(error)

  def check_goal(self):
    if not self._complete and \
      self._reward > self._current_goal*(1-self._agent.epsilon):

      self.log('Goal reward achieved.')
      if not os.path.isfile(self._curriculum_file):
        line = 'EndIter,Goal\n'
      else:
        line = ''
      line += '{},{}\n'.format(self.iterations, self._current_goal)
      with open(self._curriculum_file, 'a') as f:
        f.write(line)
      if not self._update_environment():
        # If there is no environment left, set complete flag.
        self._complete = True
    if self._complete and self._stop_when_complete:
      raise StopIteration('Training goal achieved.')

  def _update_environment(self):
    """Replaces the environments with the next one in the curriculum.
    Raises:
      StopIteration: when curriculum is finished.
    """
    if hasattr(self, '_curriculum') and self._curriculum:
      env_id, self._current_goal = self._curriculum.pop()
    else:
      return False

    self.log('Updating environment...')
    n_parallel = self._env.batch_size if self._env.multiprocessing else None
    new_env = make(
      env_id,
      n_parallel=n_parallel,
      seed=self._env_seed,
    )

    assert new_env.observation_spec == self._env.observation_spec \
      and new_env.action_spec == self._env.action_spec, \
      "All envs in curriculum must have same observation and action specs."

    del(self._env)
    self._env = new_env

    if self._replace_eval_env:
      new_env = make(
        env_id,
        n_parallel=n_parallel,
      )
      assert new_env.observation_spec == self._eval_env.observation_spec \
        and new_env.action_spec == self._eval_env.action_spec, \
        "All envs in curriculum must have same observation and action specs."

      del(self._eval_env)
      self._eval_env = new_env

    self.log('Done.')
    # Set flag to trigger environment reset on the training loop
    self._reset_env = True
    return True
