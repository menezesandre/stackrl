import os, sys, time, traceback, random
from datetime import datetime
import gin
import numpy as np
import tensorflow as tf
from siamrl.nets import PseudoSiamFCN
from siamrl.agents import DQN
from siamrl.envs import make
from siamrl.utils import Timer, AverageReward


@gin.configurable(module='siamrl')
class Training(object):
  """Implements the DQN training routine"""
  def __init__(
    self,
    env,
    n_parallel_envs=None,
    eval_env=None,
    net=PseudoSiamFCN,
    agent=DQN,
    num_eval_episodes=10,
    directory='.',
    save_evaluated_policies=False,
    log_to_file=True,
    log_interval=100,
    eval_interval=10000,
    checkpoint_interval=10000,
    memory_growth=True,
    seed=None
  ):
    """
    Args:
      env: Training environment. Either an instance of a gym Env or the id 
        of the environment on the gym registry.
      n_parallel_envs: number of environments to run in parallel. Only 
        used if env is not an Env instance. None defaults to 1.
      net: constructor for the Q network. Receives a (possibly nested) 
        tuple with input shape as argument.
      agent: constructor for the agent. Receives the Q network as 
        argument.
      num_eval_episodes: number of episodes to run on policy evaluation.
      directory: path to the directory where checkpoints, models and logs
        are to be saved
      save_evaluated_policies: whether to save the agent's net weights
        after each evaluation.
      log_to_file: whether verbose is to be printed to a file or to stdout.
      log_interval: number of iterations between logs.
      eval_interval: number of iterations between policy evaluation.
      checkpoint_interval: number of iterations between checkpoints
      seed: for the random sequence of integers used to seed all of the 
        components (env, net, agent). (Note: if not provided, None is 
        explicitly passed as seed to the components, overriding any 
        default/configuration.)
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
    if seed is not None:
      random.seed(seed)
      seed = lambda: random.randint(0,2**32-1)
    else:
      seed = lambda: None
    self._seeder = seed
    # Set global seeds.
    tf.random.set_seed(seed())
    np.random.seed(seed())

    # Environment
    self._env = make(
      env, 
      n_parallel=n_parallel_envs, 
      seed=seed()
    )
    self._eval_env = make(
      eval_env or env, 
      n_parallel=n_parallel_envs,
      block=True,
      seed=seed()
    )

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
    self._log_interval = log_interval
    self._train_file = os.path.join(directory, 'train.csv')
    # Metrics to keep a long term and a short term average of the training 
    # reward.
    self._train_lt_average_reward = AverageReward(self._env.batch_size)
    self._train_st_average_reward = AverageReward(self._env.batch_size)

    # Evaluation log
    self._eval_interval = eval_interval
    self._eval_file = os.path.join(directory, 'eval.csv')    
    self._num_eval_episodes = tf.constant(num_eval_episodes, dtype=tf.int32)
    # Metric for the evaluation average reward.
    self._eval_average_reward = AverageReward(self._eval_env.batch_size)
    # Save policy weights
    self._save_weights = save_evaluated_policies
    self._save_filepath = lambda i: os.path.join(
      directory, 
      'saved_weights', 
      str(i), 
      'weights'
    )  

    # Train checkpoints
    self._checkpoint_interval = checkpoint_interval
    self._checkpoint = tf.train.Checkpoint(agent=self._agent)
    self._checkpoint_manager = tf.train.CheckpointManager(
      self._checkpoint,
      directory=os.path.join(directory, 'checkpoint'), 
      max_to_keep=1
    )

    # To be used in subclasses
    self._callback_interval = None
    
    # Internal variables to avoid repeated operations
    self._last_checkpoint_iter = None
    self._last_save_iter = None

    # Flag to assert initialize method is called before run
    self._initialized = False

  @property
  def iterations(self):
    return self._agent.iterations.numpy()

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
      self.log('Starting from checkpoint. {}'.format(len(self._agent._replay_memory)))
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
    max_num_iterations=sys.maxsize
  ):
    if not self._initialized:
      self.initialize()
    try:
      collect_timer = Timer()
      train_timer = Timer()

      step = self._env.reset()

      for _ in range(max_num_iterations):
        # Colect experience
        with collect_timer:
          if callable(step):
            step = step() # pylint: disable=not-callable

          self._train_lt_average_reward(*step[1:])
          self._train_st_average_reward(*step[1:])

          action = self._agent.collect(*step)
          step = self._env.step(action)
        
        # Train on the sampled batch
        with train_timer:
          loss = self._agent.train()

        iters = self.iterations

        if iters % self._log_interval == 0:
          self.log_train(
            loss.numpy(),
            collect_timer(),
            train_timer()
          )
        if iters % self._checkpoint_interval == 0:
          self.checkpoint()
        if iters % self._eval_interval == 0:
          self.eval()
          if self._save_weights:
            self.save()
        if self._callback_interval and iters % self._callback_interval == 0:
          self._callback()
    except:
      self.log_exception()    
    finally:
      self.checkpoint()

  def eval(self):
    """Evaluates the current policy and writes the results."""
    self.log('Running evaluation...')
    # Reset evaluation reward and environment
    self._eval_average_reward.reset()
    o = self._eval_env.reset()
    while tf.less(
      self._eval_average_reward.episode_count, 
      self._num_eval_episodes
    ):
      o,r,t = self._eval_env.step(self._agent.policy(o))
      self._eval_average_reward(r,t)

    # If file is to be created, add header
    if not os.path.isfile(self._eval_file):
      line = 'Iter,Reward\n'
    else:
      line = ''
    # Add iteration number and results
    line += '{},{}\n'.format(
      self.iterations,
      self._eval_average_reward.result.numpy()
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

  def log_train(self, loss, collect_time, train_time):
    """Logs current step's results."""
    iters = self.iterations
    lt_avg_reward = self._train_lt_average_reward.result.numpy()
    st_avg_reward = self._train_st_average_reward.result.numpy()
    # Reset the short term average reward metric
    self._train_st_average_reward.reset(full=False)

    # If file doesn't exist, write header
    if not os.path.isfile(self._train_file):
      line = 'Iter,LTAvgReward,STAvgReward,Loss,CollectTime,TrainTime\n'
    else:
      line = ''
    line += '{},{},{},{},{},{}\n'.format(
      iters,
      lt_avg_reward,
      st_avg_reward,
      loss,
      collect_time,
      train_time
    )
    with open(self._train_file, 'a') as f:
      f.write(line)
    
    self.log('Iter %7d\tReward %.6f\tLoss %.6f'%(iters,lt_avg_reward,loss))

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

@gin.configurable(module='siamrl')
class CurriculumTraining(Training):
  """Extends Training class to implement a curriculum. A list of 
  environment ids and reward goals is provided. The agent trains on each
  environment until the train average reward reaches the goal. When
  it happens, the environment is automatically replaced by the next one."""

  def __init__(
    self, 
    env_ids,
    curriculum_goals=None,
    eval_env=None,
    directory='.',
    check_interval=1000,
    **kwargs
  ):
    """
    Args:
      env_ids: forms the curriculum. Either a list of ids of registered gym
        environments or a dict whose keys are the ids and values the reward
        goal for each environment.
      curriculum_goals: reward goals to be zipped with env_ids. Ignored if 
        env_ids is a dict.
      eval_env: id of a registered gym environment. If set, all
        parts of the training are evaluated with this environment.
        If None, each part is evaluated with an environment 
        similar to training.
      check_interval: number of iterations between checks of goal 
        completion (to move on to the next one on the curriculum). Better 
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
      if np.isscalar(goal):
        env_id, self._current_goal = self._curriculum.pop()
        if self._current_goal != goal:
          self._curriculum.append((env_id, self._current_goal))
      else:
        for g in goal:
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

    super(CurriculumTraining, self).__init__(
      env_id, 
      eval_env=eval_env,
      directory=directory, 
      **kwargs
    )

    self._callback_interval = check_interval

  @property
  def epsilon(self):
    return self._agent.epsilon

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
    """Replaces the environments with the next one in the curriculum.
    Raises:
      StopIteration: when curriculum is finished.
    """
    env_id, self._current_goal = self._curriculum.pop()

    self.log('Updating environment...')
    n_parallel = self._env.batch_size if self._env.multiprocessing else None
    new_env = make(
      env_id,
      n_parallel=n_parallel,
      seed=self._seeder()
    )

    assert new_env.observation_spec == self._env.observation_spec \
      and new_env.action_spec() == self._env.action_spec(), \
      "All envs in curriculum must have same observation and action specs."

    del(self._env)
    self._env = new_env

    if self._replace_eval_env:
      new_env = make(
        env_id,
        n_parallel=n_parallel,
        seed=self._seeder()
      )
      assert new_env.observation_spec == self._eval_env.observation_spec \
        and new_env.action_spec == self._eval_env.action_spec, \
        "All envs in curriculum must have same observation and action specs."

      del(self._eval_env)
      self._eval_env = new_env

    self.log('Done.')

  def _callback(self):
    if not self._complete and \
      self._train_lt_average_reward >= self._current_goal*(1-self.epsilon):

      self.log('Goal reward achieved.')
      if not os.path.isfile(self._curriculum_file):
        line = 'EndIter,Goal\n'
      else:
        line = ''
      line += '{},{}\n'.format(self.iterations, self._current_goal)
      with open(self._curriculum_file, 'w') as f:
        f.write(line)
      try:
        self._update_environment()
      except StopIteration:
        self._complete = True
        if self._finish_when_complete:
          raise StopIteration('Training goal achieved.')