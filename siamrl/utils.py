import sys, os, traceback, math
from datetime import datetime

import tensorflow as tf

import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

import gym

from siamrl import networks
from siamrl import envs

def register_stack_env(
  model_name='ic',
  base_size=[0.4375, 0.4375],
  resolution=2**(-9),
  time_step=1./240,
  num_objects=50,
  gravity=9.8,
  use_goal=False,
  goal_size=None,
  gui=False,
  state_reward=None,
  differential_reward=True,
  position_reward=False,
  settle_penalty=None,
  drop_penalty=0.,
  reward_scale=1.,
  dtype='float32'
):
  # Assert there are URDFs for the given name
  assert len(envs.data.getGeneratedURDF(model_name)) > 0
  ids = [env.id for env in gym.envs.registry.all() if 
      model_name.upper() in env.id]
  i = 0
  while model_name.upper()+'Stack-v%d'%i in ids:
    i +=1
  new_id = model_name.upper()+'Stack-v%d'%i
  gym.register(
      id=new_id,
      entry_point='siamrl.envs.stack:GeneratedStackEnv',
      max_episode_steps=num_objects,
      kwargs={'model_name': model_name,
              'base_size': base_size,
              'resolution': resolution,
              'time_step': time_step,
              'num_objects': num_objects,
              'gravity': gravity,
              'use_goal': use_goal,
              'goal_size': goal_size,
              'gui': gui,
              'state_reward': state_reward,
              'differential_reward': differential_reward,
              'position_reward': position_reward,
              'settle_penalty': settle_penalty,
              'drop_penalty': drop_penalty,
              'reward_scale': reward_scale,
              'dtype': dtype}
  )
  return new_id


def train(
  agent,
  train_env,
  eval_env=None,
  num_iterations = 20000,
  initial_collect_steps = 64,
  initial_collect_policy=None,
  collect_steps_per_iteration = 1,
  replay_buffer_max_length = 100,
  batch_size = 64,
  log_interval = 200,
  log_file=sys.stdout,
  use_time_stamp=True,
  num_eval_episodes = 1,
  eval_interval = 1000,
  eval_file=sys.stdout,
  ckpt_dir='./checkpoint',
  policy_dir='./policy',
  verbose=False
):
  if use_time_stamp:
    def print_log(line):
      stamp = str(datetime.now())
      log_file.write(stamp+line+'\n')
  else:
    def print_log(line):
      log_file.write(line+'\n')
  def print_eval(line):
    eval_file.write(line+'\n')
  if verbose:
    def print_verbose(line):
      print(line)
  else:
    def print_verbose(line):
      pass        
  if eval_env is None:
    eval_env = train_env

  agent.initialize()

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=train_env.batch_size,
      max_length=replay_buffer_max_length)
  replay_iter = iter(replay_buffer.as_dataset(
      num_parallel_calls=3,
      sample_batch_size=batch_size, 
      num_steps=2).prefetch(3))
  collect_driver = dynamic_step_driver.DynamicStepDriver(
      train_env,
      agent.collect_policy,
      observers=[replay_buffer.add_batch],
      num_steps=collect_steps_per_iteration)

  avg_return = tf_metrics.AverageReturnMetric(batch_size=eval_env.batch_size)
  eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      eval_env, agent.policy, observers=[avg_return], 
      num_episodes=num_eval_episodes)

  if ckpt_dir is not None:
    checkpointer = common.Checkpointer(ckpt_dir=ckpt_dir, 
        max_to_keep=1, agent=agent, policy=agent.policy,
        replay_buffer=replay_buffer, step=agent.train_step_counter)
  else:
    print_verbose('Not saving checkpoint.')
    checkpointer = None

  if policy_dir is not None:
    saver = policy_saver.PolicySaver(agent.policy)
  else:
    print_verbose('Not saving evaluated policies.')
    saver = None

  if checkpointer is None or not checkpointer.checkpoint_exists:
    print_verbose('Starting from scratch.')
    # Reset the step counter
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    print_verbose('Running initial evaluation...')
    eval_driver.run()
    print_eval('Iteration %d\tReward %f'%(0, 
        avg_return.result().numpy()))
    print_verbose('Done.')    
    # If not provided, use a random policy for the initial collect
    if initial_collect_policy is None:
      initial_collect_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec())
    # Collect transitions
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env, initial_collect_policy, 
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)
    print_verbose('Running initial collect...')
    initial_collect_driver.run()
    print_verbose('Done.')
    del(initial_collect_driver, initial_collect_policy)
  else:
    print_verbose('Starting from checkpoint.')

  # Optimize by wraping the train method in a graph
  agent.train = common.function(agent.train)
  final_time_step, policy_state = collect_driver.run()
  try:
    for _ in range(num_iterations):
      # Colect experience
      final_time_step, policy_state = collect_driver.run(
          final_time_step, policy_state)
      # Sample a batch from the replay buffer
      experience, info = next(replay_iter)
      # Train on the sampled batch
      loss_info = agent.train(experience=experience)
      step = agent.train_step_counter.numpy()

      if step % log_interval == 0:
        print_log('Iteration %d\tLoss %f'%(step, loss_info.loss))

      if step % eval_interval == 0:
        print_verbose('Running evaluation...')
        avg_return.reset()
        eval_driver.run()
        print_eval('Iteration %d\tReward %f'%(step, 
            avg_return.result().numpy()))
        if saver:
          print_verbose('Saving evaluated policy...')
          saver.save(os.path.join(policy_dir, str(step)))
        if checkpointer:
          print_verbose('Saving checkpoint...')
          checkpointer.save(step)
        print_verbose('Done.')

  except:
    print_verbose('Catched exception:')
    traceback.print_exc()
  finally:
    # Save a checkpoint and clean before exiting
    if checkpointer:
      print_verbose('Saving checkpoint.')
      checkpointer.save(agent.train_step_counter)
    del(saver, checkpointer, eval_driver, avg_return,
        collect_driver, replay_iter, replay_buffer)

def train_ddqn_on_stack_env(
  directory='.',
  model_name='ic',
  base_size=[0.4375, 0.4375],
  resolution=2**(-9),
  time_step=1./240,
  num_objects=50,
  gravity=9.8,
  use_goal=False,
  goal_size=None,
  gui=False,
  state_reward=None,
  differential_reward=True,
  position_reward=False,
  settle_penalty=None,
  drop_penalty=0.,
  reward_scale=1.,
  dtype='float32',
  num_parallel_envs=1,
  learning_rate=0.00001,
  target_update_period=10000,
  save_policies=False,
  plot=False,
  num_iterations = 20000,
  initial_collect_steps = 64,
  initial_collect_policy=None,
  collect_steps_per_iteration = 1,
  replay_buffer_max_length = 100,
  batch_size = 64,
  log_interval = 200,
  log_file=sys.stdout,
  use_time_stamp=True,
  num_eval_episodes = 1,
  eval_interval = 1000,
  verbose=False,
  **kwargs
):
  params = locals()

  # Create the directory if it doesn't exist
  if not os.path.isdir(directory):
    os.makedirs(directory)
  with open(os.path.join(directory,'params'), 'a') as f:
    f.write(str(params)+'\n')
  ckpt_dir = os.path.join(directory, 'checkpoint')
  if save_policies:
    policy_dir = os.path.join(directory, 'policy')  
  else:
    policy_dir = None
  eval_file_name = os.path.join(directory, 'eval.log')

  env_id = register_stack_env(
    model_name=model_name,
    base_size=base_size,
    resolution=resolution,
    time_step=time_step,
    num_objects=num_objects,
    gravity=gravity,
    use_goal=use_goal,
    goal_size=goal_size,
    gui=gui,
    state_reward=state_reward,
    differential_reward=differential_reward,
    position_reward=position_reward,
    settle_penalty=settle_penalty,
    drop_penalty=drop_penalty,
    reward_scale=reward_scale,
    dtype=dtype
  )
  # Load an environment for training and other for evaluation
  if num_parallel_envs > 1:
    constructors = [lambda: suite_gym.load(env_id)]*num_parallel_envs
    train_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(constructors))
    eval_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(constructors))
    num_eval_episodes = math.ceil(num_eval_episodes/num_parallel_envs)
  else:
    train_env = tf_py_environment.TFPyEnvironment(
        suite_gym.load(env_id))
    eval_env = tf_py_environment.TFPyEnvironment(
        suite_gym.load(env_id))

  # Create a Q network for the environment specs
  q_net = networks.SiamQNetwork(train_env.observation_spec(), 
      train_env.action_spec(), **kwargs)
  optimizer = tf.keras.optimizers.Adam(learning_rate)
  train_step_counter = common.create_variable('train_step_counter')
  # Create a Double DQN agent
  agent = dqn_agent.DdqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=target_update_period,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
  )

  # Train the agent
  with open(eval_file_name, 'a') as f:
    train(agent, train_env, 
      eval_env=eval_env, ckpt_dir=ckpt_dir,
      policy_dir=policy_dir, eval_file=f, 
      num_iterations=num_iterations,
      initial_collect_steps=initial_collect_steps,
      initial_collect_policy=initial_collect_policy,
      collect_steps_per_iteration=collect_steps_per_iteration,
      replay_buffer_max_length=replay_buffer_max_length,
      batch_size=batch_size,
      log_interval=log_interval,
      log_file=log_file,
      use_time_stamp=use_time_stamp,
      num_eval_episodes=num_eval_episodes,
      eval_interval=eval_interval,
      verbose=verbose
    )
  if plot:
    # Plot the evolution of the policy evaluations
    plot_log(eval_file_name, os.path.join(directory, 'plot.png'), show=True)


try:
  import matplotlib.pyplot as plt
  def plot_log(log_file_name, plot_file_name, show=False):
    if not os.path.isfile(log_file_name):
      print('No file named '+log_file_name)
      return
    data = {}
    with open(log_file_name) as f:
      l = next(f).split()
      for i in range(len(l)//2):
        data[l[2*i]] = [eval(l[2*i+1])]
      for l in f:
        l = l.split()
        for i in range(len(l)//2):
          data[l[2*i]].append(eval(l[2*i+1]))
      
    x_key = list(data.keys())[0]
    y_keys = list(data.keys())[1:]
    fig, axs = plt.subplots(len(y_keys), 1, sharex=True)
    if not hasattr(axs, '__iter__'):
      axs = [axs]

    for ax, y_key in zip(axs, y_keys):
      ax.plot(data[x_key], data[y_key])
      ax.set_ylabel(y_key)
    axs[-1].set_xlabel(x_key)
      
    fig.savefig(plot_file_name)
    if show:
      fig.show()
except ImportError as e:
  exception = e
  def plot_log(**kwargs):
    raise exception