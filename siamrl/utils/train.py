import sys, os, _io, traceback, math
from datetime import datetime
import gin

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

from siamrl import networks
from siamrl.utils import register

@gin.configurable
def train(
  agent,
  train_env,
  eval_env=None,
  num_iterations = 100000,
  initial_collect_steps = 1000,
  initial_collect_policy=None,
  collect_steps_per_iteration = 1,
  replay_buffer_max_length = 100000,
  batch_size = 64,
  log_interval = 500,
  log_file=sys.stdout,
  use_time_stamp=True,
  num_eval_episodes = 10,
  eval_interval = 5000,
  eval_file=sys.stdout,
  ckpt_dir='./checkpoint',
  policy_dir='./policy',
  best_policy=True,
  verbose=False,
  exception_actions=[],
  finally_actions=[]
):
# Define the logging functions
  def stamp():
    return str(datetime.now())+': ' if use_time_stamp else ''
  if isinstance(log_file,_io.TextIOWrapper):
    def print_log(line, v=False):
      if not v or verbose:
        log_file.write(stamp()+line+'\n')
    def print_exc():
      traceback.print_exc(file=log_file)
  else:    
    def print_log(line, v=False):
      if not v or verbose:
        with open(log_file, 'a') as f:
          f.write(stamp()+line+'\n')
    def print_exc():
      with open(log_file, 'a') as f:
        traceback.print_exc(file=f)
  if isinstance(eval_file, _io.TextIOWrapper):
    def print_eval(line):
      eval_file.write(line+'\n')
  else:
    def print_eval(line):
      with open(eval_file, 'a') as f:
        f.write(line+'\n')
#
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
    print_log('Not saving checkpoint.', v=True)
    checkpointer = None

  if policy_dir is not None:
    saver = policy_saver.PolicySaver(agent.policy)
  else:
    print_log('Not saving evaluated policies.', v=True)
    saver = None

  if checkpointer is None or not checkpointer.checkpoint_exists:
    print_log('Starting from scratch.', v=True)
    # Reset the step counter
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    print_log('Running initial evaluation...', v=True)
    eval_driver.run()
    print_eval('Iteration %d\tReward %f'%(0, 
        avg_return.result().numpy()))
    print_log('Done.', v=True)
    # If not provided, use a random policy for the initial collect
    if initial_collect_policy is None:
      initial_collect_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec())
    # Collect transitions
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env, initial_collect_policy, 
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)
    print_log('Running initial collect...', v=True)
    initial_collect_driver.run()
    print_log('Done.', v=True)
    del(initial_collect_driver, initial_collect_policy)
  else:
    print_log('Starting from checkpoint.', v=True)

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
        print_log('Running evaluation...', v=True)
        avg_return.reset()
        eval_driver.run()
        print_eval('Iteration %d\tReward %f'%(step, 
            avg_return.result().numpy()))
        if saver:
          print_log('Saving evaluated policy...', v=True)
          saver.save(os.path.join(policy_dir, str(step)))
        if checkpointer:
          print_log('Saving checkpoint...', v=True)
          checkpointer.save(step)
        print_log('Done.', v=True)

  except:
    print_log('Catched exception:', v=True)
    print_exc()
    try:
      for action in exception_actions:
        action()
    except:
      print_exc()

  finally:
    # Save a checkpoint and clean before exiting
    if checkpointer:
      print_log('Saving checkpoint.', v=True)
      checkpointer.save(agent.train_step_counter)
    del(saver, checkpointer, eval_driver, avg_return,
        collect_driver, replay_iter, replay_buffer)
    try:
      for action in finally_actions:
        action()
    except:
      print_exc()


@gin.configurable
def ddqn(
  directory='.',
  register_env=register.stack_env,
  num_parallel_envs=1,
  q_network=networks.SiamQNetwork,
  learning_rate=0.00001,
  target_update_period=10000,
  save_policies=False,
  plot=False,
  log_to_file=False
):
  # Create the directory if it doesn't exist
  if not os.path.isdir(directory):
    os.makedirs(directory)
  ckpt_dir = os.path.join(directory, 'checkpoint')
  if save_policies:
    policy_dir = os.path.join(directory, 'policy')  
  else:
    policy_dir = None

  log_file = os.path.join(directory, 'train.log') if \
    log_to_file else sys.stdout
  eval_file = os.path.join(directory, 'eval.log')

  env_id = register_env()
  # Load an environment for training and other for evaluation
  if num_parallel_envs > 1:
    constructors = [lambda: suite_gym.load(env_id)]*num_parallel_envs
    train_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(constructors))
    eval_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(constructors))
  else:
    train_env = tf_py_environment.TFPyEnvironment(
        suite_gym.load(env_id))
    eval_env = tf_py_environment.TFPyEnvironment(
        suite_gym.load(env_id))

  # Create a Q network for the environment specs
  q_net = q_network(train_env.observation_spec(), 
      train_env.action_spec())
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
  train(agent, train_env, eval_env=eval_env, 
    ckpt_dir=ckpt_dir, policy_dir=policy_dir, 
    eval_file=eval_file, log_file=log_file
  )
  if plot:
    # Plot the evolution of the policy evaluations
    plot.from_log(eval_file, os.path.join(directory, 'plot.png'), show=True)

