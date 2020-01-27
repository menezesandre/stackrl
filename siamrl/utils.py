import sys, os, traceback
import matplotlib.pyplot as plt

import tensorflow as tf

import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from siamrl import networks

def initial_collect(env, buffer, steps, policy=None):
  if policy is None:
    policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
        env.action_spec())

  driver = dynamic_step_driver.DynamicStepDriver(
      env, policy, observers=[buffer.add_batch], num_steps=steps)
  driver.run()

def train(agent,
          train_env,
          eval_env,
          num_iterations = 20000,
          initial_collect_steps = 1000 ,
          collect_steps_per_iteration = 1,
          replay_buffer_max_length = 100000,
          batch_size = 64,
          log_interval = 200,
          log_file=sys.stdout,
          num_eval_episodes = 10,
          eval_interval = 1000,
          eval_file=sys.stdout,
          ckpt_dir='./checkpoint',
          policy_dir='./policy'
          ):
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

  avg_return = tf_metrics.AverageReturnMetric()
  eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      eval_env, agent.policy, observers=[avg_return], 
      num_episodes=num_eval_episodes)

  if ckpt_dir is not None:
    checkpointer = common.Checkpointer(ckpt_dir=ckpt_dir, 
        max_to_keep=1, agent=agent, replay_buffer=replay_buffer)
  else:
    checkpointer = None

  if policy_dir is not None:
    saver = policy_saver.PolicySaver(agent.policy)
  else:
    saver = None

  if checkpointer is None or not checkpointer.checkpoint_exists:
    # Reset the step counter
    agent.train_step_counter.assign(0)
    # Evaluate the agent's policy once before training.
    eval_driver.run()
    eval_file.write('Iteration %d\tReward %f\n'%(0, 
        avg_return.result().numpy()))
    # Collect some transitions before start training
    initial_collect(train_env, replay_buffer, initial_collect_steps)

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
        log_file.write('Iteration %d\tLoss %f\n'%(step, loss_info.loss))

      if step % eval_interval == 0:
        avg_return.reset()
        eval_driver.run()
        eval_file.write('Iteration %d\tReward %f\n'%(step, 
            avg_return.result().numpy()))
        if saver:
          saver.save(os.path.join(policy_dir, str(step)))
        if checkpointer:
          checkpointer.save(step)
  except:
    traceback.print_exc()

  # Save a checkpoint before exiting
  if checkpointer:
    checkpointer.save(step)

def train_ddqn(env_name='RockStack-v1',
               net=networks.SiamQNetwork,
               learning_rate=0.00001,
               target_update_period=10000,
               directory='.',
               **kwargs
               ):
  # Load an environment for training and other for evaluation
  train_env = tf_py_environment.TFPyEnvironment(
      suite_gym.load(env_name))
  eval_env = tf_py_environment.TFPyEnvironment(
      suite_gym.load(env_name))

  # Cretate a Q network for the environment specs
  q_net = net(train_env.observation_spec(), 
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
      train_step_counter=train_step_counter)

  # Create the directory if it doesn't exist
  if not os.path.isdir(directory):
    os.makedirs(directory)
  ckpt_dir = os.path.join(directory, 'checkpoint')
  policy_dir = os.path.join(directory, 'policy')
  eval_file_name = os.path.join(directory, 'eval.log')

  # Train the agent
  with open(eval_file_name, 'a') as f:
    train(agent, train_env, eval_env, ckpt_dir=checkpoint_dir,
        policy_dir=policy_dir, plot_dir=directory, eval_file=f,
        **kwargs)
  # Plot the evolution of the policy evaluations
  plot_log(eval_file_name, os.path.join(directory, 'plot.png'), show=True)

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
  