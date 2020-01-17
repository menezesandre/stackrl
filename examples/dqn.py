import tensorflow as tf
import siamrl
import tf_agents

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from siamrl.networks import SiamQNetwork

num_iterations = 32

initial_collect_steps = 32
collect_steps_per_iteration = 1
replay_buffer_max_length = 32

batch_size = 1
learning_rate = 1e-3
log_interval = 1

num_eval_episodes = 1
eval_interval = 50

if __name__ == '__main__':

  train_env = tf_py_environment.TFPyEnvironment(suite_gym.load('RockStack-v1'))
  eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load('RockStack-v1'))

  q_net = SiamQNetwork(train_env.observation_spec(), 
      train_env.action_spec())

  optimizer = tf.keras.optimizers.Adam(learning_rate)

  train_step_counter = tf.Variable(0)

  agent = dqn_agent.DqnAgent(
      train_env.time_step_spec(),
      train_env.action_spec(),
      q_network=q_net,
      optimizer=optimizer,
      td_errors_loss_fn=common.element_wise_squared_loss,
      train_step_counter=train_step_counter)

  agent.initialize()

  eval_policy = agent.policy
  collect_policy = agent.collect_policy

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=agent.collect_data_spec,
      batch_size=train_env.batch_size,
      max_length=replay_buffer_max_length)

  replay_observer = [replay_buffer.add_batch]

  dataset = replay_buffer.as_dataset(num_parallel_calls=1,
      sample_batch_size=batch_size, num_steps=2)#prefetch(3)
    
  iterator = iter(dataset)

  train_metrics = [
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.EnvironmentSteps(),
      tf_metrics.AverageReturnMetric(),
  ]

  driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=collect_steps_per_iteration)

  # (Optional) Optimize by wrapping some of the code in a graph using TF function.
  #agent.train = common.function(agent.train)
  # Reset the train step
  agent.train_step_counter.assign(0)

  avg_return = tf_metrics.AverageReturnMetric()
  eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      eval_env, agent.policy, observers=[avg_return], 
      num_episodes=num_eval_episodes)

  # Evaluate the agent's policy once before training.
  eval_driver.run()
  returns = [avg_return.result().numpy()]

  episode_len = []

  final_time_step, policy_state = driver.run()
  for i in range(initial_collect_steps):
    final_time_step, _ = driver.run(final_time_step, 
        policy_state)

  for i in range(num_iterations):
    print(i)
    final_time_step, _ = driver.run(final_time_step, policy_state)
    experience, _ = next(iterator)
    print('before')
    train_loss = agent.train(experience=experience) #memory leak
    step = agent.train_step_counter.numpy()
    print('after')

    if step % log_interval == 0:
      print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
      eval_driver.run()
      returns.append(avg_return.result().numpy())
      print('step = {0}: Average Return = {1}'.format(step, results[-1]))


  plt.plot(episode_len)
  plt.show()

