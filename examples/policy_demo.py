import glob, os
import siamrl, tf_agents
from siamrl import baselines #CCoeffPolicy, GradCorrPolicy
from tf_agents.environments import suite_gym
from tf_agents.policies import random_py_policy
from tf_agents.policies import fixed_policy
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy
from tf_agents.metrics import py_metrics, py_metric
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment

if __name__=='__main__':
  weights = glob.glob('weights/*.h5')

  py_env = suite_gym.load('RockStack-v0')
  tf_env = tf_py_environment.TFPyEnvironment(py_env)
  q_net = []
#  for weight in weights:
#    q_net.append(SiamQNetwork(env.observation_spec(), 
#      env.action_spec()))
#    q_net.net.load_weights(weight)

  py_policies = [
    random_py_policy.RandomPyPolicy(
      time_step_spec=py_env.time_step_spec(),
      action_spec=py_env.action_spec()),
    baselines.CCoeffPolicy(
      time_step_spec=py_env.time_step_spec(),
      action_spec=py_env.action_spec()),
    baselines.GradCorrPolicy(
      time_step_spec=py_env.time_step_spec(),
      action_spec=py_env.action_spec())]
#  tf_policies= [
#    fixed_policy.FixedPolicy(
#      time_step_spec=tf_env.time_step_spec(),
#      action_spec=tf_env.action_spec()),
#    greedy_policy.GreedyPolicy(q_policy.QPolicy(
#      env.time_step_spec(), env.action_spec(), q_net)))
#    ]

  for policy in py_policies:
    metric = py_metrics.AverageReturnMetric()
    driver = py_driver.PyDriver(
        py_env, policy, [metric], max_episodes=3)

    initial_time_step = env.reset()
    final_time_step, _ = driver.run(initial_time_step)
    print(policy.__class__.__name__)
    print('- Average Return: ', metric.result())

