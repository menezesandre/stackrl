import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import siamrl, tf_agents
from siamrl import baselines #CCoeffPolicy, GradCorrPolicy
from tf_agents.environments import suite_gym
from tf_agents.policies import random_py_policy
from tf_agents.metrics import py_metrics, py_metric
from tf_agents.drivers import py_driver

if __name__=='__main__':
  env = suite_gym.load('RockStack-v1')
  policies = [
    random_py_policy.RandomPyPolicy(
      time_step_spec=env.time_step_spec(),
      action_spec=env.action_spec()),
    baselines.CCoeffPolicy(
      time_step_spec=env.time_step_spec(),
      action_spec=env.action_spec()),
    baselines.GradCorrPolicy(
      time_step_spec=env.time_step_spec(),
      action_spec=env.action_spec())]

  for policy in policies:
    metric = py_metrics.AverageReturnMetric()
    driver = py_driver.PyDriver(
        env, policy, [metric], max_episodes=3)

    initial_time_step = env.reset()
    final_time_step, _ = driver.run(initial_time_step)
    print(policy.__class__.__name__)
    print('- Average Return: ', metric.result())

"""
Return (32 objects, avg of 5):
RandomPyPolicy
- Average Return:  -7.459245
CCoeffPolicy
- Average Return:  8.345945
GradCorrPolicy
- Average Return:  1.0572164
"""