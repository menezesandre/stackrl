import siamrl, tf_agents
from siamrl.baselines import CCoeffPolicy, GradCorrPolicy
from tf_agents.environments import suite_gym
from tf_agents.eval import metric_utils

if __name__ == '__main__':
  metrics = []
  env = suite_gym.load('RockStack-v0')
  policy = GradCorrPolicy(env.time_step_spec(), env.action_spec(), normed=True)
  metric_utils.compute(metrics, env, policy)
