from gym import register
from siamrl.envs import data
from siamrl.envs.data import getDataPath
from siamrl.envs import stack

MAX_EPISODE_STEPS = 50

"""
Environments for visualization
---------------------------------------
"""
register(
    id ='CubeStack-v0',
    entry_point='siamrl.envs.stack:CubeStackEnv',
    max_episode_steps = MAX_EPISODE_STEPS,
    kwargs = {'num_objects': MAX_EPISODE_STEPS,
              'gui': True,
              'info': True}
)

register(
    id ='BrickStack-v0',
    entry_point='siamrl.envs.stack:GeneratedStackEnv',
    max_episode_steps = MAX_EPISODE_STEPS,
    kwargs = {'model_name': 'b',
              'num_objects': MAX_EPISODE_STEPS,
              'gui': True,
              'info': True}
)

register(
    id ='RockStack-v0',
    entry_point='siamrl.envs.stack:GeneratedStackEnv',
    max_episode_steps = MAX_EPISODE_STEPS,
    kwargs = {'model_name': 'i',
              'num_objects': MAX_EPISODE_STEPS,
              'gui': True,
              'info': True}
)

"""
Environment for training
------------------------
P is the maximum value of the settle penalty function and is 
  also used as the drop penalty value (a very long time to settle
  is never worst than a droped object).
H is the number of simulation steps that corresponds to half of
  the maximum settle penalty.
S is the factor by which the total reward is multiplied before
  being returned

This values are based on average baseline results: H is close
to the average simulation steps per environment step for the 
best (lowest value) method (CCOEFF). P is double the average 
step increment of the [max/mean] elevation. With this values,
average step reward should be close to zero for the baselines.
Positive reward means an improvement over baseline. S is set to
make the rewards order 1

HEIGHT_REWARD = lambda x: 1.
EXP_SETTLE_FUNC = True

H = 120
# Scale the penalty and the total reward according to the 
# reward mode
if HEIGHT_REWARD is None:
  P = 1.
elif HEIGHT_REWARD == 'mean':
  P = 0.0025
elif HEIGHT_REWARD == 'max':
  P = 0.0056
else:
  P = 1.
S = 1./P

if EXP_SETTLE_FUNC:
  SETTLE_FUNC = lambda x: P*(1-2**(-x/H))
  #time ~ 1.8507048749597744e-07
else:
  SETTLE_FUNC = lambda x: P*(x/(x+H))
  #time ~ 7.933870666723427e-08

register(
    id ='RockStack-v1',
    entry_point='siamrl.envs.stack:GeneratedStackEnv',
    max_episode_steps = MAX_EPISODE_STEPS,
    kwargs = {'model_name': 'ic',
              'num_objects': MAX_EPISODE_STEPS,
              'state_reward': 'avg_occ',#HEIGHT_REWARD,
              'differential_reward': True,
              'settle_penalty': None,#SETTLE_FUNC,
              'drop_penalty': 0.,#P,
              'reward_scale': 100.}#S}
)
"""