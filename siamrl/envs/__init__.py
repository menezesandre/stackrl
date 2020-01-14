from gym import register
from siamrl.envs import data
from siamrl.envs.data import getDataPath
from siamrl.envs import stack

MAX_EPISODE_STEPS = 50

register(
    id ='CubeStack-v0',
    entry_point='siamrl.envs.stack:CubeStackEnv',
    max_episode_steps = MAX_EPISODE_STEPS,
    kwargs = {'num_objects': MAX_EPISODE_STEPS,
              'gui': True}
)

register(
    id ='BrickStack-v0',
    entry_point='siamrl.envs.stack:GeneratedStackEnv',
    max_episode_steps = MAX_EPISODE_STEPS,
    kwargs = {'model_name': 'b',
              'num_objects': MAX_EPISODE_STEPS,
              'gui': True}
)

register(
    id ='RockStack-v0',
    entry_point='siamrl.envs.stack:GeneratedStackEnv',
    max_episode_steps = MAX_EPISODE_STEPS,
    kwargs = {'model_name': 'i',
              'num_objects': MAX_EPISODE_STEPS,
              'gui': True}
)

"""
P is the maximum value of the settle penalty function and is 
  also used as the drop penalty value
H is the number of simulation steps that corresponds to half
  of the maximum settle penalty penalty
"""
P = 0.01
H = 100

register(
    id ='RockStack-v1',
    entry_point='siamrl.envs.stack:GeneratedStackEnv',
    max_episode_steps = MAX_EPISODE_STEPS,
    kwargs = {'model_name': 'ic',
              'num_objects': MAX_EPISODE_STEPS,
              'reward_mode': 'avg',
              'drop_penalty': P,
              'settle_penalty': lambda x: P*x/(x+H)}
)
