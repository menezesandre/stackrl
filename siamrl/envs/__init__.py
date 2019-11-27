from gym import register
from siamrl.envs.data import getDataPath, getRockFiles
from siamrl.envs.stack import MAX_EPISODE_STEPS as STACK_MAX_EPISODE_STEPS

register(
    id ='CubeStack-v0',
    entry_point='siamrl.envs.stack:CubeStackEnvFlatAction',
    max_episode_steps = STACK_MAX_EPISODE_STEPS,
)

register(
    id ='CubeStack-human-v0',
    entry_point='siamrl.envs.stack:CubeStackEnv',
    max_episode_steps = STACK_MAX_EPISODE_STEPS,
    kwargs = {'gui': True}
)

register(
    id ='RockStack-v0',
    entry_point='siamrl.envs.stack:RockStackEnvFlatAction',
    max_episode_steps = STACK_MAX_EPISODE_STEPS,
)

register(
    id ='RockStack-human-v0',
    entry_point='siamrl.envs.stack:RockStackEnv',
    max_episode_steps = STACK_MAX_EPISODE_STEPS,
    kwargs = {'gui': True}
)
