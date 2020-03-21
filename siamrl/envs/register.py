import gym
from siamrl import envs
import gin

@gin.configurable
def stack_env(
  model_name='ic',
  base_size=[0.4375, 0.4375],
  resolution=2**(-9),
  time_step=1./240,
  sim_steps=None,
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
  seed=None,
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
      kwargs={
        'model_name': model_name,
        'base_size': base_size,
        'resolution': resolution,
        'time_step': time_step,
        'sim_steps': sim_steps,
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
        'seed': seed,
        'dtype': dtype
      }
  )
  return new_id

@gin.configurable
def goal_env(
  model_name='ic',
  base_size=[0.4375, 0.4375],
  resolution=2**(-9),
  num_objects=50,
  goal_size=None,
  gui=False,
  reward_scale=1.,
  seed=None,
  dtype='float32'
):
  # Assert there are URDFs for the given name
  assert len(envs.data.getGeneratedURDF(model_name)) > 0
  ids = [env.id for env in gym.envs.registry.all() if 
      model_name.upper() in env.id]
  i = 0
  while model_name.upper()+'Goal-v%d'%i in ids:
    i +=1
  new_id = model_name.upper()+'Goal-v%d'%i
  gym.register(
      id=new_id,
      entry_point='siamrl.envs.stack:GoalEnv',
      max_episode_steps=num_objects,
      kwargs={
        'model_name': model_name,
        'base_size': base_size,
        'resolution': resolution,
        'num_objects': num_objects,
        'goal_size': goal_size,
        'gui': gui,
        'reward_scale': reward_scale,
        'seed': seed,
        'dtype': dtype
      }
  )
  return new_id
