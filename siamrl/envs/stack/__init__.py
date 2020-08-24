from siamrl.envs.stack.env import register

# Stack-v0
register(
  urdfs='[4-9]?',
  reward_params=2,
  dtype='uint8',
)

# Stack-v1
register(
  urdfs='[4-9]?',
  reward_params=2,
  dtype='uint8',
  entry_point='started',
)

# Stack-v2
register(
  urdfs='[4-9]?',
  reward_params=2,
  dtype='uint8',
  entry_point='test',
)
