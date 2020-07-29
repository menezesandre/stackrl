from siamrl.envs import data
from siamrl.envs import stack
from siamrl.envs.utils import make, assert_registered

# Stack-v0
stack.register(
  urdfs='3?',
  reward_params=2,
  dtype='uint8',
)

# Stack-v1
stack.register(
  urdfs='3?',
  reward_params=2,
  dtype='uint8',
  ordering_freedom=True,
  orientation_freedom=3,
  entry_point='test',
)