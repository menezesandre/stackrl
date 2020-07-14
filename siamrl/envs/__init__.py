from siamrl.envs import data
from siamrl.envs import stack
from siamrl.envs.utils import make, assert_registered

# Stack-v0
stack.register(
  use_gui=True, 
  positions_weight=1.,
  flat_action=False
)

# Stack-v1
stack.register(
  max_z=0.5,
  goal_size_ratio=.25,
  occupation_ratio_weight=10.,
  dtype='uint8',
)

# Stack-v2
stack.register(
  max_z=0.5,
  goal_size_ratio=.25,
  occupation_ratio_weight=10.,
  dtype='uint8',
  ordering_freedom=True,
  orientation_freedom=3,
  entry_point='test',
)