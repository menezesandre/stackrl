from siamrl.envs import data
from siamrl.envs import stack
from siamrl.envs.utils import make, assert_registered

# Register Stack-v0
stack.register(
  use_gui=True, 
  positions_weight=1.,
  flat_action=False
)

stack.register(
  max_z=0.5,
  goal_size_ratio=.25,
  occupation_ratio_weight=10.,
  dtype='uint8',
)