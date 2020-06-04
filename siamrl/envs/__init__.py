from siamrl.envs import data
from siamrl.envs import stack
from siamrl.envs.utils import make, assert_registered

# Register Stack-v0
stack.register(
  use_gui=True, 
  positions_weight=1.,
  flat_action=False
)