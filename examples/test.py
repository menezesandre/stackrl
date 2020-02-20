import siamrl
from siamrl import utils

utils.train_ddqn_on_stack_env(
  time_step=1./60, 
  resolution=2**(-8), 
  use_goal=True, 
  position_reward=0.5, 
  dtype='float16'
)
