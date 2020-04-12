from siamrl.envs import data
from siamrl.envs import stack

# Register Stack-v0
stack.register(
    use_gui=True, 
    positions_weight=1.,
    flat_action=False)