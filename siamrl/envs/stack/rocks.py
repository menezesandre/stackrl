from siamrl.envs.stack.base import BaseStackEnv, BaseStackEnvFlatAction
from siamrl.envs import getRockFiles
class RockStackEnv(BaseStackEnv):
  metadata = {'render.modes': ['human', 'depth_array'],
              '__init__.modes': ['all', 'convex']}
  def __init__(self, mode='convex', **kwargs):
    self.mode = mode
    super(RockStackEnv, self).__init__(**kwargs)
    
  def _get_urdf_list(self):
    return getRockFiles(mode=self.mode)

class RockStackEnvFlatAction(BaseStackEnvFlatAction):
  metadata = {'render.modes': ['human', 'depth_array'],
              '__init__.modes': ['all', 'convex']}
  def __init__(self, mode='convex', **kwargs):
    self.mode = mode
    super(RockStackEnv, self).__init__(**kwargs)
    
  def _get_urdf_list(self):
    return getRockFiles(mode=self.mode)
