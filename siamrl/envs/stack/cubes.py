from siamrl.envs.stack.base import BaseStackEnv, BaseStackEnvFlatAction

class CubeStackEnv(BaseStackEnv):
  def _get_urdf_list(self):
    return ['cube.urdf']

class CubeStackEnvFlatAction(BaseStackEnvFlatAction):
  def _get_urdf_list(self):
    return ['cube.urdf']

