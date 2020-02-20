from siamrl.envs.utils.camera import Camera, ElevationCamera
# If trimesh is not instaled, propagate the exception to the 
# module's function calls
try:
  from siamrl.envs.utils import model_generator as generate
except ImportError as e:
  import types
  exception = e
  def f(**kwargs):
    raise exception
  generate = types.SimpleNamespace(fromBox=f, 
                                   fromIcosphere=f,
                                   random_scale_matrix=f)
