from siamrl.envs import stack
from siamrl.train import Training
import gin

if __name__=='__main__':
  try:
    gin.parse_config_file('config.gin')
  except OSError:
    pass
    
  env_id = stack.register()
  train = Training(env_id)
  train.run()
