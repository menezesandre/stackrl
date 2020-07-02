import os
import sys

import gin

import siamrl

# Default args
directory,config_file,curriculum,env = '.','config.gin',False,None
# Parse arguments
argv = sys.argv[:0:-1]
while argv:
  arg=argv.pop()
  if arg == '-d':
    directory = argv.pop()
  elif arg == '--config':
    config_file = argv.pop()
  elif arg == '--curriculum':
    curriculum = True
  elif arg == '-e':
    env = argv.pop()
# Parse config file
if config_file:
  try:
    gin.parse_config_file(os.path.join(directory, config_file))
  except OSError:
    gin.parse_config_file(config_file)
# If env is not provided, register it with args binded from config_file
if not env:
  if curriculum:
    env = siamrl.envs.stack.curriculum()
  else:
    env = siamrl.envs.stack.register()
# Run training
train = siamrl.Training(env, directory=directory)
train.run()
