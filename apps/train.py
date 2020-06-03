#!/home/andre/Desktop/Siam-RL/.venv/bin/python3
"""Train a (pseudo) siamese DQN agent on the stack environment.
  Args:
    -d: directory for log and checkpoints. (default: .)
    -n: number of iterations. (default: None)
    --config: name of the configuration file. (default: config.gin)
    --curriculum: set to use curriculum training.
    --stop: set to stop training when curriculum is complete (only used is curriculum is set).
"""
import sys, os, warnings
from siamrl.envs import stack
from siamrl.train import Training
import gin
import gin.tf

args = ['-d', '-n', '--config', '--curriculum', '--stop']
directory = '.'
num_iter = None
config_file = 'config.gin'
curriculum = False
stop = None

if __name__=='__main__':
# Parse arguments
  argv = sys.argv[:0:-1]
  while argv:
    arg=argv.pop()
    if arg=='-d':
      try:
        directory = argv.pop()
        assert directory not in args
      except (IndexError, AssertionError):
        raise ValueError('No value provided for argument -d (directory).')
    elif arg=='-n':
      try:
        num_iter = int(argv.pop())
        assert num_iter > 0
      except IndexError:
        raise ValueError('No value provided for argument -n (number of iterations).')
      except (ValueError, AssertionError):
        raise ValueError("Invalid value '{}' for argument -n (number of iterations). Must be a positive integer.".format(
        sys.argv[sys.argv.index('-n')+1]))
    elif arg=='--config':
      try:
        config_file = argv.pop()
        assert config_file not in args
      except (IndexError, AssertionError):
        raise ValueError('No value provided for argument -c (config file).')
    elif arg == '--curriculum':
      curriculum = True
    elif arg =='--stop':
      stop = True
# Parse config file
  try:
    gin.parse_config_file(os.path.join(directory, config_file))
  except OSError:
    try:
      gin.parse_config_file(config_file)
    except OSError:
      sys.stderr.write('Warning: No configuraiton file found. Using all parameter defaults.\n')

# Run training
  if curriculum:
    env = stack.curriculum()
  else:
    env = stack.register()

  train = Training(env, directory=directory)

  kwargs = {}
  if num_iter:
    kwargs['max_num_iterations'] = num_iter
  if stop:
    kwargs['stop_when_complete'] = stop

  train.run(**kwargs)
  





