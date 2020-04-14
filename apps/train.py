#!/home/andre/Desktop/Siam-RL/.venv/bin/python3
"""Train a (pseudo) siamese DQN agent on the stack environment.
  Args:
    -d: directory for log and checkpoints. (default: .)
    -n: number of iterations. (default: None)
    --config: name of the configuration file. (default: config.gin)
    --curriculum: set to use curriculum training.
    --finish: set to stop training when curriculum is complete (only used is curriculum is set).
"""
import sys, os, warnings
from siamrl.envs import stack
from siamrl.train import Training, CurriculumTraining
import gin
import gin.tf

args = ['-d', '-n', '--config', '--curriculum', '--finish']

if __name__=='__main__':
# Parse arguments
  if '-d' in sys.argv:
    try:
      directory = sys.argv[
        sys.argv.index('-d')+1
      ]
      assert directory not in args
    except (IndexError, AssertionError):
      raise ValueError('No value provided for argument -d (directory).')
  else:
    directory = '.'
  if '-n' in sys.argv:
    try:
      num_iter = int(sys.argv[
        sys.argv.index('-n')+1
      ])
      assert num_iter > 0
    except IndexError:
      raise ValueError('No value provided for argument -n (number of iterations).')
    except (ValueError, AssertionError):
      raise ValueError("Invalid value '{}' for argument -n (number of iterations). Must be a positive integer.".format(
        sys.argv[sys.argv.index('-n')+1]))
  else:
    num_iter = None
  if '--config' in sys.argv:
    try:
      config_file = sys.argv[
        sys.argv.index('--config')+1
      ]
      assert config_file not in args
    except (IndexError, AssertionError):
      raise ValueError('No value provided for argument -c (config file).')
    try:
      gin.parse_config_file(config_file)
    except OSError:
      try:
        gin.parse_config_file(os.path.join(directory, config_file))
      except OSError:
        raise ValueError("Invalid value '{}' for argument --config (config file). File doesn't exist in present directory".format(config_file)+(" nor in '{}'.".format(directory) if directory != '.' else "."))
  else:
    try:
      gin.parse_config_file(os.path.join(directory, 'config.gin'))
    except:
      sys.stderr.write('Warning: No configuraiton file parsed. Using all parameter defaults.\n')
  curriculum = '--curriculum' in sys.argv
  if curriculum:
    finish = '--finish' in sys.argv
  else:
    finish = None

# Run training
  if curriculum:
    env_ids = stack.curriculum()
    train = CurriculumTraining(env_ids, directory=directory)
  else:
    env_id = stack.register()
    train = Training(env_id, directory=directory)

  kwargs = {}
  if num_iter:
    kwargs['max_num_iterations'] = num_iter
  if finish is not None:
    kwargs['finish_when_complete'] = finish

  train.run(**kwargs)
  