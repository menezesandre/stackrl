#!/home/andre/Desktop/Siam-RL/.venv/bin/python3
"""Generates random objects for the StackEnv.
  Args:
    -n: number of objects. (default: 5000)
    -d: directory to save the generated objects. (default: siamrl/envs/data/generated)
    --split: fraction of the objects to be used in evaluation environments. (default: 0.1)
    --seed: seed to the generator. (default: None)
"""
import sys
from numpy import random
from siamrl.envs import data
from siamrl.envs.data import generator

args = ['-n', '-d', '--split', '--seed']

if __name__=='__main__':
# Parse arguments
  if '-n' in sys.argv:
    try:
      n = int(sys.argv[
        sys.argv.index('-n')+1
      ])
      assert n > 0
    except IndexError:
      raise ValueError('No value provided for argument -n (number of generated objects).')
    except (ValueError, AssertionError):
      raise ValueError("Invalid value '{}' for argument -n (number of generated objects). Must be a positive integer.".format(
        sys.argv[sys.argv.index('-n')+1]))
  else:
    n = 5000
  if '-d' in sys.argv:
    try:
      directory = sys.argv[
        sys.argv.index('-d')+1
      ]
      assert directory not in args
    except (IndexError, AssertionError):
      raise ValueError('No value provided for argument -d (directory).')
  else:
    directory = data.path('generated')
  if '--seed' in sys.argv:
    try:
      seed = int(sys.argv[
        sys.argv.index('--seed')+1
      ])
      assert seed >= 0 and seed < 2**32
    except IndexError:
      raise ValueError('No value provided for argument --seed.')
    except (ValueError, AssertionError):
      raise ValueError("Invalid value '{}' for argument --seed . Must be an integer between 0 and 2**32-1.".format(
        sys.argv[sys.argv.index('--seed')+1]))
  else:
    seed = None
  if '--split' in sys.argv:
    try:
      split = float(sys.argv[
        sys.argv.index('--split')+1
      ])
      assert split >= 0 and split < 1
    except IndexError:
      raise ValueError('No value provided for argument --split.')
    except (ValueError, AssertionError):
      raise ValueError("Invalid value '{}' for argument --split. Must be a float between 0 and 1.".format(
        sys.argv[sys.argv.index('--split')+1]))
  else:
    split = 0.1

  n_test = int(n*split)
  n_train = n - n_test
  seed_test = seed if seed is None else seed+1
  generator.from_icosphere(n=n_train, directory=directory, name='train', seed=seed)
  generator.from_icosphere(n=n_test, directory=directory, name='test', seed=seed_test)
  