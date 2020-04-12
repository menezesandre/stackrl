import sys, os
from siamrl.envs import stack
from siamrl.train import Training


if __name__=='__main__':
  if '-d' in sys.argv:
    try:
      directory = sys.argv[
        sys.argv.index('-d')+1
      ]
      assert directory[0] != '-'
    except IndexError, AssertionError:
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
    except ValueError, AssertionError:
      raise ValueError('Invalid value for argument -n (number of iterations).')

  else:
    num_iter = None
        
  print(directory, num_iter)