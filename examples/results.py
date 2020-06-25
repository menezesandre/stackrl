import sys

import siamrl

if __name__ == '__main__':
  path = sys.argv[1]

  siamrl.utils.plot_results(path)