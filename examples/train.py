from siamrl.utils import train
import gin
from glob import glob
import sys, os

if len(sys.argv) > 1:
  config_dir = sys.argv[1]
else:
  config_dir = '.'
config_list = sorted(glob(os.path.join(config_dir,'*.gin')))
for config_file in config_list:
  gin.parse_config_file(config_file)
  train.ddqn()
