from siamrl.utils import train
import gin

gin.parse_config_file('config.gin')
train.ddqn()
