from gym.envs.registration import register

register(
    id ='Rocks-v0',
    entry_point='siamrl.envs:RocksEnv'
)
