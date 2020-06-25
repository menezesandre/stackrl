import siamrl

if __name__ == '__main__':
  kwargs = {
    'episode_length':16,
    # urdfs:'train',
    # object_max_dimension:0.125,
    # sim_time_step:1/60.,
    # gravity:9.8,
    # num_sim_steps:None,
    # velocity_threshold:0.01,
    # smooth_placing : True,
    # observable_size_ratio:4,
    # resolution_factor:5,
    'max_z':0.5,
    'goal_size_ratio':.25,
    'occupation_ratio_weight':10.,
    # occupation_ratio_param:False,
    # positions_weight:0.,
    # positions_param:0.,
    # n_steps_weight:0.,
    # n_steps_param:0.,
    # contact_points_weight:0.,
    # contact_points_param:0.,
    # differential:True,
    # flat_action:True,
    'seed':11,
    'dtype':'uint8',
  }

  env_id = siamrl.envs.stack.register(**kwargs)
  
  siamrl.baselines.test(env_id, method='ccoeff', verbose=True, gui=True)