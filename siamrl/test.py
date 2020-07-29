from datetime import datetime
import os
import sys
import time

import gin
import gym
import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb

from siamrl import baselines
from siamrl import envs
from siamrl import load_policy
from siamrl import heatmap


MAX_N_VISUALIZE = 5

def write_raw(fname, **kwargs):
  """Store a set of named arrays in fname.
  
  Each line has the following format, where shape and flat data are coma 
    separated: <name>\t<dtype>\t<shape>\t<flat data>
  """
  with open(fname, 'w') as f:
    for name,value in kwargs.items():
      if not isinstance(value, np.ndarray):
        value = np.array(value)
      f.write('{}\t{}\t{}\t{}\n'.format(
        name,
        value.dtype,
        ','.join([str(i) for i in value.shape]),
        ','.join(value.ravel().astype(str)),
      ))

def read_raw(fname):
  """Parse named arrays writen by write_raw in fname."""
  returns = {}
  with open(fname) as f:
    for line in f:
      name,dtype,shape,data = line[:-1].split('\t')
      shape = tuple(int(i) for i in shape.split(','))
      data = np.array(data.split(','))
      returns[name] = data.astype(dtype).reshape(shape)
  return returns

def write_results(fname, force=False, **kwargs):
  """Write the data in kwargs to fname.
  If a file with matching header already exists, the data is appended.
  If one of the columns is 'names', lines with matching names are replaced.

  Args:
    fname: name of the file.
    force: whether to overwrite an existing file if keys don't match the
      header. If False, throws ValueError in that case.
    kwargs: keys are the header and values are iterables of the same length
      to fill the columns.

  """
  # Unify header stiles (e.g. returns_mean is turned to ReturnsMean)
  # str[:1].upper()+str[1:] is used instead of str.capitalize() to avoid lowering
  # all remaining characters. 
  kwargs = {
    ''.join([i[:1].upper()+i[1:] for i in k.split('_')]):v for k,v in kwargs.items()
  }

  if os.path.isfile(fname):
    try:
      with open(fname) as f:
        # Check if kwargs match header of existing file
        header_line = f.readline()
        header = header_line.split(',')
        if set(header) != set(kwargs):
          raise ValueError("kwargs don't match the existing file's header.")

        if 'Names' in header:
          # Check for repeated names to discard old lines.
          i = header.index('Names')
          lines_to_keep = [header_line]
          rewrite = False
          for line in f:
            if line.split(',')[i] in kwargs['Names']:
              # Only rewrite if there is at least one line to discard
              rewrite = True
            else:
              lines_to_keep.append(line)
        else:
          rewrite = False

      if rewrite:
        # Rewrite the file without the repeated lines
        with open(fname, 'w') as f:
          for line in lines_to_keep:
            f.write(line)

      # Reorder kwargs to match the header
      kwargs = {k:kwargs[k] for k in header}
      # Append new data
      with open(fname, 'a') as f:
        for values in zip(*tuple(kwargs.values())):
          values = [str(v) for v in values]
          f.write(','.join(values)+'\n')

      return

    except ValueError as e:
      # If force is True, supress the exception and overwrite.
      if not force:
        raise e

  with open(fname, 'w') as f:
    # Write header
    f.write(','.join(kwargs.keys())+'\n')
    # Write values
    for values in zip(*tuple(kwargs.values())):
      values = [str(v) for v in values]
      f.write(','.join(values)+'\n')

def run(
  env,
  policies={},
  num_steps=1024,
  verbose=True,
  visualize=False,
  sleep=None,
  seed=11,
):
  """Run a test to collect metrics from the policies.

  Args:
    env: instance of the environment to be used. Must implement the gym.Env
      interface.
    policies: policies to be used. Either a dict with policy names as keys
      and policy functions (callable) as values, an iterable with policy 
      functions, or a single policy function. Policy functions must return
      a tuple with best action and estimated action values.
    num_steps: number of environment steps to run with each policy.
    verbose: whether to print results as the test proceeds.
    visualize: whether to show a window with a visualization of the 
      observations and value maps of each policy.
    show: whether to show plots of the results on the end of the test.
    save: whether to save results and plots. If None, results are saved
      if verbose is False and plots are saved if show is False.
    sleep: time interval (in seconds, between 0 and 1) between steps. 
      Only used if visualize or gui are True.
    seed: seed for the environment.
  
  Returns:
    dictionary of arrays, with keys:
      names: names of the policies;
      actions: best actions, with shape (num_policies, total_num_steps,2);
      values: predicted values, with shape 
        (num_policies, total_num_steps, num_actions);
      rewards: step rewards, with shape (num_policies, num_steps);
      episode_starts: indexes of the steps corresponding to an episode 
        boundary, with shape (num_episodes+1,).
  """
  # Set policies dictionary
  if not isinstance(policies, dict):
    try:
      policies = {str(k):v for k,v in enumerate(policies)}
    except TypeError:
      policies = {'policy':policies}
  # Check all policies are callable
  for k,v in policies.items():
    if not callable(v):
      raise TypeError("Invalid type {} for element {} of argument policies. Must be callable.".format(type(v), k))
  # Set sleep time
  if sleep is None or sleep < 0:
    sleep = 0.5 if visualize else 0.

  if visualize:
    # Visualize at most the values from MAX_N_VISUALIZE policies simultaneously
    n_visualize = min(len(policies), MAX_N_VISUALIZE)
    _, axs = plt.subplots(
      2,1+n_visualize, 
      gridspec_kw={'height_ratios':[4, 1]}
    )
  # Shape of the value maps
  vshape = (
    env.observation_space[0].shape[0]-env.observation_space[1].shape[0]+1,
    env.observation_space[0].shape[1]-env.observation_space[1].shape[1]+1
  )
  # Auxiliary variables
  num_policies = len(policies)
  total_num_steps = num_policies*num_steps
  names = np.array(list(policies.keys()))
  episode_starts = []
  # Initialize variables to store results
  rewards = np.zeros((num_policies,num_steps))
  values = np.zeros((num_policies, total_num_steps, np.prod(vshape)))
  actions = np.zeros((num_policies, total_num_steps, 2))

  # RUN TESTS
  itime = time.time()
  for i in range(total_num_steps):
    if i % num_steps == 0:
      index = i//num_steps
      if verbose:
        print(names[index].capitalize())
      # Reseed and reset environment
      env.seed(seed)
      o=env.reset()
      episode_starts.append(i)
      num_done = 0

    for j,k in enumerate(names):
      a,v = policies[k](o)
      if j == index:
        # Action from current policy, to be performed on environment
        action = a
      actions[j,i] = np.unravel_index(a, vshape)
      values[j,i] = v

    if visualize:
      rgb = env.render(mode='rgb_array')
      # Show observation
      for j in range(2):
        axs[j][0].cla()
        axs[j][0].imshow(rgb[j])

      min_j = np.clip(index - n_visualize//2, 0, num_policies - n_visualize)
      max_j = min_j + n_visualize
      # Show value maps
      for k,j in enumerate(range(min_j, max_j)):
        axs[0][k+1].cla()
        axs[0][k+1].imshow(values[j,i].reshape(vshape))
        # Show values one standard deviation above mean
        threshold = values[j,i].mean() + values[j,i].std()
        axs[1][k+1].cla()
        axs[1][k+1].imshow(np.where(
          values[j,i]>threshold, 
          values[j,i], 
          threshold,
        ).reshape(vshape))
        # Label the policies
        axs[1][k+1].set_xlabel(names[j])

      # Label current policy
      axs[0][index+1].set_title('current policy')
        
      # Remove all axis ticks
      plt.xticks([])
      plt.yticks([])
      # Show figure and pause for the time left in the sleep period.
      plt.pause(max(1e-12, sleep-(time.time()-itime)))
    elif sleep:
      time.sleep(max(0, sleep-(time.time()-itime)))
    itime = time.time()

    o,r,d,_ = env.step(action)
    rewards[index,i%num_steps] = r

    if d:
      num_done += 1
      if verbose:
        print('  Episode #{}: Return {}'.format(
          num_done,
          rewards[index, episode_starts[-1]%num_steps:i%num_steps+1].sum(),
        ))
      episode_starts.append(i+1)
      o=env.reset()
  
  episode_starts.append(total_num_steps)
  # Remove repeated episode starts 
  # (happens when an episode end coincides with the end of a policy's num_steps)
  episode_starts = np.unique(episode_starts)

  if visualize:
    plt.close()

  return {
    'names': names,
    'actions': actions, 
    'values': values, 
    'rewards': rewards, 
    'episode_starts': episode_starts,
  }

def analyse(
  names, 
  actions, 
  values, 
  rewards, 
  episode_starts, 
  verbose=True, 
  show=False, 
  save_results=None, 
  save_plots=None, 
  dirname='.'
):
  # Set save status
  if save_results is None:
    save_results = not verbose
  if save_plots is None:
    save_plots = not show
  # # Time stamped folder
  # tdirname = os.path.join(dirname, datetime.now().strftime("%y%m%d-%H%M%S"))
  # os.makedirs(tdirname)

  num_policies = names.size
  num_steps = rewards.shape[-1]
  total_num_steps = values.shape[1]

  # Check if episode length is constant
  if np.all(
    episode_starts == 
    np.arange(0, total_num_steps+1, total_num_steps/(episode_starts.size - 1))
  ):
    episode_length = int(total_num_steps/(episode_starts.size - 1))
    # Episode returns
    returns = rewards.reshape((num_policies, -1, episode_length)).sum(axis=-1)
  else:
    episode_length = None
    # Compute episode returns with known rewards and episode boundaries
    returns = [list() for _ in range(names.size)]
    for i in range(len(episode_starts)-1):
      start,end = episode_starts[i:i+2]
      delta = end-start
      j = start//num_steps
      start = start%num_steps
      returns[j].append(rewards[j,start:start+delta].sum())
    returns = np.array(returns)

  # Returns distribution
  returns_mean = returns.mean(axis=-1)
  returns_std = returns.std(axis=-1)

  if verbose:
    print('Average returns (+/- std dev):')
    for n,r,rd in zip(names,returns_mean,returns_std):
      print('  {}: {} (+/-{})'.format(n,r,rd))

  # Plot returns distribution
  plt.errorbar(names, returns_mean, yerr=(returns_mean-returns.min(axis=-1), returns.max(axis=-1)-returns_mean), fmt='none', ecolor='b', elinewidth=8, alpha=0.25, label='Range')
  plt.errorbar(names, returns_mean, yerr=returns_std, fmt='bo', capsize=4, label='Mean +/- std dev')
  plt.xlabel('Policy')
  plt.ylabel('Return')
  plt.legend()
  if save_plots:
    plt.savefig(os.path.join(dirname, 'returns.pdf'))
    plt.savefig(os.path.join(dirname, 'returns.png'))
  if show:
    plt.show()
  else:
    plt.close()

  # Histogram of best policy by episode
  plt.hist(names[returns.argmax(axis=0)], bins='auto')
  plt.xlabel('Policy')
  plt.ylabel('# episodes with best return')
  if save_plots:
    plt.savefig(os.path.join(dirname, 'best_hist.pdf'))
    plt.savefig(os.path.join(dirname, 'best_hist.png'))
  if show:
    plt.show()
  else:
    plt.close()

  # Rewards distribution
  rewards_mean = rewards.mean(axis=-1)
  plt.errorbar(names, rewards_mean, yerr=(rewards_mean-rewards.min(axis=-1), rewards.max(axis=-1)-rewards_mean), fmt='none', ecolor='b', elinewidth=8, alpha=0.25, label='Range')
  plt.errorbar(names, rewards_mean, yerr=rewards.std(axis=-1), fmt='bo', capsize=4, label='Mean +/- std dev')
  plt.xlabel('Policy')
  plt.ylabel('Reward')
  plt.legend()
  if save_plots:
    plt.savefig(os.path.join(dirname, 'rewards.pdf'))
    plt.savefig(os.path.join(dirname, 'rewards.png'))
  if show:
    plt.show()
  else:
    plt.close()

  # Reward distribution allong episode
  if episode_length:
    rewards = rewards.reshape((num_policies, -1, episode_length))
    rewards_mean = rewards.mean(axis=1)
    rewards_std = rewards.std(axis=1)
    rewards_min = rewards.min(axis=1)
    rewards_max = rewards.max(axis=1)

    # Plot distribution for each policy
    for i in range(num_policies):
      plt.errorbar(range(1,episode_length+1), rewards_mean[i], yerr=(rewards_mean[i]-rewards_min[i], rewards_max[i]-rewards_mean[i]), fmt='none', ecolor='b', elinewidth=8, alpha=0.25, label='Range')
      plt.errorbar(range(1,episode_length+1), rewards_mean[i], yerr=rewards_std[i], fmt='bo', capsize=4, label='Mean +/- std dev')
      plt.xlabel('Step')
      plt.ylabel('Reward')
      plt.legend()
      plt.title(names[i])
      if save_plots:
        plt.savefig(os.path.join(dirname, 'rewards_{}.pdf'.format(names[i])))
        plt.savefig(os.path.join(dirname, 'rewards_{}.png'.format(names[i])))
      if show:
        plt.show()
      else:
        plt.close()

    # Plot means of all policies
    for i in range(num_policies):
      plt.plot(range(1,episode_length+1), rewards_mean[i], label=names[i])

    plt.legend()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    if save_plots:
      plt.savefig(os.path.join(dirname, 'rewards_all.pdf'))
      plt.savefig(os.path.join(dirname, 'rewards_all.png'))
    if show:
      plt.show()
    else:
      plt.close()

  # Distance between actions
  actions_distance = np.linalg.norm(
    np.expand_dims(actions, axis=0) - np.expand_dims(actions, axis=1),
    axis=-1,
  )
  im,_ = heatmap.heatmap(actions_distance.mean(axis=-1), names, names, cbarlabel='Mean distance (pixels)')
  heatmap.annotate_heatmap(im)
  if save_plots:
    plt.savefig(os.path.join(dirname, 'distance_heatmap.pdf'))
    plt.savefig(os.path.join(dirname, 'distance_heatmap.png'))
  if show:
    plt.show()
  else:
    plt.close()

  # Histogram of distances for each pair of policies
  for i in range(num_policies-1):
    for j in range(i+1, num_policies):
      n = np.sort((names[i], names[j]))
      plt.hist(actions_distance[i,j], bins='auto')
      plt.xlabel('Distance between {} and {}'.format(*n))
      plt.ylabel('Frequency')
      
      if save_plots:
        plt.savefig(os.path.join(dirname, 'distance_hist_{}_{}.pdf'.format(*n)))
        plt.savefig(os.path.join(dirname, 'distance_hist_{}_{}.png'.format(*n)))
      if show:
        plt.show()
      else:
        plt.close()

  # Correlation between value functions
  corrcoefs = np.corrcoef(values.reshape((num_policies, -1)))
  im,_ = heatmap.heatmap(corrcoefs, names, names, cbarlabel='Correlation coefficients')
  heatmap.annotate_heatmap(im)
  if save_plots:
    plt.savefig(os.path.join(dirname, 'correlation_heatmap.pdf'))
    plt.savefig(os.path.join(dirname, 'correlation_heatmap.png'))
  if show:
    plt.show()
  else:
    plt.close()

  # Stepwise values distribution
  values_mean = values.mean(axis=-1)
  values_std = values.std(axis=-1)
  # Overlap between values above mean for different functions
  values_above_mean = values > np.expand_dims(values_mean, axis=-1)
  overlap_above_mean = (
    # Intersection
    np.count_nonzero(np.logical_and(
      values_above_mean.reshape((1,num_policies,-1)),
      values_above_mean.reshape((num_policies,1,-1)),
    ), axis=-1) /
    # Union
    np.count_nonzero(np.logical_or(
      values_above_mean.reshape((1,num_policies,-1)),
      values_above_mean.reshape((num_policies,1,-1)),
    ), axis=-1)    
  )

  im,_ = heatmap.heatmap(overlap_above_mean, names, names, cbarlabel='Overlap of values above mean')
  heatmap.annotate_heatmap(im)
  if save_plots:
    plt.savefig(os.path.join(dirname, 'overlap_mean_heatmap.pdf'))
    plt.savefig(os.path.join(dirname, 'overlap_mean_heatmap.png'))
  if show:
    plt.show()
  else:
    plt.close()

  # Overlap between values one std deviation above mean for different functions  
  values_above_std = values > np.expand_dims(values_mean+values_std, axis=-1)
  overlap_above_std = (
    # Intersection
    np.count_nonzero(np.logical_and(
      values_above_std.reshape((1,num_policies,-1)),
      values_above_std.reshape((num_policies,1,-1)),
    ), axis=-1) /
    # Union
    np.count_nonzero(np.logical_or(
      values_above_std.reshape((1,num_policies,-1)),
      values_above_std.reshape((num_policies,1,-1)),
    ), axis=-1)    
  )

  im,_ = heatmap.heatmap(overlap_above_std, names, names, cbarlabel='Overlap of values one std dev above mean')
  heatmap.annotate_heatmap(im)
  if save_plots:
    plt.savefig(os.path.join(dirname, 'overlap_std_heatmap.pdf'))
    plt.savefig(os.path.join(dirname, 'overlap_std_heatmap.png'))
  if show:
    plt.show()
  else:
    plt.close()

  # Action values (max value for each step)
  action_values = values.max(axis=-1)

  action_value = action_values.mean(axis=-1)
  action_value_std = action_values.std(axis=-1)

  for i in range(num_policies):
    plt.hist(values[i].ravel(), bins='auto', density=True, label='All values')
    plt.xlabel('Values (estimated by {})'.format(names[i]))
    plt.ylabel('Frequency')
    plt.legend()
    if save_plots:
      plt.savefig(os.path.join(dirname, 'value_hist_{}.pdf'.format(names[i])))
      plt.savefig(os.path.join(dirname, 'value_hist_{}.png'.format(names[i])))
    if show:
      plt.show()
    else:
      plt.close()

    plt.hist(action_values[i], bins='auto', density=True, label='Action values')
    plt.xlabel('Action values (estimated by {})'.format(names[i]))
    plt.ylabel('Frequency')
    plt.legend()
    if save_plots:
      plt.savefig(os.path.join(dirname, 'action_value_hist_{}.pdf'.format(names[i])))
      plt.savefig(os.path.join(dirname, 'action_value_hist_{}.png'.format(names[i])))
    if show:
      plt.show()
    else:
      plt.close()

  # Action values distribution along episode
  if episode_length:
    action_values = action_values.reshape((num_policies, -1, episode_length))
    action_values_mean = action_values.mean(axis=1)
    action_values_std = action_values.std(axis=1)
    action_values_min = action_values.min(axis=1)
    action_values_max = action_values.max(axis=1)
  
    for i in range(num_policies):
      plt.errorbar(range(1,episode_length+1), action_values_mean[i], yerr=(action_values_mean[i]-action_values_min[i], action_values_max[i]-action_values_mean[i]), fmt='none', ecolor='b', elinewidth=8, alpha=0.25, label='Range')
      plt.errorbar(range(1,episode_length+1), action_values_mean[i], yerr=action_values_std[i], fmt='bo', capsize=4, label='Mean +/- std dev')
      plt.xlabel('Step')
      plt.ylabel('Value (estimated by {})'.format(names[i]))
      plt.legend()
      if save_plots:
        plt.savefig(os.path.join(dirname, 'action_values_{}.pdf'.format(names[i])))
        plt.savefig(os.path.join(dirname, 'action_values_{}.png'.format(names[i])))
      if show:
        plt.show()
      else:
        plt.close()

  # Save results
  if save_results:
    write_results(
      fname=os.path.join(dirname, 'results.csv'),
      names=names,
      return_=returns_mean,
      return_std=returns_std,
      action_value=action_value,
      action_value_std=action_value_std,
    )
    write_raw(
      fname=os.path.join(dirname, 'data'),
      actions_distance=actions_distance,
      corrcoefs=corrcoefs,
      overlap_above_mean=overlap_above_mean,
      overlap_above_std=overlap_above_std,
    )


def test(
  env,
  policies={},
  num_steps=1024,
  verbose=True,
  visualize=False,
  sleep=None,
  show=False,
  save=None,
  seed=11,
):
  """ Run tests and and analyse results.

  Args:
    env: instance of the environment to be used. Must implement the gym.Env
      interface.
    policies: policies to be used. Either a dict with policy names as keys
      and policy functions (callable) as values, an iterable with policy 
      functions, or a single policy function. Policy functions must return
      a tuple with best action and estimated action values. Note: results 
      are only saved if policy names are provided (i.e. policies is a dict).
    num_steps: number of environment steps to run with each policy.
    verbose: whether to print results as the test proceeds.
    visualize: whether to show a window with a visualization of the 
      observations and value maps of each policy.
    sleep: time interval (in seconds, geq 0) between steps. If None, defaults
      to 0.5 if visualize is True, 0. otherwise (no sleep).
    show: whether to show plots of the results on the end of the test.
    save: whether to save results and plots. If None, results are saved
      if verbose is False and plots are saved if show is False.
    seed: seed for the environment.
  """
  # Set save status
  if save is not None:
    save_results, save_plots = save, save
  else:
    save_results, save_plots = not verbose, not show
  # Only save results if policies are named  
  save_results = save_results and isinstance(policies, dict)

  # Results directory
  dirname = os.path.join(
    # Package root folder (Siam-RL/)
    os.path.dirname(os.path.dirname(__file__)),
    'data',
    'test',
    # Path from env id
    envs.utils.as_path(env.unwrapped.spec.id),
    # Path from seed and num_steps
    '{}-{}'.format(seed, num_steps),
    # Path from time stamp
    datetime.now().strftime("%y%m%d-%H%M%S"),
  )
  # Create directory if necessary
  if (save_results or save_plots) and not os.path.isdir(dirname):
    os.makedirs(dirname)
  # Raw data file
  data_fname = os.path.join(dirname, 'raw_data')

  # Check if results are already present. Tests are only run for 
  # policies that weren't tested on this setting before.
  # if (
  #   isinstance(policies, dict) and 
  #   os.path.isfile(data_fname)
  # ):
  #   previous_data = read_raw(data_fname)

  #   previous_indexes = []
  #   for i,k in enumerate(previous_data.get('names', [])):
  #     print(k, policies.keys())
  #     if k in policies:
  #       previous_indexes.append(i)
  #       policies.pop(k)
  #       if verbose:
  #         print('Reusing previous test data for {}.'.format(k))
    
  #   previous_indexes = np.array(previous_indexes)
  # else:
  #   previous_data, previous_indexes = None, None

  data = run(
    env=env,
    policies=policies,
    num_steps=num_steps,
    verbose=verbose,
    visualize=visualize,
    sleep=sleep,
    seed=seed,
  )
  if save_results:
    # Write raw results
    # new_data = {}
    # if previous_data:
    #   # Append results to previous data to rewrite the file
    #   for key in data:
    #     if data[key]:
    #       new_data[key] = np.concatenate((data[key], previous_data[key]))
    #     else:
    #       new_data
    # else:
    #   new_data = data
    write_raw(data_fname, **data)

  # if previous_indexes:
  #   # Join data from previous tests
  #   for key in data:
  #     data[key] = np.concatenate((data[key], previous_data[key][previous_indexes])) 
  
  analyse(
    **data,
    verbose=verbose, 
    show=show, 
    save_plots=save_plots, 
    save_results=save_results,
    dirname=dirname,
  )
  

# -----------------------------------------------------------------------

# def test_(
#   env_id, 
#   path='.',
#   iters=None,
#   num_steps=1024,
#   compare_with=None,
#   verbose=True,
#   visualize=False,
#   gui=False, 
#   sleep=0.5, 
#   seed=11
# ):
#   sleep = 1. if sleep > 1 else 0. if sleep < 0 else sleep

#   env = gym.make(env_id, use_gui=gui)
#   policies = load_policy(env.observation_space, path=path, iters=iters, value=True)
#   if not isinstance(policies, list):
#     policies = [policies]
#     iters = [iters]

#   n_policies = len(policies)

#   if compare_with:

#     if not isinstance(compare_with, list):
#       compare_with = [compare_with]

#     hmean_overlap = tuple(tuple([] for _ in policies) for _ in compare_with)
#     hstd_overlap = tuple(tuple([] for _ in policies) for _ in compare_with)
#     distance = tuple(tuple([] for _ in policies) for _ in compare_with)
#     correlation = tuple(tuple([] for _ in policies) for _ in compare_with)
      

#     for m in compare_with:
#       policies.append(baselines.Baseline(m, value=True))
#       iters.append(m)

#   if visualize:    
#     plot_all = len(policies) > 1 and len(policies) < 5
#     if plot_all:
#       fig, axs = plt.subplots(
#         2,1+len(policies), 
#         gridspec_kw={'height_ratios':[4, 1]}
#       )
#     else:
#       fig, axs = plt.subplots(
#         2,2, 
#         gridspec_kw={'height_ratios':[4, 1]}
#       )
#   v_shape = (
#     env.observation_space[0].shape[0]-env.observation_space[1].shape[0]+1,
#     env.observation_space[0].shape[1]-env.observation_space[1].shape[1]+1
#   )

#   for i in range(n_policies):
#     if verbose and iters[i]:
#       print('{}'.format(iters[i]))

#     tr = 0.
#     tv = 0.
#     ne = 0
#     env.seed(seed)
#     o = env.reset()

#     if gui:
#       pb.removeAllUserDebugItems()
#       pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
#       pb.resetDebugVisualizerCamera(.5, 90, -75, (0.25,0.25,0))
#       time.sleep(5*sleep)
  
#     for n in range(num_steps):
#       if visualize and plot_all or compare_with:
#         actions_values = [p(o) for p in policies]
#         a,v = actions_values[i]
        
#         actions = [np.unravel_index(action, v_shape) for action,_ in actions_values]
#         values = [(value.reshape(v_shape) - np.mean(value))/np.std(value) for _,value in actions_values]
#       else:
#         a,v = policies[i](o)

#       if compare_with:
#         for j, (baction,bvalue) in enumerate(zip(
#           actions[-len(compare_with):], 
#           values[-len(compare_with):], 
#         )):
#           for k, (action,value) in enumerate(zip(
#             actions[:-len(compare_with)], 
#             values[:-len(compare_with)], 
#           )):
#             # Overlap between values above mean
#             hmean_overlap[j][k].append(
#               np.count_nonzero(np.logical_and(bvalue>0,value>0)) /
#               np.count_nonzero(np.logical_or(bvalue>0,value>0))
#             )
#             # Overlap between values one standard deviation above mean
#             hstd_overlap[j][k].append(
#               np.count_nonzero(np.logical_and(bvalue>1,value>1)) /
#               np.count_nonzero(np.logical_or(bvalue>1,value>1))
#             )
#             # Euclidian distance (in pixels) between actions to compare
#             distance[j][k].append(np.linalg.norm(np.subtract(baction,action)))
#             # Correlation coefficient between values
#             correlation[j][k].append(np.corrcoef(bvalue.ravel(), value.ravel())[0,1])

#       if visualize:
#         o0, o1 = env.render(mode='rgb_array')
#         axs[0][0].cla()
#         axs[0][0].imshow(o0)
#         axs[1][0].cla()
#         axs[1][0].imshow(o1)
#         if plot_all:
#           for j,value in enumerate(values):
#             axs[0][j+1].cla()
#             axs[0][j+1].imshow(value)
#             axs[1][j+1].cla()
#             axs[1][j+1].imshow(np.where(value>1, value, 1))
#             if j == i:
#               axs[1][j+1].set_xlabel('Current policy')
#         else:
#           threshold = np.mean(v) + np.std(v)
#           value = v.reshape(v_shape)

#           axs[0][1].cla()
#           axs[0][1].imshow(value)
#           axs[1][1].cla()
#           axs[1][1].imshow(np.where(value>threshold, value, threshold))
          
#         # Remove all axis ticks
#         plt.xticks([])
#         plt.yticks([])
#         # Show figure and pause for the time left to in the sleep period.
#         plt.pause(sleep-(datetime.now().microsecond/1e6)%sleep)
#       elif gui:
#         time.sleep(sleep-(datetime.now().microsecond/1e6)%sleep)

#       o,r,d,_ = env.step(a)
#       tr += r
#       tv += v[a]
#       if d:
#         ne+=1
#         if verbose:
#           print('  Current average ({}): Reward {}, Value {}'.format(
#             ne,
#             tr/ne,
#             tv/n
#           ))
#           if compare_with:
#             for j in range(len(compare_with)):
#               print('  Compare with {}: Overlap {} {}, Distance {}, Correlation {}'.format(
#                 iters[n_policies+j],
#                 np.mean(hmean_overlap[j][i]),
#                 np.mean(hstd_overlap[j][i]),
#                 np.mean(distance[j][i]),
#                 np.mean(correlation[j][i]),
#               ))
#         o=env.reset()

#     if verbose:
#       print('Final average: Reward {}, Value {}'.format(tr/ne, tv/ne))
  
#   if visualize:
#     plt.close()
  
#   if compare_with:

#     for j in range(len(policies)-len(compare_with)):
#       for i in range(len(compare_with)):
#         print('{} with {}'.format(iters[j], iters[n_policies+i]))
#         print('  Overlap between regions with value higher than mean: avg {} (std {})'.format(
#           np.mean(hmean_overlap[i][j]),
#           np.std(hmean_overlap[i][j]),
#         ))
#         print('  Overlap between regions with value at least 1 std dev above mean: avg {} (std {})'.format(
#           np.mean(hstd_overlap[i][j]),
#           np.std(hstd_overlap[i][j]),
#         ))
#         print('  Correlation coefficients between value maps: avg {} (std {})'.format(
#           np.mean(correlation[i][j]),
#           np.std(correlation[i][j]),
#         ))
#         print('  Distance between actions: avg {} (std {})'.format(
#           np.mean(distance[i][j]),
#           np.std(distance[i][j]),
#         ))
#         plt.hist(distance[i][j], bins=16)
#         plt.xlabel('Distance')
#         plt.ylabel('Frequency')
#         plt.show()


if __name__ == '__main__':
  # Default args
  path,config_file,env = '.','config.gin',None
  # Parse arguments
  argv = sys.argv[:0:-1]
  kwargs = {}
  while argv:
    arg=argv.pop()
    if arg == '--config':
      config_file = argv.pop()
    elif arg == '-e':
      env = argv.pop()
    elif arg == '--iters':
      kwargs['iters'] = argv.pop().split(',')
    elif arg == '-n':
      kwargs['num_steps'] = int(argv.pop())
    elif arg == '--compare':
      kwargs['compare_with'] = argv.pop().split(',')
    elif arg == '-q':
      kwargs['verbose'] = False
    elif arg == '-v':
      kwargs['visualize'] = True
    elif arg == '--gui':
      kwargs['gui'] = True
    elif arg == '-t':
      kwargs['sleep'] = float(argv.pop())
    elif arg == '--seed':
      kwargs['seed'] = int(argv.pop())
    else:
      path = arg
  # If env is not provided, register it with args binded from config_file
  if not env:
    if config_file:
      try:
        gin.parse_config_file(os.path.join(path, config_file))
      except OSError:
        gin.parse_config_file(config_file)
    env = envs.stack.register()

  test(env, path, **kwargs)