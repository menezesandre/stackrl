from collections import defaultdict
from datetime import datetime
import glob
import os
import time
import types

import gym
try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None
import numpy as np
import pybullet as pb
try:
  from scipy import ndimage
except ImportError:
  ndimage = None

import siamrl
from siamrl import envs
try:
  from siamrl import heatmap
except ImportError:
  heatmap = None

MAX_LINE_VISUALIZE = 3

def clean(path=None, extensions='.npz'):
  """Remove all files with extension accessible from root path.
  Returns:
    list off all the removed files."""
  path = path or siamrl.datapath('test')
  removed = []

  if os.path.isfile(path):
    if path.endswith('.npz'):
      os.remove(path)
      return removed.append(path)
  elif os.path.isdir(path):
    for p in glob.glob(os.path.join(path,'*')):
      removed += clean(p)

  return removed

def write(fname, force=False, **kwargs):
  """Write the data in kwargs to fname.
  If a file with matching header already exists, the data is appended. if
    one of the columns is 'keys', lines with matching keys are replaced.
    In that case, if one of the columns is 'priority', a line with higher
    priority is not replaced.

  Args:
    fname: name of the file.
    force: whether to overwrite an existing file if keys don't match the
      header. If False, throws ValueError in that case.
    kwargs: keys are the header and values are iterables of the same length
      to fill the columns. If any of the values is scalar, it is broadcast to
      the same length as the other values.

  """
  # Check size to broadcast scalars
  for v in kwargs.values():
    if not np.isscalar(v):
      size = len(v)
      break
  # Unify header stiles (e.g. 'action_value' is turned to 'ActionValue')
  # str[:1].upper()+str[1:] is used instead of str.capitalize() to avoid lowering
  # all remaining characters. 
  kwargs = {
    ''.join([i[:1].upper()+i[1:] for i in k.split('_')]):
    np.array([v]*size) if np.isscalar(v) else np.array(v) 
    for k,v in kwargs.items()
  }

  if os.path.isfile(fname):
    try:
      with open(fname) as f:
        # Check if kwargs match header of existing file
        header_line = f.readline()
        header = header_line[:-1].split(',') # don't include last char (\n)
        if set(header) != set(kwargs):
          raise ValueError("kwargs don't match the existing file's header.")

        # Only rewrite if there is at least one line to replace
        rewrite = False
        # New data lines to discard (in case old ones have higher priority)
        new_lines_to_discard = []

        if 'Keys' in header:
          # Check for repeated keys to discard old lines.
          ik = header.index('Keys')
          if 'Priority' in header:
            ip = header.index('Priority')
          else:
            ip = None

          lines_to_keep = [header_line]
          for line in f:
            sline = line[:-1].split(',')
            if sline[ik] in kwargs['Keys']:
              if ip is not None:
                # Line of the new data corresponding to this key
                i = np.where(kwargs['Keys']==sline[ik])[0][0]
                # Compare priorities
                if float(sline[ip]) > kwargs['Priority'][i]:
                  lines_to_keep.append(line)
                  new_lines_to_discard.append(i)
                else:
                  rewrite = True
              else:
                rewrite = True
            else:
              lines_to_keep.append(line)

      if rewrite:
        # Rewrite the file without the repeated lines
        with open(fname, 'w') as f:
          for line in lines_to_keep:
            f.write(line)

      # Reorder kwargs to match the header
      kwargs = {k:kwargs[k] for k in header}
      # Append new data
      with open(fname, 'a') as f:
        for i, values in enumerate(zip(*tuple(kwargs.values()))):
          if i not in new_lines_to_discard:
            values = [str(v) for v in values]
            f.write(','.join(values)+'\n')

      return

    except ValueError as e:
      # If force is True, supress the exception and overwrite.
      if not force:
        raise e

  # Create directory if necessary
  if not os.path.isdir(os.path.dirname(fname)):
    os.makedirs(os.path.dirname(fname))

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
      keys: names of the policies;
      actions: best actions, with shape (num_policies, total_num_steps,2);
      values: predicted values, with shape 
        (num_policies, total_num_steps, num_actions);
      rewards: step rewards, with shape (num_policies, num_steps);
      episode_bounds: indexes of the steps corresponding to an episode 
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

  # Shape of the value maps
  vshape = (
    env.observation_space[0].shape[-3]-env.observation_space[1].shape[-3]+1,
    env.observation_space[0].shape[-2]-env.observation_space[1].shape[-2]+1
  )
  
  multi_action = (
    isinstance(env.action_space, gym.spaces.Tuple) and 
    len(env.observation_space[0].shape) == 4
  )

  # Auxiliary variables
  num_policies = len(policies)
  total_num_steps = num_policies*num_steps
  keys = np.array(list(policies.keys()))
  episode_bounds = []
  # Initialize variables to store results
  rewards = np.zeros((num_policies,num_steps), dtype='float32')
  values = np.zeros(
    (num_policies, total_num_steps, np.prod(vshape)), 
    dtype='float32'
  )
  actions = np.zeros(
    (num_policies, total_num_steps, 2), 
    dtype='uint8' if max(vshape) < 2**8 else 'uint16',
  )

  if visualize:
    if plt is None:
      raise ImportError("matplotlib must be instaled to run 'run' with visualize=True.")
    # Visualize at most two lines of MAX_LINE_VISUALIZE value maps simultaneously
    n_visualize = min(num_policies, 2*MAX_LINE_VISUALIZE)
    if n_visualize > MAX_LINE_VISUALIZE:
      n_line_visualize = MAX_LINE_VISUALIZE
      _, axs = plt.subplots(
        2,1+n_line_visualize, 
      )
    else:
      n_line_visualize = n_visualize
      _, axs = plt.subplots(
        2,1+n_visualize, 
        gridspec_kw={'height_ratios':[4, 1]},
      )
    for axline in axs:
      for ax in axline:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
          # Prepare spines to mark policy being used
          spine.set_color((0.993248,0.906157,0.143936, 1.))
          spine.set_linewidth(3)
          spine.set_visible(False)

  # Run tests
  itime = time.time()
  for i in range(total_num_steps):
    if i % num_steps == 0:
      index = i//num_steps
      if verbose:
        print(keys[index].capitalize())
      # Reseed and reset environment
      env.seed(seed)
      o=env.reset()
      episode_bounds.append(i)
      num_done = 0

    for j,k in enumerate(keys):
      a,v = policies[k](o)
      if j == index:
        # Action from current policy, to be performed on environment
        action = a
      if multi_action:
        ai,aj = a
        actions[j,i] = np.unravel_index(aj, vshape)
        values[j,i] = v[ai]
      else:
        actions[j,i] = np.unravel_index(a, vshape)
        values[j,i] = v

    if visualize:
      rgb = env.render(mode='rgb_array')
      # Show observation
      for axl, img, title in zip(axs, rgb, ('Overhead view', 'Object view')):
        # Clear axis and remove ticks
        axl[0].cla()
        axl[0].set_xticks([])
        axl[0].set_yticks([])

        axl[0].imshow(img)
        axl[0].set_title(title)

      min_j = np.clip(index - n_visualize//2, 0, num_policies - n_visualize)
      max_j = min_j + n_visualize
      # Show value maps
      for k,j in enumerate(range(min_j, max_j)):
        ax = axs[k//MAX_LINE_VISUALIZE][1 + k%MAX_LINE_VISUALIZE]
        # Clear axis and remove ticks
        ax.cla()
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(values[j,i].reshape(vshape))
        ax.set_title(keys[j])
        if j == index:
          # Store axis of current policy to highlight
          current_ax = ax

      # Highlight current policy
      for spine in current_ax.spines.values():
        spine.set_visible(True)

      plt.pause(max(1e-12, sleep-(time.time()-itime)))

      for spine in current_ax.spines.values():
        spine.set_visible(False)
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
          rewards[index, episode_bounds[-1]%num_steps:i%num_steps+1].sum(),
        ))
      episode_bounds.append(i+1)
      o=env.reset()
  
  episode_bounds.append(total_num_steps)
  episode_bounds = np.array(
    episode_bounds, 
    dtype='uint16' if total_num_steps < 2**16 else 'uint32'
  )
  # Remove repeated episode starts 
  # (happens when an episode end coincides with the end of a policy's num_steps)
  episode_bounds = np.unique(episode_bounds)

  if visualize:
    plt.close()

  return {
    'keys': keys,
    'actions': actions, 
    'values': values, 
    'rewards': rewards, 
    'episode_bounds': episode_bounds,
  }

def analyse(
  keys, 
  actions, 
  values, 
  rewards, 
  episode_bounds, 
  show=False, 
  save=None, 
  dirname='.'
):
  # Set save status
  if save is None:
    save = not show and plt is not None

  num_policies = keys.size
  num_steps = rewards.shape[-1]
  total_num_steps = values.shape[1]

  # Check if episode length is constant
  if np.all(
    episode_bounds == 
    np.arange(0, total_num_steps+1, total_num_steps/(episode_bounds.size - 1))
  ):
    episode_length = int(total_num_steps/(episode_bounds.size - 1))
    # Episode returns
    returns = rewards.reshape((num_policies, -1, episode_length)).sum(axis=-1)
  else:
    episode_length = None
    # Compute episode returns with known rewards and episode boundaries
    returns = [list() for _ in range(keys.size)]
    for i in range(len(episode_bounds)-1):
      start,end = episode_bounds[i:i+2]
      delta = end-start
      j = start//num_steps
      start = start%num_steps
      returns[j].append(rewards[j,start:start+delta].sum())
    returns = np.array(returns)

  # Returns distribution
  returns_mean = returns.mean(axis=-1)
  returns_std = returns.std(axis=-1)

  # Action values (max value for each step)
  action_values = values.max(axis=-1)

  action_value = action_values.mean(axis=-1)
  action_value_std = action_values.std(axis=-1)

  #Plots
  if save or show:
    if plt is None:
      raise ImportError("matplotlib must be installed to run analyse with show=True or save=True.")
    if not os.path.isdir(dirname):
      os.makedirs(dirname)
    # Plot returns distribution
    plt.errorbar(keys, returns_mean, yerr=(returns_mean-returns.min(axis=-1), returns.max(axis=-1)-returns_mean), fmt='none', ecolor='b', elinewidth=8, alpha=0.25, label='Range')
    plt.errorbar(keys, returns_mean, yerr=returns_std, fmt='bo', capsize=4, label='Mean +/- std dev')
    plt.xlabel('Policy')
    plt.ylabel('Return')
    plt.legend(loc='best')
    if save:
      plt.savefig(os.path.join(dirname, 'returns.pdf'))
      plt.savefig(os.path.join(dirname, 'returns.png'))
    if show:
      plt.show()
    else:
      plt.close()

    # Rewards distribution
    rewards_mean = rewards.mean(axis=-1)
    plt.errorbar(keys, rewards_mean, yerr=(rewards_mean-rewards.min(axis=-1), rewards.max(axis=-1)-rewards_mean), fmt='none', ecolor='b', elinewidth=8, alpha=0.25, label='Range')
    plt.errorbar(keys, rewards_mean, yerr=rewards.std(axis=-1), fmt='bo', capsize=4, label='Mean +/- std dev')
    plt.xlabel('Policy')
    plt.ylabel('Reward')
    plt.legend(loc='best')
    if save:
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
        plt.legend(loc='best')
        plt.title(keys[i])
        if save:
          plt.savefig(os.path.join(dirname, 'rewards_{}.pdf'.format(keys[i])))
          plt.savefig(os.path.join(dirname, 'rewards_{}.png'.format(keys[i])))
        if show:
          plt.show()
        else:
          plt.close()

      # Plot means of all policies
      if num_policies > 1:
        for i in range(num_policies):
          if ndimage is not None:
            plt.plot(
              range(1,episode_length+1), 
              ndimage.gaussian_filter1d(
                rewards_mean[i], 
                episode_length*2**(-4), 
                mode='nearest'
              ),
              label=keys[i],
            )
          else:
            plt.plot(
              range(1,episode_length+1), 
              rewards_mean[i],
              label=keys[i],
            )

        plt.legend(loc='best')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        if save:
          plt.savefig(os.path.join(dirname, 'rewards_all.pdf'))
          plt.savefig(os.path.join(dirname, 'rewards_all.png'))
        if show:
          plt.show()
        else:
          plt.close()
      
      del(rewards_mean,rewards_max,rewards_min,rewards_std)
    else:
      del(rewards_mean)
    del(rewards)

    if num_policies > 1:
      # Histogram of best policy by episode
      plt.hist(keys[returns.argmax(axis=0)], bins='auto')
      plt.xlabel('Policy')
      plt.ylabel('# episodes with best return')
      if save:
        plt.savefig(os.path.join(dirname, 'best_hist.pdf'))
        plt.savefig(os.path.join(dirname, 'best_hist.png'))
      if show:
        plt.show()
      else:
        plt.close()
      del(returns)

      # Distance between actions
      actions = actions.astype('int32')
      actions_distance = np.linalg.norm(
        np.expand_dims(actions, axis=0) - np.expand_dims(actions, axis=1),
        axis=-1,
      )
      im,_ = heatmap.heatmap(actions_distance.mean(axis=-1), keys, keys, cbarlabel='Mean distance (pixels)')
      heatmap.annotate_heatmap(im)
      if save:
        plt.savefig(os.path.join(dirname, 'distance_heatmap.pdf'))
        plt.savefig(os.path.join(dirname, 'distance_heatmap.png'))
      if show:
        plt.show()
      else:
        plt.close()

      # Histogram of distances for each pair of policies
      for i in range(num_policies-1):
        for j in range(i+1, num_policies):
          n = np.sort((keys[i], keys[j]))
          plt.hist(actions_distance[i,j], bins='auto')
          plt.xlabel('Distance between {} and {}'.format(*n))
          plt.ylabel('Frequency')
          
          if save:
            plt.savefig(os.path.join(dirname, 'distance_hist_{}_{}.pdf'.format(*n)))
            plt.savefig(os.path.join(dirname, 'distance_hist_{}_{}.png'.format(*n)))
          if show:
            plt.show()
          else:
            plt.close()
      del(actions, actions_distance)

      # Correlation between value functions <--- Killed here
      corrcoefs = np.corrcoef(values.reshape((num_policies, -1)))
      im,_ = heatmap.heatmap(corrcoefs, keys, keys, cbarlabel='Correlation coefficients')
      heatmap.annotate_heatmap(im)
      if save:
        plt.savefig(os.path.join(dirname, 'correlation_heatmap.pdf'))
        plt.savefig(os.path.join(dirname, 'correlation_heatmap.png'))
      if show:
        plt.show()
      else:
        plt.close()
      del(corrcoefs)

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

      im,_ = heatmap.heatmap(overlap_above_mean, keys, keys, cbarlabel='Overlap of values above mean')
      heatmap.annotate_heatmap(im)
      if save:
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

      im,_ = heatmap.heatmap(overlap_above_std, keys, keys, cbarlabel='Overlap of values one std dev above mean')
      heatmap.annotate_heatmap(im)
      if save:
        plt.savefig(os.path.join(dirname, 'overlap_std_heatmap.pdf'))
        plt.savefig(os.path.join(dirname, 'overlap_std_heatmap.png'))
      if show:
        plt.show()
      else:
        plt.close()
      del(values_mean,values_std,values_above_mean,values_above_std,overlap_above_mean,overlap_above_std)

    for i in range(num_policies):
      plt.hist(values[i].ravel(), bins='auto')
      plt.xlabel('Values (estimated by {})'.format(keys[i]))
      plt.ylabel('Frequency')
      if save:
        plt.savefig(os.path.join(dirname, 'value_hist_{}.pdf'.format(keys[i])))
        plt.savefig(os.path.join(dirname, 'value_hist_{}.png'.format(keys[i])))
      if show:
        plt.show()
      else:
        plt.close()

      plt.hist(action_values[i], bins='auto')
      plt.xlabel('Action values (estimated by {})'.format(keys[i]))
      plt.ylabel('Frequency')
      if save:
        plt.savefig(os.path.join(dirname, 'action_value_hist_{}.pdf'.format(keys[i])))
        plt.savefig(os.path.join(dirname, 'action_value_hist_{}.png'.format(keys[i])))
      if show:
        plt.show()
      else:
        plt.close()
    del(values)

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
        plt.ylabel('Value (estimated by {})'.format(keys[i]))
        plt.legend(loc='best')
        if save:
          plt.savefig(os.path.join(dirname, 'action_values_{}.pdf'.format(keys[i])))
          plt.savefig(os.path.join(dirname, 'action_values_{}.png'.format(keys[i])))
        if show:
          plt.show()
        else:
          plt.close()

  return {
    'keys':keys,
    'return':returns_mean,
    'return_std':returns_std,
    'action_value':action_value,
    'action_value_std':action_value_std,
  }

def test(
  policies={},
  num_steps=1000,
  verbose=True,
  visualize=False,
  sleep=None,
  show=False,
  save=None,
  seed=11,
  **kwargs,
):
  """ Run tests and analyse results.

  Args:
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
    save: whether to save collectd data and plots. If None, data are saved
      if verbose is False and plots are saved if show is False. If a string
      is provided, use it as the save directory.
    seed: seed for the environment.
    kwargs: passed to envs.make to instantiate the environment.
  """
  # Set save status
  if save is not None:
    save_results, save_plots = bool(save), bool(save)
  else:
    save_results, save_plots = not verbose, not show

  if isinstance(save, str):
    basedirname = save
  else:
    basedirname = siamrl.datapath('test')
  timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
  # Only save results if policies are named  
  save_results = save_results and isinstance(policies, dict)

  env = envs.make(**kwargs, unwrapped=True)
  envpath = envs.make(**kwargs, as_path=True)

  if isinstance(env, types.GeneratorType):
    env_gen = env
    envpath_gen = envpath

    # Get first item from curriculum to use as the x axis in the plots
    xkey, xvalues = next(iter(kwargs['curriculum'].items()))
    if verbose:
      xiter = iter(xvalues)

    ydict = defaultdict(lambda: list())
    ystddict = defaultdict(lambda: list())

    for (env,_), (envpath,_) in zip(env_gen, envpath_gen):
      # Environment directory
      envdirname = os.path.join(
        basedirname,
        envpath,
      )
      # Directory for this experiment
      dirname = os.path.join(
        envdirname,
        # Folder from seed and num_steps
        '{}-{}'.format(seed, num_steps),
        # Folder from time stamp
        timestamp,
      )

      if verbose:
        print('{} = {}'.format(xkey, next(xiter)))

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
        if not os.path.isdir(dirname):
          os.makedirs(dirname)
        np.savez_compressed(os.path.join(dirname, 'data'), **data)

      data = analyse(
        **data,
        show=show, 
        save=save_plots,
        dirname=dirname,
      )

      # Update the results on the environment directory. Experiments with
      # a larger number of steps are prioritized
      write(os.path.join(envdirname,'results.csv'), **data, priority=num_steps)

      if verbose:
        print('Average returns (+/- std dev):')
        for n,r,rd in zip(data['keys'],data['return'],data['return_std']):
          print('  {}: {} (+/-{})'.format(n,r,rd))

      for n,r,rd in zip(data['keys'],data['return'],data['return_std']):
        ydict[n].append(r)
        ystddict[n].append(rd)

    if show or save_plots:
      if xkey == 'urdfs' and all(isinstance(v,int) for v in xvalues):
        # In this special case, label the x axis as irregularity
        xlabel = 'Irregularity (%)'
      else:
        xlabel = xkey

      dirname = os.path.join(basedirname, timestamp)
      if not os.path.isfile(dirname):
        os.makedirs(dirname)

      # Plot evolution of each policy's return with env parameter
      for key, yvalues in ydict.items():
        plt.errorbar(xvalues, yvalues, yerr=ystddict[key], fmt='bo', capsize=4)
        plt.xlabel(xlabel)
        plt.ylabel('Return')
        plt.title(key)
        if save_plots:
          plt.savefig(os.path.join(dirname, 'returns_{}_{}.pdf'.format(xkey, key)))
          plt.savefig(os.path.join(dirname, 'returns_{}_{}.png'.format(xkey, key)))
        if show:
          plt.show()
        else:
          plt.close()
      if len(policies) > 1:
        # Plot evolution of all policies' return with env parameter
        for key, yvalues in ydict.items():
          plt.plot(xvalues, yvalues, label=key)
        plt.xlabel(xkey)
        plt.ylabel('Return')
        plt.legend(loc='best')
        if save_plots:
          plt.savefig(os.path.join(dirname, 'returns_{}.pdf'.format(xkey)))
          plt.savefig(os.path.join(dirname, 'returns_{}.png'.format(xkey)))
        if show:
          plt.show()
        else:
          plt.close()
  else:
    # Environment directory
    basedirname = os.path.join(
      basedirname,
      envpath,
    )
    # Directory for this experiment
    dirname = os.path.join(
      basedirname,
      # Folder from seed and num_steps
      '{}-{}'.format(seed, num_steps),
      # Folder from time stamp
      timestamp,
    )

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
      if not os.path.isdir(dirname):
        os.makedirs(dirname)
      np.savez_compressed(os.path.join(dirname, 'data'), **data)

    data = analyse(
      **data,
      show=show, 
      save=save_plots,
      dirname=dirname,
    )

    # Update the results on the environment directory. Experiments with
    # a larger number of steps are prioritized
    write(os.path.join(basedirname,'results.csv'), **data, priority=num_steps)

    if verbose:
      print('Average returns (+/- std dev):')
      for n,r,rd in zip(data['keys'],data['return'],data['return_std']):
        print('  {}: {} (+/-{})'.format(n,r,rd))


def analyse_npz(fname, show=False, save=None):
  """Run analyse with data from a previous experience."""
  return analyse(**np.load(fname), show=show, save=save, dirname=os.path.dirname(fname))
