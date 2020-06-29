import os
import sys

import gin
import numpy as np

from siamrl import baselines
from siamrl import envs

try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None

def read_csv(fname, columns=None):
  with open(fname) as f:
    line = f.readline()
    if line.endswith('\n'):
      line = line[:-1]
    keys = line.split(',')

  if columns is not None:
    for c in columns:
      if c not in keys:
        raise ValueError(
          "No {} column in {}.".format(c, fname))

  return {
    key:value for key, value in zip(
      keys, 
      np.loadtxt(fname, delimiter=',', skiprows=1, unpack=True)
    ) if columns is None or key in columns
  }

def plot(fname, x_key, y_keys, split=None, baselines=None, show=False, legend='Train', save_as=None):
  if plt is None:
    raise ImportError("matplotlib must be installed to run plot.")

  if isinstance(y_keys, str):
    y_keys = [y_keys]

  data = read_csv(fname, [x_key]+y_keys)

  x = data[x_key]
  ys = [data[key] for key in y_keys]

  fig, axs = plt.subplots(len(ys),1,sharex=True)
  if len(ys) == 1:
    # To be consistent for any number of targets
    axs = (axs,)

  if split and os.path.isfile(split):
    data = np.loadtxt(
      split,
      delimiter=',',
      skiprows=1,
      unpack=True,
    )
    x_splits = np.atleast_1d(data[0])
    valid = np.atleast_1d(data[1])

    split = [0]
    i = 0
    for x_split, v in zip(x_splits, valid):
      for i in range(i,len(x)):
        if x[i] > x_split:
          if v:
            split.append(i)
          else:
            if x[0] != 0:
              split = [i]            
          break
    split.append(len(x))

    split_x = [x[split[i]:split[i+1]+1] for i in range(len(split)-1)]
    split_ys = [
      [y[split[i]:split[i+1]+1] for i in range(len(split)-1)] 
      for y in ys
    ]

        
    for ax, split_y in zip(axs, split_ys):
      for xi,yi in zip(split_x, split_y):
        ax.plot(xi,yi)
    legend = ['{} part {}'.format(legend, i) for i in range(len(split_x))]
  else:
    for ax, y in zip(axs, ys):
      ax.plot(x, y)
    legend = [legend]

  if baselines:
    for key in baselines:
      axs[-1].plot([x[0], x[-1]],[baselines[key]]*2)
      legend.append(key.capitalize())

  if len(legend) > 1:
    plt.legend(legend, loc='best')

  for ax, key, y in zip(axs, y_keys, ys):
    ax.set_ylabel(key)
    ylim = ax.get_ylim()
    _mean = np.mean(y)
    _std = np.std(y)
    if ylim[0] >= 0:
      ymin = ylim[0]
    else:
      ymin = min(0, max(ylim[0], _mean - 10*_std))
    if ylim[1] <= 0:
      ymax = ylim[1]
    else:
      ymax = max(0,min(ylim[1], _mean + 10*_std))
    ax.set_ylim(ymin,ymax)

  axs[-1].set_xlabel(x_key)

  if save_as is None:
    if not show:
      path, name = os.path.split(fname)
      name = name.split('.')[0]
      if path and path != '.':
        path, pref = os.path.split(path)
        name = '{}_{}'.format(pref, name)
      for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(path,'plots',ext,'{}.{}'.format(name,ext)))
  else:
    plt.savefig(save_as)
  if show:
    plt.show()
  else:
    plt.close()

def plot_value(path, show=False):
  data = read_csv(
    os.path.join(path, 'eval.csv'),
    ['Iter','MeanValue','StdValue','MinValue','MaxValue'],
  )

  plt.plot(
    data['Iter'], data['MeanValue'], '-b',
    data['Iter'], data['MeanValue'] + data['StdValue'], '--b',
    data['Iter'], data['MaxValue'], ':b',
    data['Iter'], data['MeanValue'] - data['StdValue'], '--b',
    data['Iter'], data['MinValue'], ':b',
  )
  plt.legend(['mean', '+/- std dev', 'max/min'])
  plt.xlabel('Iter')
  plt.ylabel('Q value')

  path, name = os.path.split(path)
  if name and name != '.':
    name += '_q_value'
  else:
    name = 'q_value'
  for ext in ['png','pdf']:
    plt.savefig(os.path.join(path,'plots',ext,'{}.{}'.format(name,ext)))
  if show:
    plt.show()
  else:
    plt.close()

def plot_train(path, **kwargs):
  return plot(
    fname=os.path.join(path, 'train.csv'),
    x_key='Iter',
    y_keys=['Loss', 'Reward'],
    split=os.path.join(path, 'curriculum.csv'),
    **kwargs,
  )

def plot_eval(path, value=False, **kwargs):
  try:
    gin.parse_config_file(os.path.join(path, 'config.gin'))
    env_id = envs.stack.register()
  except OSError:
    env_id = None

  if env_id:
    bpath = os.path.join(
      os.path.dirname(__file__),
      '..',
      'data',
      'baselines',
      envs.utils.as_path(env_id),
    )
    bfile = os.path.join(
      bpath,
      'results',
    )
    if os.path.isfile(bfile):
      baselines = {}
      with open(bfile) as f:
        for line in f:
          line = line.split(':')
          baselines[line[0]] = float(line[1])
    else:
      baselines = baselines.test(env_id)
  else:
    baselines = None 

  y_keys = ['Value', 'Reward'] if value else ['Reward']
  return plot(
    fname=os.path.join(path, 'eval.csv'),
    x_key='Iter',
    y_keys=y_keys,
    split=os.path.join(path, 'curriculum.csv'),
    baselines=baselines,
    **kwargs,
  )

def plot_results(path, value=False):
  plot_train(path)
  plot_eval(path, value=value)
  if value:
    plot_value(path)

if __name__ == '__main__':
  # Default args
  paths, value = [], False
  # Parse arguments
  argv = sys.argv[:0:-1]
  while argv:
    arg=argv.pop()
    if arg == '-v':
      value = True
    else:
      paths.append(arg)

  for path in paths:
    plot_results(path, value=value)