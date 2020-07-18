import os
import sys

import gin
import numpy as np

import siamrl
from siamrl import envs

try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None
try:
  from scipy import ndimage
except ImportError:
  ndimage = None

def read_csv(fname, columns=None, reduction=None):
  if isinstance(fname, list):
    data_list = [read_csv(f, columns=columns) for f in fname]
    if columns is None:
      # Get columns that are common across all files
      columns = data_list[0].keys()
      for d in data_list[1:]:
        columns &= d.keys()
    # Get number of common lines
    length = min([len(d[columns[0]]) for d in data_list])
    # Get reduction function
    reduction = reduction or np.mean
    if isinstance(reduction, str):
      reduction = getattr(np, reduction)

    data = {}
    for key in columns:
      array = np.array([
        d[key][:length] for d in data_list
      ])
      
      data[key] = reduction(array, axis=0)
      data[key+'_std'] = np.std(array, axis=0)
    return data

  else:
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

def get_save_name(path, name=None, ext='png'):
  if isinstance(path, list):
    if name is None:
      name = os.path.commonprefix([os.path.split(p)[-1] for p in path])
      name = name.split('.')[0]
      path = [os.path.dirname(p) for p in path]
    path = os.path.commonprefix(path)
  else:
    if name is None:
      path, name = os.path.split(path)
      name = name.split('.')[0]

  if path and path != '.':
    path, pref = os.path.split(path)
    name = '{}_{}'.format(pref, name)
  return os.path.join(path,'plots',ext,'{}.{}'.format(name,ext))

def plot(fname, x_key, y_keys, smooth=0, split=None, baselines=None, show=False, legend='Train', save_as=None):
  if plt is None:
    raise ImportError("matplotlib must be installed to run plot.")

  if isinstance(y_keys, str):
    y_keys = [y_keys]

  data = read_csv(fname, [x_key]+y_keys)

  x = data[x_key]
  if smooth:
    if ndimage is None:
      raise ImportError("scipy must be installed to run plot with smooth!=0.")

    ys = [
      ndimage.gaussian_filter1d(
        data[key], 
        smooth,
        mode='nearest',
      ) for key in y_keys
    ]
    try:
      ys_std = [
        ndimage.gaussian_filter1d(
          data[key+'_std'], 
          smooth,
          mode='nearest',
        ) for key in y_keys
      ]
    except KeyError:
      ys_std = None
  else:
    ys = [data[key] for key in y_keys]
    try:
      ys_std = [data[key+'_std'] for key in y_keys]
    except KeyError:
      ys_std = None

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
      for i,(xi,yi) in enumerate(zip(split_x, split_y)):
        ax.plot(xi,yi, label='{} part {}'.format(legend,i))
    # legend = ['{} part {}'.format(legend, i) for i in range(len(split_x))]
  else:
    if ys_std is None:
      for ax, y in zip(axs, ys):
        ax.plot(x, y, label=legend)
    else:
      for ax, y, y_std in zip(axs, ys, ys_std):
        ax.plot(x, y, label=legend)
        ax.fill_between(x, y+y_std, y-y_std, alpha=0.25, label='Std deviation')

      
  if baselines:
    for key in baselines:
      axs[-1].plot([x[0], x[-1]],[baselines[key]]*2, label=key.capitalize())
      # legend.append(key.capitalize())
    i = np.argmax(ys[-1])
    axs[-1].annotate(
      'Best: {:.6}'.format(ys[-1][i]), 
      (x[i], ys[-1][i]),
      xytext=(.8, 1.05),
      textcoords='axes fraction',
      arrowprops={'arrowstyle':'simple'}
    )

  # if ys_std:
  #   legend.append('Std deviation')

  # if len(legend) > 1:
  #   plt.legend(legend, loc='best')
  if baselines or ys_std:
    plt.legend(loc='best')
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

  if save_as:
    plt.savefig(save_as)
  elif not show:
    for ext in ['png', 'pdf']:
      plt.savefig(get_save_name(fname,ext=ext))

  if show:
    plt.show()
  else:
    plt.close()

def plot_value(path, show=False, save_as=None):
  if isinstance(path, list):
    fname = [os.path.join(p, 'eval.csv') for p in path]
  else:
    fname = os.path.join(path, 'eval.csv')
  data = read_csv(
    fname,
    ['Iter','MeanValue','StdValue','MinValue','MaxValue'],
  )

  plt.plot(
    data['Iter'], data['MeanValue'], '-b')

  plt.fill_between(data['Iter'], data['MeanValue'] + data['StdValue'], data['MeanValue'] - data['StdValue'], facecolor='b', alpha=0.25)
  plt.fill_between(data['Iter'], data['MaxValue'], data['MinValue'], facecolor='b', alpha=0.125)
  plt.legend(['mean', 'std dev', 'range'])
  plt.xlabel('Iter')
  plt.ylabel('Q value')

  if save_as:
    plt.savefig(save_as)
  elif not show:
    for ext in ['png','pdf']:
      plt.savefig(get_save_name(path, name='q_value', ext=ext))
  if show:
    plt.show()
  else:
    plt.close()

def plot_train(path, **kwargs):
  if isinstance(path, list):
    fname = [os.path.join(p, 'train.csv') for p in path]
    split = None
  else:
    fname = os.path.join(path, 'train.csv')
    split = os.path.join(path, 'curriculum.csv')
  
  return plot(
    fname=fname,
    x_key='Iter',
    y_keys=['Loss', 'Reward'],
    smooth=15,
    split=split,
    **kwargs,
  )

def plot_eval(path, value=False, all_baselines=False, **kwargs):

  if isinstance(path, list):
    fname = [os.path.join(p, 'eval.csv') for p in path]
    split = None
    config_file = os.path.join(path[0], 'config.gin')
  else:
    fname = os.path.join(path, 'eval.csv')
    split = os.path.join(path, 'curriculum.csv')
    config_file = os.path.join(path, 'config.gin')

  try:
    gin.parse_config_file(config_file)
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
      baselines = siamrl.baselines.test(env_id)
    if not all_baselines:
      keymax = max(baselines, key=baselines.get)
      keymin = min(baselines, key=baselines.get)
      baselines = {
        keymax:baselines[keymax],
        keymin:baselines[keymin],
      }

  else:
    baselines = None 

  y_keys = ['Value', 'Reward'] if value else ['Reward']

  return plot(
    fname=fname,
    x_key='Iter',
    y_keys=y_keys,
    split=split,
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
      if ',' in arg:
        arg = arg.split(',')
      paths.append(arg)

  for path in paths:
    plot_results(path, value=value)