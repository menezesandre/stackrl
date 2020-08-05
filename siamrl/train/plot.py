"""Utils to plot train results."""
import os

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

def read_csv(fname, columns=None, reduction=None, dtype=float):
  """Read one or more csv files.
  Args:
    fname: name of the csv file (str), or collection of names, to be read.
      if a list of files is provided, the returned values are a reduction 
      (e.g. mean) across all files. In that case, for each column, another
      column named '<column_name>_std' is added containing the standard 
      deviations.
    columns: list of column names to be returned. If None, all columns in 
      the file are returned.
    reduction: either a callable or the name of a numpy function. Only 
      used if fname is a collection of strings. If None, mean is used.
  Returns:
    dictionary with column names as keys and 1d arrays as values.
  """
  if isinstance(fname, str):
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
        np.loadtxt(fname, delimiter=',', skiprows=1, unpack=True, dtype=dtype)
      ) if columns is None or key in columns
    }
  else:
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

def _plot(fname, x_key, y_keys, smooth=0, split=None, baselines=None, show=False, legend='Train', save_as=None):
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

  _, axs = plt.subplots(len(ys),1,sharex=True)
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
    dirname = os.path.dirname(save_as)
    if dirname and not os.path.isdir(dirname):
      os.makedirs(dirname)
    plt.savefig(save_as)
  elif not show:
    for ext in ['png', 'pdf']:
      save_as = get_save_name(fname,ext=ext)
      dirname = os.path.dirname(save_as)
      if dirname and not os.path.isdir(dirname):
        os.makedirs(dirname)
      plt.savefig(save_as)

  if show:
    plt.show()
  else:
    plt.close()

def plot_value(path, show=False, save_as=None):
  if isinstance(path, str):
    fname = os.path.join(path, 'eval.csv')
  else:
    fname = [os.path.join(p, 'eval.csv') for p in path]
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
  
  return _plot(
    fname=fname,
    x_key='Iter',
    y_keys=['Loss', 'Return'],
    smooth=15,
    split=split,
    **kwargs,
  )

def plot_eval(path, value=False, baselines=['random', 'ccoeff'], **kwargs):

  if isinstance(path, list):
    fname = [os.path.join(p, 'eval.csv') for p in path]
    split = None
    config_file = os.path.join(path[0], 'config.gin')
  else:
    fname = os.path.join(path, 'eval.csv')
    split = os.path.join(path, 'curriculum.csv')
    config_file = os.path.join(path, 'config.gin')
  print(baselines)
  if baselines:
    # Store current config state
    _config = gin.config._CONFIG.copy()

    try:
      # Try to parse directory's config file
      gin.parse_config_file(config_file)
      with gin.config_scope('eval'):
        # Get path of the eval env used in training (using binded parameters)
        envpath = envs.make(as_path=True)
      if not isinstance(envpath, str):
        # if it's not a string (i.e. it's a generator from curriculum) 
        # don't show baselines in plot.
        envpath = None
    except OSError:
      # If no config file was parsed, don't show baselines in plot.
      envpath = None
    print(envpath)
    if envpath is not None:
      # Get full path to results file for that env
      rfname = siamrl.datapath(
        'test',
        envpath,
        'results.csv',
      )
      if os.path.isfile(rfname):
        results = read_csv(rfname, columns=['Keys', 'Return'], dtype=str)
        results = {
          k:float(v) for k,v in zip(results['Keys'], results['Return']) 
          if k in baselines
        }
      else:
        results = {}

      if len(results) != len(baselines):
        # Get results for the missing baselines
        get_results = [k for k in baselines if k not in results]
        with gin.config_scope('eval'):
          siamrl.test.test(
            policies = {k:siamrl.Baseline(method=k, value=True) for k in get_results},
            verbose=False,
          )
        new_results = read_csv(rfname, columns=['Keys', 'Return'])
        for k,v in zip(new_results['Keys'], new_results['Return']):
          if k in get_results:
            results[k] = v
      
      baselines = results
    else:
      baselines = None

    # Restore config state
    gin.config._CONFIG = _config
  else:
    baselines = None

  y_keys = ['Value', 'Return'] if value else ['Return']

  return _plot(
    fname=fname,
    x_key='Iter',
    y_keys=y_keys,
    split=split,
    baselines=baselines,
    **kwargs,
  )

def plot(path, value=False, baselines=None):
  try:
    plot_train(path)
    plot_eval(path, value=value, baselines=baselines)
    if value:
      plot_value(path)
  except FileNotFoundError as e:
    # If files don't exist in given path, use it as relative from Siam-RL/data/train
    if path.startswith(siamrl.datapath('train')):
      raise e
    else:
      return plot(siamrl.datapath('train', path), value=value, baselines=baselines)
