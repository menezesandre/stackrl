import os, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

baselines = {
  'Random (anywhere)': 0.13299907790496945,
  'Random (inside target)': 0.18017877265810966,
  'Gradient correlation (inside target)': 0.2028804305009544,
  'Correlation coefficients (inside target)': 0.2157407896593213
}
plot_baselines = False

split_curriculum = True
show = False
medfilt = False
paths = []
iter_key = 'Iter'
target_keys = []
default_target_keys = ['Loss', 'LTAvgReward']
if __name__=='__main__':
  argv = sys.argv[:0:-1]
  while argv:
    arg=argv.pop()
    if arg == '-s':
      show = True
    elif arg == '--disable-split':
      split_curriculum = False
    elif arg == '-f':
      medfilt = True
    elif arg == '--iter-key':
      iter_key = argv.pop()
    elif arg == '-b':
      plot_baselines = True
    elif arg == '--baselines':
      bfile = argv.pop()
      # TODO Load baselines from file
      plot_baselines = True
    elif arg == '-t':
      target_keys.append(argv.pop())
    elif os.path.isdir(arg) and (
      os.path.isfile(os.path.join(arg, 'train.csv')) or
      os.path.isfile(os.path.join(arg, 'eval.csv'))
    ):
      paths.append(arg)
    else:
      print('Ignoring '+arg)

  plot_baselines = len(target_keys)==1 and plot_baselines
  target_keys = target_keys or default_target_keys

  if not paths and os.path.isfile('./train.csv'):
    paths = ['.']

  for path in paths:
    ok = False
    for file_name in [
      os.path.join(path, 'eval.csv'),
      os.path.join(path, 'train.csv')
    ]:
      try:
        with open(file_name) as f:
          line = f.readline()
          if line.endswith('\n'):
            line = line[:-1]
          keys = line.split(',')
        if (iter_key in keys and all([key in keys for key in target_keys])):
          ok = True
          print("Ploting results in "+file_name)
          break
      except OSError:
        pass
    if not ok:
      print('Ignoring '+path)
      continue

    values = np.loadtxt(
      file_name,
      delimiter=',',
      skiprows=1,
      unpack=True
    )
    data = {key:value for key, value in zip(keys, values)}

    iters = data[iter_key]
    targets = [data[key] for key in target_keys]

    if medfilt:
      ks = min(101, len(iters)//100)
      if ks % 2 == 0:
        ks += 1
      for i,t in enumerate(targets):
        targets[i] = signal.medfilt(t, kernel_size=ks)

    fig, axs = plt.subplots(len(targets),1,sharex=True)
    if len(targets) == 1:
      # To be consistent for any number of targets
      axs = [axs]

    cfile = os.path.join(path,'curriculum.csv')
    if split_curriculum and os.path.isfile(cfile):
      curriculum = np.loadtxt(
        cfile,
        delimiter=',',
        skiprows=1,
        unpack=True
      )
      end_iters = curriculum[0]

      split = [0]
      i = 0
      for end_iter in end_iters:
        for i in range(i,len(iters)):
          if iters[i] > end_iter:
            split.append(i)
            break
      split.append(len(iters))
      
      split_iters = [iters[split[i]:split[i+1]+1] for i in range(len(split)-1)]
      split_targets = [
        [t[split[i]:split[i+1]+1] for i in range(len(split)-1)] 
        for t in targets
      ]
      
      for ax, target in zip(axs, split_targets):
        for x,y in zip(split_iters, target):
          ax.plot(x,y)
      legend = ['Train part %d'%i for i in range(len(split_iters))]
    else:
      for ax, target in zip(axs, targets):
        ax.plot(iters, target)
      legend = ['Train']

    if plot_baselines:
      for key in baselines:
        axs[0].plot([iters[0], iters[-1]],[baselines[key]]*2)
        legend.append(key)

    if len(legend) > 1:
      plt.legend(legend, loc='best')

    for ax, key, y in zip(axs, target_keys, targets):
      ax.set_ylabel(key)
      ylim = ax.get_ylim()
      if ylim[0] >= 0:
        ymin = ylim[0]
      else:
        ymin = min(0, max(ylim[0], 10*np.min(y[len(y)//4:])))
      if ylim[1] <= 0:
        ymax = ylim[1]
      else:
        ymax = max(0,min(ylim[1], 10*np.max(y[len(y)//4:])))
      ax.set_ylim(ymin,ymax)

    axs[-1].set_xlabel(iter_key)

    name = os.path.split(file_name)[-1].split('.')[0]
    prefix = os.path.split(path)[-1]
    if prefix == '.':
      prefix = ''
    plt.savefig(prefix+name+'.png')
    if show:
      plt.show()
    else:
      plt.close()

