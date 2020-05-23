import os, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

split_curriculum = True
show = False
if __name__=='__main__':
  if len(sys.argv) > 1:
    paths = sys.argv[1:]
  else:
    paths = ['.']
  for path in paths:
    train_file = os.path.join(path, 'train.csv')
    try:
      iters, losses, collect_times, train_times, returns = np.loadtxt(
        train_file,
        delimiter=',',
        skiprows=1,
        unpack=True
      )
    except OSError:
      print('Skiping '+path)
      continue
    ks = min(101, len(losses)//100)
    if ks % 2 == 0:
      ks += 1
    losses = signal.medfilt(losses, kernel_size=ks)
    returns = signal.medfilt(returns, kernel_size=ks)

    fig, axs = plt.subplots(2,1,sharex=True)

    targets = {}
    curriculum_file = os.path.join(path,'curriculum.csv')
    if os.path.isfile(curriculum_file) and split_curriculum:
      end_iters, goals = np.loadtxt(
        curriculum_file,
        delimiter=',',
        skiprows=1,
        unpack=True
      )
      split_indexes = []
      i = 0
      for end_iter in end_iters:
        for i in range(i,len(iters)):
          if iters[i] > end_iter:
            split_indexes.append(i)
            break

      splited_iters = [iters[:split_indexes[0]+1]]
      splited_losses = [losses[:split_indexes[0]+1]]
      splited_returns = [returns[:split_indexes[0]+1]]
      for i in range(len(split_indexes)-1):
        splited_iters.append(iters[split_indexes[i]:split_indexes[i+1]+1])
        splited_losses.append(losses[split_indexes[i]:split_indexes[i+1]+1])
        splited_returns.append(returns[split_indexes[i]:split_indexes[i+1]+1])
      splited_iters.append(iters[split_indexes[-1]:])
      splited_losses.append(losses[split_indexes[-1]:])
      splited_returns.append(returns[split_indexes[-1]:])
      
      for ax, y in zip(axs, [splited_losses, splited_returns]):
        legend = []
        for i,(its,t) in enumerate(zip(splited_iters, y)):
          # if len(its) > 1:
          ax.plot(its,t)
          legend.append('Train part %d'%i)
      fig.legend(legend)
    else:
      for ax, y in zip(axs, [losses, returns]):
        ax.plot(iters, y)

    axs[0].set_ylabel('Loss')
    axs[0].set_ylim(0, 2*np.max(losses[len(losses)//5:]))
    axs[1].set_ylabel('Reward')
    axs[1].set_xlabel('Iters')

    name = os.path.split(path)[1]
    if not name or name == '.':
      name='train.png'
    else:
      name='{}_train.png'.format(name)
    plt.savefig(name)
    if show:
      plt.show()
    else:
      plt.close()

