import os, sys
import matplotlib.pyplot as plt
import numpy as np

split_curriculum = True
show = False
baselines = {
  'Random (anywhere)': 0.13299907790496945,
  'Random (inside target)': 0.18017877265810966,
  'Gradient correlation (inside target)': 0.2028804305009544,
  'Correlation coefficients (inside target)': 0.2157407896593213
}

if __name__=='__main__':
  if len(sys.argv) > 1:
    paths = sys.argv[1:]
  else:
    paths = ['.']
  for path in paths:
    eval_file = os.path.join(path, 'eval.csv')
    try:
      eval_iters, eval_returns = np.loadtxt(
        eval_file,
        delimiter=',',
        skiprows=1,
        unpack=True
      )
    except OSError:
      print('Skiping '+path)
      continue
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
        for i in range(i,len(eval_iters)):
          if eval_iters[i] > end_iter:
            split_indexes.append(i)
            break

      splited_eval_iters = [eval_iters[:split_indexes[0]+1]]
      splited_eval_returns = [eval_returns[:split_indexes[0]+1]]
      for i in range(len(split_indexes)-1):
        splited_eval_iters.append(eval_iters[split_indexes[i]:split_indexes[i+1]+1])
        splited_eval_returns.append(eval_returns[split_indexes[i]:split_indexes[i+1]+1])
      splited_eval_iters.append(eval_iters[split_indexes[-1]:])
      splited_eval_returns.append(eval_returns[split_indexes[-1]:])

      legend = []
      for i,(iters,returns) in enumerate(zip(splited_eval_iters, splited_eval_returns)):
        # if len(iters) > 1:
        plt.plot(iters,returns)
        legend.append('Train part %d'%i)
    else:
      plt.plot(eval_iters, eval_returns)
      legend = ['Train']

    for key in baselines:
      plt.plot([eval_iters[0], eval_iters[-1]],[baselines[key]]*2)
      legend.append(key)

    plt.legend(legend)
    plt.xlabel('Iters')
    plt.ylabel('Reward')

    name = os.path.split(path)[1]
    if not name or name == '.':
      name='eval_reward.png'
    else:
      name='{}_eval_reward.png'.format(name)
    plt.savefig(name)
    if show:
      plt.show()
    else:
      plt.close()

