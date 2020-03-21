import matplotlib.pyplot as plt
import os

def from_log(log_file_name, plot_file_name, show=False):
  if not os.path.isfile(log_file_name):
    print('No file named '+log_file_name)
    return
  data = {}
  with open(log_file_name) as f:
    l = next(f).split()
    for i in range(len(l)//2):
      data[l[2*i]] = [eval(l[2*i+1])]
    for l in f:
      l = l.split()
      for i in range(len(l)//2):
        data[l[2*i]].append(eval(l[2*i+1]))
    
  x_key = list(data.keys())[0]
  y_keys = list(data.keys())[1:]
  fig, axs = plt.subplots(len(y_keys), 1, sharex=True)
  if not hasattr(axs, '__iter__'):
    axs = [axs]

  for ax, y_key in zip(axs, y_keys):
    ax.plot(data[x_key], data[y_key])
    ax.set_ylabel(y_key)
  axs[-1].set_xlabel(x_key)
    
  fig.savefig(plot_file_name)
  if show:
    fig.show()
