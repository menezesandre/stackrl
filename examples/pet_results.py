import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  if len(sys.argv) > 1:
    plot = '-p' in sys.argv
    error = '-e' in sys.argv
  else:
    plot = True
    error = True

  freqs = [30, 60, 120, 240]
  names = ['c', 'n']

  # Plot times
  if plot:
    data = {}
    for name in names:
      data[name] = {}
      for freq in freqs:
        data[name][freq] = np.loadtxt(
          '.times_{}{}.csv'.format(name,freq),
          delimiter=',',
          skiprows=2,
          unpack=True          
        )
    x = np.arange(1,1+data[name][freq].shape[1])

    # by name
    for name in names:
      fig, axs = plt.subplots(3,1,sharex=True)
      for freq in freqs:
        for i in range(3):
          axs[i].plot(x,data[name][freq][i,:])
        
      axs[0].set_ylabel('time (s)')
      axs[1].set_ylabel('steps')
      axs[2].set_ylabel('time/step (s)')
      axs[2].set_xlabel('#object')
      fig.legend(['%dHz'%f for f in freqs])

      plt.savefig('times_{}.png'.format(name))

    # by frequency
    for freq in freqs:
      fig, axs = plt.subplots(3,1,sharex=True)
      for name in names:
        for i in range(3):
          axs[i].plot(x,data[name][freq][i,:])
        
      axs[0].set_ylabel('time (s)')
      axs[1].set_ylabel('steps')
      axs[2].set_ylabel('time/step (s)')
      axs[2].set_xlabel('#object')
      fig.legend([('not ' if n=='n' else '') + 'convex' for n in names])

      plt.savefig('times_{}.png'.format(freq))

    del(data)

# Compute error
  if error:
    data = {}
    for name in names:
      data[name] = {}
      for freq in freqs:
        data[name][freq] = np.loadtxt(
          '.poses_{}{}.csv'.format(name,freq),
          delimiter=',',
          skiprows=1,
        )

    # with open('.rewards.csv') as f:
    #   keys = f.readline()[:-1].split(',')
    # rewards = np.loadtxt(
    #   '.rewards.csv',
    #   delimiter=',',
    #   skiprows=1,
    # )

    # changing frequency
    for name in names:
      ref_p = data[name][freqs[-1]][:,:3]
      ref_o = data[name][freqs[-1]][:,3:]
      # ref_r = rewards[:,keys.index(name+str(freqs[-1]))]

      for freq in freqs[:-1]:
        mae_p = np.mean(np.linalg.norm(
          ref_p - data[name][freq][:,:3], 
          axis=-1
        ))
        mae_o = np.mean(2*np.arccos(np.sum(
          ref_o*data[name][freq][:,3:], 
          axis=-1
        )))
        # mae_r = np.mean(np.abs(
        #   ref_r - rewards[:,keys.index(name+str(freq))]
        # ))
        
        # with open('error.log','a') as f:
        #   f.write('{} {}-{}: p {}; o {}; r {}\n'.format(
        #     name,
        #     freqs[-1],
        #     freq,
        #     mae_p,
        #     mae_o,
        #     mae_r
        #   ))
        with open('error.log','a') as f:
          f.write('{} {}-{}: p {}; o {}\n'.format(
            name,
            freqs[-1],
            freq,
            mae_p,
            mae_o
          ))

    # using convex hull
    for freq in freqs:
      ref_p = data[names[-1]][freq][:,:3]
      ref_o = data[names[-1]][freq][:,3:]
      # ref_r = rewards[:,keys.index(names[-1]+str(freq))]

      for name in names[:-1]:
        mae_p = np.mean(np.linalg.norm(
          ref_p - data[name][freq][:,:3], 
          axis=-1
        ))
        mae_o = np.mean(2*np.arccos(np.sum(
          ref_o*data[name][freq][:,3:], 
          axis=-1
        )))
        # mae_r = np.mean(np.abs(
        #   ref_r - rewards[:,keys.index(name+str(freq))]
        # ))
         
        with open('error.log','a') as f:
          f.write('{} {}-{}: p {}; o {}\n'.format(
            freq,
            names[-1],
            name,
            mae_p,
            mae_o
          ))
