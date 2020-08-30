"""siamrl main module."""
import argparse
from collections import defaultdict
from datetime import datetime
import os
import time
import types

import gin
try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None
import numpy as np

import siamrl

# Main functions

def train(args):
  # Parse config file
  if args.config_file:
    if os.path.isfile(os.path.join(args.directory, args.config_file)):
      gin.parse_config_file(os.path.join(args.directory, args.config_file))
    elif os.path.isfile(args.config_file):
      gin.parse_config_file(args.config_file)
    else:
      raise FileNotFoundError('Failed to find config file {}'.format(args.config_file))

  kwargs = {'directory':args.directory}
  # If env id or env args are provided, use them for the training env 
  # (otherwise, the configured defaults are used)
  make_args = {k:v for k,v in args.env_args} if args.env_args else {}
  if args.env_id:
    make_args['env'] = args.env_id
  if make_args:
    kwargs['env'] = lambda **kwa: siamrl.envs.make(**make_args, **kwa)

  # Run training
  training = siamrl.Training(**kwargs)
  training.run()

def plot(args):
  if args.combined:
    siamrl.train.plot(args.directory, value=args.value, baselines=args.baselines, show=args.show, clip=args.clip)
  else:
    for path in args.directory:
      siamrl.train.plot(path, value=args.value, baselines=args.baselines, show=args.show, clip=args.clip)

def test(args):
  make_args = {k:v for k,v in args.env_args} if args.env_args else {}
  if args.env_id:
    make_args['env'] = args.env_id
  if args.env_arg_list:
    k,v = args.env_arg_list
    make_args['curriculum'] = {k:v}

  if not make_args:
    # If no env id or args were provided and (at least) a training 
    # directory was provided, try to parse env params from the train
    # config file.
    if args.directory:
      path = args.directory[0][0]
      if os.path.isfile(os.path.join(path,'config.gin')):
        gin.parse_config_file(os.path.join(path,'config.gin'))

  # instantiate a dummy env to check specs
  dummy_env = siamrl.envs.make(**make_args, unwrapped=True)
  if isinstance(dummy_env, types.GeneratorType):
    gen = dummy_env
    dummy_env, _ = next(gen)
    del(gen)
  observation_space = dummy_env.observation_space
  del(dummy_env)

  policies = {}
  if args.directory:
    for path, iters in args.directory:
      for i in iters:
        key = os.path.split(path)[-1]
        if i is not None:
          key += '-{}'.format(i)
        policies[key] = siamrl.train.load(
          observation_space, 
          path, 
          i, 
          value=True, 
          verbose=args.verbose,
        )
  if args.baselines:
    baseline_args = {k:v for k,v in args.baseline_args} if args.baseline_args else {}
    batched = len(observation_space[0].shape) == 4

    # for k,v in args.baselines:
    #   if k == 'all':
    #   baselines.pop('all')
    #   for method in siamrl.baselines.methods:
    #     if method not in baselines:
    #       baselines[method] = {}

    for method,kwa in args.baselines:
      kwargs = baseline_args.copy()
      kwargs.update(kwa)

      name = method
      i = 0
      while name in policies:
        i += 1
        name = method+str(i)

      policies[name] = siamrl.Baseline(
        method=method, 
        value=True, 
        batched=batched, 
        batchwise=batched,
        **kwargs,
      )
  
  save = args.save_dir or args.save

  siamrl.test.test(
    **make_args,
    policies=policies, 
    num_steps=args.num_steps,
    visualize=args.visualize,
    verbose=args.verbose,
    sleep=args.sleep,
    show=args.show,
    save=save,
    seed=args.seed,
    )

def generate(args):
  if siamrl.envs.data.generate is None:
    raise ImportError('trimesh must be installed to run generate.')
  irregularity = []
  if args.irregularity:
    irregularity += args.irregularity
  if args.irregularity_range:
    irregularity += args.irregularity_range
  if not irregularity:
    # Use this range if nothing was provided.
    irregularity = np.arange(0.05,1.05,0.05)

  n_i = int((1-args.split)*args.number/len(irregularity))
  n_test = args.number-len(irregularity)*n_i

  if args.directory:
    directory = args.directory
  else:
    directory = siamrl.envs.data.path('generated')
    if args.clear:
      for fname in siamrl.envs.data.matching('generated', '*'):
        if os.path.isfile(fname):
          os.remove(fname)
      for fname in siamrl.envs.data.matching('generated', 'test', '*'):
        if os.path.isfile(fname):
          os.remove(fname)

  if args.plot_only:
    args.plot_previous = True
  else:
    if args.verbose:
      print('{}: Generating {} objects.'.format(datetime.now(), args.number))
      itime = time.perf_counter()
    for i, irr in enumerate(irregularity):
      siamrl.envs.data.generate(
        n=n_i, 
        name=str(int(100*irr)), 
        align_pai=args.align_pai, 
        directory=directory, 
        seed=args.seed, 
        show=args.show, 
        irregularity=irr,
        extents=args.extents,
        subdivisions=args.subdivisions,
      )
      if args.verbose:
        print('{}: {}/{} done.'.format(datetime.now(), (i+1)*n_i, args.number))

    if n_test:
      seed = args.seed+1 if args.seed is not None else None 
      siamrl.envs.data.generate(
        n=n_i, 
        name=str(int(100*irr)), 
        align_pai=args.align_pai, 
        directory=os.path.join(directory, 'test'), 
        seed=seed,
        show=args.show, 
        irregularity=irr,
        extents=args.extents,
      )

    if args.verbose:
      etime = time.perf_counter() - itime
      print('{}: {}/{} done. Total elapsed time: {} s ({} s/object)'.format(
        datetime.now(),
        args.number,
        args.number,
        etime, 
        etime/(args.number or 1),
      ))
  
  if args.plot or args.plot_previous:
    if plt is None:
      raise ImportError("matplotlib must be installed to run generate with --plot option.")
    plot_dir = siamrl.datapath(
      'generate',
      datetime.now().strftime('%y%m%d-%H%M%S')+'-{}'.format(n_i),
    )
    if not os.path.isdir(plot_dir):
      os.makedirs(plot_dir)

    data = defaultdict(lambda: defaultdict(lambda: list()))
    values = defaultdict(lambda: np.array([]))

    if args.plot_previous:
      fnames = siamrl.envs.data.matching(
        'generated', 
        '*.csv'
      )
      irregularity = [float(os.path.split(fname)[-1].split('.')[0])/100 for fname in fnames]
      irregularity = np.sort(irregularity)

    volume_ref = 1.

    for i in irregularity:
      fname = siamrl.envs.data.path(
        'generated', 
        '{}.csv'.format(str(int(i*100)))
      )
      with open(fname) as f:
        header = f.readline()[:-1].split(',')
      
      for k,v in zip(
        header, 
        np.loadtxt(fname, delimiter=',', skiprows=1, unpack=True)
      ):
        if k not in ['Name', 'NumVertices']:
          data[k]['mean'].append(v.mean())
          data[k]['std'].append(v.std())
          data[k]['min'].append(v.min())
          data[k]['max'].append(v.max())
          values[k] = np.concatenate([values[k], v])
      values['Irregularity'] = np.concatenate([values['Irregularity'], i*np.ones_like(v)])

      if i==0:
        volume_ref = values['Volume'][-1]

    for k in data['Volume']:
      data['Volume'][k] = np.divide(data['Volume'][k], volume_ref) 

    _, axs = plt.subplots(3, 1, sharex=True)

    for ax, k in zip(axs, data):

      ax.errorbar(irregularity, data[k]['mean'], yerr=(np.subtract(data[k]['mean'],data[k]['min']), np.subtract(data[k]['max'],data[k]['mean'])), fmt='none', ecolor='b', elinewidth=8, alpha=0.25, label='Range')
      ax.errorbar(irregularity, data[k]['mean'], yerr=data[k]['std'], fmt='bo', capsize=4, label='Mean +/- std dev')

      ax.set_ylabel(k if k != 'AspectRatio' else 'Aspect ratio')

    axs[-1].set_xlabel('Irregularity')
    plt.legend(loc='best')

    plt.savefig(os.path.join(plot_dir, 'irregularity.pdf'))
    plt.savefig(os.path.join(plot_dir, 'irregularity.png'))
    if args.show:
      plt.show()
    else:
      plt.close()

    y = np.array([v for v in values.values()])
    corrcoef = np.corrcoef(y)
    fig,ax = plt.subplots(constrained_layout=True)
    im,_ = siamrl.heatmap.heatmap(corrcoef, values.keys(), values.keys(), ax=ax, cbarlabel='Correlation coefficient')
    siamrl.heatmap.annotate_heatmap(im)
    plt.savefig(os.path.join(plot_dir, 'corrcoef.pdf'))
    plt.savefig(os.path.join(plot_dir, 'corrcoef.png'))
    if args.show:
      plt.show()
    else:
      plt.close()

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(values['AspectRatio'], values['Volume']/volume_ref, values['Rectangularity'], marker='.', c=values['Irregularity'])
    ax.set_ylabel('Volume')
    ax.set_zlabel('Rectangularity')
    ax.set_xlabel('Aspect ratio')
    ax.view_init(elev=30, azim=105)
    fig.colorbar(scatter, label='Irregularity')
    plt.savefig(os.path.join(plot_dir, 'distribution.pdf'))
    plt.savefig(os.path.join(plot_dir, 'distribution.png'))
    if args.show:
      plt.show()
    else:
      plt.close()
    


# Costum types for the parser

def test_dir_type(arg):
  """Parse a directory argument for the test command."""
  if ',' in arg:
    arg = arg.split(',')
    return (arg[0], [int(i) for i in arg[1:]])
  else:
    return (arg, [None])

def baseline_with_args(arg):
  arg = arg.split(',')
  kwargs = [key_value_type(a) for a in arg[1:]]
  return (arg[0], {k:v for k,v in kwargs}) 

def key_value_type(arg):
  """Parse an argument with the format key=value"""
  k,v = arg.split('=')
  try:
    v = eval(v)
  except (NameError, SyntaxError):
    pass
  return k,v

def key_values_list_type(arg):
  """Parse an argument with the format key=value0[,value1,...]"""
  k,v = arg.split('=')
  if ',' in v:
    v = v.split(',')
    try:
      v = [eval(i) for i in v]
    except NameError:
      pass
  elif ':' in v:
    v = list(range(*[int(i) for i in v.split(':')]))
  else:
    raise ValueError("value doesn't represent a list")
  return k,v

def range_type(arg):
  """Parse an argument representing a range in the format start:stop:step."""
  arg = [float(i) for i in arg.split(':')]
  return list(np.arange(*arg))


parser = argparse.ArgumentParser(description='siamrl main module.', prog='siamrl')
parser.add_argument('--version', action='version', version='%(prog)s '+siamrl.__version__)
subparsers = parser.add_subparsers(title='commands')

# parser for the train command
parser_train = subparsers.add_parser('train', 
  description='run a training session')
parser_train.add_argument('-d', '--directory', default='.',
  help='training directory')
parser_train.add_argument('-c', '--config-file', nargs='?', const='config.gin', 
  help='gin-config file from which to parse the training parameters')
parser_train.add_argument('-e', '--env-id', 
  help='id of the environment to be used on training')
parser_train.add_argument('--env-args', nargs='+', type=key_value_type,
  metavar='ARG=VALUE', help='arguments to pass to the environment')
parser_train.set_defaults(func=train)

# parser for the plot command
parser_plot = subparsers.add_parser('plot', 
  description='produce plots from training data')
parser_plot.add_argument('directory', nargs='+', 
  help='training directories')
parser_plot.add_argument('-b', '--baselines', nargs='+',
  help='name of the baseline policies for which to show results for comparison with training')
parser_plot.add_argument('-c', '--combined', action='store_true', 
  help='whether produce plots from combined (average) data instead of from each directory separatly.')
parser_plot.add_argument('-v', '--value', action='store_true', 
  help='whether to show value estimates in plots')
parser_plot.add_argument('-s', '--show', action='store_true', 
  help='whether to show the plots')
parser_plot.add_argument('--clip', type=int,
  help='number of standard deviations of the data to limit the y axis')
parser_plot.set_defaults(func=plot)

# parser for the test command
parser_test = subparsers.add_parser('test', 
  description='run a test session and collect metrics')
parser_test.add_argument('-e', '--env-id', 
  help='id of the environment to be used on testing')
parser_test.add_argument('--env-args', nargs='+', type=key_value_type,
  metavar='ARG=VALUE', help='arguments to pass to the environment')
parser_test.add_argument('--env-arg-list', type=key_values_list_type,
  metavar='ARG=VALUE0[,VALUE1,...]', help='run a test for each of this environment argument values')
parser_test.add_argument('-d', '--directory', nargs='+', type=test_dir_type,
  metavar='DIR[,ITER,...]', help='trained policies to test')
parser_test.add_argument('-b', '--baselines', nargs='+', type=baseline_with_args,
  help='baseline policies to test')
parser_test.add_argument('--baseline-args', nargs='+', type=key_value_type,
  metavar='ARG=VALUE', help='arguments to pass to the baselines constructor')
parser_test.add_argument('-n', '--num-steps', default=1000, type=int, 
  help='number of environment steps to run with each policy')
parser_test.add_argument('-q','--quiet', action='store_false', dest='verbose', 
  help='if set, no results are printed as the test proceeds')
parser_test.add_argument('-v', '--visualize', action='store_true', 
  help='whether to show a window with a visualization of the observations and value maps of each policy')
parser_test.add_argument('--sleep', type=float,
  help='time interval, in seconds, between steps')
parser_test.add_argument('--show', action='store_true',
  help='whether to show plots of the results on the end of the test')
parser_test.add_argument('-s', '--save', type=bool, nargs='?', const=True,
  help='whether to save collected data and plots. If not set, data are saved if verbose is disabled and plots are saved if show is disabled')
parser_test.add_argument('--save-dir',
  help='directory where to save the results (use instead of -s to save to a costum directory)')
parser_test.add_argument('--seed', default=11, type=int, 
  help='seed for the environment')
parser_test.set_defaults(func=test)

# parser for the generate command
parser_generate = subparsers.add_parser('generate',
  description='generate irregular object models to populate siamrl/envs/data')
parser_generate.add_argument('-n', '--number', type=int, default=10000,
  help='number of models to generate')
parser_generate.add_argument('-d','--directory',
  help='provide to save the models in a different directory')
parser_generate.add_argument('--split', type=float, default=0.,
  help='fraction of the generated models to be stored in a test folder')
parser_generate.add_argument('--seed', type=int,
  help='seed for the randomness in the generator')
parser_generate.add_argument('--show', action='store_true',
  help='whether to show a visualization of each generated model')
parser_generate.add_argument('-q', '--quiet', dest='verbose', action='store_false',
  help='supress progress information.')
parser_generate.add_argument('--align-pai', action='store_true',
  help='whether to align the models frame with the principal axis of inertia; otherwise models are aligned with the oriented bounding box')
parser_generate.add_argument('-r','--radius', type=float, default=2**-4,
  help='maximum radius of the models bounding sphere')
parser_generate.add_argument('--extents', nargs=3, type=float,
  help='ratios of the extents of the box that originates the models')
parser_generate.add_argument('--subdivisions', type=int, default=3,
  help='number of subdivisions to apply to the box (each subdivision replaces a face by four smaller faces)')
parser_generate.add_argument('-i', '--irregularity', nargs='+', type=float, 
  help='values for the irregularity of the models')
parser_generate.add_argument('--irregularity-range', type=range_type, 
  metavar='START:STOP:STEP', help='range of values for the irregularity of the models')
parser_generate.add_argument('--plot', action='store_true',
  help='whether to produce plots of the shape metrics distribution vs. irregularity')
parser_generate.add_argument('--plot-previous', action='store_true',
  help='whether to include previous models in the directory in the produced plots')
parser_generate.add_argument('--plot-only', action='store_true',
  help='only produce plots from previous models (no generation)')
parser_generate.add_argument('--clear', action='store_true',
  help='whether to remove previous models from the directory (only used for the default directory)')
parser_generate.set_defaults(func=generate)


# Parse args and run function
args = parser.parse_args()
args.func(args)

