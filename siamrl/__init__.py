from siamrl import envs
from siamrl import networks
from siamrl import utils
from siamrl import train
# If opencv is not instaled, propagate the exception to the 
# module's function calls
try:
  from siamrl import baselines
except ImportError as e:
  import types
  exception = e
  def f(**kwargs):
    raise exception
  baselines = types.SimpleNamespace(CCoeffPolicy=f, 
                                   GradCorrPolicy=f)
