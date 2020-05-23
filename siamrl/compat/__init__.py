try:
  from siamrl.compat import external_configurables as _
  from siamrl.compat import networks
  from siamrl.compat import utils
  from siamrl.compat.train import Training, CurriculumTraining
except ImportError:
  pass