from siamrl.utils import mail

try:
  from siamrl.utils import plot
except ImportError as e:
  import types
  exception = e
  def f(**kwargs):
    raise exception
  plot = types.SimpleNamespace(from_log=f)
