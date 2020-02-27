from siamrl.utils import train, register, mail

try:
  from siamrl.utils import plot
except ImportError as e:
  import types
  exception = e
  def f(**kwargs):
    raise exception
  plot = types.SimpleNamespace(from_log=f)