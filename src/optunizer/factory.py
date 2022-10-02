import importlib
import inspect


def factory(kwargs):
  if 'class' in kwargs:
    n = kwargs.pop('class')
  elif 'function' in kwargs:
    n = kwargs.pop('function')
  else:
    raise ValueError(kwargs)
  ts = n.split('.')  # tokens
  m, c = '.'.join(ts[:-1]), ts[-1]  # module, class/function/callable
  m = importlib.import_module(m)
  c = getattr(m, c)
  if inspect.isclass(c):
    return c(**kwargs), {}
  elif inspect.isfunction(c):
    return c, kwargs
  else:
    raise ValueError(c)
