import os
import importlib
import inspect

from dotenv import load_dotenv, dotenv_values
import yaml


def parse_string(s):
  if s is not None:
    if s.isnumeric():
      s = int(s)
    else:
      try:
        s = float(s)
      except ValueError:
        pass
  return s


def parse_config(config):
  if config is not None:
    with open(config) as f:
      kwargs = yaml.safe_load(f)
  else:
    kwargs = {}
  env_kwargs = {
    **dotenv_values(os.getenv('OPTUNA_SHARED', '.env')),  # load shared development variables
    **dotenv_values(os.getenv('OPTUNA_SECRET', '.env.secret')),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
  }
  env_kwargs = {k.lower()[7:]: parse_string(v) for k, v in env_kwargs.items() 
                if k.startswith('OPTUNA_')}
  kwargs = {**kwargs, **env_kwargs}
  return kwargs


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
