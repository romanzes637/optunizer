import os
import sys

from dotenv import load_dotenv, dotenv_values
import yaml

from optunizer.factory import factory


if __name__ == '__main__':
  load_dotenv()  # load '.env' file
  if len(sys.argv) > 1:
    c = sys.argv[1]
  else:
    c = os.getenv('OPTUNA_CONFIG', 'optunizer.yaml')
  with open(c) as f:
    kwargs = yaml.safe_load(f)
  s = os.getenv('OPTUNA_SECRET', '.env.secret')
  s = dotenv_values(s)
  s = {k.lower(): v for k, v in s.items()}
  kwargs['secret'] = s
  c, kwargs = factory(kwargs)
  c(**kwargs)
