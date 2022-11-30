import os
from pathlib import Path
import sys

from optunizer.factory import factory, parse_config


if __name__ == '__main__':
  if len(sys.argv) > 1:
    c = sys.argv[1]
  else:
    c = os.getenv('OPTUNA_CONFIG', None)
  if c is not None and c == 'app':
    sys.argv[1] = str(Path(__file__).resolve().parent / 'app.py')
    cmd = f"streamlit run {' '.join(sys.argv[1:])}"
    print(cmd)
    os.system(cmd)
  else:
    kwargs = parse_config(c)
    c, kwargs = factory(kwargs)
    c(**kwargs)
