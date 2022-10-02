# optunizer
Optuna extension for JSON and YAML configuration files

## Installation
```sh
pip install optunizer
```

## Running
0. Suppose you have some script/program (e.g. `main.py`) with config in YAML/JSON file (e.g. `config.yaml`) that returns some output (e.g. `metrics.json`)
* main.py
```python
# Read config.yaml
...
# Do some stuff
...
# Write metrics.json
...
```
* config.yaml
```yaml
param1: 2
param2: 0.5
param3: c
```
* metrics.json
```json
{
  "metric": 0.3
}
```
1. Make optunizer config file, e.g. `optunizer.yaml`
```yaml
attrs:  # track all fields in files
  config.yaml: true
  metrics.json: true
  optunizer_sysinfo.json: true
class: optunizer.optimizer.Optimizer
load_if_exists: true
export_csv: optunizer_results.csv
export_metrics: optunizer_metrics.json
export_sysinfo: optunizer_sysinfo.json
study: optunizer_test
objectives:  # Specify objectives, e.g. fields in metrics.json file
  metric@metrics.json: minimize
params:  # Specify params, e.g. fields in config.yaml file
  param1@config.yaml:
    method: suggest_int
    method_kwargs:
      high: 3
      low: 0
  param2@config.yaml:
    method: suggest_float
    method_kwargs:
      high: 1.0
      low: 0.01
      log: true
  param3@config.yaml:
    method: suggest_categorical
    method_kwargs:
      choices: [a, b, c]
pruner: PatientPruner
pruner_kwargs:  # Specify pruner, e.g. PatientPruner with NopPruner subpruner
  min_delta: 0
  patience: 0
  wrapped_pruner: NopPruner
  wrapped_pruner_kwargs: {}
sampler: PartialFixedSampler
sampler_kwargs:   # Specify sampler, e.g. PartialFixedSampler with GridSampler subsampler
  # base_sampler: RandomSampler
  # base_sampler_kwargs: {}
  base_sampler: GridSampler
  base_sampler_kwargs:
    search_space:
      param1@config.yaml: [0, 1, 2]
      param2@config.yaml: [0.01, 0.5]
  fixed_params:
    param3@config.yaml: a
subprocess_kwargs:  # Specify your command
  args:
  - python
  - main.py
  - config.yaml
```

2. Run optunizer
```sh
OPTUNA_CONFIG=optunizer.yaml python -m optunizer
```

3. There are several useful environment variables, that could be set in command line, `.env` or `.env.secret` files
```sh
OPTUNA_CONFIG=optunizer.yaml
OPTUNA_SECRET=.env.secret
OPTUNA_URL=postgresql+psycopg2://USER:PASSWORD@IP:PORT/DB  # see https://docs.sqlalchemy.org/en/14/core/engines.html
OPTUNA_STUDY=STUDY_NAME
OPTUNA_TRIALS=3
OPTUNA_TIMEOUT=3600
OPTUNA_LOAD_IF_EXISTS=1
OPTUNA_EXPORT_CSV=CSV_FILE_NAME
OPTUNA_EXPORT_METRICS=METRICS_FILE_NAME
OPTUNA_EXPORT_SYSINFO=SYSINFO_FILE_NAME
```
