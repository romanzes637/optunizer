"""Optuna Optimizer

https://optuna.readthedocs.io/en/stable/reference/samplers.html
https://optuna.readthedocs.io/en/stable/reference/pruners.html
https://optuna.readthedocs.io/en/stable/reference/distributions.html
https://optuna.readthedocs.io/en/stable/reference/storages.html
"""
import json
import os
import subprocess
import copy
import io
from pathlib import Path
import sys
from pprint import pprint
from inspect import signature
import socket
import platform

import yaml
import optuna


class Optimizer:
  def __init__(
    self, url=None, study=None, n_trials=None, timeout=None,
    load_if_exists=None, export_csv=None, results_only=None, 
    export_metrics=None, export_sysinfo=None,
    sampler=None, sampler_kwargs=None, pruner=None, pruner_kwargs=None,
    subprocess_kwargs=None, stdout_kwargs=None, stderr_kwargs=None,
    params=None, objectives=None, attrs=None, lsep=None, gsep=None, secret=None):
    url = os.getenv('OPTUNA_URL') if url is None else url
    url = secret.get('optuna_url') if url is None else url
    self.url = url
    self.study = os.getenv('OPTUNA_STUDY') if study is None else study
    if n_trials is None:
      n_trials = os.getenv('OPTUNA_TRIALS', '')
      n_trials = int(n_trials) if n_trials.strip() else None
    self.n_trials = n_trials
    if timeout is None:
      timeout = os.getenv('OPTUNA_TIMEOUT', '')
      timeout = int(timeout) if timeout.strip() else None
    self.timeout = timeout
    if load_if_exists is None:
      load_if_exists = int(os.getenv('OPTUNA_LOAD_IF_EXISTS', 0))
    self.load_if_exists = load_if_exists
    if export_csv is None:
      export_csv = os.getenv('OPTUNA_EXPORT_CSV', '')
      export_csv = export_csv if export_csv.strip() else None
    self.export_csv = export_csv
    if results_only is None:
      results_only = int(os.getenv('OPTUNA_RESULTS_ONLY', 0))
    self.results_only = results_only
    if export_metrics is None:
      export_metrics = os.getenv('OPTUNA_EXPORT_METRICS', '')
      export_metrics = int(export_metrics) if export_metrics.strip() else None
    self.export_metrics = export_metrics
    if export_sysinfo is None:
      export_sysinfo = os.getenv('OPTUNA_EXPORT_SYSINFO', '')
      export_sysinfo = int(export_sysinfo) if export_sysinfo.strip() else None
    self.export_sysinfo = export_sysinfo
    self.export_sysinfo = export_sysinfo
    self.sampler = sampler
    self.sampler_kwargs = sampler_kwargs
    self.pruner = pruner
    self.pruner_kwargs = pruner_kwargs
    self.subprocess_kwargs = {} if subprocess_kwargs is None else subprocess_kwargs
    self.stdout_kwargs = {} if stdout_kwargs is None else stdout_kwargs
    self.stderr_kwargs = {} if stderr_kwargs is None else stderr_kwargs
    if not isinstance(self.subprocess_kwargs, list):
      self.subprocess_kwargs = [self.subprocess_kwargs]
    if not isinstance(self.stdout_kwargs, list):
      self.stdout_kwargs = [self.stdout_kwargs for _ in self.subprocess_kwargs]
    if not isinstance(self.stderr_kwargs, list):
      self.stderr_kwargs = [self.stderr_kwargs for _ in self.subprocess_kwargs]
    self.params = {} if params is None else params
    self.objectives = {} if objectives is None else objectives
    self.attrs = {} if attrs is None else attrs
    self.lsep = '/' if lsep is None else lsep
    self.gsep = '@' if gsep is None else gsep

  @staticmethod
  def initialize_sampler(sampler, sampler_kwargs):
    if sampler is None:
      return sampler
    elif sampler != 'PartialFixedSampler':
      c = getattr(optuna.samplers, sampler)  # class
      s = signature(c)
      filtered_kwargs = {k: v for k, v in sampler_kwargs.items() 
                         if k in s.parameters}
      return c(**filtered_kwargs)
    else:
      if 'base_sampler_kwargs' in sampler_kwargs:
        ss_kwargs = copy.deepcopy(sampler_kwargs)
        ss = Optimizer.initialize_sampler(
          ss_kwargs['base_sampler'],
          ss_kwargs['base_sampler_kwargs'])
        ss_kwargs['base_sampler'] = ss
        ss_kwargs.pop('base_sampler_kwargs')
        return Optimizer.initialize_sampler(sampler, ss_kwargs)
      else:
        c = getattr(optuna.samplers, sampler)  # class
        s = signature(c)
        filtered_kwargs = {k: v for k, v in sampler_kwargs.items() 
                           if k in s.parameters}
        return c(**filtered_kwargs)

  @staticmethod
  def initialize_pruner(pruner, pruner_kwargs):
    if pruner is None:
      return pruner
    elif pruner != 'PatientPruner':
      c = getattr(optuna.pruners, pruner)  # class
      s = signature(c)
      filtered_kwargs = {k: v for k, v in pruner_kwargs.items() 
                         if k in s.parameters}
      return c(**filtered_kwargs)
    else:
      if 'wrapped_pruner_kwargs' in pruner_kwargs:
        pp_kwargs = copy.deepcopy(pruner_kwargs)
        pp = Optimizer.initialize_pruner(
          pp_kwargs['wrapped_pruner'],
          pp_kwargs['wrapped_pruner_kwargs'])
        pp_kwargs['wrapped_pruner'] = pp
        pp_kwargs.pop('wrapped_pruner_kwargs')
        return Optimizer.initialize_pruner(pruner, pp_kwargs)
      else:
        c = getattr(optuna.pruners, pruner)  # class
        s = signature(c)
        filtered_kwargs = {k: v for k, v in pruner_kwargs.items() 
                           if k in s.parameters}
        return c(**filtered_kwargs)
  
  @staticmethod
  def suggest(trial, method, method_kwargs):
    c = getattr(trial, method)  # function
    s = signature(c)
    filtered_kwargs = {k: v for k, v in method_kwargs.items() 
                       if k in s.parameters}
    return c(**filtered_kwargs)
  
  @staticmethod
  def split_path(p, s='@'):
    tokens = p.split(s)
    if len(tokens) == 2:
      local_path, global_path = tokens
    elif len(tokens) == 1:
      local_path, global_path = None, tokens[0]
    else:
      raise ValueError(p, s)
    return local_path, global_path
  
  @staticmethod
  def parse_file(p, r=None, n=None, s='.'):
    p = Path(p)
    if p.suffix == '.json':
      with open(p) as f:
        if p.suffix == '.json':
          d = json.load(f)
    elif p.suffix == '.yaml':
      with open(p) as f:
        d = yaml.safe_load(f)
    else:
      raise NotImplementedError(p)   
    return Optimizer.parse(d, r, n, s)
  
  @staticmethod
  def parse(d, r=None, n=None, s='.'):
    r = {} if r is None else r
    if isinstance(d, dict):
      for k, v in d.items():
        nn = s.join([n, k]) if n is not None else k
        r = Optimizer.parse(v, r, nn, s)
    elif isinstance(d, list):
      for i, v in enumerate(d):
        nn = s.join([n, str(i)]) if n is not None else str(i)
        r = Optimizer.parse(v, r, nn, s)
    else:
      r[n] = d
    return r
 
  @staticmethod
  def update_file(p, v, n, s='.'):
    p = Path(p)
    with open(p) as f:
      if p.suffix == '.json':
        d = json.load(f)
      elif p.suffix == '.yaml':
        d = yaml.safe_load(f)
      else:
        raise NotImplementedError(p)  
    Optimizer.update(d, v, n, s)
    with open(p, 'w') as f:
      if p.suffix == '.json':
        json.dump(d, f)
      elif p.suffix == '.yaml':
        yaml.safe_dump(d, f)
      else:
        raise NotImplementedError(p) 
    
  @staticmethod
  def update(d, v, n, s='.'):
    ns = n.split(s)
    if isinstance(d, dict):
      k = ns[0]
    elif isinstance(d, list):
      k = int(ns[0])
    else:
      raise ValueError(d)
    if len(ns) > 1:
      Optimizer.update(d[k], v, s.join(ns[1:]), s)
    else:
      if isinstance(d[k], dict):
        for kk in d[k].keys():
          Optimizer.update(d[k], v, kk, s)
      elif isinstance(d[k], list):
        for i, _ in enumerate(d[k]):
          Optimizer.update(d[k], v, str(i), s)
      elif isinstance(d[k], (str, int, float, bool)) or d[k] is None:
        d[k] = v
      else:
        raise ValueError(k, d[k])
  
  class Logger:
    def __init__(self, export_metrics=None, export_sysinfo=None):
      self.export_metrics = export_metrics
      self.export_sysinfo = export_sysinfo
    
    @staticmethod
    def get_ip():
      ip = '127.0.0.1'
      if platform.system() == 'Windows':  # VPN
        ip = socket.gethostbyname(socket.getfqdn())
      else:  # Linux, Darwin
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
          s.connect(('10.255.255.255', 1))
          ip = s.getsockname()[0]
        except Exception:
          pass
        finally:
          s.close()
      return ip
    
    @staticmethod
    def get_sysinfo():
      d = {'platform': platform.system(),
           'release': platform.release(),
           'version': platform.version(),
           'architecture': platform.machine(),
           'processor': platform.processor(),
           'hostname': socket.gethostname(),
           'fqdn': socket.getfqdn(),
           'ip': Optimizer.Logger.get_ip()}
      return d
    
    @staticmethod
    def get_best_trial(study):
      d = {}
      if study._is_multi_objective():
        t = study.best_trials[0]
      else:
        t = study.best_trial
      d = {'number': t.number,
           'state': t.state,
           'value': t.value,
           'values': t.values,
           'datetime_start': t.datetime_start,
           'datetime_complete': t.datetime_complete,
           'params': t.params,
           'user_attrs': t.user_attrs}
      return d
    
    @staticmethod
    def dump(d, p):
      p = Path(p)
      p.parent.mkdir(parents=True, exist_ok=True)
      with open(p, 'w') as f:
        if p.suffix == '.json':
          json.dump(d, f, indent=2, sort_keys=True, default=str)
        elif p.suffix == '.yaml':
          yaml.safe_dump(d, f)
        else:
          raise NotImplementedError(p)
    
    def __call__(self, study=None, trial=None):
      if self.export_metrics is not None:
        d = {'best': self.get_best_trial(study)}
        self.dump(d, self.export_metrics)
      if self.export_sysinfo is not None:
        d = self.get_sysinfo()
        self.dump(d, self.export_sysinfo)
  
  def subprocess(self):
    subprocess_kwargs = copy.deepcopy(self.subprocess_kwargs)
    stdout_kwargs = copy.deepcopy(self.stdout_kwargs)
    stderr_kwargs = copy.deepcopy(self.stderr_kwargs)
    results = []
    for sub_kws, out_kws, err_kws in zip(
      subprocess_kwargs, stdout_kwargs, stderr_kwargs):
      stdout = sub_kws.get('stdout', None)
      if stdout is not None:
        if stdout in ['PIPE', 'STDOUT', 'DEVNULL']:
          stdout = getattr(subprocess, stdout)
        else:  # FILE
          stdout = open(file=stdout, **out_kws)
        sub_kws['stdout'] = stdout
      stderr = sub_kws.get('stderr', None)
      if stderr is not None:
        if stderr in ['PIPE', 'STDOUT', 'DEVNULL']:
          stderr = getattr(subprocess, stderr)
        else:  # FILE
          stderr = open(file=stderr, **err_kws)
        sub_kws['stderr'] = stderr
      result = subprocess.run(**sub_kws)
      if isinstance(stdout, io.IOBase) and not stdout.closed:
        stdout.close()
      if isinstance(stderr, io.IOBase) and not stderr.closed:
        stderr.close()
      results.append(result)
    return results
  
  def set_params(self, trial):
    for path, kwargs in self.params.items():
      local_path, global_path = self.split_path(p=path, s=self.gsep)
      if local_path is not None:
        kwargs['method_kwargs'].setdefault('name', path)
        v = self.suggest(trial, kwargs['method'], kwargs['method_kwargs'])
        self.update_file(p=global_path, v=v, n=local_path, s=self.lsep)
      else:
        raise NotImplementedError(path)
  
  def set_attrs(self, trial):
    attrs = {}
    for path, flag in self.attrs.items():
      if not flag:
        continue
      local_path, global_path = self.split_path(p=path, s=self.gsep)
      r = self.parse_file(p=global_path, s=self.lsep)
      if local_path is None:
        for k, v in r.items():
          attrs[self.gsep.join([k, global_path])] = v
      else:
        for k, v in r.items():
          if k.startswith(local_path):
            attrs[self.gsep.join([k, global_path])] = v
    print(f'Attributes: {len(attrs)}')
    pprint(attrs)
    for k, v in attrs.items():
      trial.set_user_attr(k, v)
  
  def get_objectives(self, trial):
    objectives = {}
    for path, direction in self.objectives.items():
      local_path, global_path = self.split_path(p=path, s=self.gsep)
      r = self.parse_file(p=global_path, s=self.lsep)
      if local_path is not None:
        objectives[path] = r[local_path]
      else:
        raise NotImplementedError(path)
    return tuple(objectives.values())
  
  def objective(self, trial):
    self.set_params(trial)
    r = self.subprocess()
    if all([x.returncode == 0 for x in r]):
      self.set_attrs(trial)
      return self.get_objectives(trial)
    else:
      return None

  def __call__(self):
    if not self.results_only:
      s = self.initialize_sampler(self.sampler, self.sampler_kwargs)
      p = self.initialize_pruner(self.pruner, self.pruner_kwargs)
      directions = list(self.objectives.values())
      if len(directions) == 1:  # single objective
        study = optuna.create_study(storage=self.url,
                                    study_name=self.study,
                                    load_if_exists=self.load_if_exists,
                                    sampler=s, pruner=p,
                                    direction=directions[0])
      else:  # multi-objective
        study = optuna.create_study(storage=self.url,
                                    study_name=self.study,
                                    load_if_exists=self.load_if_exists,
                                    sampler=s, pruner=p,
                                    directions=directions)
      # Log sysinfo
      Optimizer.Logger(export_metrics=None, export_sysinfo=self.export_sysinfo)()
      # Log metrics after each trial
      callbacks = [Optimizer.Logger(export_metrics=self.export_metrics,
                                    export_sysinfo=None)]
      # Optimize
      study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout,
                     callbacks=callbacks)
    else:
      study = optuna.load_study(storage=self.url, study_name=self.study)
    if self.export_csv is not None:
      df = study.trials_dataframe()
      p = Path(self.export_csv)
      p.parent.mkdir(parents=True, exist_ok=True)
      df.to_csv(self.export_csv, index=False)
