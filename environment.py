"""MASAEnv — authoritative, args-driven SimPy environment (safe refactor).

This file provides an environment surface that:
- Accepts a centralized `args` namespace (from MARL.common.arguments) and
  maps canonical fields (n_agents, num_operators, n_actions, episode_limit,
  obs/state dims, job generation bounds, reward params) into the runtime.
- Respects auto_* flags (auto_load_config, auto_build, auto_start_arrivals)
  so construction is pure by default and eager behaviors are opt-in.
- Delegates YAML parsing / machine registry building to utils when available,
  and keeps minimal safe fallbacks so tests can import the module during
  staged refactor.

The implementation intentionally keeps side-effects low (logging only) and
tries to use `utils` components when present. The goal is to be a stable
authoritative surface for the rest of the code while we iteratively move
full runtime behavior into this module.
"""
from __future__ import annotations
import logging
import random
import time
from collections import deque
from typing import Optional, Any, Dict, List

import numpy as np
import simpy
import os
import json
from utils.env_obs import build_agent_obs, build_state_vector  # type: ignore
try:
    from utils.io_control import allow_history_writes
except Exception:
    def allow_history_writes():
        return False
try:
    from utils.operator import Operators  # type: ignore
except Exception:
    Operators = None


# NOTE: training hyperparameters and reward shaping defaults have been
# moved to MARL.common.arguments (arguments.py). This module no longer
# defines DEFAULT_ENV_PARAMS or any training-default mirrors; MASAEnv
# consumes hyperparameters from an injected `args` namespace (preferably
# from MARL.common.arguments.get_common_args()).


# IMPORTANT: Do NOT import or call argument parsing here. The environment
# must be driven only by injected configuration (args namespace or explicit
# keyword arguments). Orchestrators (main/Runner) are the single source of
# truth for runtime configuration.


# Try to pull richer helpers from utils; fall back to minimal local shims.
try:
    from utils.workcenter import WorkCenters  # type: ignore
except Exception as e:
    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
    class WorkCenters:
        def __init__(self):
            self.machine_registry = {}
            self.machine_list = []
            self.machine_index = {}
            self.eligible_operator_groups_by_wc = {}


try:
    from utils.jobagent import JobAgent  # type: ignore
except Exception as e:
    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
    class JobAgent:
        def __init__(self, job_id: int, operations: List):
            self.id = int(job_id)
            self.operations = list(operations)
            self.current_op_idx = 0
            self.remaining_time = 0.0
            self.wait_time = 0.0
            self.finished = False
            self.arrival_time = 0.0

        def current_op(self):
            if self.finished or self.current_op_idx >= len(self.operations):
                return None
            return self.operations[self.current_op_idx]

        def progress_ratio(self):
            return float(self.current_op_idx) / max(1.0, float(len(self.operations)))


LOG = logging.getLogger(__name__)


class MASAEnv:
    """Argument-driven MASA environment with safe fallbacks.

    Constructor accepts:
    - args: optional namespace (prefer MARL.common.arguments.get_common_args())
    - seed: deterministic seed
    - reward_weights: optional overrides
    - config_path: path to YAML (if auto_load_config True)
    - strict_mode: fail-fast on config gaps when True
    - auto_load_config, auto_build, auto_start_arrivals: opt-in eager behavior

    Public API kept intentionally compatible with existing code/tests:
    - reset(), step(), wait_for_decisions(), pop_decision_reward(),
      get_env_info(), add_job(), print_jobs_human_readable().
    """

    def __init__(
        self,
        args: Optional[Any] = None,
        seed: int = 42,
        reward_weights: Optional[Dict[str, float]] = None,
        config_path: Optional[str] = None,
        strict_mode: bool = True,
        auto_load_config: bool = False,
        auto_build: bool = True,
        auto_start_arrivals: bool = False,
        use_yaml_config: bool = False,
        **kwargs,
    ):
        # Do NOT perform any global argument parsing here. The env must be
        # provided an `args` namespace or explicit keyword arguments. If both
        # are missing for a required field, fail fast so callers (orchestrator)
        # can correct the injection.
        # Ensure we have a canonical args namespace so hyperparameter defaults
        # live in MARL.common.arguments. If the caller didn't provide `args`,
        # import the common args default set.
        if args is None:
            try:
                from MARL.common.arguments import get_common_args  # type: ignore
                args = get_common_args()
            except Exception:
                # leave args as None if import fails; _resolve will still
                # attempt kwargs before failing when required fields missing
                args = None
        self.args = args

        # -----------------------
        # Safe seed initialization
        # -----------------------
        try:
            # prefer seed from args if provided
            if args is not None and hasattr(args, 'seed') and getattr(args, 'seed') is not None:
                self.seed = int(getattr(args, 'seed'))
                print(f"[Env Init] Using provided seed: {self.seed}")
            else:
                # fallback: use constructor seed if given
                if seed is not None:
                    try:
                        self.seed = int(seed)
                        print(f"[Env Init] Using constructor seed: {self.seed}")
                    except Exception:
                        self.seed = None
                else:
                    self.seed = None

            if self.seed is None:
                # deterministic generation from system clock
                self.seed = int(time.time() * 1000) % (2 ** 32)
                print(f"[Env Init] No seed provided — generated new seed: {self.seed}")
                print(f"[Reproducibility Tip] To reproduce this exact run, re-launch with --seed {self.seed}")

        except Exception:
            self.seed = int(seed) if seed is not None else int(time.time() * 1000) % (2 ** 32)

        # set global RNGs to avoid None or conflicting seeds downstream
        try:
            random.seed(self.seed)
        except Exception:
            pass
        try:
            np.random.seed(self.seed)
        except Exception:
            pass

        # maintain both RandomState and random.Random for backward compat
        self._np_rng = np.random.RandomState(self.seed)
        self._py_rng = random.Random(self.seed)
        # Control whether MASAEnv should prefer YAML-defined processing times
        # over the internal WorkCenters fallback.
        self.use_yaml_config = bool(use_yaml_config)
        # Control whether MASAEnv should write a runtime config dump to disk.
        # Opt-in only: default is False. Caller may set args.dump_config=True
        # or pass dump_config=True in kwargs to enable.
        try:
            self.dump_config = bool(getattr(args, 'dump_config', False)) or bool(kwargs.get('dump_config', False))
        except Exception:
            self.dump_config = bool(kwargs.get('dump_config', False))

        # If a config_path is provided, attempt to load YAML and populate
        # common missing kwargs so callers can pass only config_path in tests.
        # This is optional and defensive: if PyYAML is not available or the
        # file cannot be read, we silently continue and leave resolution to
        # explicit kwargs or args.
        if config_path and isinstance(config_path, str):
            try:
                try:
                    import yaml  # type: ignore
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    yaml = None
                cfg = None
                if yaml is not None and os.path.exists(config_path):
                    with open(config_path, 'r') as fh:
                        cfg = yaml.safe_load(fh) or {}
                if isinstance(cfg, dict):
                    # infer number of operators
                    if 'num_operators' not in kwargs and 'num_ops' not in kwargs:
                        ops = cfg.get('operators')
                        if isinstance(ops, list):
                            kwargs['num_operators'] = int(len(ops))

                    # infer number of workcenters / actions
                    if 'num_wcs' not in kwargs and 'n_actions' not in kwargs:
                        wcs = cfg.get('work_centers') or cfg.get('workcenters')
                        if isinstance(wcs, dict):
                            kwargs['num_wcs'] = int(len(wcs))
                        else:
                            machines = cfg.get('machines', {})
                            if isinstance(machines, dict):
                                wc_set = set()
                                for mdata in machines.values():
                                    if isinstance(mdata, dict):
                                        wc = mdata.get('wc') or mdata.get('workcenter') or mdata.get('work_center')
                                        if wc is not None:
                                            wc_set.add(str(wc))
                                if wc_set:
                                    kwargs['num_wcs'] = int(len(wc_set))

                    # task generator seq length -> job_min_ops / job_max_ops
                    tg = cfg.get('task_generator', {}) or {}
                    seq = tg.get('seq_length') if isinstance(tg, dict) else None
                    if isinstance(seq, dict):
                        if 'job_min_ops' not in kwargs:
                            kwargs['job_min_ops'] = int(seq.get('min', 1))
                        if 'job_max_ops' not in kwargs:
                            kwargs['job_max_ops'] = int(seq.get('max', max(1, kwargs.get('job_min_ops', 5))))
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                # keep behavior robust for environments without yaml or malformed files
                try:
                    self.logger.debug("config_path load skipped or failed: %s", config_path)
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    pass

        # At this point `cfg` may contain the parsed YAML (if any). Merge the
        # in-module DEFAULT_ENV_PARAMS with the provided config/YAML when the
        # caller requested auto-loading of config values. This produces a
        # merged `self.config` that fills missing sections from defaults.
        try:
            # prefer explicit kwargs['config'] if provided, else use parsed cfg
            initial_cfg = kwargs.get('config', None) if isinstance(kwargs.get('config', None), dict) else (cfg if 'cfg' in locals() and isinstance(cfg, dict) else None)
            # Only perform the merge if the caller opted into auto_load_config.
            if auto_load_config:
                # When auto-loading config, adopt the provided YAML/config
                # without merging training/training-default mirrors — all
                # hyperparameters are owned by MARL.common.arguments.
                self.config = initial_cfg or {}
                logging.getLogger(__name__).debug("Loaded config into MASAEnv.config")
            else:
                # preserve explicit config if provided, otherwise keep None
                self.config = initial_cfg
        except Exception:
            # fall back to any explicit config or None
            self.config = kwargs.get('config', None)

        # Ensure self.config is a dict for safe access
        if not isinstance(self.config, dict):
            self.config = kwargs.get('config', {}) or {}

        # If requested, prefer reading processing_time_means from YAML files
        # (cfg parsed earlier). Only do this when use_yaml_config is True.
        try:
            if self.use_yaml_config and isinstance(cfg, dict):
                proc_from_yaml = cfg.get('processing_time_means') if isinstance(cfg, dict) else None
                if isinstance(proc_from_yaml, dict):
                    self.config.setdefault('processing_time_means', proc_from_yaml)
        except Exception:
            pass

        # authoritative parameters from args, but allow kwargs to override common synonyms
        # Helper to resolve a parameter from kwargs first, then args namespace.
        def _resolve(name_variants, cast=int, default=None):
            # name_variants: iterable of possible keys to look up in kwargs
            for n in name_variants:
                if n in kwargs:
                    return cast(kwargs[n])
            if args is not None:
                for n in name_variants:
                    if hasattr(args, n):
                        return cast(getattr(args, n))
            if default is not None:
                try:
                    return cast(default)
                except Exception:
                    return default
            raise ValueError(f"MASAEnv requires one of {list(name_variants)} to be provided via args or kwargs")

        # Core counts / shapes — fail fast if not injected
        # Allow reasonable defaults so tests and lightweight callers can
        # construct environments without full args injection.
        self.num_jobs = _resolve(('n_agents', 'num_jobs'), int, default=0)
        self.num_ops = _resolve(('num_operators', 'num_ops'), int, default=1)
        # n_actions / num_wcs: allow 'n_actions' or 'num_wcs'
        self.num_wcs = _resolve(('num_wcs', 'n_actions'), int, default=1)
        # Observation/state shapes are owned by arguments.py (args.obs_shape / args.state_shape)
        try:
            if 'obs_dim_agent' in kwargs:
                self.obs_dim_agent = int(kwargs.get('obs_dim_agent'))
            elif args is not None and hasattr(args, 'obs_shape'):
                self.obs_dim_agent = int(getattr(args, 'obs_shape'))
            else:
                self.obs_dim_agent = 11
        except Exception:
            self.obs_dim_agent = 11

        try:
            if 'state_dim' in kwargs:
                self.state_dim = int(kwargs.get('state_dim'))
            elif args is not None and hasattr(args, 'state_shape'):
                self.state_dim = int(getattr(args, 'state_shape'))
            else:
                self.state_dim = 64
        except Exception:
            self.state_dim = 64
        # Increase default episode length so initial jobs have time to start
        # and progress. Callers may still override via args or kwargs.
        self.episode_limit = _resolve(('episode_limit',), int, default=300)

        # reward params — prefer explicit reward_weights dict, else require
        # presence in args or kwargs
        if reward_weights:
            self.alpha = float(reward_weights.get('alpha'))
            self.beta = float(reward_weights.get('beta'))
            self.gamma = float(reward_weights.get('gamma'))
            self.delta = float(reward_weights.get('delta'))
            self.c_time = float(reward_weights.get('c_time'))
        else:
            # Read reward shaping hyperparameters from the args namespace
            try:
                if args is not None:
                    self.alpha = float(getattr(args, 'reward_alpha', 0.0))
                    self.beta = float(getattr(args, 'reward_beta', 0.0))
                    self.gamma = float(getattr(args, 'reward_gamma', 0.0))
                    self.delta = float(getattr(args, 'reward_delta', 0.0))
                    self.c_time = float(getattr(args, 'reward_c_time', 0.0))
                else:
                    # fallback for callers not providing args
                    self.alpha = float(_resolve(('reward_alpha', 'alpha'), float, default=0.0))
                    self.beta = float(_resolve(('reward_beta', 'beta'), float, default=0.0))
                    self.gamma = float(_resolve(('reward_gamma', 'gamma'), float, default=0.0))
                    self.delta = float(_resolve(('reward_delta', 'delta'), float, default=0.0))
                    self.c_time = float(_resolve(('reward_c_time', 'c_time'), float, default=0.0))
            except Exception:
                # last-resort defaults
                self.alpha = 0.0
                self.beta = 0.0
                self.gamma = 0.0
                self.delta = 0.0
                self.c_time = 0.0

        # job generation params
        self.job_min_ops = int(_resolve(('job_min_ops',), int, default=1))
        self.job_max_ops = int(_resolve(('job_max_ops',), int, default=5))

        # options controlling eager behaviors
        self.strict_mode = bool(strict_mode)
        self.auto_load_config = bool(auto_load_config)
        self.auto_build = bool(auto_build)
        self.auto_start_arrivals = bool(auto_start_arrivals)

        # logging (do not configure root logger here)
        self.logger = LOG

        # Do NOT perform any YAML/config loading here. If a higher-level
        # orchestrator wishes to build a config and inject `workcenters_meta`
        # or `config`, it should pass it via kwargs (e.g., workcenters_meta=...).
        # `self.config` was set earlier according to auto_load_config and parsed YAML.
        self.config_path = config_path

        # Build workcenters metadata using WorkCenters or fallback shim
        try:
            # Prefer factory from utils if available
            # Allow callers to inject a pre-built WorkCenters instance via
            # kwargs; otherwise, construct an empty WorkCenters (no YAML parsing
            # or environment reads here).
            wc = kwargs.get('workcenters_meta', None)
            if wc is None:
                wc = WorkCenters()
            self.workcenters_meta = wc
        except Exception as e:
            self.logger.warning("WorkCenters construction failed: %s", e)
            self.workcenters_meta = WorkCenters()

        # Placeholder for Operators manager. The concrete Operators instance
        # is created after the SimPy Environment is initialized below so we
        # can pass the real env for per-operator resources. Initialize to
        # None for now.
        self.operators = None

        # At this point workcenters_meta exists. If processing_time_means has
        # still not been provided (or use_yaml_config was False), synthesize
        # processing_time_means from the WorkCenters fallback DEFAULT_PROCESSING_TIMES.
        try:
            if not isinstance(self.config.get('processing_time_means', None), dict):
                # Try workcenters_meta first, then class-level DEFAULT_PROCESSING_TIMES
                wc_defaults = None
                try:
                    # first check if the WorkCenters instance exposes a fallback
                    wc_defaults = getattr(self.workcenters_meta, 'DEFAULT_PROCESSING_TIMES', None)
                except Exception:
                    wc_defaults = None
                if wc_defaults is None:
                    try:
                        # DEFAULT_PROCESSING_TIMES lives at the module level in
                        # utils.workcenter. Import the module and read it.
                        import utils.workcenter as _wc_mod  # type: ignore
                        wc_defaults = getattr(_wc_mod, 'DEFAULT_PROCESSING_TIMES', None)
                    except Exception:
                        wc_defaults = None

                if isinstance(wc_defaults, dict):
                    # transpose machine->op->val into op->machine->val
                    proc_by_op = {}
                    for mname, ops_map in wc_defaults.items():
                        for opname, v in (ops_map or {}).items():
                            try:
                                proc_by_op.setdefault(opname, {})[mname] = float(v)
                            except Exception:
                                proc_by_op.setdefault(opname, {})[mname] = v
                    self.config.setdefault('processing_time_means', proc_by_op)
                else:
                    self.config.setdefault('processing_time_means', {})
        except Exception:
            self.config.setdefault('processing_time_means', {})

        # ensure machine_list exists
        try:
            if not getattr(self.workcenters_meta, 'machine_list', None):
                # synthesize machine_list if missing and num_wcs available
                if self.num_wcs is not None:
                    self.workcenters_meta.machine_list = [f"M{i}" for i in range(int(self.num_wcs))]
                else:
                    # ensure at least one machine so tests that inspect registry succeed
                    self.workcenters_meta.machine_list = ["M0"]
                self.workcenters_meta.machine_index = {m: i for i, m in enumerate(self.workcenters_meta.machine_list)}
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            self.workcenters_meta.machine_list = list(getattr(self.workcenters_meta, 'machine_registry', {}).keys())

        # If num_wcs still None, derive from machine_list length
        if self.num_wcs is None:
            try:
                self.num_wcs = len(getattr(self.workcenters_meta, 'machine_list', []) or [])
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                self.num_wcs = 1

        # SimPy runtime primitives
        self.env = simpy.Environment()
        # create per-machine resources if machine list present, else per-wc resources
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        if mlist:
            self.machine_resources = [simpy.Resource(self.env, capacity=1) for _ in range(len(mlist))]
            self.wc_resources = [simpy.Resource(self.env, capacity=1) for _ in range(int(self.num_wcs))]
        else:
            self.machine_resources = []
            self.wc_resources = [simpy.Resource(self.env, capacity=1) for _ in range(int(self.num_wcs))]

        # operator resources
        try:
            self.operator_groups = [simpy.Resource(self.env, capacity=1) for _ in range(max(1, int(self.num_ops)))]
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            self.operator_groups = [simpy.Resource(self.env, capacity=1) for _ in range(1)]

        # Instantiate Operators manager now that self.env exists. Creating
        # Operators earlier (before env was created) would prevent per-operator
        # SimPy resources from being created, so do it here and attach to env.
        try:
            if Operators is not None:
                self.operators = Operators(self.workcenters_meta, env=self.env)
            else:
                self.operators = None
        except Exception:
            self.operators = None

        # migration alias
        self.workcenters = self.workcenters_meta

        # job list and generation
        self.jobs: List[JobAgent] = []
        # pending jobs queued when capacity reached
        self.pending_jobs: List[List] = []
        # active jobs tracked to enforce capacity (n_agents)
        self.active_jobs: List[JobAgent] = []

        # initial_jobs: independent from n_agents (max capacity)
        try:
            self.initial_jobs = int(_resolve(('initial_jobs',), int, default=getattr(args, 'initial_jobs', 4)))
        except Exception:
            self.initial_jobs = int(getattr(args, 'initial_jobs', 4))

        # Ensure TaskGenerator exists and is owner-aware
        try:
            from utils.task_generator import TaskGenerator  # type: ignore
            # Pass RNGs for deterministic job creation when available
            self.job_generator = TaskGenerator(config_path=self.config_path, py_rng=self._py_rng, np_rng=self._np_rng)
            try:
                setattr(self.job_generator, '_owner_env', self)
                # ensure TaskGenerator uses the MASAEnv RNG for deterministic draws
                try:
                    setattr(self.job_generator, '_py_rng', self._py_rng)
                except Exception:
                    pass
            except Exception:
                pass
        except Exception as e:
            logging.getLogger(__name__).exception("TaskGenerator not available: %s", e, exc_info=True)
            self.job_generator = None

        # Build initial jobs deterministically via TaskGenerator when requested
        if self.auto_build:
            try:
                if self.job_generator is not None:
                    # create and add initial jobs using job_generator
                    for _ in range(max(0, int(self.initial_jobs))):
                        try:
                            ops = self.job_generator.create_job()
                            job = self.add_job(ops, start_immediately=True, set_arrival_zero=True)
                        except Exception as e:
                            logging.getLogger(__name__).exception("Failed to create initial job: %s", exc_info=True)
                else:
                    # fallback: nothing
                    self.jobs = []
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught while generating initial jobs", exc_info=True)
                self.jobs = []

        # Startup summary: print and optionally persist to historydata
        try:
            summary = {
                'seed': int(getattr(self, 'seed', -1)),
                'n_agents_capacity': int(getattr(self, 'num_jobs', 0)),
                'initial_jobs_requested': int(getattr(self, 'initial_jobs', 0)),
                'initial_jobs_created': len(self.jobs),
                'job_min_ops': int(getattr(self, 'job_min_ops', 0)),
                'job_max_ops': int(getattr(self, 'job_max_ops', 0)),
                'episode_limit': int(getattr(self, 'episode_limit', 0)),
            }
            # print concise startup summary to stdout for visibility
            try:
                print(f"[Env Summary] seed={summary['seed']} capacity={summary['n_agents_capacity']} initial_created={summary['initial_jobs_created']}")
            except Exception:
                pass

            # Persist if allowed by global gate
            try:
                if allow_history_writes():
                    hist_dir = getattr(self.args, 'history_dir', os.path.join('my_data_and_graph', 'historydata')) if getattr(self, 'args', None) is not None else os.path.join('my_data_and_graph', 'historydata')
                    os.makedirs(hist_dir, exist_ok=True)
                    path = os.path.join(hist_dir, 'env_summary.json')
                    with open(path, 'w') as fh:
                        json.dump(summary, fh, indent=2, sort_keys=True)
                    logging.getLogger(__name__).debug('Wrote env summary to %s', path)
            except Exception:
                logging.getLogger(__name__).exception('Failed to write env_summary.json', exc_info=True)
        except Exception:
            logging.getLogger(__name__).exception('Failed to build env startup summary', exc_info=True)

        # bookkeeping and reward caches
        self.t = 0.0
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self._completed_now_cache = 0
        self._recent_rewards = deque(maxlen=20)
        self.done = False

        # decision batching
        self.pending_decisions: List[Dict] = []
        self.decisions_ready = simpy.Event(self.env)
        self.gantt_records: List = []

        # Optionally dump merged configuration for runtime debugging/validation.
        # This is opt-in only (see self.dump_config). When disabled, no file is
        # written to avoid treating the dump as a data source.
        try:
            if bool(getattr(self, 'dump_config', False)):
                dump_dir = os.path.join('my_data_and_graph', 'historydata')
                os.makedirs(dump_dir, exist_ok=True)
                dump_path = os.path.join(dump_dir, 'env_config_dump.json')
                # Use json.dump with default=str to ensure non-serializable objects don't break the dump.
                with open(dump_path, 'w') as fh:
                    json.dump(self.config if self.config is not None else {}, fh, indent=2, sort_keys=True, default=str)
                logging.getLogger(__name__).debug('Wrote env config dump to %s', dump_path)
        except Exception:
            logging.getLogger(__name__).exception('Failed to write env_config_dump.json', exc_info=True)

    # ---------------- Public API ----------------
    def reset(self):
        """Reset runtime state; keep configuration and metadata intact."""
        self.env = simpy.Environment()
        # recreate resources
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        if mlist:
            self.machine_resources = [simpy.Resource(self.env, capacity=1) for _ in range(len(mlist))]
            self.wc_resources = [simpy.Resource(self.env, capacity=1) for _ in range(int(self.num_wcs))]
        else:
            self.machine_resources = []
            self.wc_resources = [simpy.Resource(self.env, capacity=1) for _ in range(int(self.num_wcs))]

        self.operator_groups = [simpy.Resource(self.env, capacity=1) for _ in range(max(1, int(self.num_ops)))]

        # recreate Operators manager on reset as well
        try:
            if Operators is not None:
                self.operators = Operators(self.workcenters_meta, env=self.env)
            else:
                self.operators = None
        except Exception:
            self.operators = None

        # reset bookkeeping
        self.t = 0.0
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self._completed_now_cache = 0
        self._recent_rewards.clear()
        self.done = False
        self.pending_decisions = []
        self.decisions_ready = simpy.Event(self.env)
        self.gantt_records = []

        # regenerate jobs if desired
        if self.auto_build:
            try:
                self._generate_initial_jobs()
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                self.jobs = []

        # start job processes
        try:
            for job in self.jobs:
                self.env.process(self._job_process(job))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass

        # Optionally start an external TaskGenerator if config requests arrivals.
        try:
            tg_conf = self.config.get('task_generator') if isinstance(self.config, dict) else None
            if tg_conf and float(tg_conf.get('arrival_lambda', 0)) > 0:
                try:
                    from utils.task_generator import TaskGenerator  # type: ignore
                    # Pass the MASAEnv instance (self) so TaskGenerator can call
                    # env.add_job on the environment surface. TaskGenerator.start
                    # will use the internal simpy.Environment for scheduling.
                    self._task_generator = TaskGenerator(config_path=self.config_path, py_rng=self._py_rng, np_rng=self._np_rng)
                    # Allow the TaskGenerator to reference the MASAEnv instance
                    # for dynamic job injection without changing the start() call
                    # signature expected by tests (which expect start(env, lam)
                    # to be invoked with the SimPy environment).
                    try:
                        setattr(self._task_generator, '_owner_env', self)
                        try:
                            setattr(self._task_generator, '_py_rng', self._py_rng)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Start the arrival process on the underlying simpy.Environment
                    self._task_generator.start(self.env, float(tg_conf.get('arrival_lambda')))
                except Exception as e:
                    logging.getLogger(__name__).exception("TaskGenerator start failed", exc_info=True)
                    self._task_generator = None
        except Exception:
            # defensive: don't let arrival logic break reset
            self._task_generator = None

        return self._build_all_agent_obs(), {"state_vec": None, "avail_actions": None}

    def step(self, action=None):
        """Compatibility step API: advance sim by a time-quantum and return (obs, reward, done, info)."""
        try:
            self.env.run(until=self.env.now + 1.0)
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass
        self.t = float(getattr(self.env, 'now', 0.0))
        try:
            reward = float(self.pop_decision_reward())
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            reward = 0.0
        obs = self._build_all_agent_obs()
        info = {"state_vec": None, "avail_actions": None}
        done = (self.t >= self.episode_limit) or all(j.finished for j in self.jobs)
        self.done = self.done or done
        return obs, float(reward), bool(done), info

    def wait_for_decisions(self):
        """Run the sim until at least one decision is pending or episode ends.

        Returns: (batch, sim_time)
        """
        if self.done:
            return [], float(self.t)

        if not self.pending_decisions:
            step = float(getattr(self, '_debug_run_step', 1.0))
            # advance in small steps until decisions appear or episode ends
            # Prefer consult an owner wrapper (if TaskGenerator or other
            # wrapper set `_owner_env`) for stop-condition attributes. Also
            # guard against cases where the underlying simpy.Environment has
            # no scheduled events (env.now does not advance) which would
            # otherwise cause an infinite loop here.
            wrapper = getattr(getattr(self, '_task_generator', None), '_owner_env', None) or getattr(self, '_owner_env', None)
            def _get_attr_from_env(attr, default=None):
                try:
                    if hasattr(self, attr):
                        return getattr(self, attr)
                except Exception:
                    pass
                try:
                    if wrapper is not None and hasattr(wrapper, attr):
                        return getattr(wrapper, attr)
                except Exception:
                    pass
                try:
                    if hasattr(self.env, attr):
                        return getattr(self.env, attr)
                except Exception:
                    pass
                return default

            no_progress = 0
            max_no_progress = int(getattr(self, '_wait_no_progress_limit', 3))
            try:
                while not self.pending_decisions and not bool(_get_attr_from_env('done', False)):
                    prev_now = float(getattr(self.env, 'now', 0.0))
                    # advance sim by a step; if nothing is scheduled, env.now
                    # may remain unchanged — detect and bail out after a few
                    # attempts to avoid infinite loops during tests.
                    self.env.run(until=prev_now + step)
                    new_now = float(getattr(self.env, 'now', prev_now))
                    if self.pending_decisions:
                        break
                    if new_now == prev_now:
                        no_progress += 1
                        if no_progress >= max_no_progress:
                            # nothing scheduled and no pending decisions —
                            # return empty batch so callers treat as done
                            logging.getLogger(__name__).debug("wait_for_decisions: no scheduled events after %d attempts; returning empty batch", no_progress)
                            return [], float(getattr(self.env, 'now', 0.0))
                    else:
                        no_progress = 0
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

        self.t = float(self.env.now)
        batch = list(self.pending_decisions)
        self.pending_decisions = []
        self.decisions_ready = simpy.Event(self.env)
        return batch, float(self.t)

    def pop_decision_reward(self) -> float:
        """Return shaped reward computed since last pop."""
        try:
            completed = float(self._completed_now_cache)
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            completed = 0.0
        self._completed_now_cache = 0
        sim_t = float(self.env.now) if hasattr(self, 'env') else 1.0
        avg_wait = float(self.total_wait_time) / max(1.0, sim_t)
        wip = float(self._wip())
        try:
            idle_ops = sum(1 for g in getattr(self, 'operator_groups', []) if self._resource_free(g))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            idle_ops = 0
        reward = (self.alpha * completed) - (self.beta * avg_wait) - (self.gamma * wip) - (self.delta * float(idle_ops))
        try:
            self._recent_rewards.append(reward)
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass
        return float(reward)

    # ---------------- SimPy job process (simple, robust) ----------------
    def _job_process(self, job: JobAgent):
        while not job.finished and self.env.now < self.episode_limit:
            op = job.current_op()
            if op is None:
                job.finished = True
                break

            # normalize op formats: support legacy (allowed_wcs, dur) and
            # canonical (op_type, allowed_wcs, per_wc_durations)
            op_type = None
            allowed_wcs = []
            per_wc = None
            base_dur = None
            if isinstance(op, (list, tuple)):
                if len(op) == 2:
                    allowed_wcs, base_dur = op
                elif len(op) >= 3:
                    op_type = op[0]
                    allowed_wcs = op[1]
                    per_wc = op[2]

            # create decision item — prefer workcenters_meta helper if present
            try:
                if hasattr(self.workcenters_meta, 'create_decision_item'):
                    decision_item, resume_evt = self.workcenters_meta.create_decision_item(self, job, op)
                else:
                    raise AttributeError
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                resume_evt = simpy.Event(self.env)
                decision_item = {
                    'job_id': job.id,
                    'obs': self._build_agent_obs(job),
                    'avail_row': self._avail_row_for_job(job),
                    'allowed_wcs': allowed_wcs,
                    'per_machine_durations': {},
                    'resume': lambda choice: None,
                }

            self.pending_decisions.append(decision_item)
            try:
                if not getattr(self.decisions_ready, 'triggered', False):
                    self.decisions_ready.succeed()
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            chosen_idx = (yield resume_evt)
            if chosen_idx is None:
                yield self.env.timeout(1e-9)
                continue

            # compute duration
            try:
                if int(chosen_idx) in decision_item.get('per_machine_durations', {}):
                    dur = float(decision_item.get('per_machine_durations', {}).get(int(chosen_idx), 0.0))
                elif per_wc is not None:
                    chosen_wc = int(allowed_wcs[int(chosen_idx)]) if allowed_wcs else 0
                    dur = float(per_wc.get(chosen_wc, 0.0)) if isinstance(per_wc, dict) else float(per_wc)
                elif base_dur is not None:
                    dur = float(base_dur)
                else:
                    dur = 0.0
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                dur = 0.0

            # acquire resources and execute
            try:
                mr = self.machine_resources[int(chosen_idx)] if self.machine_resources else self.wc_resources[int(0)]
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                mr = None
            if mr is None:
                yield self.env.timeout(1e-9)
                continue

            # pick operator group index based on eligible operator groups for the
            # chosen machine's workcenter. If multiple eligible groups exist,
            # select one using the environment RNG so operator assignment is
            # non-deterministic but reproducible when env._py_rng is seeded.
            try:
                # determine the workcenter index for the chosen machine (if possible)
                try:
                    mlist = getattr(getattr(self, 'workcenters_meta', None), 'machine_list', []) or []
                    machine_name = mlist[int(chosen_idx)] if mlist and 0 <= int(chosen_idx) < len(mlist) else f"M{int(chosen_idx)}"
                    wc_for_machine = getattr(self.workcenters_meta, 'workcenter_for_machine', None)
                    if callable(wc_for_machine):
                        wc_idx = int(wc_for_machine(machine_name))
                    else:
                        # fallback: treat chosen_idx as workcenter id
                        wc_idx = int(chosen_idx)
                except Exception:
                    wc_idx = int(chosen_idx)

                eligible_groups = getattr(getattr(self, 'workcenters_meta', None), 'eligible_operator_groups_by_wc', {}).get(int(wc_idx), []) or []
                if eligible_groups:
                    try:
                        # use env RNG when available for reproducible selection
                        if hasattr(self, '_py_rng') and getattr(self, '_py_rng', None) is not None:
                            selected_grp = int(self._py_rng.choice(list(eligible_groups)))
                        else:
                            # fallback deterministic pick
                            selected_grp = int(list(eligible_groups)[0])
                    except Exception:
                        selected_grp = int(list(eligible_groups)[0])
                else:
                    selected_grp = 0

                # Select a concrete operator and wait for availability using
                # the Operators manager. We no longer use a numeric "group:0"
                # fallback — if no operator is immediately free we wait in a
                # short loop until one becomes available. This enforces true
                # operator-level exclusivity via each Operator.resource.
                available_operator = None
                try:
                    if getattr(self, 'operators', None) is not None:
                        # Try to find a free operator that can perform this job
                        # at the desired workcenter. If none available, wait
                        # in short increments until one is free.
                        try:
                            available_operator = self.operators.find_free_operator(job.id, wc_idx)
                        except Exception:
                            available_operator = None
                        wait_count = 0
                        while available_operator is None:
                            # Wait a short amount of simulated time and retry
                            yield self.env.timeout(0.1)
                            wait_count += 1
                            try:
                                available_operator = self.operators.find_free_operator(job.id, wc_idx)
                            except Exception:
                                available_operator = None
                            # safety: if no operators manager exists or endless wait,
                            # continue looping (tests expect determinism so this
                            # should resolve when operators finish jobs)
                    else:
                        available_operator = None
                except Exception:
                    available_operator = None

                # If we found an operator object, request its per-operator
                # resource and the machine resource before starting the op.
                try:
                    if available_operator is not None and getattr(available_operator, 'resource', None) is not None:
                        try:
                            logging.getLogger(__name__).debug("Waiting -> starting job=%s on machine=%s by operator=%s time=%s", getattr(job, 'id', None), int(chosen_idx), getattr(available_operator, 'operator_id', None), float(self.env.now))
                        except Exception:
                            pass
                        with available_operator.resource.request() as opres_req, mr.request() as mc_req:
                            yield opres_req; yield mc_req
                            # We now hold the operator and machine resources.
                            wait_dur = self.env.now - getattr(job, 'arrival_time', self.env.now)
                            if wait_dur > 0:
                                job.wait_time += wait_dur
                                self.total_wait_time += wait_dur
                            op_start = float(self.env.now)
                            job.remaining_time = dur
                            # assign and mark busy via Operator.assign_job()
                            try:
                                available_operator.assign_job(job.id, wc_idx, start_time=op_start)
                            except Exception:
                                pass
                            try:
                                yield self.env.timeout(dur)
                            finally:
                                op_end = float(self.env.now)
                                # Persist a gantt record using the concrete operator id
                                op_id_for_record = str(getattr(available_operator, 'operator_id', 'UNKNOWN'))
                                try:
                                    # Print to stdout to ensure instrumentation is visible
                                    # in captured console output during test/dev runs.
                                    print(f"[GANTT-APPEND] concrete branch -> op_id_for_record={op_id_for_record!r} type={type(op_id_for_record)}")
                                    # Also write a small debug trace to the history directory so
                                    # it's persistent even if stdout is buffered or truncated.
                                    try:
                                        dbg_dir = getattr(self, 'history_dir', 'my_data_and_graph/historydata')
                                        import os, json
                                        os.makedirs(dbg_dir, exist_ok=True)
                                        dbg_path = os.path.join(dbg_dir, 'gantt_append_debug.log')
                                        # include episode stamp if available for easier grouping
                                        ep_stamp = getattr(self, 'current_episode', None)
                                        try:
                                            rec = [op_start, op_end, int(op_type) if op_type is not None else job.current_op_idx, int(chosen_idx), int(job.id), op_id_for_record, float(job.arrival_time), float(dur)]
                                            if ep_stamp is not None:
                                                rec.append(int(ep_stamp))
                                        except Exception:
                                            rec = [op_start, op_end, int(op_type) if op_type is not None else job.current_op_idx, int(chosen_idx), int(job.id), op_id_for_record, float(job.arrival_time), float(dur)]
                                        with open(dbg_path, 'a', encoding='utf-8') as df:
                                            df.write(json.dumps({'time': float(self.env.now), 'branch': 'concrete', 'op_id': op_id_for_record, 'op_id_type': str(type(op_id_for_record)), 'episode': ep_stamp, 'record': rec}) + '\n')
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                try:
                                    logging.getLogger(__name__).debug("Appending gantt record with operator id (concrete branch): %r (type=%s)", op_id_for_record, type(op_id_for_record))
                                except Exception:
                                    pass
                                # attach episode stamp to the record if present on env
                                try:
                                    ep_stamp = getattr(self, 'current_episode', None)
                                    rec_tuple = (op_start, op_end, int(op_type) if op_type is not None else job.current_op_idx, int(chosen_idx), int(job.id), op_id_for_record, float(job.arrival_time), float(dur))
                                    if ep_stamp is not None:
                                        rec_tuple = rec_tuple + (int(ep_stamp),)
                                    self.gantt_records.append(rec_tuple)
                                except Exception:
                                    # fallback to legacy append
                                    try:
                                        self.gantt_records.append((op_start, op_end, int(op_type) if op_type is not None else job.current_op_idx, int(chosen_idx), int(job.id), op_id_for_record, float(job.arrival_time), float(dur)))
                                    except Exception:
                                        pass
                                try:
                                    available_operator.release(end_time=op_end)
                                except Exception:
                                    pass
                    else:
                        # No concrete operator manager or resource available.
                        # In this case we fall back to waiting for the machine
                        # and execute without a concrete operator — but we do
                        # not record numeric/group fallbacks. Record operator as
                        # 'UNASSIGNED' to make missing assignment explicit.
                        try:
                            logging.getLogger(__name__).warning("No Operators manager or resources available; running job=%s without concrete operator at time=%s", getattr(job, 'id', None), float(self.env.now))
                        except Exception:
                            pass
                        with mr.request() as mc_req:
                            yield mc_req
                            wait_dur = self.env.now - getattr(job, 'arrival_time', self.env.now)
                            if wait_dur > 0:
                                job.wait_time += wait_dur
                                self.total_wait_time += wait_dur
                            op_start = float(self.env.now)
                            job.remaining_time = dur
                            try:
                                yield self.env.timeout(dur)
                            finally:
                                op_end = float(self.env.now)
                                op_id_for_record = 'UNASSIGNED'
                                try:
                                    # Print to stdout to ensure instrumentation is visible
                                    # in captured console output during test/dev runs.
                                    print(f"[GANTT-APPEND] unassigned branch -> op_id_for_record={op_id_for_record!r} type={type(op_id_for_record)}")
                                    # Persist debug append info to disk as well so we can
                                    # inspect appended values regardless of stdout capture.
                                    try:
                                        dbg_dir = getattr(self, 'history_dir', 'my_data_and_graph/historydata')
                                        import os, json
                                        os.makedirs(dbg_dir, exist_ok=True)
                                        dbg_path = os.path.join(dbg_dir, 'gantt_append_debug.log')
                                        ep_stamp = getattr(self, 'current_episode', None)
                                        try:
                                            rec = [op_start, op_end, int(op_type) if op_type is not None else job.current_op_idx, int(chosen_idx), int(job.id), op_id_for_record, float(job.arrival_time), float(dur)]
                                            if ep_stamp is not None:
                                                rec.append(int(ep_stamp))
                                        except Exception:
                                            rec = [op_start, op_end, int(op_type) if op_type is not None else job.current_op_idx, int(chosen_idx), int(job.id), op_id_for_record, float(job.arrival_time), float(dur)]
                                        with open(dbg_path, 'a', encoding='utf-8') as df:
                                            df.write(json.dumps({'time': float(self.env.now), 'branch': 'unassigned', 'op_id': op_id_for_record, 'op_id_type': str(type(op_id_for_record)), 'episode': ep_stamp, 'record': rec}) + '\n')
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                try:
                                    logging.getLogger(__name__).debug("Appending gantt record with operator id (unassigned branch): %r (type=%s)", op_id_for_record, type(op_id_for_record))
                                except Exception:
                                    pass
                                try:
                                    ep_stamp = getattr(self, 'current_episode', None)
                                    rec_tuple = (op_start, op_end, int(op_type) if op_type is not None else job.current_op_idx, int(chosen_idx), int(job.id), op_id_for_record, float(job.arrival_time), float(dur))
                                    if ep_stamp is not None:
                                        rec_tuple = rec_tuple + (int(ep_stamp),)
                                    self.gantt_records.append(rec_tuple)
                                except Exception:
                                    try:
                                        self.gantt_records.append((op_start, op_end, int(op_type) if op_type is not None else job.current_op_idx, int(chosen_idx), int(job.id), op_id_for_record, float(job.arrival_time), float(dur)))
                                    except Exception:
                                        pass
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    try:
                        yield self.env.timeout(max(1e-9, float(dur)))
                    except Exception:
                        return
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                # safety: advance a tiny amount to avoid deadlock
                try:
                    yield self.env.timeout(max(1e-9, float(dur)))
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    return

            job.current_op_idx += 1
            job.remaining_time = 0.0
            if job.current_op_idx >= len(job.operations):
                job.finished = True
                self.completed_jobs += 1
                self._completed_now_cache += 1
                # Capacity management: when a job finishes, free an active slot
                try:
                    if job in self.active_jobs:
                        try:
                            self.active_jobs.remove(job)
                        except Exception:
                            pass
                except Exception:
                    pass
                # If there are pending jobs, start the next one deterministically
                try:
                    if getattr(self, 'pending_jobs', None):
                        if len(self.pending_jobs) > 0:
                            next_job = self.pending_jobs.pop(0)
                            try:
                                # schedule and mark active
                                self.env.process(self._job_process(next_job))
                                self.active_jobs.append(next_job)
                            except Exception:
                                logging.getLogger(__name__).exception("Failed to start pending job", exc_info=True)
                except Exception:
                    logging.getLogger(__name__).exception("Failed in capacity dispatch", exc_info=True)

        if all(j.finished for j in self.jobs) or self.env.now >= self.episode_limit:
            self.done = True
            try:
                if not getattr(self.decisions_ready, 'triggered', False):
                    self.decisions_ready.succeed()
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

    # ---------------- Observation / State / Avail -----------------
    def _build_all_agent_obs(self):
        # Strict delegation to utils.env_obs.build_agent_obs. Missing helper
        # will raise ImportError at module import time so errors are explicit.
        return [build_agent_obs(self, j) for j in self.jobs]

    def _build_agent_obs(self, job: JobAgent):
        # Strict delegation to canonical helper. Let exceptions propagate for
        # clearer debugging when the helper is missing or fails.
        return build_agent_obs(self, job)

    def _build_avail_actions(self):
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        n_m = len(mlist) if mlist else int(self.num_wcs)
        avail = np.zeros((len(self.jobs), n_m), dtype=np.int32)
        for idx, j in enumerate(self.jobs):
            if j.finished:
                continue
            row = self._avail_row_for_job(j)
            if row is None:
                continue
            try:
                avail[idx, :] = row
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass
        return avail

    def _avail_row_for_job(self, job: JobAgent):
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        if mlist:
            row = np.zeros((len(mlist),), dtype=np.int32)
        else:
            row = np.zeros((int(self.num_wcs),), dtype=np.int32)
        op = job.current_op()
        if op is None:
            return row
        if isinstance(op, (list, tuple)) and len(op) == 2:
            allowed_wcs, _ = op
        else:
            try:
                _, allowed_wcs, _ = op
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                allowed_wcs = []
        try:
            for wc in allowed_wcs:
                if 0 <= int(wc) < row.shape[0]:
                    row[int(wc)] = 1
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass
        return row

    # ---------------- Helpers -----------------
    def _wip(self):
        return sum(1 for j in self.jobs if not j.finished)

    def _resource_free(self, res):
        try:
            return len(res.users) < res.capacity
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            return True

    # ---------------- Utilization helpers used by utils/env_obs.py -------
    def _util_machines(self) -> float:
        """Return a [0,1] utilization estimate for machines."""
        try:
            mres = getattr(self, 'machine_resources', []) or []
            if not mres:
                return 0.0
            busy = 0
            for r in mres:
                try:
                    if len(getattr(r, 'users', [])) > 0:
                        busy += 1
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    continue
            return float(busy) / max(1.0, float(len(mres)))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            return 0.0

    def _util_ops(self) -> float:
        """Return a [0,1] utilization estimate for operator groups."""
        try:
            ores = getattr(self, 'operator_groups', []) or []
            if not ores:
                return 0.0
            busy = 0
            for r in ores:
                try:
                    if len(getattr(r, 'users', [])) > 0:
                        busy += 1
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    continue
            return float(busy) / max(1.0, float(len(ores)))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            return 0.0

    def _build_state_vector(self):
        """Return the global state vector; used by tests and env_obs helper.

        Prefer utils.env_obs.build_state_vector when available; otherwise
        compute a small fallback identical in shape to expectations.
        """
        try:
            from utils.env_obs import build_state_vector  # type: ignore
            return np.asarray(build_state_vector(self))
        except Exception as e:
            # Safe fallback: log a warning and return a zero vector with the
            # expected state dimension so callers don't break.
            try:
                import numpy as _np
                print(f"[Warning] build_state_vector fallback due to: {e}")
                return _np.zeros((int(getattr(self, 'state_dim', 0)),), dtype=_np.float32)
            except Exception:
                return np.zeros((int(getattr(self, 'state_dim', 0)),), dtype=np.float32)

    def add_job(self, ops_sequence: List, start_immediately: bool = True, set_arrival_zero: bool = False):
        """Add a job (ops_sequence) to the environment.

        Args:
            ops_sequence: canonical list of ops (op_type, allowed_wcs, per_wc_durations)
            start_immediately: when True, schedule the job process on the SimPy env
            set_arrival_zero: when True, force arrival_time to 0.0 (used for initial jobs)
        Returns:
            JobAgent instance
        """
        jid = len(self.jobs)
        job = JobAgent(jid, ops_sequence)
        try:
            if set_arrival_zero:
                job.arrival_time = 0.0
            else:
                job.arrival_time = float(self.env.now)
        except Exception:
            job.arrival_time = 0.0
        # Append to master job list (arrival order)
        self.jobs.append(job)

        # Capacity enforcement: self.num_jobs represents the capacity (n_agents)
        try:
            capacity = int(getattr(self, 'num_jobs', 0)) or int(getattr(self.args, 'n_agents', 0))
        except Exception:
            capacity = int(getattr(self.args, 'n_agents', 0)) if getattr(self, 'args', None) is not None else 0

        if start_immediately:
            # If we have room, start the job and record it as active. Otherwise
            # queue it in pending_jobs to be started when capacity frees.
            try:
                if capacity <= 0 or len(self.active_jobs) < int(capacity):
                    # start now
                    try:
                        self.env.process(self._job_process(job))
                        self.active_jobs.append(job)
                    except Exception:
                        logging.getLogger(__name__).exception("Exception caught while starting job", exc_info=True)
                else:
                    # queue for later start
                    try:
                        self.pending_jobs.append(job)
                    except Exception:
                        logging.getLogger(__name__).exception("Failed to queue pending job", exc_info=True)
            except Exception:
                logging.getLogger(__name__).exception("Exception caught in capacity check", exc_info=True)

        return job

    def get_env_info(self):
        num_m = len(getattr(self.workcenters_meta, 'machine_list', []) or [])
        n_actions = int(num_m) if num_m > 0 else int(self.num_wcs)
        return {
            "n_actions": n_actions,
            "n_agents": int(self.num_jobs),
            "state_shape": int(self.state_dim),
            "obs_shape": int(self.obs_dim_agent),
            "episode_limit": int(self.episode_limit),
        }

    # ------------------ Job generation ------------------
    def _generate_initial_jobs(self):
        """Create a small deterministic initial job set based on WorkCenters capabilities.

        This function is intentionally robust: it uses the available machine
        registry when present and falls back to simple op pools when not.
        """
        self.jobs = []
        # build op pool from capabilities in machine registry
        ops_pool = set()
        try:
            for _, mdata in getattr(self.workcenters_meta, 'machine_registry', {}).items():
                for c in mdata.get('capabilities', []):
                    ops_pool.add(int(c))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass
        if not ops_pool:
            ops_pool = set(range(0, max(1, int(getattr(self, 'num_ops', 1)))))

        for jid in range(int(self.num_jobs)):
            try:
                num_ops = int(self._py_rng.randint(self.job_min_ops, max(self.job_min_ops, self.job_max_ops) + 1))
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                num_ops = max(1, int(getattr(self, 'job_min_ops', 1)))
            ops = []
            for _ in range(num_ops):
                try:
                    op_type = int(self._py_rng.choice(list(ops_pool)))
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    op_type = 0
                # allowed_wcs: WCs that have at least one machine with this capability
                allowed_wcs = set()
                try:
                    for mname, mdata in getattr(self.workcenters_meta, 'machine_registry', {}).items():
                        if op_type in list(mdata.get('capabilities', [])):
                            allowed_wcs.add(int(mdata.get('workcenter', 0)))
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    pass
                if not allowed_wcs:
                    allowed_wcs = {0}
                allowed_wcs = sorted(list(allowed_wcs))

                # Strict mode: derive per-machine durations exclusively from
                # processing_time_means in env.config. For each allowed workcenter
                # select a representative machine name and read its configured
                # duration for the operation. Missing mappings raise an error.
                per_wc_durations = {}
                try:
                    proc_means = self.config.get('processing_time_means', {}) if isinstance(self.config, dict) else {}
                    op_name = f"Op{op_type+1}"
                    op_map = proc_means.get(op_name, {}) if isinstance(proc_means, dict) else {}
                    for wc in allowed_wcs:
                        # find a machine name that belongs to this workcenter
                        machine_name = None
                        for mname, mdata in getattr(self.workcenters_meta, 'machine_registry', {}).items():
                            if int(mdata.get('workcenter', -1)) == int(wc):
                                machine_name = mname
                                break
                        if not machine_name:
                            raise ValueError(f"No machine found for workcenter {wc} when resolving durations for {op_name}")

                        if machine_name in op_map:
                            per_wc_durations[int(wc)] = float(op_map.get(machine_name))
                        else:
                            # strict mode: do not attempt extended fallbacks; require
                            # explicit machine key in processing_time_means
                            try:
                                LOG.error("Missing mapping while resolving durations: op_name=%s machine_name=%s proc_means_keys=%s op_map_keys=%s", op_name, machine_name, list(proc_means.keys()) if isinstance(proc_means, dict) else None, list(op_map.keys()) if isinstance(op_map, dict) else None)
                            except Exception:
                                pass
                            raise ValueError(f"Missing duration for {op_name} on machine {machine_name} (workcenter {wc})")
                except Exception:
                    raise

                ops.append((int(op_type), list(allowed_wcs), per_wc_durations))
            self.jobs.append(JobAgent(jid, ops))

    def print_jobs_human_readable(self):
        for job in self.jobs:
            for i, op in enumerate(job.operations):
                try:
                    op_type, allowed_wcs, per_wc = op
                    print(f"Job {job.id} Op{i} -> op_type={op_type} allowed_wcs={allowed_wcs} per_wc={per_wc}")
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    print(f"Job {job.id} Op{i} -> {op}")

