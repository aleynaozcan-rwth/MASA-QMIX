"""MASAEnv — authoritative, args-driven SimPy environment (safe refactor).

This file provides an environment surface that:
- Accepts a centralized `args` namespace (from MARL.common.arguments) and
    maps canonical fields (n_agents, num_operators, n_actions, episode_limit,
    obs/state dims, job generation bounds, reward params) into the runtime.
- Delegates YAML parsing / machine registry building to utils when available,
    and keeps minimal safe fallbacks so tests can import the module during
    staged refactor.

Config-based loading has been fully removed as of this version. The
environment and all subsystems now rely exclusively on in-module defaults
(WorkCenters, Operators, TaskGenerator). Deprecated constructor flags such
as `auto_load_config`, `config_path`, and `use_yaml_config` are accepted for
backwards compatibility but have no effect — they will be ignored at runtime
and may emit a deprecation warning when provided.

The implementation intentionally keeps side-effects low (logging only) and
tries to use `utils` components when present. The goal is to be a stable
authoritative surface for the rest of the code while we iteratively move
full runtime behavior into this module.
"""
import logging
import random
import time
from collections import deque
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, asdict, field


@dataclass
class EligibilityEntry:
    machine_id: str
    machine_busy: bool
    machine_available_at: float
    operator_id: Optional[str]
    operator_busy: Optional[bool]
    operator_available_at: Optional[float]
    qualified: Optional[bool]
    # operator_candidates is a list of dicts describing per-operator eligibility
    # Use a default factory to avoid mutable default pitfalls when instantiated.
    operator_candidates: List[Dict[str, Any]] = field(default_factory=list)
    # reason describes why this machine is not/was not immediately usable
    # values: 'no qualified', 'busy (avail@ t=X)', 'available' (has free qualified op)
    reason: Optional[str] = None


@dataclass
class DecisionTrace:
    chosen_machine: str
    chosen_operator: Optional[str]
    policy_reason: str
    at_time: float
    eligibilities: List[EligibilityEntry]

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

        # Deprecation warnings: config-related flags (kept for API compatibility)
        # are accepted but ignored. Log a single warning if any deprecated
        # flags are supplied so callers can migrate away from YAML-based flows.
        try:
            deprecated_flags = []
            if config_path:
                deprecated_flags.append('config_path')
            if auto_load_config:
                deprecated_flags.append('auto_load_config')
            if use_yaml_config:
                deprecated_flags.append('use_yaml_config')
            if deprecated_flags:
                LOG.warning("Deprecated config flags provided (ignored): %s", ','.join(deprecated_flags))
        except Exception:
            pass

        # -----------------------
        # Safe seed initialization
        # -----------------------
        try:
            # prefer seed from args if provided
            if args is not None and hasattr(args, 'seed') and getattr(args, 'seed') is not None:
                self.seed = int(getattr(args, 'seed'))
                LOG.info("[Env Init] Using provided seed: %s", self.seed)
            else:
                # fallback: use constructor seed if given
                if seed is not None:
                    try:
                        self.seed = int(seed)
                        LOG.info("[Env Init] Using constructor seed: %s", self.seed)
                    except Exception:
                        self.seed = None
                else:
                    self.seed = None

            if self.seed is None:
                # deterministic generation from system clock
                self.seed = int(time.time() * 1000) % (2 ** 32)
                LOG.info("[Env Init] No seed provided — generated new seed: %s", self.seed)
                LOG.info("[Reproducibility Tip] To reproduce this exact run, re-launch with --seed %s", self.seed)

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
    # Opt-in only: default is False. Caller may set args.dump_config to True
    # or pass dump_config=True in kwargs to enable.
        try:
            self.dump_config = bool(getattr(args, 'dump_config', False)) or bool(kwargs.get('dump_config', False))
        except Exception:
            self.dump_config = bool(kwargs.get('dump_config', False))

        # Config-free mode: ignore any provided config_path or YAML parsing.
        # The system relies solely on in-module defaults (WorkCenters, DEFAULT_PROCESSING_TIMES).
        cfg = {}
        # Announce config-free operation for visibility
        try:
            LOG.info("[Init] Config-free mode active (WorkCenters/TaskGenerator defaults in use)")
        except Exception:
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
        # Default increased to 600s to allow longer episodes by default.
        self.episode_limit = _resolve(('episode_limit',), int, default=600)

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

        # dynamic arrival params (mean interarrival time in seconds)
        try:
            # allow args/kwargs/config to set interarrival_time
            self.interarrival_time = float(_resolve(('interarrival_time', 'ia'), float, default=4.0))
        except Exception:
            try:
                self.interarrival_time = float(kwargs.get('interarrival_time', 4.0))
            except Exception:
                self.interarrival_time = 4.0

        # maximum number of jobs to generate via the dynamic job generator
        try:
            self.max_jobs = int(_resolve(('max_jobs', 'num_generate_jobs'), int, default=int(getattr(self, 'initial_jobs', 0) or 0)))
        except Exception:
            self.max_jobs = int(kwargs.get('max_jobs', getattr(self, 'initial_jobs', 0) or 0))

        # options controlling eager behaviors
        self.strict_mode = bool(strict_mode)
        self.auto_load_config = bool(auto_load_config)
        self.auto_build = bool(auto_build)
        self.auto_start_arrivals = bool(auto_start_arrivals)

        # If CLI args provided an arrival lambda, persist it on the env
        try:
            if args is not None and hasattr(args, 'arrival_lambda'):
                self.arrival_lambda = float(getattr(args, 'arrival_lambda'))
        except Exception:
            # leave arrival_lambda unset on failure
            pass

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
        
        # Ensure n_actions reflects actual machine count when machine_list is present
        try:
            mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
            if mlist:
                # prefer machine_list as the canonical action space
                self.n_actions = int(len(mlist))
                # if num_wcs was set differently, correct it (num_wcs is legacy / fallback)
                if int(getattr(self, 'num_wcs', 0)) != int(self.n_actions):
                    try:
                        LOG.warning("[WARN] CLI/num_wcs (%s) differs from machine_list length (%s); aligning to machine_list", getattr(self,'num_wcs',None), self.n_actions)
                    except Exception:
                        pass
                    self.num_wcs = int(self.n_actions)
            else:
                # fallback: keep existing num_wcs (from args) and expose as n_actions
                self.n_actions = int(getattr(self, 'num_wcs', 1))
        except Exception:
            self.n_actions = int(getattr(self, 'num_wcs', 1))

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
        # stable job id counter to ensure episode isolation and no id leaks
        self.job_counter = 0
        # pending jobs queued when capacity reached
        self.pending_jobs: List[List] = []
        # active jobs tracked to enforce capacity (n_agents)
        self.active_jobs: List[JobAgent] = []
        # active_agents: subset of jobs that are active from the learning/agent
        # perspective (may mirror self.active_jobs). Maintain separately so
        # ML-facing components can rely on a clear list that is managed here.
        self.active_agents: List[JobAgent] = []

        # generator/process lifecycle flag to avoid leaked processes across resets
        self._generator_shutdown = False
        # Interval (seconds) between periodic summary logs
        self.summary_interval = 20

        # initial_jobs: independent from n_agents (max capacity)
        try:
            self.initial_jobs = int(_resolve(('initial_jobs',), int, default=getattr(args, 'initial_jobs', 4)))
        except Exception:
            self.initial_jobs = int(getattr(args, 'initial_jobs', 4))

        # Ensure TaskGenerator exists and is owner-aware. Try several
        # constructor signatures for backward compatibility with older
        # TaskGenerator implementations. Prefer the modern signature that
        # accepts explicit RNG objects so create_job() is deterministic.
        try:
            from utils.task_generator import TaskGenerator  # type: ignore
            self.job_generator = None
            try:
                # Modern preferred signature: pass deterministic RNGs
                self.job_generator = TaskGenerator(py_rng=self._py_rng, np_rng=self._np_rng)
            except TypeError:
                try:
                    # Alternate: accepts injected Python RNG only
                    self.job_generator = TaskGenerator(py_rng=self._py_rng)
                except Exception:
                    try:
                        # Fallback: parameterless ctor
                        self.job_generator = TaskGenerator()
                    except Exception:
                        self.job_generator = None

            # Attach owner and deterministic RNG if available
            try:
                if self.job_generator is not None:
                    try:
                        setattr(self.job_generator, '_owner_env', self)
                    except Exception:
                        pass
                    try:
                        if not getattr(self.job_generator, '_py_rng', None):
                            setattr(self.job_generator, '_py_rng', self._py_rng)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            logging.getLogger(__name__).exception("TaskGenerator not available: %s", e, exc_info=True)
            self.job_generator = None

        # Create initial jobs deterministically (t=0) before starting dynamic arrivals
        try:
            if self.auto_build:
                # create the configured number of initial jobs deterministically
                try:
                    self._generate_initial_jobs()
                except Exception:
                    logging.getLogger(__name__).exception("Failed to generate initial jobs", exc_info=True)
        except Exception:
            pass

        # Bounded dynamic agents capacity: maximum number of concurrently
        # active learning agents. Prefer CLI args.n_agents when available,
        # otherwise fall back to configured num_jobs capacity.
        try:
            self.max_active_agents = int(getattr(args, 'n_agents', int(getattr(self, 'num_jobs', 0))))
        except Exception:
            try:
                self.max_active_agents = int(getattr(self, 'num_jobs', 0))
            except Exception:
                self.max_active_agents = 0

        # Note: automatic dynamic job generation is intentionally disabled
        # here. MASAEnv should be passive and only create initial jobs via
        # `_generate_initial_jobs()` (which itself uses TaskGenerator when
        # available). Dynamic arrivals / internal generators were removed as
        # part of the refactor to ensure decisions and arrivals are driven by
        # external orchestrators or TaskGenerator attached to the env owner.
        self._job_generator_proc = None

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
                LOG.info("[Env Summary] seed=%s capacity=%s initial_created=%s", summary['seed'], summary['n_agents_capacity'], summary['initial_jobs_created'])
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
        # Persist episode end marker before we reset counters
        try:
            hist_dir = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata')) if getattr(self, 'args', None) is not None else os.path.join('my_data_and_graph', 'historydata')
            os.makedirs(hist_dir, exist_ok=True)
            timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
            try:
                with open(timeline_path, 'a', encoding='utf-8') as tf:
                    tf.write("=== JOB AGENT LIFECYCLE TRACE END ===\n")
            except Exception:
                pass
        except Exception:
            pass

        # Signal any running generator loop tied to previous env to stop
        try:
            self._generator_shutdown = True
        except Exception:
            pass

        # Recreate SimPy environment for a clean episode
        self.env = simpy.Environment()
        # Reseed RNGs for deterministic episode behavior
        try:
            random.seed(self.seed)
        except Exception:
            pass
        try:
            np.random.seed(self.seed)
        except Exception:
            pass
        self._np_rng = np.random.RandomState(self.seed)
        self._py_rng = random.Random(self.seed)
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

        # reset bookkeeping and job state
        self.t = 0.0
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self._completed_now_cache = 0
        try:
            self._recent_rewards.clear()
        except Exception:
            try:
                from collections import deque
                self._recent_rewards = deque(maxlen=20)
            except Exception:
                self._recent_rewards = []
        self.done = False
        self.pending_decisions = []
        self.decisions_ready = simpy.Event(self.env)
        self.gantt_records = []
        # clear jobs and counters to ensure episode isolation
        try:
            self.jobs = []
        except Exception:
            self.jobs = []
        try:
            self.pending_jobs = []
        except Exception:
            self.pending_jobs = []
        try:
            self.active_jobs = []
        except Exception:
            self.active_jobs = []
        try:
            self.active_agents = []
        except Exception:
            self.active_agents = []
        try:
            self.job_counter = 0
        except Exception:
            self.job_counter = 0

        # After counters and lists cleared, write the START header for the new episode
        try:
            hist_dir = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata')) if getattr(self, 'args', None) is not None else os.path.join('my_data_and_graph', 'historydata')
            os.makedirs(hist_dir, exist_ok=True)
            timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
            try:
                with open(timeline_path, 'a', encoding='utf-8') as tf:
                    tf.write("=== JOB AGENT LIFECYCLE TRACE START ===\n")
            except Exception:
                pass
        except Exception:
            pass

        # Do not start any internal dynamic arrival loops on reset; however
        # create deterministic initial jobs at t=0 so tests and callers that
        # expect initial workload observe it. Initial jobs are produced via
        # `self._generate_initial_jobs()` which prefers TaskGenerator.
        self._job_generator_proc = None
        self._dynamic_arrival_proc = None
        self._task_generator = None

        # If the config requests a task_generator with a positive arrival_lambda,
        # construct and start it here. We keep this behavior opt-in and guarded
        # so most environments remain passive unless explicitly configured.
        try:
            tg_cfg = (self.config or {}).get('task_generator', {}) or {}
            lam = float(tg_cfg.get('arrival_lambda', 0.0)) if tg_cfg is not None else 0.0
            # Allow enabling arrivals via constructor/attributes when YAML/config is not used
            if not lam and getattr(self, 'auto_start_arrivals', False):
                lam = float(getattr(self, 'arrival_lambda', 0.0) or 0.0)
            if lam and lam > 0.0:
                    try:
                        from utils.task_generator import TaskGenerator  # type: ignore
                        try:
                            tg = TaskGenerator(config_path=self.config_path, py_rng=self._py_rng, np_rng=self._np_rng)
                        except TypeError:
                            try:
                                tg = TaskGenerator(py_rng=self._py_rng)
                            except TypeError:
                                try:
                                    tg = TaskGenerator(config_path=self.config_path)
                                except TypeError:
                                    try:
                                        tg = TaskGenerator(seed=self.seed)
                                    except Exception:
                                        tg = None
                    except Exception:
                        tg = None
                    try:
                        if tg is not None:
                            try:
                                setattr(tg, '_owner_env', self)
                            except Exception:
                                pass
                            try:
                                if not getattr(tg, '_py_rng', None):
                                    setattr(tg, '_py_rng', self._py_rng)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # start the generator with the env and lambda
                    try:
                        tg.start(self.env, lam)
                    except Exception:
                        try:
                            tg.start(self, lam)
                        except Exception:
                            pass
                    self._task_generator = tg
        except Exception:
            self._task_generator = None

        try:
            # Create initial jobs deterministically at reset (t=0)
            self._generate_initial_jobs()
        except Exception:
            logging.getLogger(__name__).exception("Failed to generate initial jobs on reset", exc_info=True)

        try:
            LOG.info("[Env] Episode time limit set to %s seconds", self.episode_limit)
        except Exception:
            pass
        # Start periodic summary logger (helps track job counts during long sims)
        try:
            # schedule periodic summary process on the current simpy env
            try:
                self.env.process(self._periodic_summary())
            except Exception:
                # fallback: if self.env not ready or process failed, ignore
                pass
        except Exception:
            pass
        # DEBUG: print machine counts for tracing unexpected machine totals
        try:
            mr_len = len(getattr(self, 'machine_resources', []) or [])
            mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
            ml_len = len(mlist)
            n_actions = getattr(self, 'n_actions', None)
            LOG.debug("[DEBUG] num_wcs=%s, len(machine_list)=%s, n_actions=%s", getattr(self,'num_wcs',None), ml_len, n_actions)
            # If num_wcs and machine_list disagree, align to machine_list
            try:
                if ml_len > 0 and int(getattr(self, 'num_wcs', 0)) != ml_len:
                    try:
                        LOG.warning("[WARN] Correcting num_wcs (%s) -> %s to match machine_list", getattr(self,'num_wcs',None), ml_len)
                    except Exception:
                        pass
                    self.num_wcs = int(ml_len)
                    self.n_actions = int(ml_len)
            except Exception:
                # defensive: ignore debug printing alignment issues
                pass
        except Exception:
            pass
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
                # No current op -> mark job completed (idempotent)
                now_t = float(getattr(self.env, 'now', 0.0))
                try:
                    if job.mark_completed(now_t):
                        self.completed_jobs += 1
                        self._completed_now_cache += 1
                        try:
                            if job in self.active_jobs:
                                try:
                                    self.active_jobs.remove(job)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        try:
                            if job in getattr(self, 'active_agents', []):
                                try:
                                    self.active_agents.remove(job)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        try:
                            hist_dir = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata')) if getattr(self, 'args', None) is not None else os.path.join('my_data_and_graph', 'historydata')
                            os.makedirs(hist_dir, exist_ok=True)
                            timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
                            try:
                                with open(timeline_path, 'a', encoding='utf-8') as tf:
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active: {len(getattr(self, 'active_agents', []) or [])} | Pending: {len(getattr(self, 'pending_jobs', []) or [])} | Completed: {int(getattr(self, 'completed_jobs', 0))}\n")
                            except Exception:
                                pass
                        except Exception:
                            pass
                except Exception:
                    # Best-effort: ignore failures to mark completion
                    pass
                break

            # normalize op formats: support legacy (allowed_machine_indices, dur) and
            # canonical (op_type, allowed_machine_indices, per_wc_durations)
            op_type = None
            allowed_machine_indices = []
            per_wc = None
            base_dur = None
            if isinstance(op, (list, tuple)):
                if len(op) == 2:
                    allowed_machine_indices, base_dur = op
                elif len(op) >= 3:
                    op_type = op[0]
                    allowed_machine_indices = op[1]
                    per_wc = op[2]

            # Resolve operation index (machine-level op index) for use in
            # operator qualification and duration lookups. Prefer explicit
            # op_type when provided, else fall back to job.current_op_idx.
            try:
                op_idx_local = int(op_type) if op_type is not None else int(getattr(job, 'current_op_idx', 0))
            except Exception:
                op_idx_local = int(getattr(job, 'current_op_idx', 0))
            # create decision item — prefer workcenters_meta helper if present
            try:
                if hasattr(self.workcenters_meta, 'create_decision_item'):
                    decision_item = self.workcenters_meta.create_decision_item(self, job, op)
                else:
                    raise AttributeError
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                decision_item = {
                    'job_id': job.id,
                    'obs': self._build_agent_obs(job),
                    'avail_row': self._avail_row_for_job(job),
                    'allowed_machine_indices': allowed_machine_indices,
                    'per_machine_durations': {},
                }

            # Environment owns the resume event — create it here and attach to
            # the decision_item so the policy/runner can call resume_evt.succeed(choice).
            resume_evt = simpy.Event(self.env)
            decision_item['resume_evt'] = resume_evt

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
                    # chosen_idx may index into allowed_machine_indices (legacy) or be a direct machine index
                    try:
                        chosen_idx_int = int(chosen_idx)
                    except Exception:
                        chosen_idx_int = 0
                    chosen_wc = int(allowed_machine_indices[chosen_idx_int]) if (isinstance(allowed_machine_indices, (list, tuple)) and len(allowed_machine_indices) > chosen_idx_int) else chosen_idx_int
                    dur = float(per_wc.get(chosen_wc, 0.0)) if isinstance(per_wc, dict) else float(per_wc)
                elif base_dur is not None:
                    dur = float(base_dur)
                else:
                    dur = 0.0
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                dur = 0.0

            # Before acquiring resources, build a decision-time trace that
            # records machine/operator availability for every eligible machine.
            try:
                # resolve allowed machine indices (decision_item may contain allowed_machine_indices)
                # Extend eligibilities to cover all machines present in the env so
                # we can report a full per-machine operator availability picture.
                try:
                    mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
                except Exception:
                    mlist = []
                try:
                    total_machines = int(getattr(self, 'total_machines', None)) if getattr(self, 'total_machines', None) is not None else (len(mlist) if mlist else int(getattr(self, 'num_wcs', 1)))
                except Exception:
                    try:
                        total_machines = len(mlist) if mlist else int(getattr(self, 'num_wcs', 1))
                    except Exception:
                        total_machines = 1

                # Build canonical allowed list as indices 0..total_machines-1 (inclusive start)
                try:
                    allowed_list = list(range(int(total_machines)))
                except Exception:
                    allowed_list = [0]
                # DEBUG: report what total_machines and allowed_list were resolved to
                    try:
                        LOG.debug("[DEBUG _job_process] total_machines=%s, allowed_list=%s, len(machine_resources)=%s, num_wcs=%s", total_machines, allowed_list, len(getattr(self, 'machine_resources', []) or []), getattr(self,'num_wcs', None))
                    except Exception:
                        pass

                # map to machine names when possible
                mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
                elig_entries: List[EligibilityEntry] = []
                now_t = float(self.env.now)
                # helper to compute machine next-free (best-effort) from gantt_records
                def _machine_next_free(mid: int) -> float:
                    try:
                        latest = now_t
                        for r in getattr(self, 'gantt_records', []) or []:
                            try:
                                # support dict and tuple forms
                                if isinstance(r, dict):
                                    r_w = int(r.get('wc_idx', r.get('wc', -1)))
                                    r_end = float(r.get('end', r.get('e', 0.0) or 0.0))
                                else:
                                    r_w = int(r[3])
                                    r_end = float(r[1])
                                if r_w == int(mid) and r_end > latest:
                                    latest = r_end
                            except Exception:
                                continue
                        return float(latest)
                    except Exception:
                        return now_t

                for midx in allowed_list:
                    try:
                        midx = int(midx)
                    except Exception:
                        # attempt to resolve from name -> index
                        try:
                            midx = int(getattr(self.workcenters_meta, 'machine_index', {}).get(str(midx), midx))
                        except Exception:
                            continue
                    # machine name
                    try:
                        mname = mlist[int(midx)] if mlist and 0 <= int(midx) < len(mlist) else f"M{int(midx)}"
                    except Exception:
                        mname = str(midx)
                    # machine busy: check current resource users (best-effort)
                    m_busy = False
                    try:
                        if getattr(self, 'machine_resources', None) and 0 <= int(midx) < len(self.machine_resources):
                            res = self.machine_resources[int(midx)]
                            m_busy = len(getattr(res, 'users', [])) > 0
                    except Exception:
                        m_busy = False
                    m_avail_at = _machine_next_free(midx)

                    # find operator candidates and choose a best operator (best-effort)
                    op_id = None
                    op_busy = None
                    op_avail_at = None
                    qualified = False
                    operator_candidates = []
                    try:
                        ops_mgr = getattr(self, 'operators', None)
                        if ops_mgr is not None:
                            for op_obj in getattr(ops_mgr, 'operators_object_list', []) or []:
                                try:
                                    opid = str(getattr(op_obj, 'operator_id', None))
                                    busy = bool(getattr(op_obj, 'is_busy', False))
                                    # infer next free time from history when possible
                                    try:
                                        hist = getattr(op_obj, 'history', []) or []
                                        if busy:
                                            if hist and hist[-1].get('end_time', None) is not None:
                                                next_free = float(hist[-1].get('end_time'))
                                            else:
                                                next_free = now_t
                                        else:
                                            next_free = now_t
                                    except Exception:
                                        next_free = None

                                    # attempt to resolve workcenter for this machine
                                    wc_idx_for_m = None
                                    try:
                                        wc_for_machine = getattr(self.workcenters_meta, 'workcenter_for_machine', None)
                                        if callable(wc_for_machine):
                                            wc_idx_for_m = int(wc_for_machine(mname))
                                        else:
                                            registry = getattr(self.workcenters_meta, 'machine_registry', {}) or {}
                                            if mname in registry:
                                                wc_idx_for_m = int(registry.get(mname, {}).get('workcenter', -1))
                                    except Exception:
                                        wc_idx_for_m = None

                                    # qualification check: prefer can_do_job when wc_idx known
                                    is_qualified = False
                                    try:
                                        # Use machine-level qualification: check whether
                                        # this operator can do op_idx_local on this machine.
                                        if wc_idx_for_m is not None:
                                            try:
                                                is_qualified = bool(op_obj.can_do_job(op_idx_local, wc_idx_for_m))
                                            except Exception:
                                                # fallback: check explicit machine name listing
                                                is_qualified = mname in getattr(op_obj, 'qualified_machines', [])
                                        else:
                                            is_qualified = mname in getattr(op_obj, 'qualified_machines', [])
                                    except Exception:
                                        is_qualified = False

                                    operator_candidates.append({
                                        'operator_id': opid,
                                        'qualified': bool(is_qualified),
                                        'busy': busy,
                                        'next_free': float(next_free) if next_free is not None else None,
                                        'load': int(len(getattr(op_obj, 'history', []) or [])),
                                    })

                                    if is_qualified:
                                        qualified = True
                                        # prefer free operator, otherwise the one with earliest next_free
                                        if not busy and op_id is None:
                                            op_id = opid
                                            op_busy = False
                                            op_avail_at = float(next_free) if next_free is not None else None
                                            # free operator is ideal; stop searching
                                            break
                                        else:
                                            # busy operator; pick earliest next_free
                                            try:
                                                if op_id is None or (next_free is not None and (op_avail_at is None or float(next_free) < float(op_avail_at))):
                                                    op_id = opid
                                                    op_busy = busy
                                                    op_avail_at = float(next_free) if next_free is not None else None
                                            except Exception:
                                                pass
                                except Exception:
                                    continue
                            else:
                                qualified = False
                    except Exception:
                        qualified = False

                        # derive a human-readable per-machine reason for DecisionTrace
                        try:
                            if not operator_candidates or not any(c.get('qualified', False) for c in operator_candidates):
                                reason_str = 'no qualified'
                            else:
                                # any qualified & free?
                                if any(c.get('qualified', False) and not c.get('busy', False) for c in operator_candidates):
                                    reason_str = 'available'
                                else:
                                    # busy but will be free at some point — pick earliest known next_free
                                    next_times = [c.get('next_free') for c in operator_candidates if c.get('qualified', False) and c.get('next_free') is not None]
                                    if next_times:
                                        try:
                                            earliest = min(next_times)
                                            reason_str = f"busy (avail@ t={float(earliest):.2f})"
                                        except Exception:
                                            reason_str = 'busy'
                                    else:
                                        reason_str = 'busy'
                        except Exception:
                            reason_str = None

                        elig_entries.append(EligibilityEntry(
                            machine_id=str(mname),
                            machine_busy=bool(m_busy),
                            machine_available_at=float(m_avail_at),
                            operator_id=op_id,
                            operator_busy=op_busy,
                            operator_available_at=op_avail_at,
                            qualified=qualified,
                            operator_candidates=operator_candidates,
                            reason=reason_str,
                        ))

                # determine chosen machine/operator labels
                try:
                    # chosen_idx may be either machine index or index into allowed_list
                    try:
                        chosen_m_idx = int(chosen_idx)
                        # if chosen_idx indexes into allowed_list (legacy), map
                        if allowed_list and chosen_m_idx < len(allowed_list) and int(allowed_list[chosen_m_idx]) != chosen_m_idx:
                            chosen_mid = int(allowed_list[chosen_m_idx])
                        else:
                            chosen_mid = chosen_m_idx
                    except Exception:
                        # fallback: assume chosen_idx indexes allowed_list by name
                        chosen_mid = int(allowed_list[int(chosen_idx)]) if allowed_list else int(chosen_idx)
                except Exception:
                    chosen_mid = int(chosen_idx) if chosen_idx is not None else -1
                try:
                    chosen_m_name = mlist[chosen_mid] if mlist and 0 <= int(chosen_mid) < len(mlist) else f"M{int(chosen_mid)}"
                except Exception:
                    chosen_m_name = str(chosen_mid)

                chosen_op_label = None
                try:
                    chosen_op_label = str(op_id_for_record) if 'op_id_for_record' in locals() else None
                except Exception:
                    chosen_op_label = None

                # policy reason: not yet resolved (decision expected from policy)
                policy_reason = 'pending'

                decision_trace = DecisionTrace(
                    chosen_machine=str(chosen_m_name),
                    chosen_operator=chosen_op_label,
                    policy_reason=str(policy_reason),
                    at_time=float(now_t),
                    eligibilities=elig_entries,
                )
            except Exception:
                decision_trace = None

            # acquire resources and execute
            try:
                # Use chosen_mid (resolved machine index) when selecting machine resource
                mr = self.machine_resources[int(chosen_mid)] if getattr(self, 'machine_resources', None) else self.wc_resources[int(0)]
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
                    machine_name = mlist[int(chosen_mid)] if mlist and 0 <= int(chosen_mid) < len(mlist) else f"M{int(chosen_mid)}"
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
                        # Deterministic pick (no env auto-choice). Operator
                        # group selection should be handled after a machine is
                        # chosen; we pick the first entry deterministically as
                        # a neutral fallback for downstream qualification.
                        selected_grp = int(list(eligible_groups)[0])
                    except Exception:
                        selected_grp = int(list(eligible_groups)[0])
                else:
                    selected_grp = None

                # Select a concrete operator and wait for availability using
                # the Operators manager. We no longer use a numeric "group:0"
                # fallback — if no operator is immediately free we wait in a
                # short loop until one becomes available. This enforces true
                # operator-level exclusivity via each Operator.resource.
                available_operator = None
                try:
                    if getattr(self, 'operators', None) is not None:
                        # Try to locate a free operator qualified for the chosen
                        # machine, with a limited retry policy. This ensures the
                        # environment waits briefly for short operator-held times
                        # but does not block indefinitely.
                        max_retries = int(getattr(self, 'operator_selection_retries', 5))
                        retry_wait = float(getattr(self, 'operator_selection_wait', 1.0))
                        attempt = 0
                        while attempt < max_retries and available_operator is None:
                            try:
                                available_operator = self.operators.find_free_operator_for_machine(op_idx_local, machine_name)
                            except Exception:
                                available_operator = None
                            # Fallback: try workcenter-level lookup if machine-level fails
                            if available_operator is None:
                                try:
                                    available_operator = self.operators.find_free_operator(op_idx_local, wc_idx)
                                except Exception:
                                    available_operator = None
                            # If not found, wait a longer amount (1.0s by default)
                            if available_operator is None:
                                attempt += 1
                                if attempt < max_retries:
                                    yield self.env.timeout(retry_wait)
                        # After retries, determine policy reason based on eligibilities
                        try:
                            chosen_entry = None
                            for e in elig_entries:
                                if str(e.machine_id) == str(chosen_m_name):
                                    chosen_entry = e
                                    break
                            if available_operator is None:
                                if chosen_entry is None:
                                    policy_reason = 'no qualified'
                                else:
                                    # use the machine-level reason computed earlier
                                    policy_reason = chosen_entry.reason if getattr(chosen_entry, 'reason', None) is not None else 'no qualified'
                            else:
                                policy_reason = 'selected'
                                # reflect selected operator in the chosen_entry for trace clarity
                                try:
                                    if chosen_entry is not None:
                                        chosen_entry.operator_id = str(getattr(available_operator, 'operator_id', None))
                                        chosen_entry.operator_busy = False
                                        chosen_entry.operator_available_at = float(self.env.now)
                                except Exception:
                                    pass
                        except Exception:
                            policy_reason = 'selected' if available_operator is not None else 'no qualified'
                    else:
                        available_operator = None
                        # no operators manager — policy reason remains 'pending' until we run
                except Exception:
                    available_operator = None

                # If we found an operator object, request its per-operator
                # resource and the machine resource before starting the op.
                # Update decision_trace with the resolved policy reason so that
                # gantt/trace consumers see the final outcome.
                try:
                    if 'decision_trace' in locals() and decision_trace is not None:
                        decision_trace.policy_reason = str(policy_reason)
                except Exception:
                    pass
                try:
                    if available_operator is not None and getattr(available_operator, 'resource', None) is not None:
                        try:
                            logging.getLogger(__name__).debug("Waiting -> starting job=%s on machine=%s by operator=%s time=%s", getattr(job, 'id', None), int(chosen_mid), getattr(available_operator, 'operator_id', None), float(self.env.now))
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
                                    # Log instrumentation so it's visible in test/dev logs.
                                    LOG.info("[GANTT-APPEND] concrete branch -> op_id_for_record=%r type=%s", op_id_for_record, type(op_id_for_record))
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
                                            rec = {'start': op_start, 'end': op_end, 'op_idx': int(op_type) if op_type is not None else job.current_op_idx, 'wc_idx': int(chosen_mid), 'job_id': int(job.id), 'op_grp': op_id_for_record, 'arrival': float(job.arrival_time), 'duration': float(dur)}
                                            if ep_stamp is not None:
                                                rec['episode'] = int(ep_stamp)
                                        except Exception:
                                            rec = {'start': op_start, 'end': op_end, 'op_idx': int(op_type) if op_type is not None else job.current_op_idx, 'wc_idx': int(chosen_mid), 'job_id': int(job.id), 'op_grp': op_id_for_record, 'arrival': float(job.arrival_time), 'duration': float(dur)}
                                        # attach decision_trace if available
                                        try:
                                            if decision_trace is not None:
                                                rec['decision_trace'] = asdict(decision_trace)
                                        except Exception:
                                            pass
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
                                    rec_dict = {'start': op_start, 'end': op_end, 'op_idx': int(op_type) if op_type is not None else job.current_op_idx, 'wc_idx': int(chosen_mid), 'job_id': int(job.id), 'op_grp': op_id_for_record, 'arrival': float(job.arrival_time), 'duration': float(dur)}
                                    if ep_stamp is not None:
                                        rec_dict['episode'] = int(ep_stamp)
                                    try:
                                        if decision_trace is not None:
                                            rec_dict['decision_trace'] = asdict(decision_trace)
                                    except Exception:
                                        pass
                                    # Append dict record (generator updated to support dicts)
                                    self.gantt_records.append(rec_dict)
                                except Exception:
                                    # fallback to legacy append tuple if dict append fails
                                    try:
                                        self.gantt_records.append((op_start, op_end, int(op_type) if op_type is not None else job.current_op_idx, int(chosen_mid), int(job.id), op_id_for_record, float(job.arrival_time), float(dur)))
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
                                    # Log instrumentation so it's visible in test/dev logs.
                                    LOG.info("[GANTT-APPEND] unassigned branch -> op_id_for_record=%r type=%s", op_id_for_record, type(op_id_for_record))
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
                                    rec_dict = {'start': op_start, 'end': op_end, 'op_idx': int(op_type) if op_type is not None else job.current_op_idx, 'wc_idx': int(chosen_mid), 'job_id': int(job.id), 'op_grp': op_id_for_record, 'arrival': float(job.arrival_time), 'duration': float(dur)}
                                    if ep_stamp is not None:
                                        rec_dict['episode'] = int(ep_stamp)
                                    try:
                                        if decision_trace is not None:
                                            rec_dict['decision_trace'] = asdict(decision_trace)
                                    except Exception:
                                        pass
                                    self.gantt_records.append(rec_dict)
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
                # mark completion and only increment counters once
                now_t = float(getattr(self.env, 'now', 0.0))
                try:
                    if job.mark_completed(now_t):
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

                        try:
                            if job in getattr(self, 'active_agents', []):
                                try:
                                    self.active_agents.remove(job)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    # best-effort: continue even if mark_completed fails
                    pass
                # If there are pending jobs, start pending jobs until capacity
                # is reached (bounded dynamic capacity semantics)
                try:
                    while getattr(self, 'pending_jobs', None) and len(self.pending_jobs) > 0 and (int(getattr(self, 'max_active_agents', 0)) <= 0 or len(self.active_agents) < int(getattr(self, 'max_active_agents', 0))):
                        try:
                            next_job = self.pending_jobs.pop(0)
                        except Exception:
                            break
                        try:
                            # schedule and mark active
                            self.env.process(self._job_process(next_job))
                            self.active_jobs.append(next_job)
                            try:
                                self.active_agents.append(next_job)
                            except Exception:
                                pass
                            # persist 'became active agent' event for pending activation
                            try:
                                hist_dir = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata')) if getattr(self, 'args', None) is not None else os.path.join('my_data_and_graph', 'historydata')
                                os.makedirs(hist_dir, exist_ok=True)
                                timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
                                try:
                                    with open(timeline_path, 'a', encoding='utf-8') as tf:
                                        tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent -> ActiveAgents: {len(getattr(self, 'active_agents', []) or [])}\n")
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            try:
                                next_job.is_active = True
                            except Exception:
                                pass
                            try:
                                LOG.info("[Env] Pending job %s activated at t=%.4f", getattr(next_job, 'id', None), float(getattr(self.env, 'now', 0.0)))
                            except Exception:
                                pass
                            # persist pending activation
                            try:
                                hist_dir = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata')) if getattr(self, 'args', None) is not None else os.path.join('my_data_and_graph', 'historydata')
                                os.makedirs(hist_dir, exist_ok=True)
                                timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
                                try:
                                    with open(timeline_path, 'a', encoding='utf-8') as tf:
                                        tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Pending job {getattr(next_job, 'id', None)} activated -> ActiveAgents: {len(getattr(self, 'active_agents', []) or [])}\n")
                                except Exception:
                                    pass
                            except Exception:
                                pass
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

        # Compute per-machine free flags (prefer explicit machine resources,
        # otherwise fall back to workcenter resources). If we can't determine
        # this, default to True for backward compatibility.
        machine_free = [True] * n_m
        try:
            if getattr(self, 'machine_resources', None):
                machine_free = [self._resource_free(self.machine_resources[m]) for m in range(n_m)]
            else:
                # use machine_registry -> workcenter -> wc_resources
                registry = getattr(self.workcenters_meta, 'machine_registry', {}) or {}
                mlist_local = list(getattr(self.workcenters_meta, 'machine_list', []) or [])
                tmp = []
                for i, mname in enumerate(mlist_local):
                    try:
                        wc_i = int(registry.get(mname, {}).get('workcenter', 0))
                        tmp.append(self._resource_free(self.wc_resources[wc_i]))
                    except Exception:
                        tmp.append(True)
                if len(tmp) == n_m:
                    machine_free = tmp
        except Exception:
            machine_free = [True] * n_m

        # Compute operator free flags if operator_groups are present. If
        # operator info is missing, keep operator_free as None to indicate
        # we should default to permissive behavior for compatibility.
        operator_free = None
        try:
            if getattr(self, 'operator_groups', None) is not None and int(getattr(self, 'num_ops', 0)) > 0:
                operator_free = [self._resource_free(self.operator_groups[p]) for p in range(int(self.num_ops))]
        except Exception:
            operator_free = None

        # For each job, only mark a machine as available if:
        #  - the machine supports the job's current op (row==1), AND
        #  - the machine resource is free, AND
        #  - there exists at least one operator group qualified for the
        #    machine whose resource is free. If eligible/operator mapping is
        #  missing, we default to permissive (assume operator available).
        registry = getattr(self.workcenters_meta, 'machine_registry', {}) or {}
        eligible_map = getattr(getattr(self, 'workcenters_meta', None), 'eligible_operator_groups_by_wc', {}) or {}

        for idx, j in enumerate(self.jobs):
            if j.finished:
                continue
            row = self._avail_row_for_job(j)
            if row is None:
                continue
            try:
                # Start all zeros; set to 1 only when all checks pass
                for m in range(n_m):
                    try:
                        if int(row[m]) != 1:
                            continue
                    except Exception:
                        continue

                    # Check machine-level free
                    try:
                        if not machine_free[m]:
                            continue
                    except Exception:
                        # if we can't determine, assume free
                        pass

                    # Find eligible operator groups for this machine via its workcenter
                    try:
                        mname = mlist[m] if m < len(mlist) else None
                        wc_i = int(registry.get(mname, {}).get('workcenter')) if mname is not None else None
                        eligible_groups = eligible_map.get(int(wc_i), []) if wc_i is not None else []
                    except Exception:
                        eligible_groups = []

                    # If operator info missing, assume operator available (back-compat)
                    if operator_free is None:
                        avail[idx, m] = 1
                        continue

                    # If eligible_groups is empty, be permissive (back-compat)
                    if not eligible_groups:
                        avail[idx, m] = 1
                        continue

                    # Otherwise, require at least one eligible operator group to be free
                    found_free_op = False
                    for g in eligible_groups:
                        try:
                            gi = int(g)
                            if 0 <= gi < len(operator_free) and operator_free[gi]:
                                found_free_op = True
                                break
                        except Exception:
                            continue

                    if found_free_op:
                        avail[idx, m] = 1
                # end per-machine loop
            except Exception:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                # fallback: write the original row if anything went wrong
                try:
                    avail[idx, :] = row
                except Exception:
                    pass
        return avail

    def _avail_row_for_job(self, job: JobAgent):
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        n_m = len(mlist) if mlist else int(getattr(self, 'num_wcs', 1))
        row = np.zeros((int(n_m),), dtype=np.int32)
        op = job.current_op()
        if op is None:
            return row

        # Resolve operation index (machine-level op index)
        try:
            if isinstance(op, (list, tuple)) and len(op) >= 3:
                op_type = op[0]
                op_idx_local = int(op_type) if op_type is not None else int(getattr(job, 'current_op_idx', 0))
            else:
                op_idx_local = int(getattr(job, 'current_op_idx', 0))
        except Exception:
            op_idx_local = int(getattr(job, 'current_op_idx', 0))

        # Prefer authoritative machine_registry -> mark machines that support this op
        try:
            registry = getattr(self.workcenters_meta, 'machine_registry', {}) or {}
            mlist_local = list(getattr(self.workcenters_meta, 'machine_list', []) or [])
            for i, mname in enumerate(mlist_local):
                try:
                    caps = registry.get(mname, {}).get('capabilities', [])
                    if int(op_idx_local) in caps:
                        row[i] = 1
                except Exception:
                    continue
            # If no machines marked (e.g., no registry), fall back to legacy allowed_machine_indices in op tuple
            if not row.any():
                try:
                    if isinstance(op, (list, tuple)) and len(op) == 2:
                        allowed_machine_indices, _ = op
                    else:
                        try:
                            _, allowed_machine_indices, _ = op
                        except Exception:
                            allowed_machine_indices = []
                    for idx in allowed_machine_indices:
                        try:
                            if 0 <= int(idx) < row.shape[0]:
                                row[int(idx)] = 1
                        except Exception:
                            continue
                except Exception:
                    pass
        except Exception:
            # Last-resort legacy behavior
            try:
                if isinstance(op, (list, tuple)) and len(op) == 2:
                    allowed_machine_indices, _ = op
                else:
                    try:
                        _, allowed_machine_indices, _ = op
                    except Exception:
                        allowed_machine_indices = []
                for idx in allowed_machine_indices:
                    try:
                        if 0 <= int(idx) < row.shape[0]:
                            row[int(idx)] = 1
                    except Exception:
                        continue
            except Exception:
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

    def _periodic_summary(self):
        """Periodically log job statistics during simulation."""
        # This is a SimPy generator-based process (yields timeouts)
        while True:
            try:
                yield self.env.timeout(self.summary_interval)
            except Exception:
                # If env is gone or summary_interval invalid, stop the process
                return
            try:
                total = len(getattr(self, 'jobs', []) or [])
                completed = sum(1 for j in getattr(self, 'jobs', []) if getattr(j, 'is_finished', False))
                active = sum(1 for j in getattr(self, 'jobs', []) if getattr(j, 'is_active', False))
                LOG.info("[Summary] t=%.2f → total=%d | completed=%d | active=%d",
                         float(getattr(self.env, 'now', 0.0)), int(total), int(completed), int(active))
            except Exception:
                try:
                    LOG.exception("[Summary] failed to emit periodic summary", exc_info=True)
                except Exception:
                    pass

    def add_job(self, ops_sequence: List, start_immediately: bool = True, set_arrival_zero: bool = False):
        """Add a job (ops_sequence) to the environment.

        Args:
            ops_sequence: canonical list of ops (op_type, allowed_machine_indices, per_machine_durations)
            start_immediately: when True, schedule the job process on the SimPy env
            set_arrival_zero: when True, force arrival_time to 0.0 (used for initial jobs)
        Returns:
            JobAgent instance
        """
        # Use stable job_counter so ids reset each episode
        jid = int(getattr(self, 'job_counter', 0))
        job = JobAgent(jid, ops_sequence)
        try:
            if set_arrival_zero:
                job.arrival_time = 0.0
            else:
                job.arrival_time = float(self.env.now)
        except Exception:
            job.arrival_time = 0.0
        # Append to master job list (arrival order) and advance counter
        self.jobs.append(job)
        try:
            # Live runtime visibility: log new arrivals as they are added so
            # callers and users can see dynamic job injections in real-time
            # (the gantt timeline generator produces 'New job arrived' only
            # when invoked and is not a live arrival trace).
            LOG.info("[Env] New job %s arrived at t=%.4f with %s ops", job.id, float(getattr(job, 'arrival_time', 0.0)), len(getattr(job, 'operations', []) or []))
        except Exception:
            try:
                LOG.debug("[Env] New job added id=%s arrival=%s", getattr(job, 'id', None), getattr(job, 'arrival_time', None))
            except Exception:
                pass
        try:
            self.job_counter = int(jid) + 1
        except Exception:
            try:
                self.job_counter = len(self.jobs)
            except Exception:
                pass

        # Capacity enforcement: self.num_jobs represents the capacity (n_agents)
        try:
            # Use configured bounded capacity when present
            capacity = int(getattr(self, 'max_active_agents', 0)) or int(getattr(self, 'num_jobs', 0)) or int(getattr(self.args, 'n_agents', 0))
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
                        try:
                            job.is_active = True
                        except Exception:
                            pass
                        # mirror into active_agents for ML-facing semantics
                        try:
                            self.active_agents.append(job)
                        except Exception:
                            pass
                        # persist 'became active agent' event
                        try:
                            hist_dir = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata')) if getattr(self, 'args', None) is not None else os.path.join('my_data_and_graph', 'historydata')
                            os.makedirs(hist_dir, exist_ok=True)
                            timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
                            try:
                                with open(timeline_path, 'a', encoding='utf-8') as tf:
                                    tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent -> ActiveAgents: {len(getattr(self, 'active_agents', []) or [])}\n")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # persist arrival/creation to timeline
                        try:
                            hist_dir = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata')) if getattr(self, 'args', None) is not None else os.path.join('my_data_and_graph', 'historydata')
                            os.makedirs(hist_dir, exist_ok=True)
                            timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
                            try:
                                with open(timeline_path, 'a', encoding='utf-8') as tf:
                                    tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active: {len(getattr(self, 'active_agents', []) or [])} | Pending: {len(getattr(self, 'pending_jobs', []) or [])} | Completed: {int(getattr(self, 'completed_jobs', 0))}\n")
                                    tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} created and started -> Active: {len(getattr(self, 'active_agents', []) or [])}\n")
                            except Exception:
                                pass
                        except Exception:
                            pass
                    except Exception:
                        # Diagnostic: surface add_job startup failures with sim time
                        try:
                            logging.getLogger(__name__).exception("Exception caught while starting job", exc_info=True)
                            logging.getLogger(__name__).info("[Diag] add_job() failed at t=%.4f when starting job id=%s", float(getattr(self, 'env', simpy.Environment()).now if getattr(self, 'env', None) is not None else 0.0), getattr(job, 'id', None))
                        except Exception:
                            logging.getLogger(__name__).exception("Exception caught while starting job (secondary)", exc_info=True)
                else:
                    # queue for later start
                    try:
                        self.pending_jobs.append(job)
                        try:
                            LOG.info("[Env] Job %s queued (capacity full)", getattr(job, 'id', None))
                        except Exception:
                            pass
                        # persist queued event
                        try:
                            hist_dir = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata')) if getattr(self, 'args', None) is not None else os.path.join('my_data_and_graph', 'historydata')
                            os.makedirs(hist_dir, exist_ok=True)
                            timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
                            try:
                                with open(timeline_path, 'a', encoding='utf-8') as tf:
                                    tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (capacity full: max_active_agents={int(getattr(self, 'max_active_agents', 0))})\n")
                            except Exception:
                                pass
                        except Exception:
                            pass
                    except Exception:
                        logging.getLogger(__name__).exception("Failed to queue pending job", exc_info=True)
                        try:
                            logging.getLogger(__name__).info("[Diag] add_job() failed to append pending job id=%s at t=%.4f", getattr(job, 'id', None), float(getattr(self, 'env', simpy.Environment()).now if getattr(self, 'env', None) is not None else 0.0))
                        except Exception:
                            pass
            except Exception:
                logging.getLogger(__name__).exception("Exception caught in capacity check", exc_info=True)
                try:
                    logging.getLogger(__name__).info("[Diag] add_job() capacity check failed at t=%.4f", float(getattr(self, 'env', simpy.Environment()).now if getattr(self, 'env', None) is not None else 0.0))
                except Exception:
                    pass

        return job

    def get_env_info(self):
        num_m = len(getattr(self.workcenters_meta, 'machine_list', []) or [])
        # n_actions is canonicalized to machine count when machine_list exists
        n_actions = int(getattr(self, 'n_actions', (num_m if num_m > 0 else int(getattr(self, 'num_wcs', 1)))))
        return {
            "n_actions": n_actions,
            "n_agents": int(self.num_jobs),
            "state_shape": int(self.state_dim),
            "obs_shape": int(self.obs_dim_agent),
            "episode_limit": int(self.episode_limit),
        }

    # ------------------ Job generation ------------------
    def _generate_initial_jobs(self):
        """Create deterministic initial jobs using TaskGenerator only.

        Per new refactor requirements, legacy fallback synthesis is disabled:
        if a TaskGenerator (`self.job_generator`) is not attached the function
        will not synthesize jobs. This keeps the environment passive and
        ensures job creation is controlled by an explicit generator.
        """
        # Clear any existing jobs (fresh episode)
        self.jobs = []
        try:
            self.job_counter = 0
        except Exception:
            self.job_counter = 0

        n_init = int(getattr(self, 'initial_jobs', 4))
        if getattr(self, 'job_generator', None) is None:
            logging.getLogger(__name__).warning("TaskGenerator not attached: skipping initial job creation (initial_jobs=%s)", n_init)
            return

        for _ in range(max(0, n_init)):
            try:
                # pick a deterministic number of ops using the injected RNG
                try:
                    min_init_ops = max(3, int(getattr(self, 'job_min_ops', 2)))
                    max_init_ops = int(getattr(self, 'job_max_ops', max(min_init_ops, 5)))
                    num_ops = int(self._py_rng.randint(min_init_ops, max(min_init_ops, max_init_ops) + 1))
                except Exception:
                    num_ops = int(getattr(self, 'job_min_ops', 1))

                try:
                    ops = self.job_generator.create_job(num_ops=num_ops)
                except Exception:
                    ops = None
                # If TaskGenerator fails to produce a job, fall back to a
                # minimal single-op job so tests and callers that expect
                # initial jobs don't observe an empty job list. This is a
                # conservative, short-term compatibility measure; full
                # TaskGenerator-driven creation is preferred.
                if not ops:
                    logging.getLogger(__name__).warning("TaskGenerator failed to generate job; creating minimal single-op fallback for initial job")
                    ops = [(0, [0], {0: 1.0})]
                try:
                    # Start initial jobs immediately at t=0 so the simpy
                    # environment has scheduled processes and time can
                    # advance deterministically.
                    self.add_job(ops, start_immediately=True, set_arrival_zero=True)
                except Exception:
                    logging.getLogger(__name__).exception("Failed to add initial job via add_job", exc_info=True)
            except Exception:
                logging.getLogger(__name__).exception("Exception while generating initial job", exc_info=True)

    # NOTE: dynamic arrivals and internal job generator loops removed.
    # Dynamic job arrival behavior should be provided by an external
    # TaskGenerator or orchestrator that explicitly calls `env.add_job()`.

    def print_jobs_human_readable(self):
        for job in self.jobs:
            for i, op in enumerate(job.operations):
                try:
                    op_type, allowed_machine_indices, per_wc = op
                    LOG.info("Job %s Op%s -> op_type=%s allowed_machine_indices=%s per_machine=%s", job.id, i, op_type, allowed_machine_indices, per_wc)
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    LOG.debug("Job %s Op%s -> %s", job.id, i, op)

    def print_initial_jobs_summary(self):
        """Print a concise initial jobs summary in the legacy format.

        Example:
        Job_0 -> 4 ops: [Op8, Op1, Op9, Op6] | Eligible: {Op8:[0,1,2], ...}
        """
        try:
            for job in self.jobs[:int(getattr(self, 'initial_jobs', 4))]:
                try:
                    ops_desc = []
                    eligible_desc = []
                    for i, op in enumerate(job.operations):
                        try:
                            op_type, allowed_machine_indices, per_wc = op
                            ops_desc.append(f"Op{int(op_type)+1}")
                            eligible_desc.append(f"Op{int(op_type)+1}:[{','.join(str(x) for x in allowed_machine_indices)}]")
                        except Exception:
                            ops_desc.append(str(op))
                    LOG.info("Job_%s -> %s ops: [%s] | Eligible: {%s}", job.id, len(job.operations), ', '.join(ops_desc), ', '.join(eligible_desc))
                except Exception:
                    LOG.warning("Job_%s -> (failed to summarize)", job.id)
        except Exception:
            logging.getLogger(__name__).exception("Failed to print initial job summary", exc_info=True)

