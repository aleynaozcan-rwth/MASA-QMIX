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
import math
import random
import time
import csv

from collections import deque, Counter
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
    operator_candidates: List[Dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None


@dataclass
class DecisionTrace:
    chosen_machine: str
    chosen_operator: Optional[str]
    policy_reason: str
    at_time: float
    eligibilities: List[EligibilityEntry]


class RunningMeanStd:
    """
    Welford's online algorithm for incremental mean/variance computation.
    
    Numerically stable, O(1) memory, O(1) per-update time complexity.
    Adaptive to any reward scale changes without hyperparameter tuning.
    
    References:
    - Welford (1962) "Note on a method for calculating corrected sums of squares"
    - Knuth TAOCP Vol 2, 3rd Ed., Sec 4.2.2
    """
    def __init__(self, epsilon=1e-4):
        """
        Args:
            epsilon: Small value to initialize count (prevents division by zero)
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
    
    def update(self, x):
        """Update running statistics with new value x."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count
    
    def normalize(self, x, clip_range=10.0):
        """
        Normalize value x using current mean/std, clipped to [-clip_range, +clip_range].
        
        Args:
            x: Value to normalize
            clip_range: Clipping range (default: 10.0 for [-10, +10])
            
        Returns:
            Normalized and clipped value
        """
        std = (self.var ** 0.5) if self.var > 0 else 1.0
        normalized = (x - self.mean) / (std + 1e-8)
        return max(-clip_range, min(clip_range, normalized))


import numpy as np
import simpy
import os
import json
from utils.env_obs import build_agent_obs, build_state_vector  # type: ignore
from utils.io_control import allow_history_writes
from utils.operator import Operators  # type: ignore
from utils.workcenter import WorkCenters  # type: ignore
from utils.jobagent import JobAgent  # type: ignore


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
        config_path: Optional[str] = None,
        auto_build: bool = True,
        auto_start_arrivals: bool = False,
        use_yaml_config: bool = False,
        **kwargs,
    ):
        # Require explicit args namespace - fail fast if missing
        if args is None:
            raise ValueError(
                "MASAEnv requires explicit 'args' namespace. "
                "Pass args from MARL.common.arguments.get_common_args() or get_mixer_args()."
            )
        self.args = args

        # Ensure decision_operator_util is always defined
        self.decision_operator_util = 0.0

        if hasattr(args, 'seed') and getattr(args, 'seed') is not None:
            self.seed = int(getattr(args, 'seed'))
            LOG.info("[Env Init] Using args.seed=%s", self.seed)
        else:
            self.seed = int(seed)
            LOG.info("[Env Init] Using constructor seed=%s", self.seed)

        random.seed(self.seed)
        np.random.seed(self.seed)
        self._np_rng = np.random.RandomState(self.seed)
        self._py_rng = random.Random(self.seed)

        self.dump_config = bool(getattr(args, 'dump_config', False))

        self.config = kwargs.get('config', {}) or {}
        if not isinstance(self.config, dict):
            self.config = {}

        def _resolve(name_variants, cast=int, default=None):
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
                except (ValueError, TypeError):
                    return default
            raise ValueError(f"MASAEnv requires one of {list(name_variants)} to be provided via args or kwargs")

        self.num_jobs = _resolve(('n_agents', 'num_jobs'), int, default=0)
        self.num_ops = _resolve(('num_operators', 'num_ops'), int, default=1)
        self.num_wcs = _resolve(('num_wcs',), int, default=1)
        
        # Observation/state shapes defined by environment (not from args)
        # These values are derived from canonical observation/state builders in utils/env_obs.py
        self.obs_dim_agent = 8  # fixed by canonical obs builder (see utils/env_obs.py)
        self.state_dim = 10  # canonical state builder produces 10-element vector
        self.state_shape = self.state_dim  # alias for validation logic
        
        # Counters for global state tracking (raw counts)
        self.total_jobs_arrived = 0
        self.total_ops_arrived = 0

        if not hasattr(args, 'n_operation_types'):
            raise ValueError("MASAEnv requires args.n_operation_types")
        self.n_operation_types = int(getattr(args, 'n_operation_types'))

        if not hasattr(args, 'job_max_ops'):
            raise ValueError("MASAEnv requires args.job_max_ops")
        self.max_operations_per_job = int(getattr(args, 'job_max_ops'))

        if not hasattr(args, 'max_wait_time'):
            raise ValueError("MASAEnv requires args.max_wait_time")
        self.max_wait_time = float(getattr(args, 'max_wait_time'))

        self.mean_wait_reference = float(getattr(args, 'mean_wait_reference', 0.0)) if hasattr(args, 'mean_wait_reference') else None

        if not hasattr(args, 'avg_wait_scale'):
            raise ValueError("MASAEnv requires args.avg_wait_scale")
        self.avg_wait_scale = float(getattr(args, 'avg_wait_scale'))

        # max_jobs derived from num_jobs (already resolved above from n_agents/num_jobs)
        if self.num_jobs > 0:
            self.max_jobs = int(self.num_jobs)
        else:
            raise ValueError("MASAEnv requires num_jobs > 0 (from n_agents or num_jobs in args/kwargs)")

        if getattr(self, 'max_jobs', None) is None or int(self.max_jobs) <= 0:
            raise ValueError("Invalid environment configuration: max_jobs must be > 0")
        if getattr(self, 'max_operations_per_job', None) is None or int(self.max_operations_per_job) <= 0:
            raise ValueError("Invalid environment configuration: max_operations_per_job must be > 0")
        if getattr(self, 'n_operation_types', None) is None or int(self.n_operation_types) <= 0:
            raise ValueError("Invalid environment configuration: n_operation_types must be > 0")
        if getattr(self, 'max_wait_time', None) is None or float(self.max_wait_time) <= 0.0:
            raise ValueError("Invalid environment configuration: max_wait_time must be > 0")

        self.episode_limit = int(self.args.episode_limit)

        self.reward_w1 = args.reward_w1_completed
        self.reward_w2 = args.reward_w2_avgwait
        self.reward_w3 = args.reward_w3_wip
        self.reward_w4 = args.reward_w4_throughput_delta
        self.reward_w5 = args.reward_w5_load_variance

        self.reward_a1 = args.reward_a1_completion
        self.reward_a2 = args.reward_a2_wait
        self.reward_a3 = args.reward_a3_infeasible

        self.reward_alpha_mix = args.reward_alpha_mix
        self.lambda_m = args.reward_lambda_m
        self.lambda_o = args.reward_lambda_o

        self.log_reward_components = getattr(args, "reward_log_components", False)
        
        # [ADAPTIVE-FIX-2] GLOBAL reward normalizer (never reset per episode)
        # CRITICAL: QMIX requires globally consistent reward scaling because:
        # - Replay buffer samples mixed episodes
        # - TD learning requires stable Q_total gradients
        # - Mixer network assumes consistent reward scale
        # Episodic normalizers break this assumption and cause training instability.
        # References: PyMARL, PyMARL2, SMAC official implementations
        self.reward_normalizer = RunningMeanStd()

        self.job_min_ops = int(_resolve(('job_min_ops',), int, default=1))
        self.job_max_ops = int(_resolve(('job_max_ops',), int, default=5))

        if not hasattr(args, 'interarrival_time'):
            raise ValueError("args.interarrival_time is required for dynamic arrivals")
        self.interarrival_time = float(args.interarrival_time)

        self.auto_build = bool(auto_build)
        self.auto_start_arrivals = bool(auto_start_arrivals)

        if args is not None and hasattr(args, 'arrival_lambda'):
            try:
                self.arrival_lambda = float(args.arrival_lambda)
            except (ValueError, TypeError) as e:
                logging.getLogger(__name__).warning("Invalid arrival_lambda in args: %s", e)

        self.logger = LOG

        try:
            wc = kwargs.get('workcenters_meta', None)
            if wc is None:
                wc = WorkCenters()
            self.workcenters_meta = wc
        except Exception as e:
            raise RuntimeError(f"Failed to initialize WorkCenters: {e}")

        # Get processing_time_means from args, config, or use DEFAULT_PROCESSING_TIMES from workcenter module
        processing_time_means = getattr(args, 'processing_time_means', None)
        if processing_time_means is None:
            processing_time_means = self.config.get('processing_time_means', None)
        if processing_time_means is None:
            # Use DEFAULT_PROCESSING_TIMES directly (format: {machine_name: {OpN: duration}})
            # This will be converted to {OpN: {machine_name: duration}} for validation
            from utils.workcenter import DEFAULT_PROCESSING_TIMES
            # Convert to validation format: {OpN: {machine: duration}}
            processing_time_means = {}
            for machine_name, ops_map in DEFAULT_PROCESSING_TIMES.items():
                for op_name, duration in ops_map.items():
                    if op_name not in processing_time_means:
                        processing_time_means[op_name] = {}
                    processing_time_means[op_name][machine_name] = duration
            LOG.info("[Env] Using DEFAULT_PROCESSING_TIMES from workcenter.py")
        
        if not isinstance(processing_time_means, dict):
            raise ValueError(
                "processing_time_means must be a dict. "
                "Provide via args.processing_time_means, config, or ensure DEFAULT_PROCESSING_TIMES exists."
            )
        
        # C17 FIX: Validate processing times are positive and finite
        self._validate_processing_times(processing_time_means)
        self.processing_time_means = processing_time_means

        if not hasattr(self.workcenters_meta, 'machine_list') or not self.workcenters_meta.machine_list:
            raise ValueError(
                "MASAEnv requires workcenters_meta.machine_list to be populated. "
                "n_actions cannot be derived without knowing the machine count."
            )
        self.n_actions = int(len(self.workcenters_meta.machine_list))
        LOG.info("[Env] n_actions=%d (machines), num_wcs=%d (workcenters)", self.n_actions, self.num_wcs)

        if self.num_wcs is None or self.num_wcs <= 0:
            raise ValueError(
                "MASAEnv requires num_wcs (workcenter count) to be explicitly provided via args. "
                "Cannot infer workcenter count - it must be configured."
            )

        # SimPy runtime primitives
        self.env = simpy.Environment()
        self.machine_resources = [simpy.Resource(self.env, capacity=1) for _ in range(len(self.workcenters_meta.machine_list))]
        self.wc_resources = [simpy.Resource(self.env, capacity=1) for _ in range(int(self.num_wcs))]

        self.operator_groups = [simpy.Resource(self.env, capacity=1) for _ in range(max(1, int(self.num_ops)))]

        try:
            self.operators = Operators(self.workcenters_meta, env=self.env)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Operators: {e}. "
                "Operators are required - machines cannot operate without operators."
            )

        self.workcenters = self.workcenters_meta

        # [BUG FIX #3] Cumulative job counter for dynamic normalization
        self.total_jobs_cumulative = 0
        # [BUG FIX #1] Simple variable for step-to-step throughput delta
        self._prev_completed_count = 0
        # [BUG FIX #2] GLOBAL entropy tracking windows (not reset per episode)
        # [REBALANCE] Reduced from 200→50 so single action has 2% impact (was 0.5%)
        self.recent_machine_choices = deque(maxlen=50)  # was 200, originally 30
        self.recent_operator_choices = deque(maxlen=50)
        
        self.jobs: List[JobAgent] = []
        self.job_counter = 0
        self.pending_jobs: List[List] = []
        self.active_jobs: List[JobAgent] = []
        self.active_agents: List[JobAgent] = []

        self._generator_shutdown = False
        self.summary_interval = 20

        if not hasattr(args, 'initial_jobs'):
            raise ValueError("args.initial_jobs is required")
        self.initial_jobs = int(args.initial_jobs)

        try:
            from utils.task_generator import TaskGenerator  # type: ignore
            self.job_generator = TaskGenerator(py_rng=self._py_rng, np_rng=self._np_rng)
            # Pass processing_time_means to TaskGenerator
            self.job_generator.proc_time_means = processing_time_means
            setattr(self.job_generator, '_owner_env', self)
        except Exception as e:
            raise ImportError(
                f"Failed to initialize TaskGenerator: {e}. "
                "Ensure utils.task_generator.TaskGenerator exists and accepts (py_rng, np_rng)."
            )

        # Create initial jobs deterministically (t=0) before starting dynamic arrivals
        if not hasattr(args, 'n_agents') or args.n_agents is None:
            raise ValueError("args.n_agents is required to set max_active_agents capacity")
        self.max_active_agents = int(args.n_agents)
        
        if self.auto_build:
            # C1 FIX Rule 1: Removed exception swallowing - initial job generation must succeed
            # create the configured number of initial jobs deterministically
            self._generate_initial_jobs()

        self._job_generator_proc = None

        summary = {
            'seed': int(self.seed),
            'n_agents_capacity': int(self.num_jobs),
            'initial_jobs_requested': int(self.initial_jobs),
            'initial_jobs_created': len(self.jobs),
            'job_min_ops': int(self.job_min_ops),
            'job_max_ops': int(self.job_max_ops),
            'episode_limit': int(self.episode_limit),
        }
        LOG.info("[Env Summary] seed=%s capacity=%s initial_created=%s", summary['seed'], summary['n_agents_capacity'], summary['initial_jobs_created'])

        # C1 FIX Rule 3: Best-effort logging - log failure but continue training
        if allow_history_writes():
            try:
                hist_dir = self.args.history_dir if hasattr(self.args, 'history_dir') else os.path.join('my_data_and_graph', 'historydata')
                os.makedirs(hist_dir, exist_ok=True)
                path = os.path.join(hist_dir, 'env_summary.json')
                with open(path, 'w') as fh:
                    json.dump(summary, fh, indent=2, sort_keys=True)
            except Exception as e:
                LOG.warning(
                    "[C1] Failed to write env_summary.json (non-critical): %s. "
                    "Training continues.", e
                )
                logging.getLogger(__name__).debug('Wrote env summary to %s', path)
            except (OSError, IOError) as e:
                logging.getLogger(__name__).warning('Failed to write env_summary.json: %s', e)

        self.t = 0.0
        self.total_wait_time = 0.0
        self.wait_time_dict = {}  # job_id -> cumulative wait time
        self._completed_now_cache = 0
        self.done = False

        self.pending_decisions: List[Dict] = []
        self.decisions_ready = simpy.Event(self.env)
        self.gantt_records: List = []

        self._last_decision_info: List[Dict] = []

        if self.dump_config:
            try:
                dump_dir = os.path.join('my_data_and_graph', 'historydata')
                os.makedirs(dump_dir, exist_ok=True)
                dump_path = os.path.join(dump_dir, 'env_config_dump.json')
                with open(dump_path, 'w') as fh:
                    json.dump(self.config if self.config is not None else {}, fh, indent=2, sort_keys=True, default=str)
                logging.getLogger(__name__).debug('Wrote env config dump to %s', dump_path)
            except (OSError, IOError) as e:
                logging.getLogger(__name__).warning('Failed to write env_config_dump.json: %s', e)

    def _validate_processing_times(self, pt_means: dict):
        """Validate all processing times are positive and finite.
        
        C17 FIX: Prevents silent failures from zero, negative, or infinite
        processing times that would corrupt scheduling logic.
        
        Args:
            pt_means: dict of {op_type: {wc: duration}} processing time means
            
        Raises:
            ValueError: if any processing time is non-positive or non-finite
        """
        import numpy as np
        
        if not isinstance(pt_means, dict):
            raise ValueError(f"processing_time_means must be a dict, got {type(pt_means)}")
        
        errors = []
        for op_type, durations in pt_means.items():
            if not isinstance(durations, dict):
                errors.append(f"op_type={op_type}: durations must be dict, got {type(durations)}")
                continue
            
            for wc, dur in durations.items():
                try:
                    dur_f = float(dur)
                    if dur_f <= 0:
                        errors.append(f"op_type={op_type}, wc={wc}: processing_time must be positive, got {dur_f}")
                    if not np.isfinite(dur_f):
                        errors.append(f"op_type={op_type}, wc={wc}: processing_time must be finite, got {dur_f}")
                except (ValueError, TypeError) as e:
                    errors.append(f"op_type={op_type}, wc={wc}: cannot convert to float: {e}")
        
        if errors:
            raise ValueError(
                f"Processing time validation failed ({len(errors)} errors):\n" +
                "\n".join(f"  - {err}" for err in errors[:10]) +
                (f"\n  ... and {len(errors) - 10} more" if len(errors) > 10 else "")
            )
        
        LOG.info("[C17 FIX] Processing time validation passed: %d op_types validated", len(pt_means))

    def finish_lifecycle_trace(self):
        """Force-close an open lifecycle block if one was opened by a previous reset without an episode id."""
        ep = getattr(self, "episode_id", None)
        if ep is None:
            ep = getattr(self, 'current_episode', 'unknown')
        
        if hasattr(self, 'end_lifecycle_trace'):
            try:
                self.end_lifecycle_trace(ep)
                return
            except Exception as e:
                logging.getLogger(__name__).warning("end_lifecycle_trace failed: %s", e)
        
        if not hasattr(self, "history_dir") or self.history_dir is None:
            return
        
        try:
            os.makedirs(self.history_dir, exist_ok=True)
            with open(os.path.join(self.history_dir, "scheduling_timeline.txt"), "a", encoding='utf-8') as f:
                f.write(f"=== JOB AGENT LIFECYCLE TRACE END (EPISODE {ep}) ===\n\n")
        except (OSError, IOError) as e:
            logging.getLogger(__name__).debug("Failed to write lifecycle trace end: %s", e)

    def start_lifecycle_trace(self, ep: int):
        """Write a START lifecycle header for the given episode id."""
        hist_root = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata'))
        
        try:
            os.makedirs(hist_root, exist_ok=True)
            timeline_path = os.path.join(hist_root, 'scheduling_timeline.txt')
            
            header = f"=== EPISODE {ep} ===\n"
            contents = ''
            if os.path.exists(timeline_path):
                try:
                    with open(timeline_path, 'r', encoding='utf-8') as rf:
                        contents = rf.read()
                except (OSError, IOError):
                    pass
            
            with open(timeline_path, 'a', encoding='utf-8') as f:
                if header.strip() and header not in contents:
                    f.write(header)
                f.write(f"=== JOB AGENT LIFECYCLE TRACE START (EPISODE {ep}) ===\n")
                f.write("[Lifecycle] Environment reset: all queues cleared, waiting agents reset.\n")
        except (OSError, IOError) as e:
            logging.getLogger(__name__).debug("Failed to write lifecycle trace start: %s", e)

    def end_lifecycle_trace(self, ep: int):
        """Write an END lifecycle footer for the given episode id."""
        hist_root = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata'))
        
        if not hasattr(self, '_lifecycle_end_written'):
            self._lifecycle_end_written = set()
        
        try:
            ep_key = int(ep)
        except (ValueError, TypeError):
            ep_key = ep
        
        if ep_key in self._lifecycle_end_written:
            return
        
        try:
            os.makedirs(hist_root, exist_ok=True)
            timeline_path = os.path.join(hist_root, 'scheduling_timeline.txt')
            footer = f"=== JOB AGENT LIFECYCLE TRACE END (EPISODE {ep}) ===\n\n"
            
            # Check if footer already exists on disk
            if os.path.exists(timeline_path):
                try:
                    with open(timeline_path, 'r', encoding='utf-8') as rf:
                        if footer in rf.read():
                            self._lifecycle_end_written.add(ep_key)
                            return
                except (OSError, IOError):
                    pass
            
            # Write footer
            with open(timeline_path, 'a', encoding='utf-8') as f:
                f.write(footer)
            
            self._lifecycle_end_written.add(ep_key)
            
        except (OSError, IOError) as e:
            logging.getLogger(__name__).debug("Failed to write lifecycle trace end: %s", e)

    def reset(self):
        """Reset runtime state; keep configuration and metadata intact."""
        # IMPORTANT: Do NOT assign or guess `episode_id` here. The RolloutWorker
        # is responsible for assigning `self.episode_id` and calling
        # `start_lifecycle_trace()` before `reset()` so that lifecycle START/END
        # ordering remains correct. Removing any episode_id guessing here
        # prevents a one-episode phase shift where Episode 0 lifecycle appears
        # under Episode 1.

        # Signal any running generator loop tied to previous env to stop
        self._generator_shutdown = True

        self.env = simpy.Environment()
        random.seed(self.seed)
        np.random.seed(self.seed)
        self._np_rng = np.random.RandomState(self.seed)
        self._py_rng = random.Random(self.seed)
        # recreate resources
        self.machine_resources = [simpy.Resource(self.env, capacity=1) for _ in range(len(self.workcenters_meta.machine_list))]
        self.wc_resources = [simpy.Resource(self.env, capacity=1) for _ in range(int(self.num_wcs))]

        self.operator_groups = [simpy.Resource(self.env, capacity=1) for _ in range(max(1, int(self.num_ops)))]

        # Recreate Operators manager on reset
        try:
            self.operators = Operators(self.workcenters_meta, env=self.env)
        except Exception as e:
            raise RuntimeError(
                f"Failed to recreate Operators on reset: {e}. "
                "Operators are required for machine operations."
            )

        # reset bookkeeping and job state
        self.t = 0.0
        # legacy `completed_jobs` removed; compute finished counts from jobs when needed
        # TODO: requires test adaptation
        self.total_wait_time = 0.0
        self.wait_time_dict = {}  # job_id -> cumulative wait time
        self._completed_now_cache = 0
        # Note: internal recent_rewards removed - use metrics APIs for reward history
        self.done = False
        
        # [BUG-FIX] Window is now GLOBAL - do NOT reset here!
        # Window persists across episodes for stable entropy calculation
        # (initialized in __init__ instead)
        
        # NOTE: self.reward_normalizer is NOT reset here - it's global!
        # See __init__ for one-time initialization.
        
        # [BUG-FIX] Reset throughput delta tracker for new episode
        # Without this, first step of new episode gets huge negative delta!
        self._prev_completed_count = 0
        
        # [BUG-FIX] Reset cumulative job counter for new episode
        # This counter is used for dynamic normalization within an episode
        self.total_jobs_cumulative = 0
        
        # Reset global state counters
        self.total_jobs_arrived = 0
        self.total_ops_arrived = 0
        
        self.pending_decisions = []
        self.decisions_ready = simpy.Event(self.env)
        self.gantt_records = []
        # clear jobs and counters to ensure episode isolation
        self.jobs = []
        self.pending_jobs = []
        self.active_jobs = []
        self.active_agents = []
        self.job_counter = 0

        # lifecycle START is intentionally not written here (caller-managed)

        # Do not start any internal dynamic arrival loops on reset; however
        # create deterministic initial jobs at t=0 so tests and callers that
        # expect initial workload observe it. Initial jobs are produced via
        # `self._generate_initial_jobs()` which prefers TaskGenerator.
        self._job_generator_proc = None
        self._dynamic_arrival_proc = None
        self._task_generator = None

        tg_cfg = (self.config or {}).get('task_generator', {}) or {}
        lam = float(tg_cfg.get('arrival_lambda', 0.0)) if tg_cfg is not None else 0.0
        if not lam and getattr(self, 'auto_start_arrivals', False):
            lam = float(getattr(self, 'arrival_lambda', 0.0) or 0.0)
        
        if lam and lam > 0.0:
            try:
                from utils.task_generator import TaskGenerator  # type: ignore
                tg = TaskGenerator(py_rng=self._py_rng, np_rng=self._np_rng)
                setattr(tg, '_owner_env', self)
                tg.start(self.env, lam)
                self._task_generator = tg
            except (ImportError, TypeError, AttributeError) as e:
                # [C1] TaskGenerator is optional (Rule 3) - log warning if unavailable
                logging.getLogger(__name__).warning("[C1] Failed to initialize TaskGenerator (optional): %s", e)
                self._task_generator = None

        # [C1] Initial job generation is CRITICAL - fail-fast if generation fails
        self._generate_initial_jobs()

        LOG.info("[Env] Episode time limit set to %s seconds", self.episode_limit)
        
        try:
            self.env.process(self._periodic_summary())
        except (AttributeError, RuntimeError) as e:
            # [C1] Periodic summary is best-effort monitoring (Rule 3)
            logging.getLogger(__name__).debug("[C1] Failed to start periodic summary (monitoring only): %s", e)
        # DEBUG: print machine counts for tracing unexpected machine totals
        mr_len = len(getattr(self, 'machine_resources', []) or [])
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        ml_len = len(mlist)
        n_actions = getattr(self, 'n_actions', None)
        LOG.debug("num_wcs=%s (workcenters), len(machine_list)=%s (machines), n_actions=%s (action space)", getattr(self,'num_wcs',None), ml_len, n_actions)
        # Verify n_actions matches machine count
        if ml_len > 0 and int(getattr(self, 'n_actions', 0)) != ml_len:
            LOG.warning("Correcting n_actions (%s) -> %s to match machine_list", getattr(self,'n_actions',None), ml_len)
            self.n_actions = int(ml_len)
        # lifecycle END is intentionally not written here (caller-managed)
        return self._build_all_agent_obs(), {"state_vec": None, "avail_actions": None}

    def step(self, action=None):
        """Compatibility step API: advance sim by a time-quantum and return (obs, reward, done, info)."""
        self.env.run(until=self.env.now + 1.0)
        self.t = float(self.env.now)
        reward = float(self.pop_decision_reward())
        obs = self._build_all_agent_obs()
        info = {"state_vec": None, "avail_actions": None}
        done = (self.t >= self.episode_limit)
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
                if hasattr(self, attr):
                    return getattr(self, attr)
                if wrapper is not None and hasattr(wrapper, attr):
                    return getattr(wrapper, attr)
                if hasattr(self.env, attr):
                    return getattr(self.env, attr)
                return default

            no_progress = 0
            max_no_progress = int(getattr(self, '_wait_no_progress_limit', 3))
            max_iterations = 1000  # Safety limit to prevent infinite loops
            iterations = 0
            while not self.pending_decisions and not bool(_get_attr_from_env('done', False)):
                # Check episode limit
                if float(self.env.now) >= float(self.episode_limit):
                    self.done = True
                    logging.getLogger(__name__).debug("wait_for_decisions: episode limit reached (t=%.2f >= %.2f)", float(self.env.now), float(self.episode_limit))
                    return [], float(self.env.now)
                
                # Safety: prevent infinite loops
                iterations += 1
                if iterations >= max_iterations:
                    logging.getLogger(__name__).warning("wait_for_decisions: max iterations (%d) reached at t=%.2f, returning empty batch", max_iterations, float(self.env.now))
                    self.done = True
                    return [], float(self.env.now)
                
                prev_now = float(self.env.now)
                self.env.run(until=prev_now + step)
                new_now = float(self.env.now)
                if self.pending_decisions:
                    break
                if new_now == prev_now:
                    no_progress += 1
                    if no_progress >= max_no_progress:
                        logging.getLogger(__name__).debug("wait_for_decisions: no scheduled events after %d attempts; returning empty batch", no_progress)
                        return [], float(self.env.now)
                else:
                    no_progress = 0

        self.t = float(self.env.now)
        batch = list(self.pending_decisions)
        self.pending_decisions = []
        self.decisions_ready = simpy.Event(self.env)
        
        # ===== PADDING LOGIC: Normalize batch to n_agents =====
        # Get canonical agent count - must exist, no fallback
        if not hasattr(self, 'max_jobs'):
            raise ValueError(
                "[FIXED_AGENT_BATCH] max_jobs is not defined. "
                "Environment must be initialized with explicit agent capacity."
            )
        n_agents = int(self.max_jobs)
        real_count = len(batch)
        
        # Validate batch size doesn't exceed capacity
        if real_count > n_agents:
            raise ValueError(
                f"[FIXED_AGENT_BATCH] Decision batch size ({real_count}) exceeds "
                f"n_agents capacity ({n_agents}). This indicates a configuration error."
            )
        
        # Add explicit mask to real decision items
        for i in range(real_count):
            batch[i]['agent_mask'] = 1  # Real agent
            batch[i]['agent_index'] = i  # Original position
        
        # Pad batch to n_agents with dummy items
        if real_count < n_agents:
            dummy_obs = np.zeros(self.obs_dim_agent, dtype=np.float32)
            dummy_avail = np.zeros(len(self.machine_resources), dtype=np.int32)
            
            for i in range(real_count, n_agents):
                batch.append({
                    'job_id': -1,  # Invalid job ID indicates padding
                    'obs': dummy_obs,
                    'avail_row': dummy_avail,
                    'agent_mask': 0,  # Padded agent
                    'agent_index': i,
                    'allowed_machine_indices': [],
                    'per_machine_durations': {},
                    'resume_evt': None  # No event for padded agents
                })
        # ===== END PADDING LOGIC =====
        
        return batch, float(self.t)

    def pop_decision_reward(self) -> float:
        """Return shaped reward computed since last pop."""
        # Clear per-pop completed cache
        self._completed_now_cache = 0

        # ----- Global metrics (K1..K5) -----
        # K1: CompletedNorm
        completed_count = len([j for j in self.jobs if j.finished])
        # Use actual total jobs as denominator for dynamic arrivals
        actual_total_jobs = max(self.total_jobs_cumulative, self.max_jobs, len(self.jobs))
        CompletedNorm = float(completed_count / float(max(1, actual_total_jobs)))
        # CompletedNorm will naturally be in [0,1] since completed_count <= actual_total_jobs

        # K2: AvgWaitNorm
        jobs_len = max(1, len(self.jobs))
        avg_wait_per_job = float(self.total_wait_time) / float(jobs_len)
        max_wait = float(self.max_wait_time)
        AvgWaitNorm = float(np.clip(avg_wait_per_job / (max_wait if max_wait > 0 else 1.0), 0.0, 1.0))
        # [PHASE3-FIX] Validate AvgWaitNorm is finite
        if not np.isfinite(AvgWaitNorm):
            raise ValueError(
                f"[PHASE3] AvgWaitNorm is not finite: {AvgWaitNorm}. "
                f"avg_wait={avg_wait_per_job}, max_wait={max_wait}"
            )

        # K3: WIPNorm
        wip_count = len([j for j in self.active_agents if not j.finished])
        # Use actual total jobs as denominator for dynamic arrivals
        WIPNorm = float(wip_count) / float(max(1, actual_total_jobs))
        # WIPNorm will naturally be in [0,1] since wip_count <= actual_total_jobs

        # K4: ThroughputDelta (immediate feedback when jobs complete)
        # [BUG-FIX] Use simple previous value instead of deque for step-to-step comparison
        # Calculate delta from PREVIOUS STEP (not N steps ago!)
        completed_now = len([j for j in self.jobs if j.finished])
        
        # Dynamic total jobs for normalization (handles dynamic arrivals)
        actual_total_jobs = max(self.total_jobs_cumulative, self.max_jobs, len(self.jobs))
        
        # Delta from previous step
        throughput_delta = float(completed_now - self._prev_completed_count) / float(max(1, actual_total_jobs))
        
        # Update for next step
        self._prev_completed_count = completed_now

        # [ADAPTIVE-FIX-3] K5: LoadBalance (entropy-based, works mid-episode)
        # Use sliding window of recent choices (30 decisions ≈ 3 episodes)
        # Entropy measures uniformity: higher entropy = more balanced load distribution
        
        # Calculate entropy for machines
        balance_m = 0.0
        if hasattr(self, 'recent_machine_choices') and len(self.recent_machine_choices) > 0:
            machine_counts = Counter(self.recent_machine_choices)
            total_m = len(self.recent_machine_choices)
            entropy_m = -sum((count/total_m) * math.log(count/total_m + 1e-10) for count in machine_counts.values())
            max_entropy_m = math.log(len(machine_counts)) if len(machine_counts) > 1 else 1.0
            balance_m = entropy_m / max_entropy_m if max_entropy_m > 0 else 0.0
        
        # Calculate entropy for operators
        balance_o = 0.0
        if hasattr(self, 'recent_operator_choices') and len(self.recent_operator_choices) > 0:
            operator_counts = Counter(self.recent_operator_choices)
            total_o = len(self.recent_operator_choices)
            entropy_o = -sum((count/total_o) * math.log(count/total_o + 1e-10) for count in operator_counts.values())
            max_entropy_o = math.log(len(operator_counts)) if len(operator_counts) > 1 else 1.0
            balance_o = entropy_o / max_entropy_o if max_entropy_o > 0 else 0.0
        
        # Combine: average normalized entropy [0, 1]
        # Higher value = better load distribution
        lambda_m = float(getattr(self, 'lambda_m', 0.8))
        lambda_o = float(getattr(self, 'lambda_o', 0.2))
        load_balance_score = float((lambda_m * balance_m) + (lambda_o * balance_o))

        # Compute R_global per spec
        # [ADAPTIVE-FIX-3] LoadBalance is now a positive reward (higher entropy = better)
        # [v5-REBALANCE] Adjusted weights for action-sensitivity:
        #   - w4: 5.0→8.0 (ThroughputDelta stronger immediate feedback)
        #   - w5: 0.4→0.1 (LoadBalance reduced, was dominating 79% of reward)
        R_global = (
            (float(getattr(self, 'reward_w1', 1.0)) * float(CompletedNorm))
            - (float(getattr(self, 'reward_w2', 0.6)) * float(AvgWaitNorm))
            - (float(getattr(self, 'reward_w3', 0.3)) * float(WIPNorm))
            + (float(getattr(self, 'reward_w4', 8.0)) * float(throughput_delta))
            + (float(getattr(self, 'reward_w5', 0.1)) * float(load_balance_score))
        )
        # Quick Win C10: Validate R_global is finite
        if not np.isfinite(R_global):
            raise RuntimeError(
                f"Invalid R_global computed: {R_global}. "
                f"Components: CompletedNorm={CompletedNorm}, AvgWaitNorm={AvgWaitNorm}, "
                f"WIPNorm={WIPNorm}, throughput_delta={throughput_delta}, load_balance_score={load_balance_score}"
            )

        # [v4-FIX] R_local component REMOVED
        # Reasons:
        # 1. Technical: _last_decision_info always empty due to timing bug (cleared before use)
        # 2. Conceptual: Redundant with R_global components
        #    - job_completed overlap with CompletedNorm
        #    - wait_penalty overlap with AvgWaitNorm
        #    - infeasible penalty unnecessary (action masking handles this)
        # 3. Result: R_local_mean was always 0, contributing nothing to learning
        # 
        # Replacement: Scaled up ThroughputDelta (5.0x) provides immediate feedback
        # Future: Can add dense operation-level rewards if needed
        
        R_total = float(R_global)  # Simplified: no mixing needed
        
        # [v6-FIX] Reward scaling for stable Q-learning
        # QMIX loss explodes with large rewards (Loss was 3.8M!)
        # Scale rewards to match SMAC range: episode ~45 → ~0.9
        reward_scale = float(getattr(self, 'reward_scale', 2.0))
        R_total = R_total / reward_scale
        
        # Validate final R_total is finite
        if not np.isfinite(R_total):
            raise RuntimeError(
                f"Invalid R_total computed: {R_total}. "
                f"Components: alpha_mix={alpha_mix}, R_global={R_global}, R_local_mean={R_local_mean}"
            )

        # Diagnostics
        self.last_reward_components = {
            'CompletedNorm': float(CompletedNorm),
            'AvgWaitNorm': float(AvgWaitNorm),
            'WIPNorm': float(WIPNorm),
            'ThroughputDelta': float(throughput_delta),
            'LoadBalanceScore': float(load_balance_score),
            'R_global': float(R_global),
            'R_total': float(R_total),
            # [v4-FIX] R_local_mean removed (was always 0)
        }

        # [v4-FIX] Log reward components to CSV file for analysis
        if not hasattr(self, '_reward_log_initialized'):
            self._reward_log_initialized = False
        
        try:
            hist_dir = getattr(self, 'history_dir', os.path.join('my_data_and_graph', 'historydata'))
            os.makedirs(hist_dir, exist_ok=True)
            reward_log_path = os.path.join(hist_dir, 'reward_components.csv')
            
            # Write dynamic header with formula explanation and current coefficients on first call
            if not self._reward_log_initialized:
                w1 = float(getattr(self, 'reward_w1', 1.0))
                w2 = float(getattr(self, 'reward_w2', 0.6))
                w3 = float(getattr(self, 'reward_w3', 0.3))
                w4 = float(getattr(self, 'reward_w4', 8.0))
                w5 = float(getattr(self, 'reward_w5', 0.1))
                reward_scale = float(getattr(self, 'reward_scale', 2.0))
                lambda_m = float(getattr(self, 'lambda_m', 0.8))
                lambda_o = float(getattr(self, 'lambda_o', 0.2))
                with open(reward_log_path, 'w', encoding='utf-8') as f:
                    f.write(f'# Reward Formula: R_total = R_global / reward_scale\n')
                    f.write(f'# R_global = w1*CompletedNorm - w2*AvgWaitNorm - w3*WIPNorm + w4*ThroughputDelta + w5*LoadBalance\n')
                    f.write(f'# Coefficients: w1={w1}, w2={w2}, w3={w3}, w4={w4}, w5={w5}\n')
                    f.write(f'# reward_scale={reward_scale}\n')
                    f.write(f'# LoadBalance: lambda_m={lambda_m}, lambda_o={lambda_o}\n')
                    f.write('# CompletedNorm: fraction of jobs completed [0,1]\n')
                    f.write('# AvgWaitNorm: normalized average wait time [0,1]\n')
                    f.write('# WIPNorm: work-in-progress ratio [0,1]\n')
                    f.write('# ThroughputDelta: change in completed jobs (normalized) - immediate feedback!\n')
                    f.write('# LoadBalance: entropy-based load distribution [0,1] - higher is better\n')
                    f.write('#\n')
                    f.write('step,sim_time,CompletedNorm,AvgWaitNorm,WIPNorm,ThroughputDelta,LoadBalance,R_global,R_total\n')
                self._reward_log_initialized = True
            
            # Append data row
            if not hasattr(self, '_reward_step_counter'):
                self._reward_step_counter = 0
            self._reward_step_counter += 1
            
            sim_time = float(self.env.now) if hasattr(self, 'env') else 0.0
            with open(reward_log_path, 'a', encoding='utf-8') as f:
                f.write(f'{self._reward_step_counter},{sim_time:.2f},')
                f.write(f'{CompletedNorm:.6f},{AvgWaitNorm:.6f},{WIPNorm:.6f},')
                f.write(f'{throughput_delta:.6f},{load_balance_score:.6f},')
                f.write(f'{R_global:.6f},{R_total:.6f}\n')
        except Exception as e:
            # Best-effort logging - don't crash training if logging fails
            LOG.debug('[v4] Reward component logging failed: %s', e)

        # Optional debug logging
        if bool(getattr(self, 'log_reward_components', False)):
            LOG.debug('[REWARD COMPONENTS] %s', self.last_reward_components)

        # [v4-FIX] Reward normalizer REMOVED
        # Issue: Normalizer was compressing already weak reward signal
        # - Raw reward: -0.084 → Normalized: -0.0008 (10x smaller!)
        # - This made learning impossible (signal too weak)
        # Solution: Return raw R_total directly
        # Note: If rewards become too large in future, can re-enable with proper tuning
        
        # Keep normalizer for future use but don't apply it
        self.reward_normalizer.update(R_total)  # Track statistics only
        
        return float(R_total)  # Return raw reward without normalization

    # ---------------- SimPy job process (simple, robust) ----------------
    def _job_process(self, job: JobAgent):
        while not job.finished and self.env.now < self.episode_limit:
            op = job.current_op()
            if op is None:
                # No current op -> mark job completed (idempotent)
                now_t = float(self.env.now)
                if job.mark_completed(now_t):
                    self._completed_now_cache += 1
                    if job in self.active_jobs:
                        self.active_jobs.remove(job)
                    if job in self.active_agents:
                        self.active_agents.remove(job)

                    # Persist completion event in lifecycle trace
                    hist_dir = self.history_dir if hasattr(self, 'history_dir') else os.path.join('my_data_and_graph', 'historydata')
                    os.makedirs(hist_dir, exist_ok=True)
                    timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
                    completed_count = len([j for j in self.jobs if j.finished])
                    with open(timeline_path, 'a', encoding='utf-8') as tf:
                        tf.write(f"[t={float(now_t):.2f}] Job {job.id} completed -> Active:{len(self.active_agents)} | Pending:{len(self.pending_jobs)} | Completed:{completed_count}\n")

                    # --- JOB STATUS LOGGING ---
                    # Prevent logging completed jobs to observation log
                    if not getattr(job, 'finished', False):
                        job_status = "Completed"
                        obs_vec = self._build_agent_obs(job)
                        allowed_machine_indices_log = []
                        avail_actions_log = []
                        print(f"[DEBUG] Job {job.id} completed, logging status: {job_status} to decision_observation_metrics.csv")
                        self._write_observation_log(
                            now_t,
                            job.id,
                            allowed_machine_indices_log,
                            avail_actions_log,
                            action_idx,
                            obs_vec,
                            job_status
                        )
                break

            # normalize op formats: support legacy (allowed_machine_indices, dur) and
            # canonical (op_type, allowed_machine_indices, per_wc_durations)
            #op_type = None
            #allowed_machine_indices = []
            #per_wc = None
            #base_dur = None
            #if isinstance(op, (list, tuple)):
            #    if len(op) == 2:
            #        allowed_machine_indices, base_dur = op
            #    elif len(op) >= 3:
            #        op_type = op[0]
            #        allowed_machine_indices = op[1]
            #        per_wc = op[2]

            # NEW FIXED: operation tuple normalization – no legacy allowed_machine_indices parsing
            # Canonical format: (op_type,) only. Real allowed machines come from WorkCenter.
            try:
                op_type = int(op[0]) if isinstance(op, (list, tuple)) and len(op) >= 1 else int(job.current_op_idx)
            except:
                op_type = int(job.current_op_idx)

            allowed_machine_indices = []   # will be filled later by WorkCenter.create_decision_item
            per_wc = {}
            base_dur = 0.0


            # C1 FIX Rule 2: Removed exception swallowing - invalid op_idx must fail explicitly
            # Resolve operation index (machine-level op index) for use in
            # operator qualification and duration lookups. Prefer explicit
            # op_type when provided, else fall back to job.current_op_idx.
            op_idx_local = int(op_type) if op_type is not None else int(getattr(job, 'current_op_idx', 0))
            # C1 FIX Rule 2: Removed exception swallowing - decision_item creation must succeed
            # D: Compute machine_free status for 3-step reasoning
            machine_free_status = self._compute_machine_free_status()
            
            # create decision item — prefer workcenters_meta helper if present
            if hasattr(self.workcenters_meta, 'create_decision_item'):
                decision_item = self.workcenters_meta.create_decision_item(
                    self, job, op, machine_free_status=machine_free_status
                )
            else:
                # Fallback to manual creation (this is expected behavior, not an error)
                decision_item = {
                    'job_id': job.id,
                    'obs': self._build_agent_obs(job),
                    'avail_row': self._avail_row_for_job(job, machine_free=machine_free_status),
                    'allowed_machine_indices': allowed_machine_indices,
                    'per_machine_durations': {},
                }

            # Environment owns the resume event — create it here and attach to
            # the decision_item so the policy/runner can call resume_evt.succeed(choice).
            resume_evt = simpy.Event(self.env)
            decision_item['resume_evt'] = resume_evt

            # If this is the start of a new decision batch (pending_decisions
            # currently empty), overwrite the last decision cache so we do
            # not accumulate entries across batches.
            if not self.pending_decisions:
                self._last_decision_info = []

            self.pending_decisions.append(decision_item)
            if not self.decisions_ready.triggered:
                self.decisions_ready.succeed()

            chosen_idx = (yield resume_evt)
            # Eğer iş tamamlandıysa, karar noktasına gelmesin ve log yazılmasın
            if job.finished:
                continue
            # Eğer agent aksiyon seçemedi (chosen_idx == -1), job beklesin ve bir sonraki decision pointte tekrar denesin
            if chosen_idx is None or int(chosen_idx) == -1:
                # Log: No valid action, job waits
                decision_time = float(self.env.now)
                job_id = int(job.id)
                allowed_machine_indices_log = list(decision_item.get("allowed_machine_indices", []))
                avail_actions_all = self._build_avail_actions()
                j_index = self.jobs.index(job)
                n_machines = len(avail_actions_all[j_index])
                allowed_set = set(allowed_machine_indices_log)
                avail_actions_log = [1 if i in allowed_set else 0 for i in range(n_machines)]
                obs_vec = self._build_agent_obs(job, allowed_machine_indices=allowed_machine_indices_log)
                state_vec = self._build_global_state()
                job_status = "Completed" if job.finished else "WIP"
                action_idx = -1  # Güvenli default
                if not job.finished:
                    self._write_observation_log(
                        decision_time,
                        job_id,
                        allowed_machine_indices_log,
                        avail_actions_log,
                        action_idx,
                        obs_vec,
                        job_status
                    )
                self._write_state_log(
                    decision_time,
                    job_id,
                    allowed_machine_indices_log,
                    avail_actions_log,
                    action_idx,
                    job.current_op_idx,
                    state_vec
                )
                # Loglama fonksiyonuna reason ekle
                # scheduling_trace.csv için reason: no_valid_action_job_waits
                if hasattr(self, '_write_scheduling_trace_log'):
                    self._write_scheduling_trace_log(
                        decision_time,
                        job_id,
                        allowed_machine_indices_log,
                        avail_actions_log,
                        action_idx,
                        None,
                        'no_valid_action_job_waits'
                    )
                # Job beklesin (ör: 1 simpy time unit)
                yield self.env.timeout(1)
                continue
            
            # ===================== DECISION LOGGING =====================
            # Fail-safe: Eğer agent maskte 0 olan bir aksiyonu seçtiyse, job beklesin ve tekrar denesin
            if 'action_idx' in locals() and action_idx >= 0 and action_idx < len(avail_actions_log):
                if avail_actions_log[action_idx] == 0:
                    print(f"[FAIL-SAFE] Agent selected unavailable machine (idx={action_idx}) according to mask. Job will wait and retry.")
                    job_status = "Completed" if job.finished else "WIP"
                    self._write_observation_log(
                        decision_time,
                        job_id,
                        allowed_machine_indices_log,
                        avail_actions_log,
                        action_idx,
                        obs_vec,
                        job_status
                    )
                    self._write_state_log(
                        decision_time,
                        job_id,
                        allowed_machine_indices_log,
                        avail_actions_log,
                        action_idx,
                        job.current_op_idx,
                        state_vec
                    )
                    if hasattr(self, '_write_scheduling_trace_log'):
                        self._write_scheduling_trace_log(
                            decision_time,
                            job_id,
                            allowed_machine_indices_log,
                            avail_actions_log,
                            action_idx,
                            None,
                            'unavailable_machine_selected_job_waits'
                        )
                    yield self.env.timeout(1)
                    continue
            try:
                decision_time = float(self.env.now)
                job_id = int(job.id)

                # Allowed machines parsed from operation metadata
                #allowed_machine_indices_log = list(allowed_machine_indices) if allowed_machine_indices else []
                # ----------------------------------------------
                # Allowed machines for LOGGING (canonical version)
                # ----------------------------------------------
                #if isinstance(decision_item, dict) and 'allowed_machine_indices' in decision_item:
                #    allowed_machine_indices_log = list(decision_item['allowed_machine_indices'])
                #else:
                #    allowed_machine_indices_log = list(allowed_machine_indices) if allowed_machine_indices else []
                # ----------------------------------------------

                # Canonical allowed machine list (ONLY from decision_item)
                allowed_machine_indices_log = list(decision_item.get("allowed_machine_indices", []))

                # Availability mask for this job
                avail_actions_all = self._build_avail_actions()
                j_index = self.jobs.index(job)
                # Mask only allowed machines as 1, others as 0
                n_machines = len(avail_actions_all[j_index])
                allowed_set = set(allowed_machine_indices_log)
                avail_actions_log = [1 if i in allowed_set else 0 for i in range(n_machines)]

                # Action chosen by agent
                action_idx = int(chosen_idx)
                # Seçilen aksiyonun gerçekten uygulanabilirliğini kontrol et
                action_failed = False
                fail_reason = ''
                chosen_machine_name = None
                if action_idx >= 0 and action_idx < len(avail_actions_log):
                    # Makine ve operatör uygun mu?
                    mlist = self.workcenters_meta.machine_list if hasattr(self.workcenters_meta, 'machine_list') else list(range(len(avail_actions_log)))
                    mname = mlist[action_idx] if action_idx < len(mlist) else str(action_idx)
                    # Makine busy mi?
                    m_busy = False
                    if self.machine_resources and 0 <= action_idx < len(self.machine_resources):
                        res = self.machine_resources[action_idx]
                        m_busy = len(res.users) > 0
                    # Operatör uygun mu?
                    op_ok = False
                    if self.operators is not None:
                        registry = self.workcenters_meta.machine_registry or {}
                        wc_idx_for_m = int(registry[mname].get('workcenter', -1)) if mname in registry else None
                        for op_obj in self.operators.operators_object_list:
                            if mname in op_obj.qualified_machines and wc_idx_for_m is not None and op_obj.can_do_job(job.current_op_idx, wc_idx_for_m) and not op_obj.is_busy:
                                op_ok = True
                                break
                    if m_busy or not op_ok:
                        action_failed = True
                        fail_reason = 'action_failed_due_to_resource_contention'
                        chosen_machine_name = None
                else:
                    if action_idx == -1:
                        fail_reason = 'machine_not_available'
                        chosen_machine_name = None
                # Loglama fonksiyonuna reason ve bekleme süresi ekle
                # ...existing code...
                # DEBUG: action_idx ve maski yan yana yazdır
                print(f"[DEBUG] action_idx: {action_idx}, avail_actions_log: {avail_actions_log}")
                if action_idx < len(avail_actions_log):
                    print(f"[DEBUG] Selected machine mask value: {avail_actions_log[action_idx]}")
                    if avail_actions_log[action_idx] == 0:
                        print(f"[WARNING] Agent selected a machine (idx={action_idx}) that is not available according to mask!")
                # DEBUG: allowed_machine_indices_log ve avail_actions_log'u yan yana yazdır
                print(f"[DEBUG] allowed_machine_indices_log: {allowed_machine_indices_log}")
                print(f"[DEBUG] avail_actions_log: {avail_actions_log}")

                # Observation for this job
                obs_vec = self._build_agent_obs(job, allowed_machine_indices=allowed_machine_indices_log)

                # Global state vector
                state_vec = self._build_global_state()

                # Write logs
                job_status = "Completed" if job.finished else "WIP"
                self._write_observation_log(
                    decision_time,
                    job_id,
                    allowed_machine_indices_log,
                    avail_actions_log,
                    action_idx,
                    obs_vec,
                    job_status
                )

                self._write_state_log(
                    decision_time,
                    job_id,
                    allowed_machine_indices_log,
                    avail_actions_log,
                    action_idx,
                    obs_vec[0] + 1,
                    state_vec
                )

            except Exception as e:
                print(f"[LOGGING ERROR] Could not write decision logs for job {getattr(job, 'id', '?')}: {e}")
            # ================== END DECISION LOGGING =====================
                
            # C13 FIX: Strict machine index validation
            # Validate against actual machine count (len(machine_resources)),
            # not n_actions (which is an abstract action space concept).
            # In MASAEnv, agent selects a machine index directly.
            chosen_idx_int = int(chosen_idx)
            n_machines = len(self.machine_resources) if self.machine_resources else len(self.workcenters_meta.machine_list)
            if not (0 <= chosen_idx_int < n_machines):
                raise ValueError(
                    f"[C13 FIX] Invalid machine index: {chosen_idx_int} out of bounds "
                    f"[0, {n_machines}). Job={job.id}, op_idx={job.current_op_idx}, "
                    f"t={float(self.env.now):.2f}"
                )
            # MASK ELIGIBILITY FAIL-SAFE: Eğer agent maskte 0 olan bir aksiyonu seçtiyse, job beklesin ve tekrar denesin
            if chosen_idx_int >= 0 and chosen_idx_int < len(avail_actions_log):
                if avail_actions_log[chosen_idx_int] == 0:
                    print(f"[FAIL-SAFE] Agent selected unavailable machine (idx={chosen_idx_int}) according to mask. Job will wait and retry.")
                    job_status = "Completed" if job.finished else "WIP"
                    if not job.finished:
                        self._write_observation_log(
                            decision_time,
                            job_id,
                            allowed_machine_indices_log,
                            avail_actions_log,
                            chosen_idx_int,
                            obs_vec,
                            job_status
                        )
                    self._write_state_log(
                        decision_time,
                        job_id,
                        allowed_machine_indices_log,
                        avail_actions_log,
                        chosen_idx_int,
                        job.current_op_idx,
                        state_vec
                    )
                    if hasattr(self, '_write_scheduling_trace_log'):
                        self._write_scheduling_trace_log(
                            decision_time,
                            job_id,
                            allowed_machine_indices_log,
                            avail_actions_log,
                            chosen_idx_int,
                            None,
                            'unavailable_machine_selected_job_waits'
                        )
                    yield self.env.timeout(1)
                    continue

            # C1 FIX Rule 2 + C17: Duration computation must not use fake default 0.0
            # compute duration - fail if no valid duration found
            # FIX: Use machine index directly from chosen_idx (agent's action)
            dur = None
            if int(chosen_idx) in decision_item.get('per_machine_durations', {}):
                dur = float(decision_item['per_machine_durations'][int(chosen_idx)])
            elif per_wc is not None:
                # per_wc might be indexed by machine number (0-4) not workcenter
                # Try chosen_idx first (machine index), then allowed_machine_indices mapping
                if isinstance(per_wc, dict):
                    dur = float(per_wc.get(int(chosen_idx)))  if int(chosen_idx) in per_wc else None
                    if dur is None and isinstance(allowed_machine_indices, (list, tuple)) and len(allowed_machine_indices) > int(chosen_idx):
                        chosen_wc = int(allowed_machine_indices[int(chosen_idx)])
                        dur = float(per_wc.get(chosen_wc)) if chosen_wc in per_wc else None
                else:
                    dur = float(per_wc)
            
            if dur is None and base_dur is not None:
                dur = float(base_dur)
            
            # File: MASAEnv.py (inside MASAEnv class, _job_process method)

            # ... (duration calculation logic attempting to derive 'dur') ...
                    
            # Final fallback: use a reasonable default (e.g., 2.0) instead of failing
            if dur is None or dur <= 0:
                # -------------------------------------------------------------
                # !!! CRITICAL FIX 3: REMOVE RANDOM DURATION FALLBACK AND RAISE ERROR !!!
                # Job execution is strictly deterministic and requires positive, finite duration.
                
                error_msg = (
                    f"[CRITICAL ERROR - DURATION MISSING] No valid positive duration found for "
                    f"job={job.id}, machine={chosen_idx}. Duration must be explicitly provided and positive. "
                    f"Check TaskGenerator and processing_time_means configuration. "
                    f"per_machine_durations={decision_item.get('per_machine_durations', {})}, "
                    f"per_wc={per_wc}, base_dur={base_dur}"
                )
                logging.getLogger(__name__).error(error_msg)
                # Use ValueError as used elsewhere in this module for config errors (e.g., _validate_processing_times)
                raise ValueError(error_msg)
                # -------------------------------------------------------------

            # C17 validation: Duration must be positive and finite
            if dur <= 0 or not np.isfinite(dur):
                raise ValueError(
                    f"[C1+C17] Invalid duration {dur} for job={job.id}, machine={chosen_idx}. "
                    f"Duration must be positive and finite."
                )
            # ...

            # Before acquiring resources, build a decision-time trace that
            # records machine/operator availability for every eligible machine.
            mlist = self.workcenters_meta.machine_list or []
            total_machines = int(self.total_machines) if hasattr(self, 'total_machines') and self.total_machines is not None else (len(mlist) if mlist else self.num_wcs)
            allowed_list = list(range(total_machines))
            LOG.debug("[DEBUG _job_process] total_machines=%s, allowed_list=%s, len(machine_resources)=%s, num_wcs=%s", 
                     total_machines, allowed_list, len(self.machine_resources), self.num_wcs)

            # map to machine names when possible
            mlist = self.workcenters_meta.machine_list or []
            elig_entries: List[EligibilityEntry] = []
            now_t = float(self.env.now)
            
            # helper to compute machine next-free from gantt_records
            def _machine_next_free(mid: int) -> float:
                latest = now_t
                for r in self.gantt_records:
                    # support dict and tuple forms
                    if isinstance(r, dict):
                        r_w = int(r.get('wc_idx', r.get('wc', -1)))
                        r_end = float(r.get('end', r.get('e', 0.0) or 0.0))
                    else:
                        r_w = int(r[3])
                        r_end = float(r[1])
                    if r_w == int(mid) and r_end > latest:
                        latest = r_end
                return float(latest)

            for midx in allowed_list:
                midx = int(midx)
                # machine name
                mname = mlist[int(midx)] if mlist and 0 <= int(midx) < len(mlist) else f"M{int(midx)}"
                # machine busy: check current resource users
                m_busy = False
                if self.machine_resources and 0 <= int(midx) < len(self.machine_resources):
                    res = self.machine_resources[int(midx)]
                    m_busy = len(res.users) > 0
                m_avail_at = _machine_next_free(midx)

                # find operator candidates and choose a best operator
                op_id = None
                op_busy = None
                op_avail_at = None
                qualified = False
                operator_candidates = []
                
                if self.operators is not None:
                    for op_obj in self.operators.operators_object_list:
                        opid = str(op_obj.operator_id)
                        busy = bool(op_obj.is_busy)
                        # infer next free time from history when possible
                        hist = op_obj.history or []
                        if busy:
                            if hist and hist[-1].get('end_time', None) is not None:
                                next_free = float(hist[-1]['end_time'])
                            else:
                                next_free = now_t
                        else:
                            next_free = now_t

                        # attempt to resolve workcenter for this machine
                        wc_idx_for_m = None
                        wc_for_machine = self.workcenters_meta.workcenter_for_machine
                        if callable(wc_for_machine):
                            wc_idx_for_m = int(wc_for_machine(mname))
                        else:
                            registry = self.workcenters_meta.machine_registry or {}
                            if mname in registry:
                                wc_idx_for_m = int(registry[mname].get('workcenter', -1))

                        # [PHASE9-FIX] Task 9.2: Qualification check with TOCTOU awareness
                        # CRITICAL: For eligibility, only consider operators that are:
                        # 1. Qualified for this specific machine (not just workcenter)
                        # 2. Can perform this operation type
                        # 3. Currently FREE (not busy)
                        # This prevents selecting machines where no operators are available
                        is_qualified = False
                        # First check: operator must be qualified for this specific machine
                        if mname in op_obj.qualified_machines:
                            # Second check: operator can perform this operation at this workcenter
                            if wc_idx_for_m is not None:
                                can_do_operation = bool(op_obj.can_do_job(op_idx_local, wc_idx_for_m))
                                # Third check: operator must be FREE for eligibility
                                # (Don't mark machine as eligible if all qualified operators are busy)
                                if can_do_operation and not busy:
                                    is_qualified = True
                            else:
                                # No workcenter info, assume qualified if machine matches and free
                                if not busy:
                                    is_qualified = True

                        operator_candidates.append({
                            'operator_id': opid,
                            'qualified': bool(is_qualified),
                            'busy': busy,
                            'next_free': float(next_free),
                            'load': int(len(op_obj.history or [])),
                        })

                        if is_qualified:
                            qualified = True
                            # prefer free operator, otherwise the one with earliest next_free
                            if not busy and op_id is None:
                                op_id = opid
                                op_busy = False
                                op_avail_at = float(next_free)
                                # free operator is ideal; stop searching
                                break
                            else:
                                # busy operator; pick earliest next_free
                                if op_id is None or float(next_free) < float(op_avail_at or float('inf')):
                                    op_id = opid
                                    op_busy = busy
                                    op_avail_at = float(next_free)
                else:
                    qualified = False

                # derive a human-readable per-machine reason for DecisionTrace
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
                            earliest = min(next_times)
                            reason_str = f"busy (avail@ t={float(earliest):.2f})"
                        else:
                            reason_str = 'busy'

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
            chosen_m_idx = int(chosen_idx)
            chosen_mid = chosen_m_idx
            # if chosen_idx indexes into allowed_list (legacy), map
            #if allowed_list and chosen_m_idx < len(allowed_list) and int(allowed_list[chosen_midx]) != chosen_midx:
            #    chosen_mid = int(allowed_list[chosen_midx])
            #else:
            #    chosen_mid = chosen_m_idx
            
            chosen_m_name = mlist[chosen_mid] if mlist and 0 <= int(chosen_mid) < len(mlist) else f"M{int(chosen_mid)}"
            chosen_op_label = str(op_id_for_record) if 'op_id_for_record' in locals() else None

            # policy reason: not yet resolved (decision expected from policy)
            policy_reason = 'pending'

            decision_trace = DecisionTrace(
                chosen_machine=str(chosen_m_name),
                chosen_operator=chosen_op_label,
                policy_reason=str(policy_reason),
                at_time=float(now_t),
                eligibilities=elig_entries,
            )
            
            # Cache this decision info for reward computation
            if not hasattr(self, '_last_decision_info') or self._last_decision_info is None:
                self._last_decision_info = []
            
            job_obj = job
            avail_row = decision_item.get('avail_row') if isinstance(decision_item, dict) else None
            chosen_action_val = int(chosen_idx) if chosen_idx is not None else -1
            job_completed = job_obj.finished
            wait_time = job_obj.wait_time
            max_wait = self.max_wait_time
            wait_time_norm = wait_time / max_wait if max_wait > 0 else 0.0
            self._last_decision_info.append({
                'job_id': job_obj.id,
                'chosen_action': chosen_action_val,
                'avail_row': avail_row,
                'chosen_machine_name': chosen_m_name,
                'job_completed': job_completed,
                'wait_time_norm': wait_time_norm
            })
            
            # [ADAPTIVE-FIX-3] Track machine/operator choices for entropy calculation
            if hasattr(self, 'recent_machine_choices'):
                self.recent_machine_choices.append(chosen_m_name)
            
            # Operator tracking with validation
            if hasattr(self, 'recent_operator_choices'):
                # Try multiple sources for operator label
                op_label = None
                if 'chosen_op_label' in locals():
                    op_label = chosen_op_label
                elif isinstance(avail_row, dict) and 'operator' in avail_row:
                    op_label = avail_row['operator']
                elif hasattr(self, 'last_operator_assigned'):
                    op_label = self.last_operator_assigned
                
                # Only append if valid label found
                if op_label is not None and op_label != '':
                    self.recent_operator_choices.append(op_label)
            # [C1] Acquire machine resource - fail-fast if indexing fails
            # [PHASE5-FIX] Task 5.1: Validate machine_resources exists and is not empty
            if not hasattr(self, 'machine_resources') or not self.machine_resources:
                raise RuntimeError(
                    f"[PHASE5] No machines available. machine_resources is empty or None. "
                    f"job={job.id}, chosen_mid={chosen_mid}, t={float(self.env.now):.2f}"
                )
            
            # [PHASE5-FIX] Validate chosen_mid is within valid bounds
            if not (0 <= int(chosen_mid) < len(self.machine_resources)):
                raise ValueError(
                    f"[PHASE5] Machine index out of bounds: chosen_mid={chosen_mid}, "
                    f"valid range=[0, {len(self.machine_resources)}). "
                    f"job={job.id}, op_idx={job.current_op_idx}, t={float(self.env.now):.2f}"
                )
            
            # Use chosen_mid (resolved machine index) when selecting machine resource
            mr = self.machine_resources[int(chosen_mid)]
            if mr is None:
                raise RuntimeError(
                    f"[PHASE5] Machine resource at index {chosen_mid} is None. "
                    f"job={job.id}, t={float(self.env.now):.2f}"
                )
            
            # [PHASE1-FIX] Validate duration is positive and finite before SimPy timeout
            if dur <= 0 or not np.isfinite(dur):
                raise ValueError(
                    f"[PHASE1] Invalid duration {dur} for job={job.id}, machine={chosen_mid} at t={float(self.env.now):.2f}. "
                    f"Duration must be positive and finite. Check decision_item['per_machine_durations'], "
                    f"per_wc={per_wc}, base_dur={base_dur}"
                )

            # pick operator group index based on eligible operator groups for the
            # chosen machine's workcenter. If multiple eligible groups exist,
            # select one using the environment RNG so operator assignment is
            # non-deterministic but reproducible when env._py_rng is seeded.
            # determine the workcenter index for the chosen machine (if possible)
            mlist = getattr(getattr(self, 'workcenters_meta', None), 'machine_list', []) or []
            machine_name = mlist[int(chosen_mid)] if mlist and 0 <= int(chosen_mid) < len(mlist) else f"M{int(chosen_mid)}"
            wc_for_machine = getattr(self.workcenters_meta, 'workcenter_for_machine', None)
            if callable(wc_for_machine):
                wc_idx = int(wc_for_machine(machine_name))
            else:
                # fallback: treat chosen_idx as workcenter id
                wc_idx = int(chosen_idx)

            # B: WorkCenter-based eligibility removed - use machine-level operator selection only
            # Machine-based operator selection (seeded random)
            available_operator = None
            if self.operators is not None:
                # Try machine-specific operator first (seeded random)
                available_operator = self.operators.find_free_operator_for_machine_seeded_random(
                    op_idx_local, machine_name, self._np_rng
                )
                
                # Fallback: try workcenter-level lookup if machine-level returns None
                if available_operator is None:
                    available_operator = self.operators.find_free_operator_seeded_random(
                        op_idx_local, wc_idx, self._np_rng
                    )
                
                # If no operator available for this machine, wait for new decision point
                # Agent will re-evaluate ALL machines and select best available pair
                if available_operator is None:
                    LOG.info(
                        "[OPERATOR_UNAVAILABLE] No free operator for job=%s op_idx=%s machine=%s wc=%s at t=%.2f. "
                        "Waiting for new decision point to re-evaluate all machine-operator pairs...",
                        job.id, op_idx_local, machine_name, wc_idx, float(self.env.now)
                    )
                    # Wait briefly - this will trigger new decision point where agent can choose different machine
                    yield self.env.timeout(0.5)
                    # After timeout, loop back to wait for new agent decision
                    # The job will re-enter decision queue and agent will see updated eligibility
                    continue
            else:
                available_operator = None
            
            # Determine policy reason based on eligibilities
            chosen_entry = None
            for e in elig_entries:
                if str(e.machine_id) == str(chosen_m_name):
                    chosen_entry = e
                    break
            
            if available_operator is None:
                if chosen_entry is None:
                    policy_reason = 'no qualified'
                else:
                    policy_reason = chosen_entry.reason if chosen_entry.reason is not None else 'no qualified'
            else:
                policy_reason = 'selected'
                # C7 FIX: Log operator selection for reproducibility verification
                LOG.debug(
                    "[C7] Operator selected (seeded random): job=%s, op=%s, "
                    "operator_id=%s, machine=%s, wc=%s, t=%.2f",
                    job.id, op_idx_local, available_operator.operator_id,
                    machine_name, wc_idx, float(self.env.now)
                )
                # reflect selected operator in the chosen_entry
                if chosen_entry is not None:
                    chosen_entry.operator_id = str(available_operator.operator_id)
                    chosen_entry.operator_busy = False
                    chosen_entry.operator_available_at = float(self.env.now)

            # Update decision_trace with the resolved policy reason
            if 'decision_trace' in locals() and decision_trace is not None:
                decision_trace.policy_reason = str(policy_reason)
            
            if available_operator is not None and available_operator.resource is not None:
                logging.getLogger(__name__).debug("Waiting -> starting job=%s on machine=%s by operator=%s time=%s", 
                                                 job.id, int(chosen_mid), available_operator.operator_id, float(self.env.now))
                with available_operator.resource.request() as opres_req, mr.request() as mc_req:

                    yield opres_req; yield mc_req
                    # We now hold the operator and machine resources.
                    # Calculate ACTUAL queue wait time: time from operation ready to resource allocated
                    operation_ready = getattr(job, 'operation_ready_time', job.arrival_time)
                    wait_dur = self.env.now - operation_ready
                    if wait_dur > 0:
                        job.wait_time += wait_dur
                        self.total_wait_time += wait_dur
                        # Track per-job wait time
                        job_id = int(getattr(job, 'id', getattr(job, 'job_id', -1)))
                        if job_id >= 0:
                            self.wait_time_dict[job_id] = self.wait_time_dict.get(job_id, 0.0) + wait_dur
                    op_start = float(self.env.now)
                    job.remaining_time = dur
                    # assign and mark busy via Operator.assign_job()
                    available_operator.assign_job(job.id, wc_idx, start_time=op_start)
                    yield self.env.timeout(dur)
                    op_end = float(self.env.now)
                    
                    # Persist gantt record using concrete operator id
                    op_id_for_record = str(available_operator.operator_id)
                    LOG.info("[GANTT-APPEND] concrete branch -> op_id_for_record=%r type=%s", op_id_for_record, type(op_id_for_record))
                    
                    # Build gantt record
                    ep_stamp = getattr(self, 'current_episode', None)
                    rec_dict = {
                        'start': op_start, 
                        'end': op_end, 
                        'op_idx': int(op_type) if op_type is not None else job.current_op_idx, 
                        'wc_idx': int(chosen_mid), 
                        'job_id': int(job.id), 
                        'op_grp': op_id_for_record, 
                        'arrival': float(job.arrival_time), 
                        'duration': float(dur)
                    }
                    if ep_stamp is not None:
                        rec_dict['episode'] = int(ep_stamp)
                    if decision_trace is not None:
                        rec_dict['decision_trace'] = asdict(decision_trace)
                    
                    self.gantt_records.append(rec_dict)
                    available_operator.release(end_time=op_end)
            else:
                # No concrete Operator object available - operators are REQUIRED
                raise RuntimeError(
                    f"No qualified operator available for job={job.id} "
                    f"on machine={chosen_m_name} at t={float(self.env.now):.2f}. "
                    f"Cannot proceed without operator assignment."
                )

            job.current_op_idx += 1
            job.remaining_time = 0.0
            # Update operation_ready_time: next operation is ready NOW (previous op just finished)
            job.operation_ready_time = float(self.env.now)
            if job.current_op_idx >= len(job.operations):
                # mark completion and only increment counters once
                now_t = float(self.env.now)
                if job.mark_completed(now_t):
                    self._completed_now_cache += 1
                    # Capacity management: when a job finishes, free an active slot
                    if job in self.active_jobs:
                        self.active_jobs.remove(job)
                    if job in self.active_agents:
                        self.active_agents.remove(job)
                    # Completed status observation logu
                    obs = self._build_agent_obs(job)
                    allowed_machine_indices = []
                    idx = job.current_op_idx - 1
                    if len(job.operations) > 0 and idx >= 0 and idx < len(job.operations):
                        op_tuple = job.operations[idx]
                        if isinstance(op_tuple, (list, tuple)) and len(op_tuple) > 1:
                            allowed_machine_indices = op_tuple[1]
                    avail_actions = self._build_avail_actions()[self.jobs.index(job)] if hasattr(self, 'jobs') and job in self.jobs else []
                    action_idx = -1
                    self._write_observation_log(
                        decision_time=now_t,
                        job_id=job.id,
                        allowed_machine_indices=allowed_machine_indices,
                        avail_actions=avail_actions,
                        action_idx=action_idx,
                        obs=obs,
                        job_status="Completed"
                    )
                
                # If there are pending jobs, start pending jobs until capacity is reached
                if getattr(self, 'pending_jobs', None) and len(self.pending_jobs) > 0:
                    max_active = int(getattr(self, 'max_active_agents', 0))
                    if max_active <= 0 or len(self.active_agents) < max_active:
                        next_job = self.pending_jobs.pop(0)
                        self.active_agents.append(next_job)
                        self.active_jobs.append(next_job)
                        next_job.is_active = True
                        self.env.process(self._job_process(next_job))
                        LOG.info("[Env] Pending job %s activated at t=%.4f", next_job.id, float(self.env.now))

        if self.env.now >= self.episode_limit:
            self.done = True
            if not getattr(self.decisions_ready, 'triggered', False):
                self.decisions_ready.succeed()
    # ======================================================
    # Helper functions for decision logging
    # ======================================================

    def _colorize(self, value, color_name=None, *args, **kwargs):
        """Universal safe version: ignore colors and return raw clean value."""
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value)
            except Exception:
                return str(value)
        return value


    def _write_observation_log(
        self, decision_time, job_id, allowed_machine_indices,
        avail_actions, action_idx, obs, job_status
    ):
        """Append to decision_observation_metrics.csv (observation + job_status)."""
        # Kesin koruma: Eğer iş tamamlandıysa logu atla
        if job_status == "Completed":
            return
        log_dir = os.path.join("my_data_and_graph", "historydata")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "decision_observation_metrics.csv")

        header = [
            "decision_time",
            "job_id",
            "allowed_machine_indices",
            "avail_actions",
            "action_idx",
            "obs_current_op_type",
            "obs_total_operations",
            "obs_remaining_operations",
            "obs_wait_time",
            "obs_theoretical_machine_count",
            "obs_free_machine_count",
            "obs_n_jobs_active",
            "finished_flag",
            "job_status",
        ]

        # Maskı her zaman string olarak yaz ve free_machine_count'u masktan hesapla
        if isinstance(avail_actions, (np.ndarray, list)):
            mask_list = list(avail_actions)
        else:
            try:
                import ast
                mask_list = ast.literal_eval(str(avail_actions))
            except Exception:
                mask_list = [int(x) for x in str(avail_actions).replace('[','').replace(']','').replace(',',' ').split() if x.isdigit()]

        mask_str = str(mask_list)
        free_machine_count = float(sum([1 for x in mask_list if x == 1]))

        # Finished flag: 1 (bitmiş iş, action_idx=-1 ve job_status=Completed), 0 (diğer tüm durumlar)
        finished_flag = 1 if (action_idx == -1 and job_status == "Completed") else 0
        def to_native(val):
            # Convert numpy types to native Python types
            if hasattr(val, 'item'):
                return val.item()
            return val

        row = [
            to_native(self._colorize(decision_time, "red")),
            to_native(self._colorize(job_id, "blue")),
            to_native(self._colorize(allowed_machine_indices, "green")),
            to_native(self._colorize(mask_str, "purple")),
            to_native(self._colorize(action_idx, "orange")),
            to_native(self._colorize(float(obs[0]) + 1, "cyan")),
            to_native(self._colorize(float(obs[1]), "cyan")),
            to_native(self._colorize(float(obs[2]), "cyan")),
            to_native(self._colorize(float(obs[3]), "cyan")),
            to_native(self._colorize(float(obs[4]), "cyan")),
            to_native(self._colorize(free_machine_count, "cyan")),
            to_native(self._colorize(float(obs[6]), "cyan")),
            to_native(self._colorize(finished_flag, "yellow")),
            to_native(job_status),
        ]

        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

    def _write_state_log(
        self, decision_time, job_id, allowed_machine_indices,
        avail_actions, action_idx, obs_current_op_type, state
    ):
        """Append to decision_state_metrics.csv (10-element global state)."""
        log_dir = os.path.join("my_data_and_graph", "historydata")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "decision_state_metrics.csv")

        header = [
            "decision_time",
            "job_id",
            "obs_current_op_type",
            "allowed_machine_indices",
            "avail_actions",
            "action_idx",
            "state_n_jobs_arrived",
            "state_n_jobs_processing",
            "state_n_jobs_waiting",
            "state_n_ops_arrived",
            "state_n_ops_processing",
            "state_n_ops_waiting",
            "state_avg_machine_util",
            "state_avg_operator_util",
            "state_global_avg_wait",
            "state_episode_time_fraction",
        ]

        # Decision time operator utilization (global)
        busy_count = 0
        total_count = 0
        if hasattr(self, 'operators') and hasattr(self.operators, 'operators_object_list'):
            for op in self.operators.operators_object_list:
                total_count += 1
                if op.is_busy:
                    busy_count += 1
        self.decision_operator_util = busy_count / total_count if total_count > 0 else 0.0

        row = [
            self._colorize(decision_time, "red"),
            self._colorize(job_id, "blue"),
            self._colorize(obs_current_op_type, "cyan"),
            self._colorize(allowed_machine_indices, "green"),
            self._colorize(avail_actions, "purple"),
            self._colorize(action_idx, "orange"),
            self._colorize(float(state[0]), "yellow"),
            self._colorize(float(state[1]), "yellow"),
            self._colorize(float(state[2]), "yellow"),
            self._colorize(float(state[3]), "yellow"),
            self._colorize(float(state[4]), "yellow"),
            self._colorize(float(state[5]), "yellow"),
            self._colorize(float(state[6]), "yellow"),
            self._colorize(float(state[7]), "yellow"),
            self._colorize(float(state[8]), "yellow"),
            self._colorize(float(state[9]), "yellow"),
        ]

        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

    # ======================================================
    # END HELPER FUNCTIONS
    # ======================================================

    def _build_global_state(self):
        """Wrapper around canonical build_state_vector()."""
        return build_state_vector(self)

    # ---------------- Observation / State / Avail -----------------

    def _build_all_agent_obs(self):
        # Only include jobs that are not finished
        filtered_jobs = [(idx, j) for idx, j in enumerate(self.jobs) if not getattr(j, 'finished', False)]
        return [build_agent_obs(self, j, job_index=idx) for idx, j in filtered_jobs]

    def _build_agent_obs(self, job: JobAgent, allowed_machine_indices=None):
        # Strict delegation to canonical helper. Let exceptions propagate for
        # clearer debugging when the helper is missing or fails.
        # Find job_index for free_machine_count calculation
        job_index = None
        for idx, j in enumerate(self.jobs):
            if j is job:
                job_index = idx
                break
        return build_agent_obs(self, job, job_index=job_index, allowed_machine_indices=allowed_machine_indices)
    # File: MASAEnv.py (inside MASAEnv class, _build_avail_actions method)

    def _build_avail_actions(self):
        """Build availability matrix for all jobs and machines.
        
        STEP E FIX: Pass machine_free to _avail_row_for_job to enable correct
        check ordering (capable → machine_free → operator) and remove redundant
        operator checks in this function.
        
        Returns:
            np.array of shape (n_jobs, n_machines) with 1 for eligible, 0 otherwise
        """
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        n_m = len(mlist) if mlist else int(self.num_wcs)
        avail = np.zeros((len(self.jobs), n_m), dtype=np.int32)

        # Compute per-machine free flags (prefer explicit machine resources,
        # otherwise fall back to workcenter resources). If we can't determine
        # this, default to True for backward compatibility.
        machine_free = [True] * n_m
        if getattr(self, 'machine_resources', None):
            machine_free = [self._resource_free(self.machine_resources[m]) for m in range(n_m)]
        else:
            # use machine_registry -> workcenter -> wc_resources
            registry = getattr(self.workcenters_meta, 'machine_registry', {}) or {}
            mlist_local = list(getattr(self.workcenters_meta, 'machine_list', []) or [])
            tmp = []
            for i, mname in enumerate(mlist_local):
                wc_i = int(registry.get(mname, {}).get('workcenter', 0))
                tmp.append(self._resource_free(self.wc_resources[wc_i]))
            if len(tmp) == n_m:
                machine_free = tmp
            else:
                LOG.warning(
                    "[NO_MACHINE_RESOURCES] Cannot determine machine_free status (got %d, expected %d). "
                    "Marking all as BUSY for safety.", len(tmp), n_m
                )
                machine_free = [False] * n_m  # Conservative: all busy if can't validate

        # STEP E FIX: Pass machine_free to _avail_row_for_job so it can skip
        # expensive operator checks for busy machines. The row returned already
        # incorporates all three checks (capable, machine_free, operator_free),
        # so we just trust it directly without redundant checks.
        for idx, j in enumerate(self.jobs):
            if j.finished:
                continue
            row = self._avail_row_for_job(j, machine_free=machine_free)
            if row is None:
                continue
            
            # -------------------------------------------------------------
            # !!! CRITICAL FIX 4: REMOVE REDUNDANT FILTERING !!!
            # Trust the 'row' calculated by _avail_row_for_job as the single source
            # of truth, which already contains the final filtered mask (capable & free).
            avail[idx, :] = row
            # -------------------------------------------------------------
        
        return avail

    def _avail_row_for_job(self, job: JobAgent, machine_free=None):
        """Check which machines can execute this job's current operation.
        
        STEP E FIX: Correct check order for performance:
        1. capable (machine can do this operation type)
        2. machine_free (machine resource is available)
        3. operator_free (exists free qualified operator for this machine)
        
        Args:
            job: JobAgent to check eligibility for
            machine_free: Optional list[bool] of machine free status. If provided,
                         skips expensive operator checks for busy machines.
        
        Returns:
            np.array of shape (n_machines,) with 1 for eligible, 0 otherwise
        """
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        n_m = len(mlist) if mlist else int(self.num_wcs)
        row = np.zeros((int(n_m),), dtype=np.int32)
        op = job.current_op()
        if op is None:
            return row

        # Resolve operation index (machine-level op index)
        if isinstance(op, (list, tuple)) and len(op) >= 3:
            op_type = op[0]
            op_idx_local = int(op_type) if op_type is not None else int(job.current_op_idx)
        else:
            op_idx_local = int(job.current_op_idx)

        # Prefer authoritative machine_registry -> mark machines that support this op
        registry = getattr(self.workcenters_meta, 'machine_registry', {}) or {}
        mlist_local = list(getattr(self.workcenters_meta, 'machine_list', []) or [])
        for i, mname in enumerate(mlist_local):
            # STEP 1: Check capability (machine can do this operation type)
            caps = registry.get(mname, {}).get('capabilities', [])
            if int(op_idx_local) not in caps:
                continue  # Not capable, skip to next machine
            
            # STEP 2: Check machine free (CHEAP CHECK - do before expensive operator loop)
            if machine_free is not None and not machine_free[i]:
                continue  # Machine busy, skip expensive operator check
            
            # STEP 3: Check operator (EXPENSIVE CHECK - only for free machines)
            wc_idx = registry.get(mname, {}).get('workcenter', None)
            has_free_qualified_operator = False
            
            if self.operators is not None and wc_idx is not None:
                # Check all operators: must be qualified for THIS MACHINE and FREE
                for op_obj in self.operators.operators_object_list:
                    # Operator must be qualified for this specific machine
                    if mname in op_obj.qualified_machines:
                        # Operator must be able to do this operation type
                        if op_obj.can_do_job(op_idx_local, wc_idx):
                            # Operator must be FREE (not busy)
                            if not op_obj.is_busy:
                                has_free_qualified_operator = True
                                break
                
                # Only mark machine as available if free qualified operator exists
                if has_free_qualified_operator:
                    row[i] = 1
            else:
                # No operator system or workcenter info - cannot validate operator eligibility
                LOG.warning(
                    "[NO_OPERATOR_SYSTEM] Cannot validate operator eligibility for machine=%s op_idx=%s. "
                    "Marking machine as UNAVAILABLE for safety.",
                    mname, op_idx_local
                )
                row[i] = 0  # Conservative: unavailable if can't validate

        # All-zero mask is expected when all qualified operators/machines are busy
        # This is a normal system state (resource contention), not an error
        if not row.any():
            LOG.debug(
                "[RESOURCE_CONTENTION] No available machine-operator pairs for job=%s op_idx=%s at t=%.2f. "
                "All qualified resources currently busy. Job will wait and agent receives wait penalty.",
                getattr(job, 'id', '?'), op_idx_local, float(getattr(self, 'env', self).now)
            )

        return row

    def _compute_machine_free_status(self):
        """Return boolean list of machine free status (True = free, False = busy).
        
        D: Helper for 3-step reasoning - computes machine availability at decision time.
        """
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        n_m = len(mlist)
        machine_free = [True] * n_m
        
        if getattr(self, 'machine_resources', None):
            machine_free = [self._resource_free(self.machine_resources[m]) for m in range(n_m)]
        else:
            # Fallback: use machine_registry -> workcenter -> wc_resources
            registry = getattr(self.workcenters_meta, 'machine_registry', {}) or {}
            for i, mname in enumerate(mlist):
                wc_i = int(registry.get(mname, {}).get('workcenter', 0))
                machine_free[i] = self._resource_free(self.wc_resources[wc_i])
        
        return machine_free

    # ---------------- Helpers -----------------
    def _resource_free(self, res):
        return len(res.users) < res.capacity

    def active_jobs_count(self) -> int:
        """Return canonical number of currently active jobs (public helper)."""
        if self.active_jobs is not None:
            return int(len(self.active_jobs))
        # Fallback: compute from jobs list
        return int(sum(1 for j in self.jobs if not j.finished))
    # Note: legacy helpers `_wip`, `_util_machines`, and `_util_ops` have
    # been removed. Consumers should compute utilization/wip directly from
    # `self.machine_resources`, `self.operator_groups`, or `self.jobs`, or
    # use canonical metrics APIs instead.

    def _is_episode_done(self):
        """Check if episode has completed.
        
        C16 FIX: Used to determine when utilization can be computed accurately.
        Episode is done when:
        - Episode time limit is reached, OR
        - Environment marked as done (emergency condition)
        
        Returns:
            bool: True if episode has completed
        """
        # Check if time limit reached
        time_limit_reached = float(self.env.now) >= float(self.episode_limit)
        
        # Check if environment marked as done (emergency only)
        env_done = getattr(self, 'done', False)
        
        return time_limit_reached or env_done

    def _build_state_vector(self):
        """Return the global state vector; used by tests and env_obs helper."""
        from utils.env_obs import build_state_vector  # type: ignore
        state = np.asarray(build_state_vector(self), dtype=np.float32)
        # Phase A(A): Validate state matches expected state_shape
        expected_state_shape = getattr(self, 'state_shape', None)
        if expected_state_shape is not None:
            expected_len = int(expected_state_shape)
            if state.shape[0] != expected_len:
                raise ValueError(
                    f"State vector shape mismatch: got {state.shape[0]}, expected {expected_len}. "
                    f"Ensure build_state_vector() returns exactly state_shape dimensions."
                )
        return state

    def _periodic_summary(self):
        """Periodically log job statistics during simulation."""
        while True:
            yield self.env.timeout(self.summary_interval)
            total = len(self.jobs)
            completed = sum(1 for j in self.jobs if j.finished)
            active = sum(1 for j in self.jobs if j.is_active)
            LOG.info("[Summary] t=%.2f → total=%d | completed=%d | active=%d",
                     float(self.env.now), int(total), int(completed), int(active))

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
        jid = int(self.job_counter)
        job = JobAgent(jid, ops_sequence)
        if set_arrival_zero:
            job.arrival_time = 0.0
        else:
            job.arrival_time = float(self.env.now)
        
        # Initialize operation_ready_time: first operation is ready at arrival
        job.operation_ready_time = job.arrival_time
        
        # Append to master job list (arrival order) and advance counter
        self.jobs.append(job)
        self.total_jobs_cumulative += 1  # Track cumulative jobs for dynamic normalization
        self.total_jobs_arrived += 1  # Track jobs for state vector
        self.total_ops_arrived += len(job.operations)  # Track ops for state vector
        LOG.info("[Env] New job %s arrived at t=%.4f with %s ops", job.id, job.arrival_time, len(job.operations))
        self.job_counter = jid + 1

        # Console-friendly lifecycle debug print
        jname = getattr(job, 'name', None) if hasattr(job, 'name') else f"Job_{job.id}"
        print(f"[Lifecycle] New job added: {jname} | total_jobs={len(self.jobs)}")

        # Capacity enforcement
        capacity = self.max_active_agents or self.num_jobs or self.args.n_agents

        if start_immediately:
            # If we have room, start the job and record it as active. Otherwise
            # queue it in pending_jobs to be started when capacity frees.
            if capacity <= 0 or len(self.active_jobs) < capacity:
                # start now
                self.env.process(self._job_process(job))
                self.active_jobs.append(job)
                job.is_active = True
                # mirror into active_agents for ML-facing semantics
                self.active_agents.append(job)
                
                # Persist lifecycle events to timeline
                hist_dir = self.history_dir if hasattr(self, 'history_dir') else os.path.join('my_data_and_graph', 'historydata')
                os.makedirs(hist_dir, exist_ok=True)
                timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
                
                with open(timeline_path, 'a', encoding='utf-8') as tf:
                    tf.write(f"[t={self.env.now:.2f}] Job {job.id} became active agent -> ActiveAgents: {len(self.active_agents)}\n")
                    completed_count = len([j for j in self.jobs if j.finished])
                    tf.write(f"[t={job.arrival_time:.2f}] New job {job.id} arrived with {len(job.operations)} ops -> Active:{len(self.active_agents)} | Pending:{len(self.pending_jobs)} | Completed:{completed_count}\n")
                    
                    # Lifecycle snapshot
                    active_jobs = [j for j in self.jobs if not j.finished]
                    completed_jobs = [j for j in self.jobs if j.finished]
                    pending_jobs = [j for j in self.jobs if not j.is_active and not j.finished]
                    tf.write(f"[Lifecycle] t={self.env.now:.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={len(self.jobs)}\n")
                    
                    if len(active_jobs) > self.max_active_agents:
                        print(f"[WARN] Max active agents exceeded: {len(active_jobs)} > {self.max_active_agents}")
            else:
                # queue for later start
                self.pending_jobs.append(job)
                LOG.info("[Env] Job %s queued (capacity full)", job.id)
                
                # Persist queued event
                hist_dir = self.history_dir if hasattr(self, 'history_dir') else os.path.join('my_data_and_graph', 'historydata')
                os.makedirs(hist_dir, exist_ok=True)
                timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
                
                with open(timeline_path, 'a', encoding='utf-8') as tf:
                    completed_count = len([j for j in self.jobs if j.finished])
                    tf.write(f"[t={self.env.now:.2f}] Job {job.id} queued (pending) -> Active:{len(self.active_agents)} | Pending:{len(self.pending_jobs)} | Completed:{completed_count}\n")
                    
                    # Lifecycle snapshot for queued event
                    active_jobs = [j for j in self.jobs if not j.finished]
                    completed_jobs = [j for j in self.jobs if j.finished]
                    pending_jobs = [j for j in self.jobs if not j.is_active and not j.finished]
                    tf.write(f"[Lifecycle] t={self.env.now:.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={len(self.jobs)}\n")

        return job

    def get_env_info(self):
        """Return canonical environment dimensions for agents/mixer/buffer.
        
        This is the authoritative source for observation/state/action shapes.
        Runner must call this after environment construction and inject values
        into args before initializing agents/mixer/buffer.
        """
        return {
            "obs_shape": self.obs_dim_agent,
            "state_shape": self.state_dim,
            "n_actions": self.n_actions,
            "n_agents": self.max_jobs
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
        self.job_counter = 0

        n_init = int(getattr(self, 'initial_jobs', 4))
        # [PHASE4-FIX] Validate job generation parameters
        if n_init < 0:
            raise ValueError(f"[PHASE4] initial_jobs must be non-negative, got {n_init}")
        if n_init > 1000:
            raise ValueError(
                f"[PHASE4] initial_jobs={n_init} exceeds safety limit (1000). "
                "Check configuration."
            )
        
        if getattr(self, 'job_generator', None) is None:
            logging.getLogger(__name__).warning("TaskGenerator not attached: skipping initial job creation (initial_jobs=%s)", n_init)
            return

        for _ in range(max(0, n_init)):
            # pick a deterministic number of ops using the injected RNG
            min_init_ops = max(3, int(getattr(self, 'job_min_ops', 2)))
            max_init_ops = int(getattr(self, 'job_max_ops', max(min_init_ops, 5)))
            # [PHASE4-FIX] Validate operation count bounds
            if min_init_ops > max_init_ops:
                raise ValueError(
                    f"[PHASE4] job_min_ops ({min_init_ops}) > job_max_ops ({max_init_ops}). "
                    "Check configuration."
                )
            if max_init_ops > 100:
                raise ValueError(
                    f"[PHASE4] job_max_ops={max_init_ops} exceeds safety limit (100). "
                    "Check configuration."
                )
            num_ops = int(self._py_rng.randint(min_init_ops, max(min_init_ops, max_init_ops) + 1))

            ops = self.job_generator.create_job(num_ops=num_ops)
            # If TaskGenerator fails to produce a job, fall back to a minimal single-op job
            if not ops:
                logging.getLogger(__name__).warning("TaskGenerator failed to generate job; creating minimal single-op fallback")
                ops = [(0, [0], {0: 1.0})]
            # Start initial jobs immediately at t=0
            self.add_job(ops, start_immediately=True, set_arrival_zero=True)
            # Track arrived jobs for global state
            self.total_jobs_arrived += 1
            # self.total_ops_arrived += len(ops)  # Fazla sayımı engellemek için kaldırıldı

    # NOTE: dynamic arrivals and internal job generator loops removed.
    # Dynamic job arrival behavior should be provided by an external
    # TaskGenerator or orchestrator that explicitly calls `env.add_job()`.

    def print_jobs_human_readable(self):
        for job in self.jobs:
            for i, op in enumerate(job.operations):
                op_type, allowed_machine_indices, per_wc = op
                LOG.info("Job %s Op%s -> op_type=%s allowed_machine_indices=%s per_machine=%s", job.id, i, op_type, allowed_machine_indices, per_wc)

    def print_initial_jobs_summary(self):
        """Print a concise initial jobs summary in the legacy format.

        Example:
        Job_0 -> 4 ops: [Op8, Op1, Op9, Op6] | Eligible: {Op8:[0,1,2], ...}
        """
        for job in self.jobs[:int(getattr(self, 'initial_jobs', 4))]:
            ops_desc = []
            eligible_desc = []
            for i, op in enumerate(job.operations):
                op_type, allowed_machine_indices, per_wc = op
                ops_desc.append(f"Op{int(op_type)+1}")
                eligible_desc.append(f"Op{int(op_type)+1}:[{','.join(str(x) for x in allowed_machine_indices)}]")
            LOG.info("Job_%s -> %s ops: [%s] | Eligible: {%s}", job.id, len(job.operations), ', '.join(ops_desc), ', '.join(eligible_desc))

    def _compute_utilization_summary(self):
        """Compute utilization summary from gantt_records.
        
        C16 FIX: Should only be called at episode end for accurate results.
        During episode, utilization is meaningless (jobs still running).

        Returns a dict with keys:
          - avg_machine_utilization: fraction [0,1] averaged across machines and time
          - avg_operator_utilization: fraction [0,1] averaged across operators and time
          - average_makespan: makespan observed in this episode (seconds)
          - average_wait_time: total_wait_time / completed_jobs (seconds)
        """
        # [PHASE6-FIX] Task 6.2: Return None instead of warning if called mid-episode
        if not self._is_episode_done():
            LOG.warning(
                "[PHASE6] _compute_utilization_summary called mid-episode (t=%.2f/%.2f). "
                "Returning None. Utilization can only be computed accurately at episode end.",
                float(self.env.now), float(self.episode_limit)
            )
            # Return None to force callers to handle episode-end-only computation
            return None
        
        try:
            records = getattr(self, 'gantt_records', []) or []
            starts = []
            ends = []
            total_machine_busy = {}
            total_operator_busy = {}
            # debug flag: allow env- or args-driven enable
            try:
                log_util_debug = bool(getattr(self, 'log_util_debug', False)) or bool(getattr(self.args, 'log_util_debug', False))
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Failed to read log_util_debug flag: {e}")
                log_util_debug = False

            # NOTE: One machine can only be active with one operator at a time.
            # UNASSIGNED operators are ignored (no real human involvement).
            for r in records:
                if isinstance(r, dict):
                    s = float(r.get('start', r.get('s', 0.0)))
                    e = float(r.get('end', r.get('e', s)))
                    machine_id = r.get('wc_idx', r.get('wc', None))
                    operator_id = r.get('op_grp', r.get('op_id', None))
                else:
                    # legacy tuple: (start, end, op_idx, wc_idx, job_id, op_grp, ...)
                    s = float(r[0])
                    e = float(r[1])
                    machine_id = r[3] if len(r) > 3 else None
                    operator_id = r[5] if len(r) > 5 else None

                # Skip records with non-positive duration
                dur = max(0.0, float(e) - float(s))
                if dur <= 0.0:
                    continue

                # Skip UNASSIGNED operators entirely (user requested semantics)
                if operator_id is not None and str(operator_id) == 'UNASSIGNED':
                    # still record start/end for makespan calculation but do not credit busy-time
                    starts.append(float(s)); ends.append(float(e))
                    continue

                # compute busy time (strictly for this record's machine/operator pair)
                starts.append(float(s))
                ends.append(float(e))

                # accumulate per-machine busy time
                mid = int(machine_id) if machine_id is not None else None
                if mid is not None:
                    total_machine_busy[mid] = total_machine_busy.get(mid, 0.0) + dur

                # accumulate per-operator busy time (ignore UNASSIGNED)
                if operator_id is not None:
                    opid = str(operator_id)
                    if opid and opid != 'UNASSIGNED':
                        total_operator_busy[opid] = total_operator_busy.get(opid, 0.0) + dur

            # C16 FIX: Episode length from gantt records (actual makespan)
            # Fallback to env.now only if no records exist (edge case)
            if starts and ends:
                # Actual makespan: time from first job start to last job end
                episode_length = float(max(ends)) - float(min(starts))
            else:
                # No gantt records: use current time as fallback
                # This should only happen if no jobs were processed
                episode_length = float(getattr(self.env, 'now', 0.0))
                if episode_length > 0:
                    LOG.warning(
                        "[C16] Computing utilization with no gantt records. "
                        "Using env.now=%.2f as fallback (may be inaccurate).",
                        episode_length
                    )

            # avoid zero-length
            if episode_length <= 0:
                episode_length = 1e-9

            # compute utilizations based on machine-operator pairs
            # configured totals: prefer explicit resources when available
            total_machines = len(getattr(self, 'machine_resources', []) or [])
            if total_machines <= 0:
                total_machines = len(getattr(self.workcenters_meta, 'machine_list', []) or []) or int(getattr(self, 'num_wcs', 1))

            total_operators = len(getattr(self, 'operator_groups', []) or [])
            # if operator_groups not available or zero, infer from gantt records
            if total_operators <= 0:
                seen_ops = set()
                for r in records:
                    op = r.get('op_grp', r.get('op_id')) if isinstance(r, dict) else (r[5] if len(r) > 5 else None)
                    if op is not None and str(op) != 'UNASSIGNED':
                        seen_ops.add(str(op))
                total_operators = max(1, len(seen_ops))

            mm_total = float(sum(total_machine_busy.values()))
            avg_machine_util = mm_total / (episode_length * max(1, int(total_machines)))

            oo_total = float(sum(total_operator_busy.values()))
            avg_operator_util = oo_total / (episode_length * max(1, int(total_operators)))

            avg_makespan = float(episode_length)

            # average wait per completed job
            total_wait = float(getattr(self, 'total_wait_time', 0.0))
            completed = len([j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)])
            avg_wait_time = float(total_wait) / max(1.0, float(completed))

            # Build per-machine utilization for all configured machines
            per_machine_util = {}
            total_machines = len(self.machine_resources) if self.machine_resources else (len(self.workcenters_meta.machine_list) or self.num_wcs)

            for mid in range(max(1, total_machines)):
                busy = float(total_machine_busy.get(mid, 0.0))
                per_machine_util[mid] = float(np.clip(busy / float(episode_length), 0.0, 1.0))

            # Build per-operator utilization for all configured operators
            per_operator_util = {}
            # prefer operator objects if available
            if self.operators is not None and self.operators.operators_object_list is not None:
                op_ids = [str(o.operator_id) for o in self.operators.operators_object_list]
            else:
                # fall back to operator_groups count
                n_ops_conf = len(self.operator_groups)
                op_ids = [str(i) for i in range(max(1, n_ops_conf))]
            # ensure unique
            op_ids = list(dict.fromkeys(op_ids))

            for opid in op_ids:
                busy = float(total_operator_busy.get(opid, 0.0))
                per_operator_util[opid] = float(np.clip(busy / float(episode_length), 0.0, 1.0))

            # Debug reporting: optionally write a short summary to logfile/stdout
            if log_util_debug:
                dbg_lines = []
                # makespan summary
                makespan_line = f"[DEBUG UTIL] Computed makespan={float(episode_length):.3f}s using {len(records)} records"
                dbg_lines.append(makespan_line)
                LOG.debug(makespan_line)

                for mid in sorted(per_machine_util.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else str(x)):
                    busy = float(total_machine_busy.get(mid, 0.0))
                    util = float(per_machine_util.get(mid, 0.0))
                    mid_str = str(mid)
                    line = f"[DEBUG UTIL] Machine {mid_str} busy {busy:.2f}s of {episode_length:.2f}s -> {util:.3f}"
                    dbg_lines.append(line)
                    LOG.debug(line)

                for opid in sorted(per_operator_util.keys(), key=lambda x: str(x)):
                    busy = float(total_operator_busy.get(opid, 0.0))
                    util = float(per_operator_util.get(opid, 0.0))
                    oid_str = str(opid)
                    line = f"[DEBUG UTIL] Operator {oid_str} busy {busy:.2f}s of {episode_length:.2f}s -> {util:.3f}"
                    dbg_lines.append(line)
                    LOG.debug(line)

                # persist to history_dir if available
                hist = self.history_dir if hasattr(self, 'history_dir') else os.path.join('my_data_and_graph', 'historydata')
                os.makedirs(hist, exist_ok=True)
                dbg_path = os.path.join(hist, 'util_debug.log')
                with open(dbg_path, 'a', encoding='utf-8') as df:
                    df.write('\n'.join(dbg_lines) + "\n")

            return {
                'avg_machine_utilization': avg_machine_util,
                'avg_operator_utilization': avg_operator_util,
                'average_makespan': avg_makespan,
                'average_wait_time': avg_wait_time,
                'per_machine_utilization': per_machine_util,
                'per_operator_utilization': per_operator_util,
            }
        except Exception as e:
            # C1 Rule 2: DO NOT return fake zeros - fail-fast on invalid utilization
            raise RuntimeError(
                f"[C1] Failed to compute utilization summary: {e}. "
                f"Check gantt_records, machine_list, and operator availability."
            ) from e
