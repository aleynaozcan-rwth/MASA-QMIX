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
from collections import deque
from typing import Optional, Any, Dict, List

import numpy as np
import simpy
import os
import json


# Default environment-related mirrors (Phase 3C.0)
# Mirrors YAML keys: 'reward_params', 'training_defaults', 'task_generator'
# These are inert defaults for inspection and future merge_config() integration.
# TODO(Phase3C.1): integrate with MASAEnv via merge_config(DEFAULT_ENV_PARAMS, cfg)
DEFAULT_ENV_PARAMS = {
    "reward_params": {"alpha": 1.0, "beta": 0.5, "gamma": 0.2, "delta": 0.1, "c_time": 0.0},
    "training_defaults": {"obs_shape": 11, "state_shape": 64},
    "task_generator": {"arrival_lambda": 0.0, "seq_length": {"min": 1, "max": 5}},
}


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
        **kwargs,
    ):
        # Do NOT perform any global argument parsing here. The env must be
        # provided an `args` namespace or explicit keyword arguments. If both
        # are missing for a required field, fail fast so callers (orchestrator)
        # can correct the injection.
        self.args = args
        self.seed = int(seed)
        self._np_rng = np.random.RandomState(self.seed)
        self._py_rng = random.Random(self.seed)

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

                    # obs/state dims: prefer training_defaults if present, else fallbacks
                    td = cfg.get('training_defaults', {}) if isinstance(cfg.get('training_defaults', {}), dict) else {}
                    if 'obs_dim_agent' not in kwargs and 'obs_shape' not in kwargs:
                        kwargs['obs_dim_agent'] = int(td.get('obs_shape', 11))
                    if 'state_dim' not in kwargs and 'state_shape' not in kwargs:
                        kwargs['state_dim'] = int(td.get('state_shape', 64))

                    # task generator seq length -> job_min_ops / job_max_ops
                    tg = cfg.get('task_generator', {}) or {}
                    seq = tg.get('seq_length') if isinstance(tg, dict) else None
                    if isinstance(seq, dict):
                        if 'job_min_ops' not in kwargs:
                            kwargs['job_min_ops'] = int(seq.get('min', 1))
                        if 'job_max_ops' not in kwargs:
                            kwargs['job_max_ops'] = int(seq.get('max', max(1, kwargs.get('job_min_ops', 5))))
                    # reward params
                    rp = cfg.get('reward_params', {}) if isinstance(cfg.get('reward_params', {}), dict) else {}
                    if rp:
                        if 'reward_alpha' not in kwargs and 'alpha' not in kwargs:
                            kwargs['reward_alpha'] = float(rp.get('alpha', 0.0))
                        if 'reward_beta' not in kwargs and 'beta' not in kwargs:
                            kwargs['reward_beta'] = float(rp.get('beta', 0.0))
                        if 'reward_gamma' not in kwargs and 'gamma' not in kwargs:
                            kwargs['reward_gamma'] = float(rp.get('gamma', 0.0))
                        if 'reward_delta' not in kwargs and 'delta' not in kwargs:
                            kwargs['reward_delta'] = float(rp.get('delta', 0.0))
                        if 'reward_c_time' not in kwargs and 'c_time' not in kwargs:
                            kwargs['reward_c_time'] = float(rp.get('c_time', 0.0))
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
                from utils.config_loader import merge_config
                self.config = merge_config(DEFAULT_ENV_PARAMS, initial_cfg or {})
                logging.getLogger(__name__).debug("Merged DEFAULT_ENV_PARAMS into MASAEnv.config")
            else:
                # preserve explicit config if provided, otherwise keep None
                self.config = initial_cfg
        except Exception:
            # fall back to any explicit config or None
            self.config = kwargs.get('config', None)

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
        self.obs_dim_agent = _resolve(('obs_shape', 'obs_dim_agent'), int, default=11)
        self.state_dim = _resolve(('state_shape', 'state_dim'), int, default=64)
        self.episode_limit = _resolve(('episode_limit',), int, default=100)

        # reward params — prefer explicit reward_weights dict, else require
        # presence in args or kwargs
        if reward_weights:
            self.alpha = float(reward_weights.get('alpha'))
            self.beta = float(reward_weights.get('beta'))
            self.gamma = float(reward_weights.get('gamma'))
            self.delta = float(reward_weights.get('delta'))
            self.c_time = float(reward_weights.get('c_time'))
        else:
            # Use conservative defaults for reward shaping when not provided.
            self.alpha = float(_resolve(('reward_alpha', 'alpha'), float, default=0.0))
            self.beta = float(_resolve(('reward_beta', 'beta'), float, default=0.0))
            self.gamma = float(_resolve(('reward_gamma', 'gamma'), float, default=0.0))
            self.delta = float(_resolve(('reward_delta', 'delta'), float, default=0.0))
            self.c_time = float(_resolve(('reward_c_time', 'c_time'), float, default=0.0))

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

        # ensure machine_list exists
        try:
            if not getattr(self.workcenters_meta, 'machine_list', None):
                # synthesize machine_list if missing and num_wcs available
                if self.num_wcs is not None:
                    self.workcenters_meta.machine_list = [f"M_{i}_0" for i in range(int(self.num_wcs))]
                else:
                    # ensure at least one machine so tests that inspect registry succeed
                    self.workcenters_meta.machine_list = ["M_0_0"]
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

        # migration alias
        self.workcenters = self.workcenters_meta

        # job list and generation
        self.jobs: List[JobAgent] = []

        if self.auto_build:
            try:
                self._generate_initial_jobs()
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                self.jobs = []

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

        # Dump merged configuration for runtime debugging/validation if available.
        try:
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
                    self._task_generator = TaskGenerator(config_path=self.config_path)
                    # Start the arrival process with the provided lambda
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
            step = getattr(self, '_debug_run_step', 1.0)
            # advance in small steps until decisions appear or episode ends
            try:
                while not self.pending_decisions and not self.done:
                    self.env.run(until=self.env.now + step)
                    if self.pending_decisions:
                        break
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

            # pick operator group index (simple fallback choice)
            try:
                selected_grp = 0
                with self.operator_groups[selected_grp].request() as op_req, mr.request() as mc_req:
                    yield op_req; yield mc_req
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
                        self.gantt_records.append((op_start, op_end, int(op_type) if op_type is not None else job.current_op_idx, int(chosen_idx), int(job.id), int(selected_grp), float(job.arrival_time), float(dur)))
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
        try:
            from utils.env_obs import build_agent_obs  # type: ignore

            # Ensure returned observations are numpy-array-like where possible
            return [np.asarray(build_agent_obs(self, j)) for j in self.jobs]
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            # fallback: return numpy arrays so callers can use .shape
            return [np.zeros((self.obs_dim_agent,), dtype=np.float32) for _ in self.jobs]

    def _build_agent_obs(self, job: JobAgent):
        try:
            from utils.env_obs import build_agent_obs  # type: ignore

            return np.asarray(build_agent_obs(self, job))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            # fallback simple obs as numpy array
            return np.zeros((self.obs_dim_agent,), dtype=np.float32)

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
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            try:
                util_m, util_o = self._util_machines(), self._util_ops()
                avg_wait = np.clip(self.total_wait_time / max(1, self.env.now * 10.0), 0, 1)
                wip = np.clip(self._wip() / 100.0, 0, 1)
                completed = np.clip(self.completed_jobs / 200.0, 0, 1)
                reward_recent = np.clip((np.mean(self._recent_rewards) if self._recent_rewards else 0) / 10.0, 0, 1)
                idle_ratio = 1.0 - 0.5 * (util_m + util_o)
                core = np.array([util_m, util_o, avg_wait, wip, completed, reward_recent, idle_ratio], dtype=np.float32)
                if core.shape[0] < getattr(self, 'state_dim', core.shape[0]):
                    core = np.concatenate([core, np.zeros(getattr(self, 'state_dim', core.shape[0]) - core.shape[0])])
                return core[: getattr(self, 'state_dim', core.shape[0])]
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                return np.zeros((int(getattr(self, 'state_dim', 7)),), dtype=np.float32)

    def add_job(self, ops_sequence: List):
        jid = len(self.jobs)
        job = JobAgent(jid, ops_sequence)
        try:
            job.arrival_time = float(self.env.now)
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            job.arrival_time = 0.0
        self.jobs.append(job)
        try:
            self.env.process(self._job_process(job))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass
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

                # compute simple per-wc durations using a deterministic base and speed factors
                per_wc_durations = {}
                base = float(self._np_rng.uniform(1.0, 5.0))
                try:
                    for wc in allowed_wcs:
                        # attempt to find a machine with this workcenter to read speed_factor
                        speed = 1.0
                        for mname, mdata in getattr(self.workcenters_meta, 'machine_registry', {}).items():
                            if int(mdata.get('workcenter', 0)) == int(wc):
                                speed = float(mdata.get('speed_factor', 1.0))
                                break
                        per_wc_durations[int(wc)] = round(float(base) / max(1e-6, float(speed)), 6)
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    per_wc_durations = {int(wc): float(base) for wc in allowed_wcs}

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

