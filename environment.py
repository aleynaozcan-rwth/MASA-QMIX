# environment.py – MASA-QMIX (Step 9A – True SimPy Integration, Event-Driven)
# ---------------------------------------------------------------------------
# • JobAgent = learning agent
# • Action = WorkCenter selection (karar anında)
# • Operator & machine constraints via simpy.Resource (gerçek concurrency)
# • Observation = 11D  |  State = 64D (bozulmadı)
# • Reward shaping = aynı mantık (decision boundary’lerde hesaplanır)
# • RL zamanı = SimPy zamanı (env.now)
#
# Yeni (9A) API ilavesi (Runner otomatik algılar):
#   - wait_for_decisions()  -> (batch, sim_time)
#   - pop_decision_reward() -> float (şekillendirilmiş ödül)
# batch elemanları: {
#    "job_id", "obs", "avail_row", "allowed_wcs", "base_duration",
#    "resume"(callable), "grp_by_wc"{wc:grp}
# }
#
# Not: Mevcut reset(), get_env_info() ve builder fonksiyonları korunur.

from __future__ import annotations
import logging
import numpy as np
from collections import deque
import random
import simpy
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from utils.workcenter import WorkCenters
except Exception:
    class WorkCenters:
        def __init__(self):
            # minimal fallback for environments without utils.workcenter
            # Keep empty lists/dicts rather than legacy defaults.
            self.workcenters_list = []
            self.machine_registry = {}


# ============================================================
#                         JobAgent
# ============================================================
class JobAgent:
    def __init__(self, job_id: int, operations: List[Tuple[List[int], float]]):
        self.id = int(job_id)
        self.operations = operations
        self.current_op_idx = 0
        self.remaining_time = 0.0
        self.wait_time = 0.0
        self.finished = False
        # record arrival time (defaults to 0.0 for initial jobs)
        self.arrival_time = 0.0

    def current_op(self):
        if self.finished or self.current_op_idx >= len(self.operations):
            return None
        return self.operations[self.current_op_idx]

    def progress_ratio(self):
        return float(self.current_op_idx / max(1, len(self.operations)))


# ============================================================
#                          Env (SimPy)
# ============================================================
class MASAEnv:
    def __init__(
        self,
        num_jobs: int = 10,
        num_operators: int = 4,
        num_wcs: Optional[int] = None,
        episode_limit: int = 200,
        obs_dim_agent: int = 11,
        state_dim: int = 64,
        seed: int = 42,
        reward_weights: Optional[Dict[str, float]] = None,
        config_path: Optional[str] = None,
        strict_mode: bool = True,
    ):
        self.num_jobs = num_jobs
        self.num_ops = num_operators
        # default to None so that a provided config can deterministically set
        # the authoritative number of WorkCenters; fallbacks below ensure a
        # sensible positive value when no config is present.
        self.num_wcs = num_wcs
        self.obs_dim_agent = obs_dim_agent
        self.state_dim = state_dim
        self.episode_limit = episode_limit

        self._np_rng = np.random.RandomState(seed)
        self._py_rng = random.Random(seed)
        # strict_mode: when True, generation will raise on config gaps instead
        # of silently falling back to a default WC. Default True to fail-fast
        # and make configuration issues explicit. Set to False to restore the
        # old tolerant/deterministic fallback behavior.
        self.strict_mode = bool(strict_mode)

        # configure module logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            # basic config only if not already configured by application
            logging.basicConfig(level=logging.INFO)

    # Load optional YAML config to override WorkCenters/machine/operator definitions
        self.config = None
        # If no explicit config_path was provided, look for a default configs/env_config.yaml
        cfg_path = config_path
        if cfg_path is None:
            default_cfg = Path("configs/env_config.yaml")
            if default_cfg.exists():
                cfg_path = str(default_cfg)

        if cfg_path:
            try:
                from utils.config_loader import load_config
                self.config = load_config(cfg_path)
                self.config_path = cfg_path
            except Exception as e:
                print(f"[WARN] Could not load config {cfg_path}: {e}")
                # preserve original config_path arg if provided
                self.config_path = config_path
        else:
            # no config found / specified
            self.config_path = config_path

        # Initialize WorkCenters (will be overridden by config if provided)
        # runtime API: `workcenters_meta` holds the WorkCenters/WorkCenter registry
        # (keeps separate from `self.workcenters` which are simpy.Resource objects)
        self.workcenters_meta = WorkCenters()

        # If a config is provided, construct a simplified machine registry
        # that is compatible with existing code (keys like 'M_<wc>_0').
        if self.config is not None:
            try:
                machines_cfg = self.config.get('machines', {})
                # Determine workcenter ordering: prefer explicit work_centers section if present
                wc_cfg = self.config.get('work_centers', {})
                if wc_cfg:
                    wc_names = list(wc_cfg.keys())
                else:
                    # derive unique wc names from machine definitions preserving insertion order
                    seen = {}
                    for mname, mconf in machines_cfg.items():
                        wcn = mconf.get('wc')
                        if wcn and wcn not in seen:
                            seen[wcn] = True
                    wc_names = list(seen.keys())

                wc_name_to_idx = {name: idx for idx, name in enumerate(wc_names)}

                # Build machine_registry keyed by original machine names and group by wc
                machine_registry = {}
                wc_to_machines = {idx: [] for idx in range(len(wc_names))}
                for mname, mconf in machines_cfg.items():
                    # get declared wc name, fall back to machine name index if missing
                    wcn = mconf.get('wc')
                    if wcn is None:
                        # place into a new wc bucket per-machine
                        wci = len(wc_name_to_idx)
                        wc_name_to_idx[mname] = wci
                        wc_names.append(mname)
                        wc_to_machines[wci] = []
                    wci = wc_name_to_idx.get(wcn, wc_name_to_idx.get(mname))

                    caps = mconf.get('capable_ops', [])
                    caps_idx = []
                    for c in caps:
                        try:
                            if isinstance(c, str) and c.lower().startswith('op'):
                                caps_idx.append(int(c[2:]) - 1)
                            else:
                                caps_idx.append(int(c))
                        except Exception:
                            pass

                    machine_registry[mname] = {
                        'workcenter': int(wci),
                        'capabilities': caps_idx,
                        'speed_factor': float(mconf.get('speed_factor', 1.0)),
                    }
                    wc_to_machines[int(wci)].append(mname)

                # build eligible operator groups per wc index from operators config
                eligible = {}
                ops_cfg = self.config.get('operators', [])
                for idx in range(len(wc_names)):
                    eligible[idx] = []
                    for op_idx, opconf in enumerate(ops_cfg):
                        q = opconf.get('qualified_machines', [])
                        # if any machine in this wc is listed as qualified for the operator,
                        # mark that operator as eligible for the whole WorkCenter
                        for mname in wc_to_machines.get(idx, []):
                            if mname in q:
                                eligible[idx].append(op_idx)
                                break

                # override WorkCenters registries
                self.workcenters_meta.machine_registry = machine_registry
                # maintain a stable machine list order for global machine-level actions
                try:
                    self.workcenters_meta.machine_list = list(machine_registry.keys())
                    self.workcenters_meta.machine_index = {m: i for i, m in enumerate(self.workcenters_meta.machine_list)}
                except Exception:
                    self.workcenters_meta.machine_list = list(machine_registry.keys())
                    self.workcenters_meta.machine_index = {m: i for i, m in enumerate(self.workcenters_meta.machine_list)}
                # populate workcenters_list from the grouped wc_to_machines
                try:
                    from utils.workcenter import WorkCenter
                    wc_list = []
                    for wc_idx in range(len(wc_names)):
                        # aggregate capabilities for the WorkCenter from its machines
                        machine_names = wc_to_machines.get(wc_idx, [])
                        caps_set = set()
                        for mname in machine_names:
                            caps_set.update(machine_registry.get(mname, {}).get('capabilities', []))
                        caps = list(sorted(caps_set))
                        wc_obj = WorkCenter(wc_idx, caps)
                        # replace default single-machine mapping with the actual machines
                        wc_obj.machines = {}
                        for mi, mname in enumerate(machine_names):
                            wc_obj.machines[mname] = machine_registry.get(mname, {}).copy()
                        wc_list.append(wc_obj)
                    self.workcenters_meta.workcenters_list = wc_list
                except Exception:
                    pass

                # set both modern and legacy attributes for compatibility
                try:
                    self.workcenters_meta.eligible_operator_groups_by_wc = eligible
                except Exception:
                    pass
                # update sizes
                # num_wcs should reflect the number of configured WorkCenters
                # (wc_names) rather than the number of machines. Use wc_names
                # length as the authoritative source.
                try:
                    self.num_wcs = max(1, len(wc_names))
                except Exception:
                    self.num_wcs = max(1, len(machines_cfg))
                self.num_ops = max(1, len(ops_cfg))
                try:
                    self.logger.info("Applied config %s: num_wcs=%d num_ops=%d machines=%d", self.config_path, int(self.num_wcs), int(self.num_ops), len(machines_cfg))
                except Exception:
                    pass
            except Exception as e:
                print(f"[WARN] Could not apply config to WorkCenters: {e}")

        # --- SimPy environment & resources ---
        self.env = simpy.Environment()
        # simpy.Resource list for runtime scheduling (distinct from the WorkCenters metadata)
        # If num_wcs is still None (no config provided and no explicit arg),
        # fall back to 1 WorkCenter to keep the environment runnable.
        if self.num_wcs is None or int(self.num_wcs) <= 0:
            self.num_wcs = 1
        # Create per-machine resources where possible. Fall back to per-WC resources
        # for legacy configs that don't expose a machine_list.
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        if mlist:
            # create a SimPy Resource per machine (capacity=1)
            self.machine_resources = [simpy.Resource(self.env, capacity=1) for _ in range(len(mlist))]
            # legacy alias kept for backward compatibility
            self.wc_resources = [simpy.Resource(self.env, capacity=1) for _ in range(int(self.num_wcs))]
        else:
            # no machine registry available: fall back to per-WC resources as before
            self.machine_resources = []
            self.wc_resources = [simpy.Resource(self.env, capacity=1) for _ in range(int(self.num_wcs))]

        # operator groups remain per-operator-group resources
        self.operator_groups = [simpy.Resource(self.env, capacity=1) for _ in range(self.num_ops)]

        # Migration aliases (phase: provide backwards-compatible names while
        # moving to a clearer API):
        #  - `wc_resources` will refer to the simpy.Resource list (runtime)
        #  - `workcenters` will be exposed as the WorkCenters metadata object
        # expose metadata and keep legacy aliases
        self.workcenters = self.workcenters_meta
        # ensure there is always a machine_list attribute for downstream code
        if not getattr(self.workcenters_meta, 'machine_list', None):
            # synthesize simple machine names per WC when registry absent
            try:
                self.workcenters_meta.machine_list = [f"M_{i}_0" for i in range(int(self.num_wcs))]
                self.workcenters_meta.machine_index = {m: i for i, m in enumerate(self.workcenters_meta.machine_list)}
            except Exception:
                self.workcenters_meta.machine_list = []
                self.workcenters_meta.machine_index = {}

        # Jobs
        self.jobs: List[JobAgent] = []
        self._generate_initial_jobs()

        # If an Operators manager exists, create it and pass the WorkCenters metadata
        try:
            from utils.operator import Operators
            # pass the WorkCenters reference so Operators can consult WC capabilities
            self.operators = Operators(self.workcenters_meta)
        except Exception:
            # fallback: no Operators wrapper available; keep attribute for callers
            self.operators = None

        # If a TaskGenerator and arrival rate are provided in config, start a
        # SimPy arrival process that injects newly generated jobs into the env.
        try:
            if self.config and isinstance(self.config.get('task_generator', {}), dict):
                tg_cfg = self.config.get('task_generator', {})
                lam = float(tg_cfg.get('arrival_lambda', 0.0))
                if lam > 0.0:
                    # create task generator instance bound to the same config
                    try:
                        from utils.task_generator import TaskGenerator
                        self._task_generator = TaskGenerator(config_path=self.config_path)
                    except Exception:
                        self._task_generator = None
                    # spawn arrival loop
                    if self._task_generator is not None:
                        try:
                            self.env.process(self._dynamic_arrival_loop(lam))
                        except Exception:
                            pass
        except Exception:
            pass

        # Bookkeeping
        self.t = 0.0
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self.prev_total_remaining = 0.0
        self._completed_now_cache = 0
        self._recent_rewards = deque(maxlen=5)
        self.done = False

        # default coefficients for shaped reward (maps to alpha, beta, gamma, delta)
        # reward_t = (+alpha * completed_jobs_delta) - beta * avg_wait - gamma * WIP - delta * idle_ops
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 0.2
        self.delta = 0.1
        # default cost-per-time parameter (present in configs/env_config.yaml as c_time)
        self.c_time = 0.0
        # If a YAML config provides reward_params, apply them as authoritative defaults
        try:
            if getattr(self, 'config', None) and isinstance(self.config, dict):
                rp = self.config.get('reward_params', {}) or {}
                if rp:
                    try:
                        self.alpha = float(rp.get('alpha', self.alpha))
                    except Exception:
                        pass
                    try:
                        self.beta = float(rp.get('beta', self.beta))
                    except Exception:
                        pass
                    try:
                        self.gamma = float(rp.get('gamma', self.gamma))
                    except Exception:
                        pass
                    try:
                        self.delta = float(rp.get('delta', self.delta))
                    except Exception:
                        pass
                    try:
                        self.c_time = float(rp.get('c_time', self.c_time))
                    except Exception:
                        pass
        except Exception:
            pass
        # allow programmatic override via reward_weights dict
        if reward_weights:
            try:
                self.alpha = float(reward_weights.get('alpha', self.alpha))
                self.beta = float(reward_weights.get('beta', self.beta))
                self.gamma = float(reward_weights.get('gamma', self.gamma))
                self.delta = float(reward_weights.get('delta', self.delta))
            except Exception:
                pass
        # allow environment variable overrides for quick experiments
        try:
            import os
            if os.environ.get('EXP_ALPHA') is not None:
                self.alpha = float(os.environ.get('EXP_ALPHA'))
            if os.environ.get('EXP_BETA') is not None:
                self.beta = float(os.environ.get('EXP_BETA'))
            if os.environ.get('EXP_GAMMA') is not None:
                self.gamma = float(os.environ.get('EXP_GAMMA'))
            if os.environ.get('EXP_DELTA') is not None:
                self.delta = float(os.environ.get('EXP_DELTA'))
        except Exception:
            pass

    # 9A decision batching
        self.pending_decisions: List[Dict] = []
        self.decisions_ready = simpy.Event(self.env)
    # lightweight gantt/event records:
    # (start, end, operation, wc, job_id, operator_grp, arrival_time, duration)
        self.gantt_records = []

    # --------------------------------------------------------
    # Public API (compatible)
    # --------------------------------------------------------
    def reset(self):
        # Reset SimPy world
        self.env = simpy.Environment()
        # recreate machine and WC resources consistently with initialization
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        if mlist:
            self.machine_resources = [simpy.Resource(self.env, capacity=1) for _ in range(len(mlist))]
            self.wc_resources = [simpy.Resource(self.env, capacity=1) for _ in range(int(self.num_wcs))]
        else:
            self.machine_resources = []
            self.wc_resources = [simpy.Resource(self.env, capacity=1) for _ in range(int(self.num_wcs))]
        self.operator_groups = [simpy.Resource(self.env, capacity=1) for _ in range(self.num_ops)]

        # Maintain migration aliases after reset
        self.wc_resources = self.workcenters
        # ensure metadata remains accessible at env.workcenters
        self.workcenters = getattr(self, 'workcenters_meta', self.workcenters)

        self.t = 0.0
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self.prev_total_remaining = 0.0
        self._recent_rewards.clear()
        self._completed_now_cache = 0
        self.done = False

        self.pending_decisions = []
        self.decisions_ready = simpy.Event(self.env)
        # reset gantt records for the new episode
        self.gantt_records = []

        self._generate_initial_jobs()
        self.prev_total_remaining = self._total_remaining_work()

        # Spawn each job as a SimPy process
        for job in self.jobs:
            self.env.process(self._job_process(job))

        # Return initial obs/info for compatibility
        obs = self._build_all_agent_obs()
        info = {
            "state_vec": self._build_state_vector(),
            "avail_actions": self._build_avail_actions(),
        }
        return obs, info

    # Backwards-compatible step API (kept simple for non-event-driven callers)
    def step(self, action=None):
        """Compatibility step: advance sim by a small time quantum and return (obs, reward, done, info).
        Note: the preferred Runner API is the event-driven `wait_for_decisions()` loop.
        """
        # advance a small time slice
        try:
            self.env.run(until=self.env.now + 1.0)
        except Exception:
            # if env is not initialized or already finished, ignore
            pass
        self.t = getattr(self, 'env', simpy.Environment()).now if hasattr(self, 'env') else 0.0
        reward = 0.0
        try:
            # return shaped reward directly (no extra normalization here)
            reward = float(self.pop_decision_reward())
        except Exception:
            reward = 0.0

        obs_next = self._build_all_agent_obs()
        info = {
            "state_vec": self._build_state_vector(),
            "avail_actions": self._build_avail_actions(),
            "note": "9A event-driven ortamda step() kullanılmamalı; Runner karar döngüsünü kullanır.",
        }
        done = (self.t >= self.episode_limit) or all(j.finished for j in self.jobs)
        self.done = self.done or done
        return obs_next, float(reward), bool(done), info
    def wait_for_decisions(self):
        """Run the sim until at least one decision is pending or the episode is done.
        Uses small run-steps for responsiveness and prints debug info when waiting.
        Returns: (batch, sim_time)
        """
        if self.done:
            return [], self.t

        # if there are no pending decisions, advance the sim in bounded steps
        if not self.pending_decisions:
            step = getattr(self, "_debug_run_step", 1.0)  # seconds per mini-run (responsive default)
            try:
                while not self.pending_decisions and not self.done:
                    if not getattr(self, 'quiet_env', False):
                        print(f"[DEBUG wait_for_decisions] before run: sim.now={self.env.now} pending={len(self.pending_decisions)} decisions_ready.triggered={getattr(self.decisions_ready, 'triggered', False)}")
                    # advance at most `step` simulated time; this avoids very long env.run jumps
                    self.env.run(until=self.env.now + step)
                    # if decisions_ready event has been triggered by a job process, pending_decisions will be filled
                    if self.pending_decisions or (hasattr(self.decisions_ready, "triggered") and self.decisions_ready.triggered):
                        break
            except Exception as e:
                if not getattr(self, 'quiet_env', False):
                    print(f"[DEBUG wait_for_decisions] env.run raised: {e}")
                raise

        self.t = self.env.now
        batch = self.pending_decisions
        self.pending_decisions = []
        # reset event for next decision round
        self.decisions_ready = simpy.Event(self.env)
        if not getattr(self, 'quiet_env', False):
            print(f"[DEBUG wait_for_decisions] returning batch_len={len(batch)} sim.now={self.t}")
        return batch, self.t

    def pop_decision_reward(self) -> float:
        """Son karar sınırından bu yana biriken shaped ödülü döndür."""
        # Compute shaped reward as per specification:
        # reward_t = (+1.0 * completed_jobs_delta)
        #            - 0.5 * avg_wait_time_t
        #            - 0.2 * WIP_t
        #            - 0.1 * idle_operators_t
        try:
            completed = float(self._completed_now_cache)
        except Exception:
            completed = 0.0
        # reset completed cache after reading
        self._completed_now_cache = 0

        # average wait time (total_wait_time / sim_time)
        sim_t = float(self.env.now) if hasattr(self, 'env') else 1.0
        avg_wait = float(self.total_wait_time) / max(1.0, sim_t)

        wip = float(self._wip())

        # idle_operators_t: count of operator_groups that are currently free
        try:
            idle_ops = 0
            for g in getattr(self, 'operator_groups', []):
                if self._resource_free(g):
                    idle_ops += 1
        except Exception:
            idle_ops = 0

        # use configurable coefficients (alpha, beta, gamma, delta)
        try:
            a = float(getattr(self, 'alpha', 1.0))
        except Exception:
            a = 1.0
        try:
            b = float(getattr(self, 'beta', 0.5))
        except Exception:
            b = 0.5
        try:
            c = float(getattr(self, 'gamma', 0.2))
        except Exception:
            c = 0.2
        try:
            d = float(getattr(self, 'delta', 0.1))
        except Exception:
            d = 0.1

        reward = (a * completed) - (b * avg_wait) - (c * wip) - (d * float(idle_ops))
        # keep a recent history for observations
        try:
            self._recent_rewards.append(reward)
        except Exception:
            pass
        return float(reward)

    # ------------------------- SimPy Processes -------------------------
    def _job_process(self, job: JobAgent):
        while not job.finished and self.env.now < self.episode_limit:
            op = job.current_op()
            if op is None:
                job.finished = True
                break

            # support legacy (allowed_wcs, dur), older (op_type, allowed_wcs, base_dur)
            # and the new format: (op_type, allowed_wcs, per_wc_durations_dict)
            op_type = None
            allowed_wcs = []
            per_wc_durations = None
            base_dur = None
            if isinstance(op, (list, tuple)):
                if len(op) == 2:
                    # legacy
                    allowed_wcs, base_dur = op
                    op_type = None
                elif len(op) == 3:
                    op_type = op[0]
                    allowed_wcs = op[1]
                    third = op[2]
                    if isinstance(third, dict):
                        per_wc_durations = third
                    else:
                        # base duration (old-style)
                        base_dur = float(third)
                else:
                    # malformed; fallback
                    try:
                        allowed_wcs, base_dur = op[0], op[1]
                    except Exception:
                        allowed_wcs, base_dur = [], 0.0
            else:
                allowed_wcs, base_dur = [], 0.0
            # Karar talebi oluştur (Runner bu çağrıya karar verecek)
            resume_evt = simpy.Event(self.env)

            # compute eligible machines for this operation (global machine-level actions)
            allowed_machines = []
            allowed_machine_indices = []
            per_machine_durations = {}
            try:
                # determine operation index to lookup processing_time_means
                op_idx_local = int(op_type) if (op_type is not None) else int(getattr(job, 'current_op_idx', 0))
                # iterate over global machine registry to find machines capable of op_idx
                mlist = list(getattr(self.workcenters_meta, 'machine_list', []))
                mindex = getattr(self.workcenters_meta, 'machine_index', {})
                for mname in mlist:
                    try:
                        mreg = self.workcenters_meta.machine_registry.get(mname, {})
                        caps = mreg.get('capabilities', [])
                        if op_idx_local in caps:
                            allowed_machines.append(mname)
                            allowed_machine_indices.append(int(mindex.get(mname, len(allowed_machine_indices))))
                    except Exception:
                        continue
                # build per-machine durations from config processing_time_means when available
                if getattr(self, 'config', None):
                    try:
                        proc_means = self.config.get('processing_time_means', {})
                        op_name = f"Op{op_idx_local+1}"
                        op_means = proc_means.get(op_name, {}) if isinstance(proc_means, dict) else {}
                        for m in allowed_machines:
                            if m in op_means:
                                per_machine_durations[int(mindex.get(m))] = float(op_means.get(m))
                    except Exception:
                        pass
                # fallback: estimate per-machine duration using base and speed_factor
                for m in allowed_machines:
                    mi = int(mindex.get(m, 0))
                    if mi in per_machine_durations:
                        continue
                    try:
                        speed = float(self.workcenters_meta.machine_registry.get(m, {}).get('speed_factor', 1.0))
                        per_machine_durations[mi] = round(float(base_duration_val) / max(1e-6, speed), 6)
                    except Exception:
                        per_machine_durations[mi] = float(base_duration_val)
            except Exception:
                allowed_machines = []
                allowed_machine_indices = []
                per_machine_durations = {}

            def _resume_with(choice, _resume_evt=resume_evt):
                """
                Accept either a machine name (str), a global machine index (int),
                or legacy WorkCenter index. For modern operation we prefer returning
                a machine index (int) into workcenters_meta.machine_list. If mapping
                fails or the chosen machine is not allowed for this operation, we
                succeed with None to let the job process time out briefly.
                """
                try:
                    # machine registry lookup
                    mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
                    mreg = getattr(self.workcenters_meta, 'machine_registry', {}) or {}
                    # string machine name
                    if isinstance(choice, str):
                        if choice not in mreg:
                            _resume_evt.succeed(None)
                            return
                        mi = int(self.workcenters_meta.machine_index.get(choice, -1))
                    else:
                        # numeric choice: could be machine index or legacy WC index
                        try:
                            c = int(choice)
                        except Exception:
                            _resume_evt.succeed(None)
                            return
                        # if it's a valid machine index
                        if mlist and 0 <= c < len(mlist):
                            mi = int(c)
                        else:
                            # legacy: interpret as WorkCenter index -> pick a default machine
                            if c in allowed_wcs:
                                # pick the first machine in that WC that is allowed for this op
                                mi = None
                                for mname, md in (mreg or {}).items():
                                    try:
                                        if int(md.get('workcenter', -1)) == int(c):
                                            mi_candidate = int(self.workcenters_meta.machine_index.get(mname, -1))
                                            mi = mi_candidate
                                            break
                                    except Exception:
                                        continue
                                if mi is None:
                                    _resume_evt.succeed(None)
                                    return
                            else:
                                _resume_evt.succeed(None)
                                return

                    # final validation: machine must be among allowed_machines for this op
                    allowed_machine_indices = decision_item.get('allowed_machine_indices', [])
                    if allowed_machine_indices and (mi not in allowed_machine_indices):
                        _resume_evt.succeed(None)
                        return
                    _resume_evt.succeed(int(mi))
                except Exception:
                    _resume_evt.succeed(None)
                    return

            # compute a numeric base_duration for the Runner: prefer explicit base_dur,
            # otherwise use the mean of per-WC durations when available, else 0.0
            if base_dur is not None:
                try:
                    base_duration_val = float(base_dur)
                except Exception:
                    base_duration_val = 0.0
            elif per_wc_durations:
                try:
                    vals = [float(v) for v in per_wc_durations.values() if v is not None]
                    base_duration_val = sum(vals) / len(vals) if vals else 0.0
                except Exception:
                    base_duration_val = 0.0
            else:
                base_duration_val = 0.0

            decision_item = {
                "job_id": job.id,
                "obs": self._build_agent_obs(job),
                "avail_row": self._avail_row_for_job(job),
                "allowed_wcs": list(allowed_wcs),
                # machine-level action support
                "allowed_machines": list(allowed_machines),
                "allowed_machine_indices": list(allowed_machine_indices),
                "per_machine_durations": dict(per_machine_durations),
                "base_duration": float(base_duration_val),
                "resume": _resume_with,
                # authoritative eligible operator groups per WC
                "eligible_ops_by_wc": {wc: self.workcenters_meta.eligible_operator_groups_by_wc.get(int(wc), []) for wc in allowed_wcs},
            }
            self.pending_decisions.append(decision_item)
            if not self.decisions_ready.triggered:
                self.decisions_ready.succeed()

            chosen_machine_idx = (yield resume_evt)

            if chosen_machine_idx is None:
                # No-op: small timeout to avoid deadlock
                yield self.env.timeout(1e-9)
                continue

            # map machine index to machine name and workcenter
            try:
                mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
                mname = mlist[int(chosen_machine_idx)]
                mreg = self.workcenters_meta.machine_registry.get(mname, {})
                chosen_wc = int(mreg.get('workcenter', -1))
            except Exception:
                mname = None
                chosen_wc = None

            # determine eligible operator groups for the chosen machine's WorkCenter
            eligible_ops = self.workcenters_meta.eligible_operator_groups_by_wc.get(int(chosen_wc), []) if chosen_wc is not None else []
            # pick a specific operator-group index to request: prefer a free one
            selected_grp = None
            try:
                for g in eligible_ops:
                    if self._resource_free(self.operator_groups[g]):
                        selected_grp = int(g)
                        break
            except Exception:
                selected_grp = None
            # fallback: if none free or no eligible_ops defined, choose the first eligible or 0
            if selected_grp is None:
                try:
                    selected_grp = int(eligible_ops[0]) if eligible_ops else 0
                except Exception:
                    selected_grp = 0

            # determine duration for chosen machine: prefer per-machine durations
            try:
                if int(chosen_machine_idx) in decision_item.get('per_machine_durations', {}):
                    dur = float(decision_item.get('per_machine_durations', {}).get(int(chosen_machine_idx), 0.0))
                else:
                    # fall back to per-WC duration mapping if present
                    if per_wc_durations is not None and chosen_wc is not None:
                        dur = float(per_wc_durations.get(int(chosen_wc), 0.0))
                    elif base_dur is not None:
                        speed = self._speed_factor_for_wc(chosen_wc) if chosen_wc is not None else 1.0
                        try:
                            dur = float(base_dur) / max(1e-6, float(speed))
                        except Exception:
                            dur = float(base_dur)
                    else:
                        dur = 0.0
            except Exception:
                dur = 0.0

            start_wait = self.env.now
            # request operator-group and the chosen machine resource
            mr = None
            try:
                mr = self.machine_resources[int(chosen_machine_idx)]
            except Exception:
                # fallback to WC resource if machine_resources not available
                try:
                    mr = self.wc_resources[int(chosen_wc)]
                except Exception:
                    mr = None
            if mr is None:
                # nothing to acquire -> small timeout
                yield self.env.timeout(1e-9)
                continue
            with self.operator_groups[selected_grp].request() as op_req, mr.request() as mc_req:
                yield op_req; yield mc_req
                # bekleme süreleri
                wait_dur = self.env.now - start_wait
                if wait_dur > 0:
                    job.wait_time += wait_dur
                    self.total_wait_time += wait_dur

                # işlem süresi
                # record start time of the operation
                op_start = float(self.env.now)
                job.remaining_time = dur
                # If we have an Operators manager, attempt to assign a specific operator
                assigned_op = None
                try:
                    if getattr(self, 'operators', None) is not None:
                        # prefer machine-level operator selection when possible
                        try:
                            op_idx_local = int(op_type) if (op_type is not None) else int(getattr(job, 'current_op_idx', 0))
                        except Exception:
                            op_idx_local = int(getattr(job, 'current_op_idx', 0))
                        if mname is not None:
                            # machine-level operator lookup
                            assigned_op = self.operators.find_free_operator_for_machine(op_idx_local, mname)
                            if assigned_op is not None:
                                assigned_op.assign_job(job.id, chosen_wc, start_time=self.env.now)
                        else:
                            # fallback to legacy workcenter-level lookup
                            assigned_op = self.operators.find_free_operator(job.id, chosen_wc)
                            if assigned_op is not None:
                                assigned_op.assign_job(job.id, chosen_wc, start_time=self.env.now)
                except Exception:
                    assigned_op = None
                try:
                    yield self.env.timeout(dur)
                finally:
                    # ensure operator release even if timeout interrupted
                    if assigned_op is not None:
                        try:
                            assigned_op.release(end_time=self.env.now)
                        except Exception:
                            pass
                    # record gantt data (start, end, operation_type, wc, job, operator_group,
                    # arrival_time, duration)
                    op_end = float(self.env.now)
                    # prefer explicit op_type if present, otherwise use current op index
                    try:
                        op_tag = int(op_type) if (op_type is not None) else int(job.current_op_idx)
                    except Exception:
                        op_tag = int(getattr(job, 'current_op_idx', 0))
                    try:
                        arrival = float(getattr(job, 'arrival_time', 0.0))
                    except Exception:
                        arrival = 0.0
                    try:
                        dur_val = float(dur)
                    except Exception:
                        dur_val = float(op_end - op_start)
                    self.gantt_records.append((op_start, op_end, int(op_tag), int(chosen_wc), int(job.id), int(selected_grp), arrival, dur_val))
                job.current_op_idx += 1
                job.remaining_time = 0.0
                if job.current_op_idx >= len(job.operations):
                    job.finished = True
                    self.completed_jobs += 1
                    self._completed_now_cache += 1

        # bölüm sonu kontrol
        if all(j.finished for j in self.jobs) or self.env.now >= self.episode_limit:
            self.done = True
            if not self.decisions_ready.triggered:
                self.decisions_ready.succeed()

    # --------------------------------------------------------
    # Observation / State / Avail (korundu)
    # --------------------------------------------------------
    def _build_all_agent_obs(self):
        return [self._build_agent_obs(j) for j in self.jobs]

    def _build_agent_obs(self, job):
        progress = job.progress_ratio()
        wait_norm = np.clip(job.wait_time / 50.0, 0, 1)
        rem_norm = np.clip(job.remaining_time / 20.0, 0, 1)
        util_m, util_o = self._util_machines(), self._util_ops()
        wip_norm = np.clip(self._wip() / 80.0, 0, 1)
        time_norm = np.clip(self.env.now / self.episode_limit, 0, 1)
        reward_norm = np.clip((np.mean(self._recent_rewards) if self._recent_rewards else 0) / 10.0, 0, 1)
        completed_norm = np.clip(self.completed_jobs / 100.0, 0, 1)
        n_ops_norm = np.clip(len(job.operations) / 10.0, 0, 1)
        finished_flag = 1.0 if job.finished else 0.0

        obs = np.array([
            progress, wait_norm, rem_norm, util_m, util_o, wip_norm,
            time_norm, reward_norm, completed_norm, n_ops_norm, finished_flag
        ], dtype=np.float32)
        if obs.shape[0] < self.obs_dim_agent:
            obs = np.concatenate([obs, np.zeros(self.obs_dim_agent - obs.shape[0])])
        return obs[:self.obs_dim_agent]

    def _build_state_vector(self):
        util_m, util_o = self._util_machines(), self._util_ops()
        avg_wait = np.clip(self.total_wait_time / max(1, self.env.now * 10.0), 0, 1)
        wip = np.clip(self._wip() / 100.0, 0, 1)
        completed = np.clip(self.completed_jobs / 200.0, 0, 1)
        reward_recent = np.clip((np.mean(self._recent_rewards) if self._recent_rewards else 0) / 10.0, 0, 1)
        idle_ratio = 1.0 - 0.5 * (util_m + util_o)
        core = np.array([util_m, util_o, avg_wait, wip, completed, reward_recent, idle_ratio], dtype=np.float32)
        if core.shape[0] < self.state_dim:
            core = np.concatenate([core, np.zeros(self.state_dim - core.shape[0])])
        return core[:self.state_dim]

    def _build_avail_actions(self):
        # Build machine-level availability mask if machine registry exists.
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        n_m = len(mlist) if mlist else int(self.num_wcs)
        avail = np.zeros((self.num_jobs, n_m), dtype=np.int32)
        for j_idx, job in enumerate(self.jobs):
            if job.finished:
                continue
            row = self._avail_row_for_job(job)
            # _avail_row_for_job now returns machine-level row when possible
            if row.shape[0] == n_m:
                avail[j_idx, :] = row
            else:
                # backward compatible: row may be per-WC; keep zeros
                pass
        return avail

    def _avail_row_for_job(self, job: JobAgent):
        # If we have machine registry, return per-machine mask, else per-WC mask
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        if mlist:
            n_m = len(mlist)
            row = np.zeros((n_m,), dtype=np.int32)
        else:
            row = np.zeros((self.num_wcs,), dtype=np.int32)
        op = job.current_op()
        if op is None:
            return row
        # support legacy (allowed_wcs, dur) and new (op_type, allowed_wcs, dur)
        if isinstance(op, (list, tuple)) and len(op) == 2:
            allowed_wcs, _ = op
        else:
            try:
                _, allowed_wcs, _ = op
            except Exception:
                allowed_wcs = []
        # If machine registry exists, test per-machine constraints
        mlist = getattr(self.workcenters_meta, 'machine_list', []) or []
        mindex = getattr(self.workcenters_meta, 'machine_index', {}) or {}
        if mlist:
            try:
                op_idx_local = int(op[0]) if (isinstance(op, (list, tuple)) and len(op) >= 1) else int(getattr(job, 'current_op_idx', 0))
            except Exception:
                op_idx_local = int(getattr(job, 'current_op_idx', 0))
            for mi, mname in enumerate(mlist):
                try:
                    mreg = self.workcenters_meta.machine_registry.get(mname, {})
                    caps = mreg.get('capabilities', [])
                    if op_idx_local not in caps:
                        continue
                    # machine free?
                    machine_free = self._resource_free(self.machine_resources[mi]) if getattr(self, 'machine_resources', None) else False
                    # operator free for this machine: check any operator object qualified for this machine and free
                    op_free = False
                    try:
                        # derive workcenter for operator-group lookup
                        wc = int(mreg.get('workcenter', -1))
                        eligible = self.workcenters_meta.eligible_operator_groups_by_wc.get(int(wc), [])
                        for g in eligible:
                            if self._resource_free(self.operator_groups[g]):
                                op_free = True
                                break
                    except Exception:
                        op_free = False
                    if machine_free and op_free:
                        row[mi] = 1
                except Exception:
                    continue
            return row
        else:
            for wc in allowed_wcs:
                eligible = self.workcenters_meta.eligible_operator_groups_by_wc.get(int(wc), [])
                # require both a free machine resource and at least one free operator group
                machine_free = self._resource_free(self.wc_resources[wc])
                op_free = False
                try:
                    for g in eligible:
                        if self._resource_free(self.operator_groups[g]):
                            op_free = True
                            break
                except Exception:
                    op_free = False
                if machine_free and op_free:
                    row[wc] = 1
            return row
        return row

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _generate_initial_jobs(self):
        # Define a fixed set of operation types (e.g., 9 types)
        op_types = list(range(9))
        self.jobs = []
        # Build mapping workcenter -> machine list from the registry so we can
        # decide which WorkCenters actually contain machines that can perform
        # each operation type. This prevents assigning operations to WCs that
        # don't have any capable machine (avoids the "fully flexible" problem).
        wc_to_machines = {}
        try:
            for mname, mdata in getattr(self.workcenters_meta, 'machine_registry', {}).items():
                wc = int(mdata.get('workcenter', 0))
                wc_to_machines.setdefault(wc, []).append(mname)
        except Exception:
            # fallback: distribute machines evenly by index if registry not available
            wc_to_machines = {wc: [] for wc in range(max(1, self.num_wcs))}

        for jid in range(self.num_jobs):
            # number of operations per job: bounded by args-like defaults (2..4)
            try:
                min_ops = max(1, int(getattr(self, 'job_min_ops', 2)))
                max_ops = max(min_ops, int(getattr(self, 'job_max_ops', 4)))
            except Exception:
                min_ops, max_ops = 2, 4
            num_ops = int(self._np_rng.randint(min_ops, min(max_ops, 9) + 1))
            # sample unique operation types per job to avoid repeated identical ops
            chosen_ops = list(self._np_rng.choice(op_types, size=num_ops, replace=False))
            ops = []
            for op_type in chosen_ops:
                op_idx = int(op_type)
                # Determine allowed workcenters by checking if any machine in the
                # workcenter declares the operation in its capabilities.
                allowed_wcs = []
                for wc_idx, machine_names in wc_to_machines.items():
                    for mname in machine_names:
                        try:
                            caps = list(self.workcenters_meta.machine_registry.get(mname, {}).get('capabilities', []))
                        except Exception:
                            caps = []
                        if op_idx in caps:
                            allowed_wcs.append(int(wc_idx))
                            break

                # If no workcenter is found (config gap), fall back to a deterministic
                # workcenter (WC 0) to keep the environment deterministic and avoid
                # introducing stochasticity at generation time. Emit a clear warning
                # so the user can fix the configuration (missing capability mapping).
                if not allowed_wcs:
                    if self.strict_mode:
                        # In strict mode we raise so the user can fix the config
                        raise RuntimeError(f"No eligible WorkCenter found for Op{op_idx+1}; please check 'machines.capable_ops' and 'processing_time_means' in config")
                    # deterministic fallback instead of random choice for backward-compat
                    allowed_wcs = [0]
                    try:
                        self.logger.warning(
                            "No eligible WorkCenter found for %s; defaulting allowed_wcs=%s (check config processing_time_means / capable_ops)",
                            f"Op{op_idx+1}", allowed_wcs,
                        )
                    except Exception:
                        pass

                # base duration in [1.0, 9.0] (base for op_type)
                base = float(self._np_rng.uniform(1.0, 9.0))

                # per-WC duration: estimate based on capable machines' speed_factors
                per_wc_durations = {}
                for wc in allowed_wcs:
                    machine_names = wc_to_machines.get(int(wc), [])
                    candidate_speeds = []
                    for mname in machine_names:
                        try:
                            mreg = self.workcenters_meta.machine_registry.get(mname, {})
                            caps = mreg.get('capabilities', [])
                            if op_idx in caps:
                                candidate_speeds.append(float(mreg.get('speed_factor', 1.0)))
                        except Exception:
                            continue

                    # Prefer explicit per-machine mean processing times from config
                    # when available. The config uses keys like 'Op1'..'Op9'. If a
                    # mean is provided for one or more capable machines in this
                    # WC, use the fastest (minimum) mean as the WC-level estimate.
                    used_mean = None
                    try:
                        if getattr(self, 'config', None):
                            proc_means = self.config.get('processing_time_means', {})
                            op_name = f"Op{op_idx+1}"
                            op_means = proc_means.get(op_name, {}) if isinstance(proc_means, dict) else {}
                            found_means = []
                            for mname in machine_names:
                                if mname in op_means:
                                    try:
                                        found_means.append(float(op_means.get(mname)))
                                    except Exception:
                                        pass
                            if found_means:
                                used_mean = float(min(found_means))
                    except Exception:
                        used_mean = None

                    if used_mean is not None:
                        per_wc_durations[int(wc)] = round(float(used_mean), 6)
                    else:
                        # fallback to speed-factor based estimate (legacy behavior)
                        if candidate_speeds:
                            speed_factor = max(candidate_speeds)
                        else:
                            try:
                                speeds = [float(self.workcenters_meta.machine_registry.get(m, {}).get('speed_factor', 1.0)) for m in machine_names]
                                speed_factor = max(speeds) if speeds else 1.0
                            except Exception:
                                speed_factor = 1.0
                        est = float(base) / max(1e-6, speed_factor)
                        per_wc_durations[int(wc)] = round(est, 6)

                # NOTE: Restrict per_wc_durations output to eligible WorkCenters only.
                # Previously the code could initialize durations for all WCs; this
                # ensures we only include entries for WCs that were identified as
                # `allowed_wcs` for this operation.
                try:
                    per_wc_durations = {int(k): float(v) for k, v in per_wc_durations.items() if int(k) in allowed_wcs}
                except Exception:
                    # defensive: keep as-is if something unexpected happens
                    pass
                # store op as canonical (op_type, allowed_wcs, per_wc_durations)
                ops.append((int(op_type), list(allowed_wcs), per_wc_durations))
            self.jobs.append(JobAgent(jid, ops))

    # Small helper to format operation type indices into YAML-friendly names
    def _op_name(self, op_idx: int) -> str:
        try:
            return f"Op{int(op_idx) + 1}"
        except Exception:
            return str(op_idx)

    def print_jobs_human_readable(self):
        """Print job list with friendly op names and per-WC durations.

        Example line:
          Job 3 Op0 -> Op7 allowed_wcs=[1] per_wc={1: 5.2}
        """
        for job in self.jobs:
            for i, op in enumerate(job.operations):
                try:
                    op_type, allowed_wcs, per_wc = op
                    op_label = self._op_name(op_type) if op_type is not None else 'None'
                    print(f"Job {job.id} Op{i} -> {op_label} allowed_wcs={allowed_wcs} per_wc={per_wc}")
                except Exception:
                    print(f"Job {job.id} Op{i} -> {op}")

    def _total_remaining_work(self):
        total = 0.0
        for j in self.jobs:
            if j.finished:
                continue
            op = j.current_op()
            if op:
                # op may be (allowed_wcs, dur) or (op_type, allowed_wcs, dur)
                try:
                    if len(op) == 2:
                        total += float(op[1])
                    else:
                        third = op[2]
                        # third may be a numeric base duration or a dict of per-wc durations
                        if isinstance(third, dict):
                            vals = [float(v) for v in third.values() if v is not None]
                            total += (sum(vals) / len(vals)) if vals else 0.0
                        else:
                            total += float(third)
                except Exception:
                    try:
                        total += float(op[1])
                    except Exception:
                        pass
        return total

    def _wip(self):
        return sum(1 for j in self.jobs if not j.finished)

    # NOTE: _group_for_wc has been removed. Use
    # `workcenters_meta.eligible_operator_groups_by_wc[wc]` to get the authoritative
    # list of eligible operator-group ids for a given workcenter index.

    def _speed_factor_for_wc(self, wc: int):
        m_id = f"M_{wc}_0"
        reg = getattr(self.workcenters_meta, "machine_registry", {})
        return reg.get(m_id, {}).get("speed_factor", 1.0)

    def _util_machines(self):
        # prefer per-machine resource utilization when available
        if getattr(self, 'machine_resources', None):
            busy = sum(len(m.users) for m in self.machine_resources)
            return busy / max(1, len(self.machine_resources))
        busy = sum(len(wc.users) for wc in self.wc_resources)
        return busy / max(1, len(self.wc_resources))

    def _util_ops(self):
        busy = sum(len(g.users) for g in self.operator_groups)
        return busy / max(1, len(self.operator_groups))

    def _resource_free(self, res):
        return len(res.users) < res.capacity

    # --------------------------------------------------------
    # Dynamic job injection API
    def add_job(self, ops_sequence: list):
        """Add a new JobAgent to the environment at current sim time.

        ops_sequence: list of operation tuples expected by JobAgent/_job_process
                      (either (op_type, allowed_wcs, per_wc_durations) or
                       legacy (allowed_wcs, base_dur)).
        """
        jid = len(self.jobs)
        job = JobAgent(jid, ops_sequence)
        # record arrival time at the moment of insertion into the env
        try:
            job.arrival_time = float(self.env.now)
        except Exception:
            job.arrival_time = 0.0
        self.jobs.append(job)
        # start the job process so it participates in the SimPy world
        try:
            self.env.process(self._job_process(job))
        except Exception:
            # if env is not yet fully initialized, keep the job and it will be
            # started on reset()/init sequence
            pass
        print(f"[Env] Dynamically added Job {jid} with {len(ops_sequence)} op(s) at t={self.env.now}")
        return job

    def _dynamic_arrival_loop(self, arrival_lambda: float):
        """SimPy process: sample inter-arrival times (exponential) and inject
        new jobs generated by TaskGenerator into the env.
        """
        # defensive: if no task generator, just exit
        if getattr(self, '_task_generator', None) is None:
            return
        lam = float(arrival_lambda)
        # scale parameter for numpy exponential is 1/lambda
        scale = 1.0 / max(1e-12, lam)
        while self.env.now < self.episode_limit and not self.done:
            ia = float(self._np_rng.exponential(scale))
            # wait for next arrival
            yield self.env.timeout(ia)
            # generate ops for a single JobAgent
            try:
                ops_objs = self._task_generator.generate_constrained_task(jobagent_id=len(self.jobs))
            except Exception as e:
                print(f"[TaskGen] generation failed: {e}")
                continue
            # convert the returned job objects into the op tuples expected by _job_process
            converted_ops = []
            # build capability map: op_type -> allowed WCs
            capability_map = {op: [] for op in range(0, 32)}
            try:
                for mid, mdata in getattr(self.workcenters_meta, 'machine_registry', {}).items():
                    wc = int(mdata.get('workcenter', 0))
                    caps = list(mdata.get('capabilities', []))
                    for c in caps:
                        capability_map.setdefault(int(c), []).append(int(wc))
            except Exception:
                pass

            for jobobj in ops_objs:
                # jobobj expected to be a Jobs.Job-like object with index_id and time_span
                op_type = int(getattr(jobobj, 'index_id', 0))
                allowed_wcs = list(sorted(set(capability_map.get(op_type, []))))
                if not allowed_wcs:
                    allowed_wcs = [int(self._np_rng.randint(0, max(1, self.num_wcs)))]
                # per-wc durations: use jobobj.time_span as base and derive per-wc durations
                per_wc = {}
                for wc in allowed_wcs:
                    speed = self._speed_factor_for_wc(wc)
                    per_wc[int(wc)] = round(float(getattr(jobobj, 'time_span', 1.0)) / max(1e-6, speed), 6)
                converted_ops.append((op_type, allowed_wcs, per_wc))

            # finally add the job to the environment
            try:
                self.add_job(converted_ops)
            except Exception as e:
                print(f"[Env] Failed to add dynamic job: {e}")

    def get_env_info(self):
        # prefer machine-level action space when available
        num_machines = len(getattr(self.workcenters_meta, 'machine_list', []) or [])
        n_actions = int(num_machines) if num_machines > 0 else int(self.num_wcs)
        return {
            "n_actions": n_actions,
            "n_agents": self.num_jobs,
            "state_shape": self.state_dim,
            "obs_shape": self.obs_dim_agent,
            "episode_limit": self.episode_limit,
        }
# ============================================================