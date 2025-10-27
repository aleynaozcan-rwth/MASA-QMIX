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
import numpy as np
from collections import deque
import random
import simpy
from typing import List, Dict, Optional, Tuple

try:
    from utils.site import Sites
except Exception:
    class Sites:
        def __init__(self):
            self.sites_object_list = list(range(18))
            self.machine_registry = {f"M_{i}_0": {"speed_factor": 1.0} for i in range(18)}


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
        num_wcs: int = 18,
        episode_limit: int = 200,
        obs_dim_agent: int = 11,
        state_dim: int = 64,
        seed: int = 42,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        self.num_jobs = num_jobs
        self.num_ops = num_operators
        self.num_wcs = num_wcs
        self.obs_dim_agent = obs_dim_agent
        self.state_dim = state_dim
        self.episode_limit = episode_limit

        self._np_rng = np.random.RandomState(seed)
        self._py_rng = random.Random(seed)

        self.sites = Sites()

        # --- SimPy environment & resources ---
        self.env = simpy.Environment()
        self.workcenters = [simpy.Resource(self.env, capacity=1) for _ in range(num_wcs)]
        self.operator_groups = [simpy.Resource(self.env, capacity=1) for _ in range(num_operators)]

        # Jobs
        self.jobs: List[JobAgent] = []
        self._generate_initial_jobs()

        # Bookkeeping
        self.t = 0.0
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self.prev_total_remaining = 0.0
        self._completed_now_cache = 0
        self._recent_rewards = deque(maxlen=5)
        self.done = False

        # Reward weights (unchanged)
        self.rw = {"complete": 10.0, "progress": 2, "wait": 0.05, "wip": 0.02, "idle": 0.05}
        if reward_weights:
            self.rw.update(reward_weights)

        # 9A decision batching
        self.pending_decisions: List[Dict] = []
        self.decisions_ready = simpy.Event(self.env)
        # lightweight gantt/event records: (start, end, operation, wc, job_id, operator_grp)
        self.gantt_records = []

    # --------------------------------------------------------
    # Public API (compatible)
    # --------------------------------------------------------
    def reset(self):
        # Reset SimPy world
        self.env = simpy.Environment()
        self.workcenters = [simpy.Resource(self.env, capacity=1) for _ in range(self.num_wcs)]
        self.operator_groups = [simpy.Resource(self.env, capacity=1) for _ in range(self.num_ops)]

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

    # Eski step() API’sini koruyoruz ama 9A’da Runner event-driven kullanacak.
    # Step tabanlı çağrılırsa 1 zaman birimi çalıştırır (geriye uyumluluk).
    def step(self, actions):
        # 9A modunda step kullanılmaz; ama çağrılırsa "karar yoksa" küçük bir zaman akışı sağlar.
        self.env.run(until=self.env.now + 1.0)
        self.t = self.env.now
        reward = self.pop_decision_reward()
        reward = np.clip(reward / 10.0, -10.0, 10.0)
        obs_next = self._build_all_agent_obs()
        info = {
            "state_vec": self._build_state_vector(),
            "avail_actions": self._build_avail_actions(),
            "note": "9A event-driven ortamda step() kullanılmamalı; Runner karar döngüsünü kullanır.",
        }
        done = (self.t >= self.episode_limit) or all(j.finished for j in self.jobs)
        self.done = self.done or done
        return obs_next, float(reward), bool(done), info

    # ---------------------- 9A Decision Loop API -----------------------
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
        completed = float(self._completed_now_cache)
        self._completed_now_cache = 0
        total_rem = self._total_remaining_work()
        progress_delta = max(0.0, self.prev_total_remaining - total_rem)
        self.prev_total_remaining = total_rem

        avg_wait = self.total_wait_time / max(1, self.env.now) if self.env.now > 0 else 0.0
        wip = float(self._wip())
        idle_ratio = 1.0 - 0.5 * (self._util_ops() + self._util_machines())

        r_complete = self.rw["complete"] * completed
        r_progress = self.rw["progress"] * progress_delta
        r_wait = self.rw["wait"] * avg_wait
        r_wip = self.rw["wip"] * wip
        r_idle = self.rw["idle"] * idle_ratio

        reward = r_complete + r_progress - r_wait - r_wip - r_idle
        self._recent_rewards.append(reward)
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

            def _resume_with(wc_choice: int, _resume_evt=resume_evt):
                # Runner’ın verdiği WC seçimini doğrula ve süreci devam ettir
                if wc_choice not in allowed_wcs:
                    _resume_evt.succeed(None)  # invalid → no-op (minik skip)
                    return
                _resume_evt.succeed(int(wc_choice))

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
                "base_duration": float(base_duration_val),
                "resume": _resume_with,
                "grp_by_wc": {wc: self._group_for_wc(wc) for wc in allowed_wcs},
            }
            self.pending_decisions.append(decision_item)
            if not self.decisions_ready.triggered:
                self.decisions_ready.succeed()

            chosen_wc = (yield resume_evt)

            if chosen_wc is None:
                # No-op: küçük bir zaman sıçraması ile deadlock önle
                yield self.env.timeout(1e-9)
                continue

            grp = self._group_for_wc(chosen_wc)
            # determine duration for chosen WC
            if per_wc_durations is not None:
                try:
                    dur = float(per_wc_durations.get(int(chosen_wc), 0.0))
                except Exception:
                    dur = 0.0
            elif base_dur is not None:
                # interpret machine speed as speed factor (>1 faster), so duration = base / speed
                speed = self._speed_factor_for_wc(chosen_wc)
                try:
                    dur = float(base_dur) / max(1e-6, float(speed))
                except Exception:
                    dur = float(base_dur)
            else:
                # fallback small timeout
                dur = 0.0

            start_wait = self.env.now
            with self.operator_groups[grp].request() as op_req, self.workcenters[chosen_wc].request() as mc_req:
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
                yield self.env.timeout(dur)
                # record end time and append to gantt_records
                op_end = float(self.env.now)
                op_idx = int(job.current_op_idx)
                try:
                    # prefer op_type (explicit type); fallback to sequence index
                    op_tag = op_type if (op_type is not None) else op_idx
                    # store record as (start, end, operation_type, workcenter, job_id, operator_group)
                    self.gantt_records.append((op_start, op_end, int(op_tag), int(chosen_wc), int(job.id), int(grp)))
                except Exception:
                    pass
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
        avail = np.zeros((self.num_jobs, self.num_wcs), dtype=np.int32)
        for j_idx, job in enumerate(self.jobs):
            if job.finished:
                continue
            row = self._avail_row_for_job(job)
            avail[j_idx, :] = row
        return avail

    def _avail_row_for_job(self, job: JobAgent):
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
        for wc in allowed_wcs:
            grp = self._group_for_wc(wc)
            if self._resource_free(self.workcenters[wc]) and self._resource_free(self.operator_groups[grp]):
                row[wc] = 1
        return row

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _generate_initial_jobs(self):
        # Define a fixed set of operation types (e.g., 9 types)
        op_types = list(range(9))
        self.jobs = []
        # build capability map: op_type -> allowed WCs by inspecting Sites.machine_registry
        # here we assume op_type indexes map to capability ids in machine capabilities (0..8)
        # machine_registry entries: {'M_<wc>_0': {'workcenter': wc, 'capabilities':[...], 'speed_factor':...}}
        capability_map = {op: [] for op in op_types}
        try:
            for mid, mdata in getattr(self.sites, 'machine_registry', {}).items():
                wc = int(mdata.get('workcenter', 0))
                caps = list(mdata.get('capabilities', []))
                for c in caps:
                    if c in capability_map:
                        capability_map[c].append(wc)
        except Exception:
            # fallback: spread using modulo assignment
            for op in op_types:
                capability_map[op] = [wc for wc in range(self.num_wcs) if (wc % len(op_types)) == (op % len(op_types))]

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
                # allowed WCs derived from capability map
                allowed_wcs = list(sorted(set(capability_map.get(int(op_type), []))))
                if not allowed_wcs:
                    allowed_wcs = [int(self._np_rng.randint(0, self.num_wcs))]
                # base duration in [1.0, 9.0] (base for op_type)
                base = float(self._np_rng.uniform(1.0, 9.0))
                # per-WC duration: assume duration scales inversely with machine speed
                per_wc_durations = {}
                for wc in allowed_wcs:
                    # get speed_factor from Sites registry
                    mid = f"M_{wc}_0"
                    speed = getattr(self.sites.machine_registry.get(mid, {}), 'get', None)
                    try:
                        speed_factor = float(self.sites.machine_registry.get(mid, {}).get('speed_factor', 1.0))
                    except Exception:
                        speed_factor = 1.0
                    # duration = base / speed_factor
                    est = float(base) / max(1e-6, speed_factor)
                    per_wc_durations[int(wc)] = round(est, 6)
                # store op as canonical (op_type, allowed_wcs, per_wc_durations)
                ops.append((int(op_type), list(allowed_wcs), per_wc_durations))
            self.jobs.append(JobAgent(jid, ops))

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

    @staticmethod
    def _group_for_wc(wc: int) -> int:
        if 0 <= wc <= 5:
            return 0
        if 6 <= wc <= 9:
            return 1
        if 10 <= wc <= 13:
            return 2
        return 3

    def _speed_factor_for_wc(self, wc: int):
        m_id = f"M_{wc}_0"
        reg = getattr(self.sites, "machine_registry", {})
        return reg.get(m_id, {}).get("speed_factor", 1.0)

    def _util_machines(self):
        busy = sum(len(wc.users) for wc in self.workcenters)
        return busy / max(1, len(self.workcenters))

    def _util_ops(self):
        busy = sum(len(g.users) for g in self.operator_groups)
        return busy / max(1, len(self.operator_groups))

    def _resource_free(self, res):
        return len(res.users) < res.capacity

    # --------------------------------------------------------
    def get_env_info(self):
        return {
            "n_actions": self.num_wcs,
            "n_agents": self.num_jobs,
            "state_shape": self.state_dim,
            "obs_shape": self.obs_dim_agent,
            "episode_limit": self.episode_limit,
        }
# ============================================================