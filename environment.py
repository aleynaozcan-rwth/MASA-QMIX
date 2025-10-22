# environment.py – MASA-QMIX Environment (Step 8A.6.5)
# ------------------------------------------------------
# Added: progress_ratio feature for pattern-level generalization across heterogeneous jobs
# Observation vector now 11D (was 10D)
# Purpose: allow agents to learn from relative progress rather than absolute job length.

import numpy as np
from collections import deque
import random
from typing import List, Dict, Any, Optional
from utils.site import Sites


class MASAEnv:
    """
    MASA-QMIX Environment (Step 8A.6.5)
    ------------------------------------
    Key Features:
      • Dynamic job arrivals (continuous generation)
      • Operator–WorkCenter eligibility constraints
      • RNN-compatible observation/state (obs_dim=11, state_dim=64)
      • Reward shaping (complete, progress, wait, wip, idle)
      • Added progress_ratio feature for cross-job generalization
    """

    def __init__(
        self,
        num_operators: int = 4,
        obs_dim_agent: int = 11,   # <-- updated from 10 to 11
        state_dim: int = 64,
        episode_limit: int = 200,
        seed: int = 42,
        reward_weights: Optional[Dict[str, float]] = None,
        job_spawn_baseline: float = 0.35,
    ):
        # Core dimensions
        self.obs_dim_agent = int(obs_dim_agent)
        self.state_dim = int(state_dim)
        self.episode_limit = int(episode_limit)

        # RNGs
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        # WorkCenter and machine definitions (from site.py)
        self.sites = Sites()
        self.num_wcs = len(self.sites.sites_object_list)

        # Operators
        self.num_ops = int(num_operators)
        self.operator_group = [self._group_of_site(i) for i in range(self.num_ops)]
        self.operator_busy = [False] * self.num_ops
        self.operator_task: List[Optional[Dict[str, Any]]] = [None] * self.num_ops

        # Machine queues and statuses
        self.machine_queues: List[deque] = [deque() for _ in range(self.num_wcs)]
        self.machine_busy = [False] * self.num_wcs
        self.wc_machine_id = [f"M_{i}_0" for i in range(self.num_wcs)]
        self.machine_speed = [
            float(self.sites.machine_registry[m_id]["speed_factor"])
            for m_id in self.wc_machine_id
        ]

        # Simulation time
        self.t = 0
        self.time_limit = int(episode_limit)

        # Metrics
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self.prev_total_remaining = 0.0
        self._completed_now_cache = 0.0

        # Job generation
        self.job_spawn_baseline = float(job_spawn_baseline)

        # Reward weights
        self.rw = {
            "complete": +5.0,
            "progress": +0.5,
            "wait": 0.2,
            "wip": 0.05,
            "idle": 0.1,
        }
        if reward_weights is not None:
            self.rw.update(reward_weights)

        self._last_reward_breakdown = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self):
        self._reset_system()
        obs = self._build_all_agent_obs()
        info = {
            "state_vec": self._build_state_vector(),
            "avail_actions": self._build_avail_actions(),
        }
        return obs, info

    def step(self, actions):
        """Perform one simulation step."""
        self._apply_actions(actions)
        self._advance_time()
        reward, rb = self._compute_shaped_reward()

        obs_next = self._build_all_agent_obs()
        info = {
            "state_vec": self._build_state_vector(),
            "reward_breakdown": rb,
            "avail_actions": self._build_avail_actions(),
        }
        done = self.t >= self.time_limit
        return obs_next, float(reward), bool(done), info

    # ------------------------------------------------------------------
    # Simulation Core
    # ------------------------------------------------------------------
    def _apply_actions(self, actions):
        """Assign operators to WorkCenters if eligible and available."""
        if actions is None:
            return
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        for op_id in range(self.num_ops):
            if self.operator_busy[op_id] or op_id >= len(actions):
                continue
            wc = int(actions[op_id])
            if not self._eligible(op_id, wc):
                continue
            if len(self.machine_queues[wc]) == 0 or self.machine_busy[wc]:
                continue

            job = self.machine_queues[wc].popleft()
            machine_rate = self.machine_speed[wc]
            remaining = float(job.get("dur", 0.0)) / max(1e-6, machine_rate)

            self.operator_busy[op_id] = True
            self.machine_busy[wc] = True
            self.operator_task[op_id] = {
                "wc": wc,
                "remaining": remaining,
                "started_at": self.t,
                "wait_acc": float(job.get("wait", 0.0)),
            }

    def _advance_time(self):
        """Advance simulation time by one tick."""
        self.t += 1
        completed_this_tick = 0

        # Update waiting times
        for q in self.machine_queues:
            for job in q:
                job["wait"] = job.get("wait", 0.0) + 1.0
                self.total_wait_time += 1.0

        # Process active jobs
        for op_id, task in enumerate(self.operator_task):
            if task is None:
                continue
            wc = task["wc"]
            task["remaining"] -= 1.0
            if task["remaining"] <= 1e-6:
                completed_this_tick += 1
                self.operator_busy[op_id] = False
                self.machine_busy[wc] = False
                self.operator_task[op_id] = None

        self._set_completed_this_tick(completed_this_tick)
        self._maybe_generate_jobs()

    def _maybe_generate_jobs(self, initial=False):
        """Random job generator for dynamic environment."""
        if initial:
            for wc in range(self.num_wcs):
                n = self._np_rng.binomial(n=3, p=0.6)
                for _ in range(n):
                    dur = float(self._np_rng.uniform(5.0, 20.0))
                    self.machine_queues[wc].append({"dur": dur, "wait": 0.0})
            return

        wip = self._wip()
        damp = np.clip(1.0 - (wip / 120.0), 0.2, 1.0)
        lam = self.job_spawn_baseline * damp
        trials = self.num_ops + 2
        new_jobs = self._np_rng.binomial(n=trials, p=lam)

        for _ in range(new_jobs):
            wc = int(self._np_rng.randint(0, self.num_wcs))
            dur = float(self._np_rng.uniform(5.0, 20.0))
            self.machine_queues[wc].append({"dur": dur, "wait": 0.0})

    # ------------------------------------------------------------------
    # Observation, State, and Action Mask
    # ------------------------------------------------------------------
    def _build_all_agent_obs(self):
        return [self._build_agent_obs(op) for op in range(self.num_ops)]

    def _build_agent_obs(self, op_id: int) -> np.ndarray:
        """Agent observation vector (11D) with progress ratio."""
        # --- New feature: overall progress ratio ---
        total_jobs = float(self.completed_jobs + self._wip())
        progress_ratio = float(self.completed_jobs / total_jobs) if total_jobs > 0 else 0.0
        # This captures global progress of the system (0–1) and helps generalize
        # across jobs of varying lengths and sequences.

        op_busy = 1.0 if self.operator_busy[op_id] else 0.0
        group_norm = float(op_id / max(1, (self.num_ops - 1)))

        q_lens = np.array([len(q) for q in self.machine_queues], dtype=np.float32)
        avg_q_norm = float(np.clip(q_lens.mean() / 10.0 if q_lens.size else 0.0, 0.0, 1.0))
        wip_norm = float(np.clip(self._wip() / 60.0, 0.0, 1.0))
        time_norm = float(np.clip(self.t / max(1, self.time_limit), 0.0, 1.0))
        completed_norm = float(np.clip(self.completed_jobs / 200.0, 0.0, 1.0))
        util_ops = float(np.mean([1.0 if b else 0.0 for b in self.operator_busy])) if self.num_ops > 0 else 0.0
        util_machines = float(np.mean([1.0 if b else 0.0 for b in self.machine_busy])) if self.num_wcs > 0 else 0.0

        base = np.array(
            [op_busy, group_norm, avg_q_norm, wip_norm, time_norm,
             completed_norm, util_ops, util_machines, progress_ratio],
            dtype=np.float32,
        )
        # pad or trim to obs_dim_agent
        if base.shape[0] < self.obs_dim_agent:
            base = np.concatenate([base,
                                   np.zeros((self.obs_dim_agent - base.shape[0],),
                                            dtype=np.float32)], axis=0)
        elif base.shape[0] > self.obs_dim_agent:
            base = base[:self.obs_dim_agent]
        return base

    def _build_state_vector(self):
        """Global state vector (64D)."""
        q_lens = np.array([len(q) for q in self.machine_queues], dtype=np.float32)
        q_stats = np.array([
            float(q_lens.mean() if q_lens.size else 0.0),
            float(q_lens.max() if q_lens.size else 0.0),
            float(q_lens.min() if q_lens.size else 0.0),
            float(q_lens.std() if q_lens.size else 0.0),
            float(np.percentile(q_lens, 75) if q_lens.size else 0.0),
            float(np.percentile(q_lens, 25) if q_lens.size else 0.0),
        ], dtype=np.float32)

        util_m = float(np.mean([1.0 if b else 0.0 for b in self.machine_busy])) if self.num_wcs > 0 else 0.0
        util_o = float(np.mean([1.0 if b else 0.0 for b in self.operator_busy])) if self.num_ops > 0 else 0.0
        time_norm = float(np.clip(self.t / max(1, self.time_limit), 0.0, 1.0))
        active_jobs_norm = float(np.clip(self._wip() / 200.0, 0.0, 1.0))
        completed_norm = float(np.clip(self.completed_jobs / 400.0, 0.0, 1.0))
        wait_norm = float(np.clip(self.total_wait_time / max(1.0, self.t), 0.0, 10.0))

        per_group_load = np.zeros((self.num_ops,), dtype=np.float32)
        blocks = [(0, 6), (6, 10), (10, 14), (14, 18)]
        for gid, (a, b) in enumerate(blocks[:self.num_ops]):
            sl = q_lens[a:b] if q_lens.size else np.array([], dtype=np.float32)
            per_group_load[gid] = float(sl.mean() if sl.size else 0.0)

        per_group_util = np.full((self.num_ops,), util_o, dtype=np.float32)
        K = 10
        top_q = np.sort(q_lens)[-K:] if q_lens.size else np.zeros((K,), dtype=np.float32)
        wc_busy = np.array([1.0 if b else 0.0 for b in self.machine_busy], dtype=np.float32)
        top_busy = np.sort(wc_busy)[-K:] if wc_busy.size else np.zeros((K,), dtype=np.float32)

        parts = [q_stats,
                 np.array([util_m, util_o, time_norm, active_jobs_norm, completed_norm, wait_norm], dtype=np.float32),
                 per_group_load, per_group_util, top_q, top_busy]
        s = np.concatenate(parts, axis=0)
        if s.shape[0] < self.state_dim:
            s = np.concatenate([s, np.zeros((self.state_dim - s.shape[0],), dtype=np.float32)], axis=0)
        elif s.shape[0] > self.state_dim:
            s = s[:self.state_dim]
        return s

    def _build_avail_actions(self):
        """Return a (num_ops × num_wcs) availability mask for eligible and idle WorkCenters."""
        mask = np.zeros((self.num_ops, self.num_wcs), dtype=np.int32)
        for op_id in range(self.num_ops):
            if self.operator_busy[op_id]:
                continue
            for wc in range(self.num_wcs):
                if self._eligible(op_id, wc) and len(self.machine_queues[wc]) > 0 and not self.machine_busy[wc]:
                    mask[op_id, wc] = 1
        return mask

    # ------------------------------------------------------------------
    # Reward Shaping
    # ------------------------------------------------------------------
    def _compute_shaped_reward(self):
        completed_now = self._completed_this_tick()
        total_remaining = self._total_remaining_work()
        progress_delta = max(0.0, float(self.prev_total_remaining - total_remaining))
        self.prev_total_remaining = total_remaining

        avg_wait = float(self.total_wait_time / max(1, self.t))
        wip = float(self._wip())
        idle_ratio = 1.0 - 0.5 * (self._util_ops() + self._util_machines())

        r_complete = self.rw["complete"] * completed_now
        r_progress = self.rw["progress"] * progress_delta
        r_wait = self.rw["wait"] * avg_wait
        r_wip = self.rw["wip"] * wip
        r_idle = self.rw["idle"] * idle_ratio

        reward = r_complete + r_progress - r_wait - r_wip - r_idle
        breakdown = {
            "completed_now": float(completed_now),
            "progress_delta": float(progress_delta),
            "avg_wait": float(avg_wait),
            "wip": float(wip),
            "idle_ratio": float(idle_ratio),
            "r_complete": float(r_complete),
            "r_progress": float(r_progress),
            "r_wait": float(-r_wait),
            "r_wip": float(-r_wip),
            "r_idle": float(-r_idle),
            "reward": float(reward),
        }
        self._last_reward_breakdown = breakdown
        return reward, breakdown

    # ------------------------------------------------------------------
    # Helpers & Metrics
    # ------------------------------------------------------------------
    def _reset_system(self):
        """Reset simulation state."""
        self.t = 0
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self._completed_now_cache = 0.0
        self.operator_busy = [False] * self.num_ops
        self.operator_task = [None] * self.num_ops
        self.machine_busy = [False] * self.num_wcs
        self.machine_queues = [deque() for _ in range(self.num_wcs)]
        self._maybe_generate_jobs(initial=True)
        self.prev_total_remaining = self._total_remaining_work()

    def _eligible(self, op_id: int, wc: int) -> bool:
        """Check if an operator is eligible to work on a specific WorkCenter."""
        if op_id < 0 or op_id >= self.num_ops:
            return False
        if op_id == 0 and 0 <= wc <= 5:
            return True
        if op_id == 1 and 6 <= wc <= 9:
            return True
        if op_id == 2 and 10 <= wc <= 13:
            return True
        if op_id == 3 and 14 <= wc <= 17:
            return True
        return False

    def _group_of_site(self, site_id: int) -> int:
        """Return operator group based on WorkCenter id."""
        if 0 <= site_id <= 5:
            return 0
        if 6 <= site_id <= 9:
            return 1
        if 10 <= site_id <= 13:
            return 2
        return 3

    def _total_remaining_work(self) -> float:
        total = sum(float(job.get("dur", 0.0)) for q in self.machine_queues for job in q)
        for task in self.operator_task:
            if task is not None:
                total += max(0.0, float(task.get("remaining", 0.0)))
        return float(total)

    def _wip(self) -> int:
        return int(sum(len(q) for q in self.machine_queues) + sum(1 for t in self.operator_task if t is not None))

    def _util_ops(self) -> float:
        return float(np.mean([1.0 if b else 0.0 for b in self.operator_busy])) if self.num_ops > 0 else 0.0

    def _util_machines(self) -> float:
        return float(np.mean([1.0 if b else 0.0 for b in self.machine_busy])) if self.num_wcs > 0 else 0.0

    def _set_completed_this_tick(self, v: int):
        self._completed_now_cache = float(v)
        self.completed_jobs += int(v)

    def _completed_this_tick(self) -> float:
        return float(getattr(self, "_completed_now_cache", 0.0))
