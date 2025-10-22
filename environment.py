# environment.py – MASA-QMIX (Step 8A.7 Fixed Stable Version)
# ------------------------------------------------------------
# • JobAgent = learning agent
# • Action = WorkCenter selection
# • Operator & machine constraints active
# • Observation = 11D (progress_ratio, utilization, etc.)
# • State = 64D global vector
# • Reward shaping normalized (scaled)
# • Done condition fixed → no early termination

from __future__ import annotations
import numpy as np
from collections import deque
import random
from typing import List, Dict, Any, Optional, Tuple

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
#                          Env
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
        self.machine_busy = [False] * num_wcs
        self.machine_queues = [deque() for _ in range(num_wcs)]

        # Operator groups (4)
        self.group_capacity = [1, 1, 1, 1]
        self.group_busy = [0, 0, 0, 0]

        self.jobs: List[JobAgent] = []
        self._generate_initial_jobs()

        self.t = 0
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self.prev_total_remaining = 0.0
        self._completed_now_cache = 0
        self._recent_rewards = deque(maxlen=5)

        self.rw = {"complete": 10.0, "progress": 2, "wait": 0.05, "wip": 0.02, "idle": 0.05}
        if reward_weights:
            self.rw.update(reward_weights)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------
    def reset(self):
        self.t = 0
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self._recent_rewards.clear()

        self.machine_busy = [False] * self.num_wcs
        self.machine_queues = [deque() for _ in range(self.num_wcs)]
        self.group_busy = [0, 0, 0, 0]

        self._generate_initial_jobs()
        self.prev_total_remaining = self._total_remaining_work()

        obs = self._build_all_agent_obs()
        info = {
            "state_vec": self._build_state_vector(),
            "avail_actions": self._build_avail_actions(),
        }
        return obs, info

    def step(self, actions):
        """Each JobAgent selects a WorkCenter; env applies constraints."""
        self._apply_actions(actions)
        self._advance_time()
        reward, breakdown = self._compute_shaped_reward()
        self._recent_rewards.append(reward)

        obs_next = self._build_all_agent_obs()
        info = {
            "state_vec": self._build_state_vector(),
            "reward_breakdown": breakdown,
            "avail_actions": self._build_avail_actions(),
        }

        # ✅ FIXED: Done condition (no premature termination)
        done = (self.t >= self.episode_limit) or all(j.finished for j in self.jobs)

        # Normalize reward (to ~[-10, +10])
        reward = np.clip(reward / 10.0, -10.0, 10.0)
        return obs_next, float(reward), bool(done), info

    # --------------------------------------------------------
    # Core Simulation
    # --------------------------------------------------------
    def _apply_actions(self, actions):
        if actions is None:
            return
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        for job, act in zip(self.jobs, actions):
            if job.finished:
                continue
            wc = int(act)
            if wc < 0 or wc >= self.num_wcs:
                continue
            if self.machine_busy[wc]:
                continue
            op = job.current_op()
            if op is None:
                continue
            allowed_wcs, dur = op
            if wc not in allowed_wcs:
                continue
            grp = self._group_for_wc(wc)
            if not self._operator_available(grp):
                continue

            dur = float(dur) * self._speed_factor_for_wc(wc)
            self.machine_queues[wc].append((job, dur, grp))
            job.remaining_time = dur
            self.machine_busy[wc] = True
            self.group_busy[grp] += 1

    def _advance_time(self):
        self.t += 1
        completed_now = 0
        for wc in range(self.num_wcs):
            if not self.machine_queues[wc]:
                continue
            job, rem, grp = self.machine_queues[wc][0]
            rem -= 1.0
            job.remaining_time = rem
            if rem <= 1e-6:
                job.current_op_idx += 1
                self.machine_queues[wc].popleft()
                self.group_busy[grp] = max(0, self.group_busy[grp] - 1)
                if job.current_op_idx >= len(job.operations):
                    job.finished = True
                    completed_now += 1
                    self.completed_jobs += 1
                self.machine_busy[wc] = False
            else:
                self.machine_queues[wc][0] = (job, rem, grp)
                self.machine_busy[wc] = True

        for j in self.jobs:
            if not j.finished and j.remaining_time <= 0.0:
                j.wait_time += 1.0
                self.total_wait_time += 1.0

        self._completed_now_cache = completed_now

    # --------------------------------------------------------
    # Observation / State / Avail
    # --------------------------------------------------------
    def _build_all_agent_obs(self):
        return [self._build_agent_obs(j) for j in self.jobs]

    def _build_agent_obs(self, job):
        progress = job.progress_ratio()
        wait_norm = np.clip(job.wait_time / 50.0, 0, 1)
        rem_norm = np.clip(job.remaining_time / 20.0, 0, 1)
        util_m, util_o = self._util_machines(), self._util_ops()
        wip_norm = np.clip(self._wip() / 80.0, 0, 1)
        time_norm = np.clip(self.t / self.episode_limit, 0, 1)
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
        avg_wait = np.clip(self.total_wait_time / max(1, self.t * 10.0), 0, 1)
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
            op = job.current_op()
            if op is None:
                continue
            allowed_wcs, _ = op
            for wc in allowed_wcs:
                grp = self._group_for_wc(wc)
                if not self.machine_busy[wc] and self._operator_available(grp):
                    avail[j_idx, wc] = 1
        return avail

    # --------------------------------------------------------
    # Reward Shaping
    # --------------------------------------------------------
    def _compute_shaped_reward(self):
        completed = float(self._completed_now_cache)
        total_rem = self._total_remaining_work()
        progress_delta = max(0.0, self.prev_total_remaining - total_rem)
        self.prev_total_remaining = total_rem

        avg_wait = self.total_wait_time / max(1, self.t)
        wip = float(self._wip())
        idle_ratio = 1.0 - 0.5 * (self._util_ops() + self._util_machines())

        r_complete = self.rw["complete"] * completed
        r_progress = self.rw["progress"] * progress_delta
        r_wait = self.rw["wait"] * avg_wait
        r_wip = self.rw["wip"] * wip
        r_idle = self.rw["idle"] * idle_ratio

        reward = r_complete + r_progress - r_wait - r_wip - r_idle
        return reward, {
            "completed_now": completed,
            "progress_delta": progress_delta,
            "avg_wait": avg_wait,
            "wip": wip,
            "idle_ratio": idle_ratio,
            "reward": reward,
        }

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _generate_initial_jobs(self):
        self.jobs = []
        for jid in range(self.num_jobs):
            num_ops = self._np_rng.randint(2, 5)
            ops = []
            for _ in range(num_ops):
                wc = int(self._np_rng.randint(0, self.num_wcs))
                dur = float(self._np_rng.uniform(5.0, 20.0))
                ops.append(([wc], dur))
            self.jobs.append(JobAgent(jid, ops))

    def _total_remaining_work(self):
        total = 0.0
        for j in self.jobs:
            if j.finished:
                continue
            op = j.current_op()
            if op:
                total += op[1]
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

    def _operator_available(self, grp: int):
        cap = self.group_capacity[grp]
        return self.group_busy[grp] < cap

    def _speed_factor_for_wc(self, wc: int):
        m_id = f"M_{wc}_0"
        reg = getattr(self.sites, "machine_registry", {})
        return reg.get(m_id, {}).get("speed_factor", 1.0)

    def _util_machines(self):
        return np.mean([1.0 if b else 0.0 for b in self.machine_busy])

    def _util_ops(self):
        return sum(self.group_busy) / max(1, sum(self.group_capacity))

    # --------------------------------------------------------
    def get_env_info(self):
        return {
            "n_actions": self.num_wcs,
            "n_agents": self.num_jobs,
            "state_shape": self.state_dim,
            "obs_shape": self.obs_dim_agent,
            "episode_limit": self.episode_limit,
        }
