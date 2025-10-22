# environment.py – MASA-QMIX (Step 8A.7 Fix)
# ------------------------------------------
# • Agent = Job (öğrenen varlık)
# • Action = WorkCenter seçimi (n_actions = num_wcs)
# • Operatör/Makine kısıtları env tarafında (avail mask + runtime kontrol)
# • Observation: 11D (progress_ratio dahil)  ← 8A.6.x ile uyumlu
# • State: varsayılan 64 (args.state_shape ile eşleştir)
# • Reward shaping: complete/progress – wait – wip – idle (değişmedi)

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

# ----------------------------- JobAgent -----------------------------

class JobAgent:
    """Her job, ardışık operasyonlardan oluşur; her adımda sıradaki operasyonu işler."""
    def __init__(self, job_id: int, operations: List[Tuple[List[int], float]]):
        """
        operations: [([allowed_wc_ids], duration), ...]
        """
        self.id = int(job_id)
        self.operations = operations
        self.current_op_idx = 0
        self.remaining_time = 0.0
        self.wait_time = 0.0
        self.finished = False

    def current_op(self) -> Optional[Tuple[List[int], float]]:
        if self.finished or self.current_op_idx >= len(self.operations):
            return None
        return self.operations[self.current_op_idx]

    def progress_ratio(self) -> float:
        return float(self.current_op_idx / max(1, len(self.operations)))


# ------------------------------- Env --------------------------------

class MASAEnv:
    """
    MASA-QMIX Env — JobAgent-based + Operator-constrained

    • step(actions): actions uzunluğu = num_jobs, her eleman bir WC id'si
    • avail mask: (num_jobs, num_wcs) — allowed & wc boş & uygun operator grubu boş
    • reward: shaped
    """

    def __init__(
        self,
        num_jobs: int = 10,
        num_operators: int = 4,   # 4 grup: (0:0–5, 1:6–9, 2:10–13, 3:14–17)
        num_wcs: int = 18,
        episode_limit: int = 200,
        obs_dim_agent: int = 11,  # 8A.6.x ile uyumlu
        state_dim: int = 64,      # 8A.6.x ile uyumlu
        seed: int = 42,
        reward_weights: Optional[Dict[str, float]] = None,
        job_spawn_baseline: float = 0.0,  # şimdilik dinamik üretim kapalı
    ):
        # Boyutlar
        self.num_jobs = int(num_jobs)
        self.num_ops = int(num_operators)  # operatör havuzu (grup kapasitesi)
        self.num_wcs = int(num_wcs)
        self.obs_dim_agent = int(obs_dim_agent)
        self.state_dim = int(state_dim)
        self.episode_limit = int(episode_limit)

        # RNG
        self._np_rng = np.random.RandomState(seed)
        self._py_rng = random.Random(seed)

        # Site/Makine
        self.sites = Sites()
        self.machine_busy = [False] * self.num_wcs
        # Kuyruk: (job, remaining, grp)
        self.machine_queues: List[deque] = [deque() for _ in range(self.num_wcs)]

        # Operatör grupları (kapasite/usage)
        self.group_capacity = [1, 1, 1, 1]  # her grupta 1 operatör; istersen artır
        self.group_busy = [0, 0, 0, 0]

        # Jobs
        self.jobs: List[JobAgent] = []
        self._generate_initial_jobs()

        # Simülasyon / KPI
        self.t = 0
        self.completed_jobs = 0
        self.total_wait_time = 0.0
        self.prev_total_remaining = 0.0
        self._completed_now_cache = 0
        self._recent_rewards = deque(maxlen=5)

        # Dinamik üretim param
        self.job_spawn_baseline = float(job_spawn_baseline)

        # Reward ağırlıkları
        self.rw = {"complete": 5.0, "progress": 0.5, "wait": 0.2, "wip": 0.05, "idle": 0.1}
        if reward_weights:
            self.rw.update(reward_weights)

    # --------------------------- Public API ---------------------------

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
        """Her JobAgent bir WC seçer; env operatör + makine kısıtlarını uygular."""
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
        done = self.t >= self.episode_limit
        return obs_next, float(reward), bool(done), info

    # ------------------------- Core Simulation ------------------------

    def _apply_actions(self, actions):
        """Valid (job, wc) seçimi: allowed & wc boş & operatör grubu boş."""
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

            # İş başlat
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
                # operasyon bitti
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

        # Bekleme zamanları
        for j in self.jobs:
            if not j.finished and j.remaining_time <= 0.0:
                j.wait_time += 1.0
                self.total_wait_time += 1.0

        self._completed_now_cache = completed_now

    # ------------------ Observation / State / Avail -------------------

    def _build_all_agent_obs(self) -> List[np.ndarray]:
        return [self._build_agent_obs(j) for j in self.jobs]

    def _build_agent_obs(self, job: JobAgent) -> np.ndarray:
        """
        11D Job-merkezli gözlem (8A.6.x ile uyumlu):
          0: progress_ratio
          1: job_wait_norm
          2: job_remaining_norm
          3: util_machines
          4: util_operators
          5: wip_norm
          6: time_norm
          7: recent_reward_norm
          8: completed_jobs_norm
          9: n_ops_norm
         10: finished_flag
        """
        progress = job.progress_ratio()
        wait_norm = float(np.clip(job.wait_time / 50.0, 0.0, 1.0))
        rem_norm = float(np.clip(job.remaining_time / 20.0, 0.0, 1.0))
        util_m, util_o = self._util_machines(), self._util_ops()
        wip_norm = float(np.clip(self._wip() / 80.0, 0.0, 1.0))
        time_norm = float(np.clip(self.t / max(1, self.episode_limit), 0.0, 1.0))
        recent_reward_norm = float(np.clip((np.mean(self._recent_rewards) if self._recent_rewards else 0.0) / 10.0, 0.0, 1.0))
        completed_norm = float(np.clip(self.completed_jobs / 100.0, 0.0, 1.0))
        n_ops_norm = float(np.clip(len(job.operations) / 10.0, 0.0, 1.0))
        finished_flag = 1.0 if job.finished else 0.0

        obs = np.array(
            [progress, wait_norm, rem_norm, util_m, util_o, wip_norm,
             time_norm, recent_reward_norm, completed_norm, n_ops_norm, finished_flag],
            dtype=np.float32,
        )
        if obs.shape[0] < self.obs_dim_agent:
            obs = np.concatenate([obs, np.zeros((self.obs_dim_agent - obs.shape[0],), dtype=np.float32)], axis=0)
        elif obs.shape[0] > self.obs_dim_agent:
            obs = obs[: self.obs_dim_agent]
        return obs

    def _build_state_vector(self) -> np.ndarray:
        """KPI-odaklı state (64 varsayılan; args.state_shape ile eşleştir)."""
        util_m, util_o = self._util_machines(), self._util_ops()
        avg_wait_norm = float(np.clip(self.total_wait_time / max(1, self.t * 10.0), 0.0, 1.0))
        wip_norm = float(np.clip(self._wip() / 100.0, 0.0, 1.0))
        completed_norm = float(np.clip(self.completed_jobs / 200.0, 0.0, 1.0))
        reward_recent = float(np.clip((np.mean(self._recent_rewards) if self._recent_rewards else 0.0) / 10.0, 0.0, 1.0))
        idle_ratio = 1.0 - 0.5 * (util_m + util_o)

        core = np.array([util_m, util_o, avg_wait_norm, wip_norm, completed_norm, reward_recent, idle_ratio], dtype=np.float32)
        if core.shape[0] < self.state_dim:
            core = np.concatenate([core, np.zeros((self.state_dim - core.shape[0],), dtype=np.float32)], axis=0)
        return core[: self.state_dim]

    def _build_avail_actions(self) -> np.ndarray:
        """
        (num_jobs, num_wcs) → 1 ise bu job için bu WC şu anda seçilebilir:
          • allowed (routing)
          • wc boş
          • uygun operator grubu boş
        """
        avail = np.zeros((self.num_jobs, self.num_wcs), dtype=np.int32)
        for j_idx, job in enumerate(self.jobs):
            if job.finished:
                continue
            op = job.current_op()
            if op is None:
                continue
            allowed_wcs, _ = op
            for wc in allowed_wcs:
                if 0 <= wc < self.num_wcs:
                    grp = self._group_for_wc(wc)
                    if (not self.machine_busy[wc]) and self._operator_available(grp):
                        avail[j_idx, wc] = 1
        return avail

    # ------------------------- Reward Shaping -------------------------

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
        breakdown = {
            "completed_now": completed,
            "progress_delta": float(progress_delta),
            "avg_wait": float(avg_wait),
            "wip": float(wip),
            "idle_ratio": float(idle_ratio),
            "reward": float(reward),
        }
        return float(reward), breakdown

    # ---------------------------- Helpers ----------------------------

    def _generate_initial_jobs(self):
        """Basit iş üretimi: her job 2–4 operasyon; her operasyon bir WC (routing sabit)."""
        self.jobs = []
        for jid in range(self.num_jobs):
            num_ops = int(self._np_rng.randint(2, 4 + 1))
            ops: List[Tuple[List[int], float]] = []
            for _ in range(num_ops):
                wc = int(self._np_rng.randint(0, self.num_wcs))
                base_dur = float(self._np_rng.uniform(5.0, 20.0))
                ops.append(([wc], base_dur))
            self.jobs.append(JobAgent(jid, ops))

    def _total_remaining_work(self) -> float:
        total = 0.0
        for j in self.jobs:
            if j.finished:
                continue
            op = j.current_op()
            if op:
                total += float(op[1])
        return float(total)

    def _wip(self) -> int:
        return int(sum(1 for j in self.jobs if not j.finished))

    # --- operator/machine helpers ---
    @staticmethod
    def _group_for_wc(wc: int) -> int:
        if 0 <= wc <= 5:
            return 0
        if 6 <= wc <= 9:
            return 1
        if 10 <= wc <= 13:
            return 2
        return 3  # 14..17

    def _operator_available(self, grp: int) -> bool:
        cap = self.group_capacity[grp] if 0 <= grp < len(self.group_capacity) else 1
        busy = self.group_busy[grp] if 0 <= grp < len(self.group_busy) else 0
        return busy < cap

    def _speed_factor_for_wc(self, wc: int) -> float:
        m_id = f"M_{wc}_0"
        reg = getattr(self.sites, "machine_registry", {})
        return float(reg.get(m_id, {}).get("speed_factor", 1.0))

    # --- util ---
    def _util_machines(self) -> float:
        return float(np.mean([1.0 if b else 0.0 for b in self.machine_busy])) if self.num_wcs > 0 else 0.0

    def _util_ops(self) -> float:
        total_cap = sum(self.group_capacity) if len(self.group_capacity) > 0 else 1
        total_busy = sum(self.group_busy) if len(self.group_busy) > 0 else 0
        return float(total_busy / max(1, total_cap))
        # ---------------------- Interface Helper -----------------------
    def get_env_info(self):
        """Runner/Args uyumluluğu için temel ortam bilgilerini döndürür."""
        return {
            "n_actions": self.num_wcs,
            "n_agents": self.num_jobs,
            "state_shape": self.state_dim,
            "obs_shape": self.obs_dim_agent,
            "episode_limit": self.episode_limit,
        }
