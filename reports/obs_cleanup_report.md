# Observation cleanup report

Generated programmatically. This report groups matches by risk and tag. Do not apply any code edits from this run; this is analysis-only.

## Section A — High-risk (must review)
These hits appear in or near observation-building code (builders or env surfaces).

- environment.py:384  — tag=LEGACY_FEATURE_API, match=`reward_norm`
  - context before:
    -         except Exception:
    -             self.mean_wait_reference = None
    - 
  - line:         try:

- environment.py:386  — tag=LEGACY_FEATURE_API, match=`reward_norm`
  - context before:
    - 
    -         # reward_norm_divisor: used by policies to normalize reward signals
    -         try:
  - line:         except Exception:

- environment.py:388  — tag=LEGACY_FEATURE_API, match=`reward_norm`
  - context before:
    -         try:
    -             self.reward_norm_divisor = float(getattr(args, 'reward_norm_divisor', None) or kwargs.get('reward_norm_divisor', None))
    -         except Exception:
  - line:         if not self.reward_norm_divisor or float(self.reward_norm_divisor) <= 0.0:

- environment.py:389  — tag=LEGACY_FEATURE_API, match=`reward_norm`
  - context before:
    -             self.reward_norm_divisor = float(getattr(args, 'reward_norm_divisor', None) or kwargs.get('reward_norm_divisor', None))
    -         except Exception:
    -             self.reward_norm_divisor = None
  - line:             self.reward_norm_divisor = getattr(args, 'reward_norm_divisor', 10.0)

- environment.py:390  — tag=LEGACY_FEATURE_API, match=`reward_norm`
  - context before:
    -         except Exception:
    -             self.reward_norm_divisor = None
    -         if not self.reward_norm_divisor or float(self.reward_norm_divisor) <= 0.0:
  - line: 

- environment.py:778  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    - 
    -         # bookkeeping and reward caches
    -         self.t = 0.0
  - line:         self.total_wait_time = 0.0

- environment.py:781  — tag=LEGACY_FEATURE_API, match=`_recent_rewards`
  - context before:
    -         self.completed_jobs = 0
    -         self.total_wait_time = 0.0
    -         self._completed_now_cache = 0
  - line:         self.done = False

- environment.py:1108  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    - 
    -         # reset bookkeeping and job state
    -         self.t = 0.0
  - line:         self.total_wait_time = 0.0

- environment.py:1112  — tag=LEGACY_FEATURE_API, match=`_recent_rewards`
  - context before:
    -         self.total_wait_time = 0.0
    -         self._completed_now_cache = 0
    -         try:
  - line:         except Exception:

- environment.py:1116  — tag=LEGACY_FEATURE_API, match=`_recent_rewards`
  - context before:
    -         except Exception:
    -             try:
    -                 from collections import deque
  - line:             except Exception:

- environment.py:1118  — tag=LEGACY_FEATURE_API, match=`_recent_rewards`
  - context before:
    -                 from collections import deque
    -                 self._recent_rewards = deque(maxlen=20)
    -             except Exception:
  - line:         self.done = False

- environment.py:1346  — tag=LEGACY_FEATURE_API, match=`_wip(`
  - context before:
    -         self._completed_now_cache = 0
    -         sim_t = float(self.env.now) if hasattr(self, 'env') else 1.0
    -         avg_wait = float(self.total_wait_time) / max(1.0, sim_t)
  - line:         try:

- environment.py:1354  — tag=LEGACY_FEATURE_API, match=`_recent_rewards`
  - context before:
    -             idle_ops = 0
    -         reward = (self.alpha * completed) - (self.beta * avg_wait) - (self.gamma * wip) - (self.delta * float(idle_ops))
    -         try:
  - line:         except Exception as e:

- environment.py:1369  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                 now_t = float(getattr(self.env, 'now', 0.0))
    -                 try:
    -                     if job.mark_completed(now_t):
  - line:                         self._completed_now_cache += 1

- environment.py:1395  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                             timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
    -                             try:
    -                                 with open(timeline_path, 'a', encoding='utf-8') as tf:
  - line:                             except Exception:

- environment.py:1406  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                             os.makedirs(hist_dir, exist_ok=True)
    -                             with open(os.path.join(hist_dir, 'scheduling_timeline.txt'), 'a', encoding='utf-8') as tfs:
    -                                 active_jobs = [j for j in (getattr(self, 'jobs', []) or []) if not getattr(j, 'finished', False)]
  - line:                                 pending_jobs = [j for j in (getattr(self, 'jobs', []) or []) if not getattr(j, 'is_active', False) and not getattr(j, 'finished', False)]

- environment.py:1409  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                                 completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
    -                                 pending_jobs = [j for j in (getattr(self, 'jobs', []) or []) if not getattr(j, 'is_active', False) and not getattr(j, 'finished', False)]
    -                                 total_jobs = len(getattr(self, 'jobs', []) or [])
  - line:                                 try:

- environment.py:1423  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                             timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
    -                             try:
    -                                 with open(timeline_path, 'a', encoding='utf-8') as tf:
  - line:                             except Exception:

- environment.py:1874  — tag=LEGACY_FEATURE_API, match=`job.remaining_time`
  - context before:
    -                                 job.wait_time += wait_dur
    -                                 self.total_wait_time += wait_dur
    -                             op_start = float(self.env.now)
  - line:                             # assign and mark busy via Operator.assign_job()

- environment.py:2029  — tag=LEGACY_FEATURE_API, match=`job.remaining_time`
  - context before:
    -                                             job.wait_time += wait_dur
    -                                             self.total_wait_time += wait_dur
    -                                         op_start = float(self.env.now)
  - line:                                         try:

- environment.py:2102  — tag=LEGACY_FEATURE_API, match=`job.remaining_time`
  - context before:
    -                     return
    - 
    -             job.current_op_idx += 1
  - line:             if job.current_op_idx >= len(job.operations):

- environment.py:2108  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                 now_t = float(getattr(self.env, 'now', 0.0))
    -                 try:
    -                     if job.mark_completed(now_t):
  - line:                         self._completed_now_cache += 1

- environment.py:2162  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                                     timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
    -                                     try:
    -                                         with open(timeline_path, 'a', encoding='utf-8') as tf:
  - line:                                     except Exception:

- environment.py:2376  — tag=LEGACY_FEATURE_API, match=`_wip(`
  - context before:
    -         return row
    - 
    -     # ---------------- Helpers -----------------
  - line:         return sum(1 for j in self.jobs if not j.finished)

- environment.py:2390  — tag=LEGACY_FEATURE_API, match=`_wip(`
  - context before:
    -         """Return canonical number of currently active jobs (public helper).
    - 
    -         Prefer an explicit `active_jobs` list if present; otherwise fall back
  - line:         elsewhere in the codebase.

- environment.py:2399  — tag=LEGACY_FEATURE_API, match=`_wip(`
  - context before:
    -         except Exception:
    -             pass
    -         try:
  - line:         except Exception:

- environment.py:2404  — tag=LEGACY_FEATURE_API, match=`_util_machines(`
  - context before:
    -             return 0
    - 
    -     # ---------------- Utilization helpers used by utils/env_obs.py -------
  - line:         """Return a [0,1] utilization estimate for machines."""

- environment.py:2423  — tag=LEGACY_FEATURE_API, match=`_util_ops(`
  - context before:
    -             logging.getLogger(__name__).exception("Exception caught", exc_info=True)
    -             return 0.0
    - 
  - line:         """Return a [0,1] utilization estimate for operator groups."""

- environment.py:2577  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                             timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
    -                             try:
    -                                 with open(timeline_path, 'a', encoding='utf-8') as tf:
  - line:                                     tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")

- environment.py:2578  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                             try:
    -                                 with open(timeline_path, 'a', encoding='utf-8') as tf:
    -                                     tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
  - line:                             except Exception:

- environment.py:2589  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                             os.makedirs(hist_dir, exist_ok=True)
    -                             with open(os.path.join(hist_dir, 'scheduling_timeline.txt'), 'a', encoding='utf-8') as tf2:
    -                                 active_jobs = [j for j in (getattr(self, 'jobs', []) or []) if not getattr(j, 'finished', False)]
  - line:                                 pending_jobs = [j for j in (getattr(self, 'jobs', []) or []) if not getattr(j, 'is_active', False) and not getattr(j, 'finished', False)]

- environment.py:2592  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                                 completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
    -                                 pending_jobs = [j for j in (getattr(self, 'jobs', []) or []) if not getattr(j, 'is_active', False) and not getattr(j, 'finished', False)]
    -                                 total_jobs = len(getattr(self, 'jobs', []) or [])
  - line:                                 # Optional warn

- environment.py:2623  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                                 timeline_path = os.path.join(hist_dir, 'scheduling_timeline.txt')
    -                                 try:
    -                                     with open(timeline_path, 'a', encoding='utf-8') as tf:
  - line:                                 except Exception:

- environment.py:2634  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                                     os.makedirs(hist_dir, exist_ok=True)
    -                                     with open(os.path.join(hist_dir, 'scheduling_timeline.txt'), 'a', encoding='utf-8') as tfq:
    -                                         active_jobs = [j for j in (getattr(self, 'jobs', []) or []) if not getattr(j, 'finished', False)]
  - line:                                         pending_jobs = [j for j in (getattr(self, 'jobs', []) or []) if not getattr(j, 'is_active', False) and not getattr(j, 'finished', False)]

- environment.py:2637  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -                                         completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
    -                                         pending_jobs = [j for j in (getattr(self, 'jobs', []) or []) if not getattr(j, 'is_active', False) and not getattr(j, 'finished', False)]
    -                                         total_jobs = len(getattr(self, 'jobs', []) or [])
  - line:                                 except Exception:

- environment.py:2765  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -           - avg_machine_utilization: fraction [0,1] averaged across machines and time
    -           - avg_operator_utilization: fraction [0,1] averaged across operators and time
    -           - average_makespan: makespan observed in this episode (seconds)
  - line:         """

- environment.py:2911  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -             # average wait per completed job
    -             try:
    -                 total_wait = float(getattr(self, 'total_wait_time', 0.0))
  - line:                 avg_wait_time = float(total_wait) / max(1.0, float(completed))

- utils/env_obs.py:26  — tag=LEGACY_FEATURE_API, match=`time_norm`
  - context before:
    -     # 0 current_op_type_norm      -> job.current operation type  / env.n_operation_types
    -     # 1 total_ops_count_norm      -> len(job.operations)           / env.max_operations_per_job
    -     # 2 remaining_ops_count_norm  -> job.remaining_ops()          / env.max_operations_per_job
  - line:     # 4 n_jobs_active_norm        -> env.active_jobs_count()      / env.max_jobs

- utils/env_obs.py:56  — tag=LEGACY_FEATURE_API, match=`time_norm`
  - context before:
    -     remaining_ops_val = float(len(job.operations) - int(job.current_op_idx))
    -     remaining_ops_count_norm = np.clip(remaining_ops_val / div_ops, 0.0, 1.0)
    - 
  - line:     # --- wait_time_norm ---

- utils/env_obs.py:57  — tag=LEGACY_FEATURE_API, match=`time_norm`
  - context before:
    -     remaining_ops_count_norm = np.clip(remaining_ops_val / div_ops, 0.0, 1.0)
    - 
    -     # --- wait_time_norm ---
  - line:     wait_time_val = float(job.wait_time)

- utils/env_obs.py:60  — tag=LEGACY_FEATURE_API, match=`time_norm`
  - context before:
    -     # --- wait_time_norm ---
    -     wait_time_val = float(job.wait_time)
    -     div_wait = float(env.max_wait_time)
  - line: 

- utils/env_obs.py:75  — tag=LEGACY_FEATURE_API, match=`time_norm`
  - context before:
    -         current_op_type_norm,
    -         total_ops_count_norm,
    -         remaining_ops_count_norm,
  - line:         n_jobs_active_norm,

- utils/env_obs.py:90  — tag=LEGACY_FEATURE_API, match=`_util_machines(`
  - context before:
    - 
    -     Mirrors `environment._build_state_vector` logic.
    -     """
  - line:     # avg_wait: normalize total wait by (env.env.now * env.avg_wait_scale)

- utils/env_obs.py:102  — tag=LEGACY_FEATURE_API, match=`_wip(`
  - context before:
    -         max_jobs_div = float(getattr(env, 'max_jobs', None) or 50.0)
    -     except Exception:
    -         max_jobs_div = 50.0
  - line:     try:

- utils/env_obs.py:104  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -         max_jobs_div = 50.0
    -     wip = np.clip(env._wip() / max_jobs_div, 0, 1)
    -     try:
  - line:     except Exception:

- utils/env_obs.py:106  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
  - context before:
    -     try:
    -         completed = np.clip(env.completed_jobs / float(getattr(env, 'episode_limit', 1)), 0, 1)
    -     except Exception:
  - line:     # reward_recent: normalize using env.reward_norm_divisor when present

- utils/env_obs.py:107  — tag=LEGACY_FEATURE_API, match=`reward_norm`
  - context before:
    -         completed = np.clip(env.completed_jobs / float(getattr(env, 'episode_limit', 1)), 0, 1)
    -     except Exception:
    -         completed = np.clip(env.completed_jobs / 200.0, 0, 1)
  - line:     try:

- utils/env_obs.py:109  — tag=LEGACY_FEATURE_API, match=`reward_norm`
  - context before:
    -         completed = np.clip(env.completed_jobs / 200.0, 0, 1)
    -     # reward_recent: normalize using env.reward_norm_divisor when present
    -     try:
  - line:     except Exception:

- utils/env_obs.py:112  — tag=LEGACY_FEATURE_API, match=`_recent_rewards`
  - context before:
    -         reward_div = float(getattr(env, 'reward_norm_divisor', None) or 10.0)
    -     except Exception:
    -         reward_div = 10.0
  - line:     idle_ratio = 1.0 - 0.5 * (util_m + util_o)

- CODE_HYGIENE_REPORT.md:158  — tag=OBS_BUILDERS, match=`observations and global state vector`
  - context before:
    - Imported by: `main.py`, Runner, tools/tests
    - 
    - ## File: utils/env_obs.py
  - line: Inputs: `env`, `job` objects

- docs/module_dependency_audit.md:254  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    - 
    - utils/env_obs.py -> imports:
    -   defines:
  - line:     - func build_state_vector  (used elsewhere: yes)

- docs/module_dependency_audit.md:326  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    - 
    - tests/test_env_obs.py -> 🟢 Candidate for reintegration
    -   defines:
  - line:     - func test_build_state_vector_shape

- environment.py:60  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    - import simpy
    - import os
    - import json
  - line: try:

- environment.py:329  — tag=OBS_BUILDERS, match=`observation builder`
  - context before:
    -             self.state_dim = 64
    - 
    -         # ---------------- Normalization reference attributes ----------------
  - line:         # dynamically-normalized features (replace hard-coded divisors).

- environment.py:410  — tag=OBS_BUILDERS, match=`observation builder`
  - context before:
    - 
    -         # Strict validation of canonical normalization attributes.
    -         # These must be present and strictly positive when using the
  - line:         # configuration errors (no fallbacks here by design).

- environment.py:1464  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    -                 logging.getLogger(__name__).exception("Exception caught", exc_info=True)
    -                 decision_item = {
    -                     'job_id': job.id,
  - line:                     'avail_row': self._avail_row_for_job(job),

- environment.py:2187  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    - 
    -     # ---------------- Observation / State / Avail -----------------
    -     def _build_all_agent_obs(self):
  - line:         # will raise ImportError at module import time so errors are explicit.

- environment.py:2189  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    -     def _build_all_agent_obs(self):
    -         # Strict delegation to utils.env_obs.build_agent_obs. Missing helper
    -         # will raise ImportError at module import time so errors are explicit.
  - line: 

- environment.py:2191  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    -         # will raise ImportError at module import time so errors are explicit.
    -         return [build_agent_obs(self, j) for j in self.jobs]
    - 
  - line:         # Strict delegation to canonical helper. Let exceptions propagate for

- environment.py:2194  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    -     def _build_agent_obs(self, job: JobAgent):
    -         # Strict delegation to canonical helper. Let exceptions propagate for
    -         # clearer debugging when the helper is missing or fails.
  - line: 

- tests/test_env_obs.py:6  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    - from utils import env_obs
    - 
    - 
  - line:     env = MASAEnv(num_jobs=4, episode_limit=200, seed=123, config_path='configs/env_no_arrival.yaml')

- tests/test_env_obs.py:11  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    -     obs, info = env.reset()
    -     # pick the first non-finished job
    -     job = env.jobs[0]
  - line:     aobs2 = env_obs.build_agent_obs(env, job)

- tests/test_env_obs.py:12  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    -     # pick the first non-finished job
    -     job = env.jobs[0]
    -     aobs = env._build_agent_obs(job)
  - line:     assert isinstance(aobs, np.ndarray)

- tests/test_observation_equivalence.py:18  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    - 
    - def test_env_and_helper_agent_obs_equivalence():
    -     import numpy as _np
  - line: 

- tests/test_observation_equivalence.py:21  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    -     from utils.env_obs import build_agent_obs
    - 
    -     env = MASAEnv(auto_build=False, auto_start_arrivals=False)
  - line:     # Use a simple operation tuple: (op_type, allowed_machine_indices, per_machine_durations)

- tests/test_observation_equivalence.py:29  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    -     # Ensure job was added
    -     assert job is not None
    - 
  - line:     a_helper = build_agent_obs(env, job)

- tests/test_observation_equivalence.py:30  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    -     assert job is not None
    - 
    -     a_env = env._build_agent_obs(job)
  - line:     assert _np.allclose(_np.asarray(a_env), _np.asarray(a_helper))

- tests/test_observation_shapes.py:10  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    -     env = MASAEnv(num_jobs=1, num_operators=1, num_wcs=1, seed=0, auto_build=True)
    -     # Ensure at least one job exists
    -     assert len(env.jobs) >= 1
  - line:     state = env._build_state_vector()

- utils/env_obs.py:2  — tag=OBS_BUILDERS, match=`observation and state builder`
  - context before:
    - """utils/env_obs.py
  - line: 

- utils/env_obs.py:5  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    - Extracted observation and state builder helpers for MASAEnv.
    - 
    - Provides:
  - line:   - build_state_vector(env) -> np.ndarray (state_dim,)

- utils/env_obs.py:16  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    - from typing import Any
    - 
    - 
  - line:     """Build the per-agent observation vector.

- utils/env_obs.py:17  — tag=OBS_BUILDERS, match=`observation vector`
  - context before:
    - 
    - 
    - def build_agent_obs(env: Any, job: Any) -> np.ndarray:
  - line: 

- utils/env_obs.py:19  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    - def build_agent_obs(env: Any, job: Any) -> np.ndarray:
    -     """Build the per-agent observation vector.
    - 
  - line:     """

- utils/env_obs.py:21  — tag=OBS_BUILDERS, match=`observation vector`
  - context before:
    - 
    -     Mirrors the logic previously embedded in `environment._build_agent_obs`.
    -     """
  - line:     # Order:

- utils/workcenter.py:339  — tag=OBS_BUILDERS, match=`build_agent_obs`
  - context before:
    - 
    -         decision_item = {
    -             "job_id": getattr(job, 'id', getattr(job, 'agent_id', None)),
  - line:             "avail_row": env._avail_row_for_job(job),

- environment.py:333  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -         # dynamically-normalized features (replace hard-coded divisors).
    -         # Prefer explicit values provided via `args` or kwargs; otherwise
    -         # infer from metadata; finally fall back to conservative constants.
  - line:         try:

- environment.py:335  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -         # infer from metadata; finally fall back to conservative constants.
    -         # n_operation_types: number of distinct operation types (categorical count)
    -         try:
  - line:         except Exception:

- environment.py:337  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -         try:
    -             self.n_operation_types = int(getattr(args, 'n_operation_types', None) or kwargs.get('n_operation_types', None))
    -         except Exception:
  - line:         if not self.n_operation_types:

- environment.py:338  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -             self.n_operation_types = int(getattr(args, 'n_operation_types', None) or kwargs.get('n_operation_types', None))
    -         except Exception:
    -             self.n_operation_types = None
  - line:             # Attempt simple inference from workcenters_meta if available

- environment.py:348  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -                         caps.extend(list(md.get('capabilities', []) or []))
    -                     except Exception:
    -                         continue
  - line:             except Exception:

- environment.py:350  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -                         continue
    -                 self.n_operation_types = int(max(caps) + 1) if caps else None
    -             except Exception:
  - line:         # fallback

- environment.py:352  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -             except Exception:
    -                 self.n_operation_types = None
    -         # fallback
  - line:             self.n_operation_types = getattr(args, 'n_operation_types', None) or 10

- environment.py:353  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -                 self.n_operation_types = None
    -         # fallback
    -         if not self.n_operation_types or int(self.n_operation_types) <= 0:
  - line: 

- environment.py:355  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -         if not self.n_operation_types or int(self.n_operation_types) <= 0:
    -             self.n_operation_types = getattr(args, 'n_operation_types', None) or 10
    - 
  - line:         try:

- environment.py:357  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    - 
    -         # max_operations_per_job: used to normalize total / remaining ops
    -         try:
  - line:         except Exception:

- environment.py:359  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -         try:
    -             self.max_operations_per_job = int(getattr(args, 'job_max_ops', None) or kwargs.get('max_operations_per_job', None))
    -         except Exception:
  - line:         if not self.max_operations_per_job:

- environment.py:360  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -             self.max_operations_per_job = int(getattr(args, 'job_max_ops', None) or kwargs.get('max_operations_per_job', None))
    -         except Exception:
    -             self.max_operations_per_job = None
  - line:             try:

- environment.py:364  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -             try:
    -                 gen = getattr(self, 'job_generator', None)
    -                 if gen is not None and hasattr(gen, 'default_max_ops'):
  - line:             except Exception:

- environment.py:366  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -                 if gen is not None and hasattr(gen, 'default_max_ops'):
    -                     self.max_operations_per_job = int(getattr(gen, 'default_max_ops'))
    -             except Exception:
  - line:         if not self.max_operations_per_job or int(self.max_operations_per_job) <= 0:

- environment.py:367  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -                     self.max_operations_per_job = int(getattr(gen, 'default_max_ops'))
    -             except Exception:
    -                 self.max_operations_per_job = None
  - line:             self.max_operations_per_job = 10

- environment.py:368  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -             except Exception:
    -                 self.max_operations_per_job = None
    -         if not self.max_operations_per_job or int(self.max_operations_per_job) <= 0:
  - line: 

- environment.py:370  — tag=CANONICAL_SOURCES, match=`max_wait_time`
  - context before:
    -         if not self.max_operations_per_job or int(self.max_operations_per_job) <= 0:
    -             self.max_operations_per_job = 10
    - 
  - line:         try:

- environment.py:372  — tag=CANONICAL_SOURCES, match=`max_wait_time`
  - context before:
    - 
    -         # max_wait_time: used to normalize job.wait_time
    -         try:
  - line:         except Exception:

- environment.py:374  — tag=CANONICAL_SOURCES, match=`max_wait_time`
  - context before:
    -         try:
    -             self.max_wait_time = float(getattr(args, 'max_wait_time', None) or kwargs.get('max_wait_time', None))
    -         except Exception:
  - line:         if not self.max_wait_time or self.max_wait_time <= 0.0:

- environment.py:375  — tag=CANONICAL_SOURCES, match=`max_wait_time`
  - context before:
    -             self.max_wait_time = float(getattr(args, 'max_wait_time', None) or kwargs.get('max_wait_time', None))
    -         except Exception:
    -             self.max_wait_time = None
  - line:             self.max_wait_time = getattr(args, 'max_wait_time', 50.0)

- environment.py:376  — tag=CANONICAL_SOURCES, match=`max_wait_time`
  - context before:
    -         except Exception:
    -             self.max_wait_time = None
    -         if not self.max_wait_time or self.max_wait_time <= 0.0:
  - line: 

- environment.py:400  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -         if not self.avg_wait_scale or float(self.avg_wait_scale) <= 0.0:
    -             self.avg_wait_scale = getattr(args, 'avg_wait_scale', 10.0)
    - 
  - line:         # Enforce strict derivation: must be provided via args.n_agents and > 0.

- environment.py:403  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -         # max_jobs (job capacity): canonical link to args.n_agents (active capacity)
    -         # Enforce strict derivation: must be provided via args.n_agents and > 0.
    -         if args is not None and hasattr(args, 'n_agents') and int(getattr(args, 'n_agents')) > 0:
  - line:         else:

- environment.py:405  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -         if args is not None and hasattr(args, 'n_agents') and int(getattr(args, 'n_agents')) > 0:
    -             self.max_jobs = int(getattr(args, 'n_agents'))
    -         else:
  - line:         # --------------------------------------------------------------------

- environment.py:412  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -         # These must be present and strictly positive when using the
    -         # observation builders that rely on them. Fail fast to surface
    -         # configuration errors (no fallbacks here by design).
  - line:             raise ValueError("Invalid environment configuration: max_jobs must be > 0")

- environment.py:413  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -         # observation builders that rely on them. Fail fast to surface
    -         # configuration errors (no fallbacks here by design).
    -         if getattr(self, 'max_jobs', None) is None or int(self.max_jobs) <= 0:
  - line:         if getattr(self, 'max_operations_per_job', None) is None or int(self.max_operations_per_job) <= 0:

- environment.py:414  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -         # configuration errors (no fallbacks here by design).
    -         if getattr(self, 'max_jobs', None) is None or int(self.max_jobs) <= 0:
    -             raise ValueError("Invalid environment configuration: max_jobs must be > 0")
  - line:             raise ValueError("Invalid environment configuration: max_operations_per_job must be > 0")

- environment.py:415  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -         if getattr(self, 'max_jobs', None) is None or int(self.max_jobs) <= 0:
    -             raise ValueError("Invalid environment configuration: max_jobs must be > 0")
    -         if getattr(self, 'max_operations_per_job', None) is None or int(self.max_operations_per_job) <= 0:
  - line:         if getattr(self, 'n_operation_types', None) is None or int(self.n_operation_types) <= 0:

- environment.py:416  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -             raise ValueError("Invalid environment configuration: max_jobs must be > 0")
    -         if getattr(self, 'max_operations_per_job', None) is None or int(self.max_operations_per_job) <= 0:
    -             raise ValueError("Invalid environment configuration: max_operations_per_job must be > 0")
  - line:             raise ValueError("Invalid environment configuration: n_operation_types must be > 0")

- environment.py:417  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -         if getattr(self, 'max_operations_per_job', None) is None or int(self.max_operations_per_job) <= 0:
    -             raise ValueError("Invalid environment configuration: max_operations_per_job must be > 0")
    -         if getattr(self, 'n_operation_types', None) is None or int(self.n_operation_types) <= 0:
  - line:         if getattr(self, 'max_wait_time', None) is None or float(self.max_wait_time) <= 0.0:

- environment.py:418  — tag=CANONICAL_SOURCES, match=`max_wait_time`
  - context before:
    -             raise ValueError("Invalid environment configuration: max_operations_per_job must be > 0")
    -         if getattr(self, 'n_operation_types', None) is None or int(self.n_operation_types) <= 0:
    -             raise ValueError("Invalid environment configuration: n_operation_types must be > 0")
  - line:             raise ValueError("Invalid environment configuration: max_wait_time must be > 0")

- environment.py:419  — tag=CANONICAL_SOURCES, match=`max_wait_time`
  - context before:
    -         if getattr(self, 'n_operation_types', None) is None or int(self.n_operation_types) <= 0:
    -             raise ValueError("Invalid environment configuration: n_operation_types must be > 0")
    -         if getattr(self, 'max_wait_time', None) is None or float(self.max_wait_time) <= 0.0:
  - line: 

- environment.py:473  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -                 self.interarrival_time = 4.0
    - 
    -         # maximum number of jobs to generate via the dynamic job generator
  - line:         # overwrite it here; dynamic generation bounds (num_generate_jobs)

- environment.py:2386  — tag=CANONICAL_SOURCES, match=`active_jobs_count`
  - context before:
    -             logging.getLogger(__name__).exception("Exception caught", exc_info=True)
    -             return True
    - 
  - line:         """Return canonical number of currently active jobs (public helper).

- utils/env_obs.py:23  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -     """
    -     # Build 6-element observation vector using env-level normalization refs.
    -     # Order:
  - line:     # 1 total_ops_count_norm      -> len(job.operations)           / env.max_operations_per_job

- utils/env_obs.py:24  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -     # Build 6-element observation vector using env-level normalization refs.
    -     # Order:
    -     # 0 current_op_type_norm      -> job.current operation type  / env.n_operation_types
  - line:     # 2 remaining_ops_count_norm  -> job.remaining_ops()          / env.max_operations_per_job

- utils/env_obs.py:25  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -     # Order:
    -     # 0 current_op_type_norm      -> job.current operation type  / env.n_operation_types
    -     # 1 total_ops_count_norm      -> len(job.operations)           / env.max_operations_per_job
  - line:     # 3 wait_time_norm            -> job.wait_time                / env.max_wait_time

- utils/env_obs.py:26  — tag=CANONICAL_SOURCES, match=`max_wait_time`
  - context before:
    -     # 0 current_op_type_norm      -> job.current operation type  / env.n_operation_types
    -     # 1 total_ops_count_norm      -> len(job.operations)           / env.max_operations_per_job
    -     # 2 remaining_ops_count_norm  -> job.remaining_ops()          / env.max_operations_per_job
  - line:     # 4 n_jobs_active_norm        -> env.active_jobs_count()      / env.max_jobs

- utils/env_obs.py:27  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -     # 1 total_ops_count_norm      -> len(job.operations)           / env.max_operations_per_job
    -     # 2 remaining_ops_count_norm  -> job.remaining_ops()          / env.max_operations_per_job
    -     # 3 wait_time_norm            -> job.wait_time                / env.max_wait_time
  - line:     # 5 finished_flag             -> 1.0 if job.finished else 0.0

- utils/env_obs.py:31  — tag=CANONICAL_SOURCES, match=`current_operation`
  - context before:
    -     # 5 finished_flag             -> 1.0 if job.finished else 0.0
    - 
    -     # --- current_op_type_norm ---
  - line:     cur_op = job.current_operation()

- utils/env_obs.py:32  — tag=CANONICAL_SOURCES, match=`current_operation`
  - context before:
    - 
    -     # --- current_op_type_norm ---
    -     # numerator: job.current_operation().type or job.operations[job.current_op_idx][0]
  - line:     if cur_op is not None:

- utils/env_obs.py:41  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -     else:
    -         op_type_val = float(job.operations[job.current_op_idx][0])
    - 
  - line:     div_op_types = float(env.n_operation_types)

- utils/env_obs.py:42  — tag=CANONICAL_SOURCES, match=`n_operation_types`
  - context before:
    -         op_type_val = float(job.operations[job.current_op_idx][0])
    - 
    -     # divisor: env.n_operation_types (explicit canonical source)
  - line:     current_op_type_norm = np.clip(op_type_val / div_op_types, 0.0, 1.0)

- utils/env_obs.py:48  — tag=CANONICAL_SOURCES, match=`max_operations_per_job`
  - context before:
    -     # --- total_ops_count_norm ---
    -     # --- total_ops_count_norm ---
    -     total_ops = float(len(job.operations))
  - line:     total_ops_count_norm = np.clip(total_ops / div_ops, 0.0, 1.0)

- utils/env_obs.py:51  — tag=CANONICAL_SOURCES, match=`remaining_ops`
  - context before:
    -     div_ops = float(env.max_operations_per_job)
    -     total_ops_count_norm = np.clip(total_ops / div_ops, 0.0, 1.0)
    - 
  - line:     # --- remaining_ops_count_norm ---

- utils/env_obs.py:52  — tag=CANONICAL_SOURCES, match=`remaining_ops`
  - context before:
    -     total_ops_count_norm = np.clip(total_ops / div_ops, 0.0, 1.0)
    - 
    -     # --- remaining_ops_count_norm ---
  - line:     remaining_ops_val = float(len(job.operations) - int(job.current_op_idx))

- utils/env_obs.py:53  — tag=CANONICAL_SOURCES, match=`remaining_ops`
  - context before:
    - 
    -     # --- remaining_ops_count_norm ---
    -     # --- remaining_ops_count_norm ---
  - line:     remaining_ops_count_norm = np.clip(remaining_ops_val / div_ops, 0.0, 1.0)

- utils/env_obs.py:54  — tag=CANONICAL_SOURCES, match=`remaining_ops`
  - context before:
    -     # --- remaining_ops_count_norm ---
    -     # --- remaining_ops_count_norm ---
    -     remaining_ops_val = float(len(job.operations) - int(job.current_op_idx))
  - line: 

- utils/env_obs.py:59  — tag=CANONICAL_SOURCES, match=`max_wait_time`
  - context before:
    -     # --- wait_time_norm ---
    -     # --- wait_time_norm ---
    -     wait_time_val = float(job.wait_time)
  - line:     wait_time_norm = np.clip(wait_time_val / div_wait, 0.0, 1.0)

- utils/env_obs.py:64  — tag=CANONICAL_SOURCES, match=`active_jobs_count`
  - context before:
    - 
    -     # --- n_jobs_active_norm ---
    -     # --- n_jobs_active_norm ---
  - line:     div_jobs = float(env.max_jobs)

- utils/env_obs.py:65  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -     # --- n_jobs_active_norm ---
    -     # --- n_jobs_active_norm ---
    -     active_jobs_val = float(env.active_jobs_count())
  - line:     n_jobs_active_norm = np.clip(active_jobs_val / div_jobs, 0.0, 1.0)

- utils/env_obs.py:74  — tag=CANONICAL_SOURCES, match=`remaining_ops`
  - context before:
    -     obs = np.array([
    -         current_op_type_norm,
    -         total_ops_count_norm,
  - line:         wait_time_norm,

- utils/env_obs.py:97  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -     except Exception:
    -         avg_wait_div = 10.0
    -     avg_wait = np.clip(env.total_wait_time / max(1, env.env.now * avg_wait_div), 0, 1)
  - line:     try:

- utils/env_obs.py:99  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -     avg_wait = np.clip(env.total_wait_time / max(1, env.env.now * avg_wait_div), 0, 1)
    -     # wip normalized by env.max_jobs when available
    -     try:
  - line:     except Exception:

- utils/env_obs.py:101  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -     try:
    -         max_jobs_div = float(getattr(env, 'max_jobs', None) or 50.0)
    -     except Exception:
  - line:     wip = np.clip(env._wip() / max_jobs_div, 0, 1)

- utils/env_obs.py:102  — tag=CANONICAL_SOURCES, match=`max_jobs`
  - context before:
    -         max_jobs_div = float(getattr(env, 'max_jobs', None) or 50.0)
    -     except Exception:
    -         max_jobs_div = 50.0
  - line:     try:

## Section B — Medium (likely used in downstream logic)

- MARL/common/arguments.py:113  — tag=LEGACY_FEATURE_API, match=`reward_norm`
- MARL/policy/qmix.py:78  — tag=LEGACY_FEATURE_API, match=`reward_norm`
- MARL/runner.py:1193  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- MARL/runner.py:1244  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- MARL/runner.py:1246  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- MARL/runner.py:1493  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- MARL/runner.py:1533  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- MARL/runner.py:1534  — tag=LEGACY_FEATURE_API, match=`_util_machines(`
- MARL/runner.py:1535  — tag=LEGACY_FEATURE_API, match=`_util_ops(`
- MARL/runner.py:1549  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- MARL/runner.py:1574  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- MARL/runner.py:1880  — tag=LEGACY_FEATURE_API, match=`_util_machines(`
- MARL/runner.py:1881  — tag=LEGACY_FEATURE_API, match=`_util_ops(`
- my_data_and_graph/metrics.py:108  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- my_data_and_graph/metrics.py:170  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- my_data_and_graph/metrics.py:198  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- my_data_and_graph/metrics.py:199  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- tmp_smoke_test.py:14  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- tmp_test_gantt_decision_trace.py:35  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- utils/gantt.py:644  — tag=LEGACY_FEATURE_API, match=`completed_jobs`
- MARL/common/replay_buffer.py:382  — tag=ELEVEN_DIM_SHAPES, match=`obs_dim = 11`
- MARL/policy/qmix.py:5  — tag=ELEVEN_DIM_SHAPES, match=`11D`
- MARL/runner.py:795  — tag=ELEVEN_DIM_SHAPES, match=`11D`
- SYSTEM_STATUS.md:72  — tag=ELEVEN_DIM_SHAPES, match=`obs_dim=11`

## Section C — Low / docs
Comments or docs that mention old 11-D obs; non-functional but should be updated for clarity.


## New 6-D feature canonical source map
- current_op_type_norm → n_operation_types / job.current_operation or similar
- total_ops_count_norm → max_operations_per_job / len(job.operations)
- remaining_ops_count_norm → max_operations_per_job / job.remaining_ops() or current index
- wait_time_norm → max_wait_time / job.wait_time
- n_jobs_active_norm → max_jobs / env.active_jobs_count()
- finished_flag → job.finished or job.is_completed()


## Proposed changes (preview only, DO NOT APPLY)
For each HIGH risk hit, consider removing references to legacy API in observation builders and replace with canonical attributes listed above. If helper functions are used elsewhere, keep helper but remove from obs paths.