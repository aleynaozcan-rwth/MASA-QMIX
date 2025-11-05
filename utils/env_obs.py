"""utils/env_obs.py
Extracted observation and state builder helpers for MASAEnv.

Provides:
  - build_agent_obs(env, job) -> np.ndarray (obs_dim_agent,)
  - build_state_vector(env) -> np.ndarray (state_dim,)

These are pure helpers that read the environment's public attributes and
produce the exact same arrays previously built inside `environment._build_*`.
"""
from __future__ import annotations
import numpy as np
from typing import Any


def build_agent_obs(env: Any, job: Any) -> np.ndarray:
    """Build the per-agent observation vector.

    Mirrors the logic previously embedded in `environment._build_agent_obs`.
    """
    progress = job.progress_ratio()
    wait_norm = np.clip(job.wait_time / 50.0, 0, 1)
    rem_norm = np.clip(job.remaining_time / 20.0, 0, 1)
    util_m, util_o = env._util_machines(), env._util_ops()
    wip_norm = np.clip(env._wip() / 80.0, 0, 1)
    time_norm = np.clip(env.env.now / env.episode_limit, 0, 1)
    reward_norm = np.clip((np.mean(env._recent_rewards) if env._recent_rewards else 0) / 10.0, 0, 1)
    completed_norm = np.clip(env.completed_jobs / 100.0, 0, 1)
    n_ops_norm = np.clip(len(job.operations) / 10.0, 0, 1)
    finished_flag = 1.0 if job.finished else 0.0

    obs = np.array([
        progress, wait_norm, rem_norm, util_m, util_o, wip_norm,
        time_norm, reward_norm, completed_norm, n_ops_norm, finished_flag
    ], dtype=np.float32)
    # Ensure the returned observation has exactly env.obs_dim_agent elements
    obs = np.pad(obs, (0, max(0, getattr(env, 'obs_dim_agent', obs.shape[0]) - len(obs))))[: getattr(env, 'obs_dim_agent', obs.shape[0])]
    return obs


def build_state_vector(env: Any) -> np.ndarray:
    """Build the global state vector (compact summary used by MARL code).

    Mirrors `environment._build_state_vector` logic.
    """
    util_m, util_o = env._util_machines(), env._util_ops()
    avg_wait = np.clip(env.total_wait_time / max(1, env.env.now * 10.0), 0, 1)
    wip = np.clip(env._wip() / 100.0, 0, 1)
    completed = np.clip(env.completed_jobs / 200.0, 0, 1)
    reward_recent = np.clip((np.mean(env._recent_rewards) if env._recent_rewards else 0) / 10.0, 0, 1)
    idle_ratio = 1.0 - 0.5 * (util_m + util_o)
    core = np.array([util_m, util_o, avg_wait, wip, completed, reward_recent, idle_ratio], dtype=np.float32)
    # Ensure the returned state vector has exactly env.state_dim elements
    core = np.pad(core, (0, max(0, getattr(env, 'state_dim', core.shape[0]) - len(core))))[: getattr(env, 'state_dim', core.shape[0])]
    return core
