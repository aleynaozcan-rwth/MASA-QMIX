"""utils/env_obs.py
Canonical observation and state helpers for MASAEnv.

This module implements a strict, minimal 7-element per-agent observation
compatible with the MASAEnv canonical attributes. All observations and state
vectors are normalized to [0, 1] with parametric scaling factors.

Observation layout (length 7, dtype float32, all normalized to [0, 1]):
 0. current_op_type           -> operation type index (normalized)
 1. total_operations          -> total ops in job (normalized)
 2. remaining_operations      -> ops remaining (normalized)
 3. wait_time                 -> individual job wait (normalized)
 4. theoretical_machine_count -> eligible machines (normalized)
 5. free_machine_count        -> available machines (normalized)
 6. n_jobs_active             -> concurrent jobs (normalized)

State vector layout (length 10, dtype float32, all normalized to [0, 1]):
 0-5. Job/ops counts (normalized)
 6-7. Utilizations (already [0, 1])
 8.   Cumulative wait time (normalized)
 9.   Episode progress (already [0, 1])

Normalization includes overflow detection and weak signal warnings.
"""
from __future__ import annotations 
import numpy as np 
from typing import Any 
import ast
import logging

LOG = logging.getLogger(__name__)


def _validate_normalization(raw_val: float, normalized_val: float, factor: float, 
                            name: str, env: Any, threshold_overflow: float = 0.95,
                            threshold_weak: float = 0.05) -> None:
    """
    Validate normalization and log warnings for overflow or weak signal.
    
    Args:
        raw_val: Original raw value before normalization
        normalized_val: Value after normalization and clipping
        factor: Normalization factor used
        name: Name of the dimension for logging
        env: Environment instance for checking logging flags
        threshold_overflow: Normalized value threshold for overflow warning (default: 0.95)
        threshold_weak: Normalized value threshold for weak signal info (default: 0.05)
    """
    args = getattr(env, 'args', None)
    if args is None:
        return
    
    log_overflow = getattr(args, 'norm_log_overflow', False)
    log_weak = getattr(args, 'norm_log_weak_signal', False)
    
    # Check for overflow (value too large, hitting ceiling)
    if log_overflow and normalized_val >= threshold_overflow and raw_val > 0:
        LOG.warning(
            f"[NORMALIZATION OVERFLOW] {name}={raw_val:.2f} / factor={factor:.1f} "
            f"= {normalized_val:.3f} (≥{threshold_overflow}). "
            f"Consider increasing normalization factor for {name}!"
        )
    
    # Check for weak signal (value too small, underutilized range)
    if log_weak and 0 < normalized_val < threshold_weak and raw_val > 0:
        LOG.info(
            f"[NORMALIZATION WEAK] {name}={raw_val:.2f} / factor={factor:.1f} "
            f"= {normalized_val:.3f} (<{threshold_weak}). "
            f"Consider decreasing normalization factor for {name}."
        )


def _require_positive (env :Any ,attr :str ):
    if not hasattr (env ,attr ):
        raise AttributeError (f"Environment missing required attribute: {attr }")
    val =getattr (env ,attr )
    try :
        f =float (val )
    except Exception :
        raise ValueError (f"Environment attribute {attr } is not a number: {val }")
    if f <=0.0 :
        raise ValueError (f"Environment attribute {attr } must be > 0. Found: {val }")
    return f 


def build_agent_obs(env: Any, job: Any, job_index: Any = None, allowed_machine_indices=None) -> np.ndarray:
    """
    Build and return the canonical 7-element per-agent observation with full normalization to [0, 1].
    
    Observation layout (all normalized to [0, 1]):
      0. current_op_type           -> operation type index (normalized)
      1. total_operations          -> total ops in job (normalized)
      2. remaining_operations      -> ops left (normalized)
      3. wait_time                 -> individual job wait (normalized)
      4. theoretical_machine_count -> eligible machines (normalized)
      5. free_machine_count        -> available machines (normalized)
      6. n_jobs_active             -> concurrent jobs (normalized)
    """
    
    # Get problem-specific parameters from environment
    max_jobs = float(getattr(env, 'max_jobs', 10))
    n_operation_types = int(getattr(env, 'n_operation_types', 10))
    max_operations_per_job = int(getattr(env, 'max_operations_per_job', 6))
    n_machines = len(getattr(env, 'machine_resources', [5]))  # Count actual machines
    
    # Get normalization hyperparameters/multipliers from args (with fallback defaults)
    args = getattr(env, 'args', None)
    if args is not None:
        norm_job_wait_max = float(getattr(args, 'norm_job_wait_max', 80.0))
        norm_active_mult = float(getattr(args, 'norm_active_multiplier', 1.5))
    else:
        # Fallback defaults if args not available
        norm_job_wait_max = 80.0
        norm_active_mult = 1.5
    
    # Calculate ADAPTIVE normalization factors (based on actual problem size)
    NORM_OP_TYPE_MAX = float(n_operation_types)  # Use actual number of op types
    NORM_OPS_PER_JOB = float(max_operations_per_job)  # Use actual max ops per job
    NORM_MACHINES_MAX = float(n_machines)  # Use actual number of machines
    MAX_ACTIVE_FACTOR = max_jobs * norm_active_mult

    # RAW VALUES COMPUTATION
    op = job.current_op()
    if op is not None:
        current_op_type = float(op[0])
    else:
        last_op = job.operations[-1]
        current_op_type = float(last_op[0])

    # allowed_machine_indices parametresi öncelikli
    ami = allowed_machine_indices if allowed_machine_indices is not None else (op[1] if op is not None and len(op) > 1 else [])
    if isinstance(ami, str):
        import ast
        ami = ast.literal_eval(ami)
    if not isinstance(ami, (list, tuple)):
        ami = []
    theoretical_machine_count = float(len(ami))

    total_operations = float(len(job.operations))
    idx = int(getattr(job, 'current_op_idx', 0))
    remaining_operations = float(max(0, len(job.operations) - idx))
    wait_time = float(job.wait_time)
    avail_actions = env._build_avail_actions()
    free_machine_count = float(np.sum(avail_actions[job_index]))
    n_jobs_active = float(env.active_jobs_count())
    
    # NORMALIZE ALL ELEMENTS TO [0, 1] WITH CLIPPING
    obs_element_0 = np.clip(current_op_type / max(1.0, NORM_OP_TYPE_MAX), 0.0, 1.0)
    obs_element_1 = np.clip(total_operations / max(1.0, NORM_OPS_PER_JOB), 0.0, 1.0)
    obs_element_2 = np.clip(remaining_operations / max(1.0, NORM_OPS_PER_JOB), 0.0, 1.0)
    obs_element_3 = np.clip(wait_time / max(1.0, norm_job_wait_max), 0.0, 1.0)
    obs_element_4 = np.clip(theoretical_machine_count / max(1.0, NORM_MACHINES_MAX), 0.0, 1.0)
    obs_element_5 = np.clip(free_machine_count / max(1.0, NORM_MACHINES_MAX), 0.0, 1.0)
    obs_element_6 = np.clip(n_jobs_active / max(1.0, MAX_ACTIVE_FACTOR), 0.0, 1.0)
    
    # VALIDATION: Check for overflow/weak signal (only for key dimensions)
    # Note: We validate selectively to avoid log spam, focusing on critical dimensions
    _validate_normalization(wait_time, obs_element_3, norm_job_wait_max, 'obs[3]:wait_time', env)
    _validate_normalization(n_jobs_active, obs_element_6, MAX_ACTIVE_FACTOR, 'obs[6]:jobs_active', env)
    
    obs = np.array([
        obs_element_0,
        obs_element_1,
        obs_element_2,
        obs_element_3,  # CRITICAL: wait_time normalized
        obs_element_4,
        obs_element_5,
        obs_element_6,
    ], dtype=np.float32)

    if obs.shape[0] != 7:
        raise RuntimeError("Canonical observation must be length 7")

    return obs




def _count_processing_ops(env: Any) -> int:
    """Count how many operations are currently being processed."""
    count = 0
    for job in env.jobs:
        if not job.finished and hasattr(job, 'is_active') and job.is_active:
            # If job is active, its current op is being processed
            count += 1
    return count


def _calculate_machine_utilization(env: Any) -> float:
    """Calculate average machine utilization [0-1]."""
    if not hasattr(env, 'machine_resources') or not env.machine_resources:
        return 0.0
    
    total_busy = 0
    total_capacity = 0
    for machine_res in env.machine_resources:
        busy = len(machine_res.users)
        capacity = machine_res.capacity
        total_busy += busy
        total_capacity += capacity
    
    if total_capacity == 0:
        return 0.0
    return min(1.0, total_busy / total_capacity)


def _calculate_operator_utilization(env: Any) -> float:
    """Calculate average operator utilization [0-1]."""
    if not hasattr(env, 'operator_groups') or not env.operator_groups:
        return 0.0
    
    total_busy = 0
    total_capacity = 0
    for op_res in env.operator_groups:
        busy = len(op_res.users)
        capacity = op_res.capacity
        total_busy += busy
        total_capacity += capacity
    
    if total_capacity == 0:
        return 0.0
    return min(1.0, total_busy / total_capacity)


def build_state_vector(env: Any) -> np.ndarray:
    """Build 10-element global state vector with full normalization to [0, 1].
    
    State layout (length 10, dtype float32, ALL NORMALIZED to [0, 1]):
     0. n_jobs_arrived       -> total jobs arrived (normalized)
     1. n_jobs_processing    -> jobs currently active (normalized)
     2. n_jobs_waiting       -> jobs waiting (normalized)
     3. n_ops_arrived        -> total operations arrived (normalized)
     4. n_ops_processing     -> operations being processed (normalized)
     5. n_ops_waiting        -> operations waiting (normalized)
     6. avg_machine_util     -> machine utilization [0-1] (already normalized)
     7. avg_operator_util    -> operator utilization [0-1] (already normalized)
     8. global_avg_wait      -> cumulative wait time (normalized)
     9. episode_time_fraction -> time progress [0-1] (already normalized)
    
    Returns:
        10-element numpy array (float32), all elements in [0, 1]
    """
    # Get problem parameters for dynamic normalization factors
    episode_limit = float(getattr(env, 'episode_limit', 50.0))
    max_jobs = float(getattr(env, 'max_jobs', 10))
    
    # Get normalization hyperparameters from args (with fallback defaults)
    args = getattr(env, 'args', None)
    if args is not None:
        norm_jobs_mult = float(getattr(args, 'norm_jobs_multiplier', 4.0))
        norm_ops_per_job = float(getattr(args, 'norm_ops_per_job', 12.0))
        norm_wait_mult = float(getattr(args, 'norm_wait_multiplier', 1.2))
        norm_active_mult = float(getattr(args, 'norm_active_multiplier', 1.5))
    else:
        # Fallback defaults if args not available
        norm_jobs_mult = 4.0
        norm_ops_per_job = 12.0
        norm_wait_mult = 1.2
        norm_active_mult = 1.5
    
    # Calculate dynamic normalization factors (adapts to problem size)
    MAX_JOBS_FACTOR = max_jobs * norm_jobs_mult
    MAX_OPS_FACTOR = max_jobs * norm_ops_per_job
    MAX_WAIT_FACTOR = episode_limit * max_jobs * norm_wait_mult
    MAX_ACTIVE_FACTOR = max_jobs * norm_active_mult
    
    # Get current simulation time
    now = float(getattr(env.env, 'now', 0.0)) if hasattr(env, 'env') else 0.0
    
    # RAW COUNTS (to be normalized)
    n_jobs_arrived = float(len(env.jobs))
    n_jobs_processing = float(env.active_jobs_count())
    n_jobs_waiting = float(max(0, n_jobs_arrived - n_jobs_processing))
    
    n_ops_arrived = float(getattr(env, 'total_ops_arrived', 0))
    n_ops_processing = float(_count_processing_ops(env))
    n_ops_waiting = float(max(0, n_ops_arrived - n_ops_processing))
    
    # ALREADY NORMALIZED UTILIZATIONS (6-7): [0-1]
    avg_machine_util = _calculate_machine_utilization(env)
    avg_operator_util = float(env.decision_operator_util)
    
    # RAW CUMULATIVE WAIT (to be normalized)
    global_avg_wait = float(getattr(env, 'total_wait_time', 0.0))
    
    # ALREADY NORMALIZED TIME FRACTION (9): [0-1]
    episode_time_fraction = min(1.0, now / max(1.0, episode_limit))
    
    # NORMALIZE ALL ELEMENTS TO [0, 1] WITH CLIPPING
    state_element_0 = np.clip(n_jobs_arrived / max(1.0, MAX_JOBS_FACTOR), 0.0, 1.0)
    state_element_1 = np.clip(n_jobs_processing / max(1.0, MAX_ACTIVE_FACTOR), 0.0, 1.0)
    state_element_2 = np.clip(n_jobs_waiting / max(1.0, MAX_JOBS_FACTOR), 0.0, 1.0)
    state_element_3 = np.clip(n_ops_arrived / max(1.0, MAX_OPS_FACTOR), 0.0, 1.0)
    state_element_4 = np.clip(n_ops_processing / max(1.0, MAX_ACTIVE_FACTOR), 0.0, 1.0)
    state_element_5 = np.clip(n_ops_waiting / max(1.0, MAX_OPS_FACTOR), 0.0, 1.0)
    state_element_8 = np.clip(global_avg_wait / max(1.0, MAX_WAIT_FACTOR), 0.0, 1.0)
    
    # VALIDATION: Check for overflow/weak signal (only for normalized elements)
    _validate_normalization(n_jobs_arrived, state_element_0, MAX_JOBS_FACTOR, 'state[0]:jobs_arrived', env)
    _validate_normalization(n_jobs_processing, state_element_1, MAX_ACTIVE_FACTOR, 'state[1]:jobs_processing', env)
    _validate_normalization(n_jobs_waiting, state_element_2, MAX_JOBS_FACTOR, 'state[2]:jobs_waiting', env)
    _validate_normalization(n_ops_arrived, state_element_3, MAX_OPS_FACTOR, 'state[3]:ops_arrived', env)
    _validate_normalization(n_ops_processing, state_element_4, MAX_ACTIVE_FACTOR, 'state[4]:ops_processing', env)
    _validate_normalization(n_ops_waiting, state_element_5, MAX_OPS_FACTOR, 'state[5]:ops_waiting', env)
    _validate_normalization(global_avg_wait, state_element_8, MAX_WAIT_FACTOR, 'state[8]:cumulative_wait', env)
    
    state = np.array([
        state_element_0,
        state_element_1,
        state_element_2,
        state_element_3,
        state_element_4,
        state_element_5,
        avg_machine_util,  # Already [0, 1]
        avg_operator_util,  # Already [0, 1]
        state_element_8,
        episode_time_fraction,  # Already [0, 1]
    ], dtype=np.float32)
    
    if state.shape[0] != 10:
        raise RuntimeError('Canonical state must be length 10')
    return state
