"""utils/env_obs.py
Canonical observation and state helpers for MASAEnv.

This module implements a strict, minimal 7-element per-agent observation
compatible with the MASAEnv canonical attributes. The observation builder
does NOT contain any legacy features or silent fallbacks — if a
required canonical attribute is missing or invalid the builder raises.

Observation layout (length 7, dtype float32, all integers cast to float):
 0. current_op_type           -> job.current_op_idx (integer)
 1. total_operations          -> len(job.operations) (integer)
 2. remaining_operations      -> remaining ops count (integer)
 3. wait_time                 -> job.wait_time (float)
 4. theoretical_machine_count -> count of machines qualified for current operation
 5. free_machine_count        -> count of qualified machines currently available
 6. n_jobs_active             -> env.active_jobs_count() (integer)

The state builder here is intentionally small and only uses canonical
environment attributes. It will raise on missing/invalid attributes so
issues surface early.
"""
from __future__ import annotations 
import numpy as np 
from typing import Any 
import ast


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
    Build and return the canonical 7-element per-agent observation.
    Observation layout:
      0. current_op_type
      1. total_operations
      2. remaining_operations
      3. wait_time
      4. theoretical_machine_count
      5. free_machine_count
      6. n_jobs_active
    """

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

    obs = np.array([
        current_op_type,
        total_operations,
        remaining_operations,
        wait_time,
        theoretical_machine_count,
        free_machine_count,
        n_jobs_active,
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
    """Build 10-element global state vector (hybrid: raw counts + normalized utils).
    
    State layout (length 10, dtype float32):
     0. n_jobs_arrived       -> total jobs arrived (raw count)
     1. n_jobs_processing    -> jobs currently active (raw count)
     2. n_jobs_waiting       -> jobs waiting (raw count)
     3. n_ops_arrived        -> total operations arrived (raw count)
     4. n_ops_processing     -> operations being processed (raw count)
     5. n_ops_waiting        -> operations waiting (raw count)
     6. avg_machine_util     -> machine utilization [0-1] (normalized)
     7. avg_operator_util    -> operator utilization [0-1] (normalized)
     8. global_avg_wait      -> cumulative wait time (raw)
     9. episode_time_fraction -> time progress [0-1] (normalized)
    
    Returns:
        10-element numpy array (float32)
    """
    # Get current simulation time
    now = float(getattr(env.env, 'now', 0.0)) if hasattr(env, 'env') else 0.0
    
    # RAW COUNTS (0-5): No normalization
    n_jobs_arrived = float(getattr(env, 'total_jobs_arrived', 0))
    n_jobs_processing = float(env.active_jobs_count())
    n_jobs_waiting = float(max(0, n_jobs_arrived - n_jobs_processing))
    
    n_ops_arrived = float(getattr(env, 'total_ops_arrived', 0))
    n_ops_processing = float(_count_processing_ops(env))
    n_ops_waiting = float(max(0, n_ops_arrived - n_ops_processing))
    
    # NORMALIZED UTILIZATIONS (6-7): Percentage [0-1]
    avg_machine_util = _calculate_machine_utilization(env)
    avg_operator_util = _calculate_operator_utilization(env)
    
    # RAW TIME (8): Total cumulative wait time
    global_avg_wait = float(getattr(env, 'total_wait_time', 0.0))
    
    # NORMALIZED TIME FRACTION (9): Episode progress [0-1]
    episode_limit = float(getattr(env, 'episode_limit', 1.0))
    episode_time_fraction = min(1.0, now / max(1.0, episode_limit))
    
    state = np.array([
        n_jobs_arrived,
        n_jobs_processing,
        n_jobs_waiting,
        n_ops_arrived,
        n_ops_processing,
        n_ops_waiting,
        avg_machine_util,
        avg_operator_util,
        global_avg_wait,
        episode_time_fraction,
    ], dtype=np.float32)
    
    if state.shape[0] != 10:
        raise RuntimeError('Canonical state must be length 10')
    return state
