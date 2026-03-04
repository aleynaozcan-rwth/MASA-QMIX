"""RL-facing action masking helpers (MARL/common/mask_utils.py)

Archived/clarified candidate: Reintegration candidate — action/availability
mask helpers used by rollout and replay logic.

Role:
    - This module provides the RL-facing action masking layer. It translates
        workcenter/operator eligibility and current availability into deterministic
        index maps and boolean masks (numpy arrays) that indicate which
        (machine,operator) action pairs are valid for a given job/operation.
    - NOTE: The canonical availability representation is machine-major
      (`avail_row` / `avail_actions`) which contains one entry per machine.
      Operator-granular flattened masks (historically named `avail_mask`) are
      deprecated here and consumers should deterministically expand the
      machine-major row via `np.repeat(avail_row, num_ops)` if operator-level
      action slots are required. Mask builders in this module are deterministic
      and do not perform operator selection.
    - Rollout and policy code should use these helpers to produce tensors or
        flattened action vectors consumed by agents and replay buffers.

Notes:
    - Keep this module small and dependency-free (numpy only) so it remains easy
        to audit or move into a shared utilities module if desired.
    - If functionality is merged elsewhere during refactors, ensure callers are
        updated before removing this file.
"""
from typing import List, Tuple, Dict, Iterable
import numpy as np


def build_index_map(num_machines: int, num_operators: int) -> List[Tuple[int, int]]:
    """Return deterministic list of (machine_idx, operator_idx) pairs.

    Ordering is machine-major: (m0,o0),(m0,o1),...,(m1,o0),(m1,o1),...
    """
    idx = []
    for m in range(num_machines):
        for p in range(num_operators):
            idx.append((m, p))
    return idx


def build_mask_for_job(
    index_map: List[Tuple[int, int]],
    allowed_machine_indices: Iterable[int],
    operator_to_machines: Dict[int, Iterable[int]],
    machine_free: Iterable[bool],
    operator_free: Iterable[bool],
) -> np.ndarray:
    """Compute boolean mask over index_map where True means action is feasible.

    allowed_machine_indices: iterable of machine indices allowed for this op (machine-level indices)
    operator_to_machines: mapping operator_idx -> iterable of machine indices that operator can work on
    machine_free: sequence of bools length num_machines indicating free machines
    operator_free: sequence of bools length num_operators indicating free operators
    """
    allowed_set = set(int(x) for x in allowed_machine_indices)
    num_actions = len(index_map)
    mask = np.zeros((num_actions,), dtype=np.bool_)
    for i, (m, p) in enumerate(index_map):
        if m not in allowed_set:
            continue
        # operator must be qualified for machine
        quals = operator_to_machines.get(int(p), [])
        if int(m) not in set(quals):
            continue
        # machine and operator must be free
        if not machine_free[int(m)]:
            continue
        if not operator_free[int(p)]:
            continue
        mask[i] = True
    return mask


def build_machine_major_mask(env, job_or_job_id) -> List[int]:
    """Return a machine-major availability row for the given job.

    The returned list has length == number of machines (env.workcenters_meta.machine_list
    when present, otherwise env.num_wcs) and contains 1 for available machines and 0
    for unavailable ones. Availability means: machine supports the op, machine resource
    is free, and at least one qualified operator group is free. If operator information
    is missing, the function falls back to permissive behavior (treat machine as
    available when machine supports the op and machine resource appears free).

    Accepts either a JobAgent instance or an integer job_id.
    """
    try:
        # Prefer env-provided batch avail builder if available
        if hasattr(env, '_build_avail_actions'):
            try:
                matrix = env._build_avail_actions()
            except Exception:
                matrix = None
        else:
            matrix = None
    except Exception:
        matrix = None

    # Resolve job index
    job_idx = None
    try:
        if isinstance(job_or_job_id, int):
            job_idx = int(job_or_job_id)
        else:
            # assume a JobAgent-like object with .id
            job_idx = int(getattr(job_or_job_id, 'id', None))
    except Exception:
        job_idx = None

    if matrix is not None and job_idx is not None:
        # [C1] Matrix lookup is CRITICAL - fail-fast if indexing fails
        row = matrix[int(job_idx)]
        return [int(bool(x)) for x in list(row)]

    # Fallback: try env._avail_row_for_job(job)
    if job_idx is not None:
        # try to retrieve JobAgent by id from env.jobs
        job_obj = None
        for j in getattr(env, 'jobs', []) or []:
            # [C1] Job ID comparison - fail-fast if getattr fails
            if int(getattr(j, 'id', getattr(j, 'job_id', -1))) == int(job_idx):
                job_obj = j
                break
        
        if job_obj is not None:
            row = env._avail_row_for_job(job_obj)
        else:
            # last resort: if env.jobs indexed by id equals index
            try:
                job_obj = getattr(env, 'jobs', [])[int(job_idx)]
                row = env._avail_row_for_job(job_obj)
            except (IndexError, AttributeError) as e:
                # [C1] Job lookup failed - raise explicit error (Rule 2: no fake defaults)
                raise RuntimeError(
                    f"[C1] Failed to find job with idx={job_idx} in env.jobs. "
                    f"Cannot compute availability mask without valid job reference."
                ) from e
    else:
        # job_or_job_id might be a JobAgent-like
        row = env._avail_row_for_job(job_or_job_id)
    
    if row is None:
        # [C1] No availability row found - raise explicit error (Rule 2: no fake defaults)
        raise RuntimeError(
            f"[C1] Could not determine availability mask for job {job_or_job_id}. "
            f"env._avail_row_for_job returned None. Check job state and machine configuration."
        )
    
    return [int(bool(x)) for x in list(row)]
    #--------------------------------------------------------------
