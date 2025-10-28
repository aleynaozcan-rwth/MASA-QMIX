"""Mask utilities for deterministic (machine,operator) index mapping.

Provides helpers to build a consistent index map for (machine,operator)
pairs and to compute boolean masks of feasible actions for a job/operation.
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
    allowed_wcs: Iterable[int],
    operator_to_machines: Dict[int, Iterable[int]],
    machine_free: Iterable[bool],
    operator_free: Iterable[bool],
) -> np.ndarray:
    """Compute boolean mask over index_map where True means action is feasible.

    allowed_wcs: iterable of machine indices allowed for this op
    operator_to_machines: mapping operator_idx -> iterable of machine indices that operator can work on
    machine_free: sequence of bools length num_machines indicating free machines
    operator_free: sequence of bools length num_operators indicating free operators
    """
    allowed_set = set(int(x) for x in allowed_wcs)
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
