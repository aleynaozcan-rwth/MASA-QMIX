"""
Wrappers for some common utility helpers
"""
import math

def left_planes(chosen_plane, current_idle_planes):
    """Return the list of idle planes that are NOT in the chosen set."""
    res = []
    for eve in current_idle_planes:
        if eve not in chosen_plane:
            res.append(eve)
    return res


def min_but_zero(state_left_time):
    """
    Return the minimum value among non-zero elements of state_left_time.
    If all elements are zero (or the list is empty), return 0.
    """
    non_zero_list = []
    for eve in state_left_time:
        if eve != 0:
            non_zero_list.append(eve)
    if len(non_zero_list) != 0:
        return min(non_zero_list)
    else:
        return 0


# Subtract min_time from every non-zero value in state_left_time (zeros stay zero)
def advance_by_min_time(min_time, state_left_time):
    res = []
    for eve in state_left_time:
        if eve != 0:
            assert eve >= min_time
            res.append(eve - min_time)
        else:
            res.append(0)
    return res


# ============================================================
# === Temporary Compatibility Stub ===========================
# ============================================================
def count_path_on_road(path):
    """
    Placeholder for legacy function (used by SDrules in shortestDistence.py).
    Currently returns a simple path length to preserve compatibility.

    TODO:
        - Replace with actual road/path distance logic if SDrules is reactivated
          for rule-based scheduling or visualization modes.
    """
    if path is None:
        return 0
    try:
        return len(path)
    except Exception:
        return 0
