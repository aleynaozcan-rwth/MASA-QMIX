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

# NOTE: count_path_on_road removed in Step 1B (distance logic fully deleted)
