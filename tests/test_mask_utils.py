import numpy as np
from MARL.common.mask_utils import build_index_map, build_mask_for_job


def run_checks():
    num_m = 3
    num_p = 2
    idx_map = build_index_map(num_m, num_p)

    # operators: p0 can work on machines 0,1 ; p1 on machines 1,2
    op_to_m = {0: [0, 1], 1: [1, 2]}
    machine_free = [True, True, True]
    operator_free = [True, True]

    # allowed machines only {1}
    mask = build_mask_for_job(idx_map, [1], op_to_m, machine_free, operator_free)
    # index_map ordering: (0,0),(0,1),(1,0),(1,1),(2,0),(2,1)
    # feasible pairs that include machine 1: (1,0) index 2 and (1,1) index 3 -> both True
    assert mask.shape[0] == num_m * num_p
    assert mask[2] and mask[3]

    # if machine 1 busy
    machine_free = [True, False, True]
    mask2 = build_mask_for_job(idx_map, [1], op_to_m, machine_free, operator_free)
    assert not mask2[2] and not mask2[3]

    # if operator 0 busy, but machine 0 allowed
    machine_free = [True, True, True]
    operator_free = [False, True]
    mask3 = build_mask_for_job(idx_map, [0,1,2], op_to_m, machine_free, operator_free)
    # (0,0) should be False because op0 busy
    assert not mask3[0]

    print('test_mask_utils: all checks passed')


if __name__ == '__main__':
    run_checks()
