import numpy as np
from MARL.common.mask_utils import build_machine_major_mask


class DummyWorkcentersMeta:
    def __init__(self, machine_list, eligible_operator_groups_by_wc):
        self.machine_list = machine_list
        self.eligible_operator_groups_by_wc = eligible_operator_groups_by_wc


class DummyEnv:
    def __init__(self):
        # topology
        self.num_ops = 2
        self.num_wcs = 5
        self.workcenters_meta = DummyWorkcentersMeta(
            machine_list=[f"M{i}" for i in range(self.num_wcs)],
            # per-machine eligible operator indices
            eligible_operator_groups_by_wc={
                0: [0],      # machine0 qualified operator 0
                1: [1],      # machine1 qualified operator 1
                2: [0, 1],   # machine2 both operators
                3: [],       # machine3 no qualified operator
                4: [0],      # machine4 operator 0
            },
        )

        # runtime resources (free flags) — True means free, False means busy
        # Make operator 0 free, operator 1 busy to test operator-filtering
        self.operator_free = [True, False]
        # Machine free flags: mark machine 2 busy, others free
        self.machine_free = [True, True, False, True, True]

        # simple jobs list for fallback lookup by id (not used in this test heavily)
        self.jobs = []

    def _resource_free(self, resource):
        # not used; present for API compatibility
        return bool(resource)

    def _build_avail_actions(self):
        """Return list-of-rows where each row is machine-major availability.

        Availability rule: machine available iff machine_free[m] is True and
        at least one operator qualified for that machine is free.
        """
        # we only construct a single job row (job id 0) for testing
        rows = []
        row = []
        for m in range(self.num_wcs):
            eligible = self.workcenters_meta.eligible_operator_groups_by_wc.get(m, [])
            # if no eligible operators -> unavailable
            if not eligible:
                row.append(0)
                continue
            # if machine is busy -> unavailable
            if not self.machine_free[m]:
                row.append(0)
                continue
            # if any qualified operator is free -> available
            any_free = False
            for p in eligible:
                # clamp p in operator_free range
                if int(p) < len(self.operator_free) and self.operator_free[int(p)]:
                    any_free = True
                    break
            row.append(1 if any_free else 0)
        rows.append(row)
        # additional rows to test negative cases:
        # job 1: all operators busy
        old_op = list(self.operator_free)
        self.operator_free = [False, False]
        row2 = []
        for m in range(self.num_wcs):
            eligible = self.workcenters_meta.eligible_operator_groups_by_wc.get(m, [])
            if not eligible or not self.machine_free[m]:
                row2.append(0)
                continue
            any_free = False
            for p in eligible:
                if int(p) < len(self.operator_free) and self.operator_free[int(p)]:
                    any_free = True
                    break
            row2.append(1 if any_free else 0)
        rows.append(row2)
        # restore operator_free for safety
        self.operator_free = old_op
        # job 2: job with no eligible machines -> empty row
        rows.append([0] * self.num_wcs)
        return rows

    def _avail_row_for_job(self, job):
        # simplistic fallback: use _build_avail_actions row 0
        try:
            return self._build_avail_actions()[0]
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
            return [1] * self.num_wcs


def test_machine_availability_semantics():
    env = DummyEnv()
    # env._build_avail_actions returns rows list indexed by job id
    matrix = env._build_avail_actions()
    assert isinstance(matrix, list)

    # job 0: operator 0 free, operator1 busy
    expected_job0 = [1, 0, 0, 0, 1]
    assert matrix[0] == expected_job0

    # build_machine_major_mask should return same semantics for job 0
    mask0 = build_machine_major_mask(env, 0)
    assert mask0 == expected_job0

    # job 1: all operators busy => all machines unavailable
    mask1 = build_machine_major_mask(env, 1)
    assert mask1 == [0, 0, 0, 0, 0]

    # job 2: explicitly no eligible machines -> unavailable
    mask2 = build_machine_major_mask(env, 2)
    assert mask2 == [0, 0, 0, 0, 0]
