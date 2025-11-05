import random
from utils.task_generator import TaskGenerator
from utils.job import Jobs


def _make_stub_workcenters():
    class WC:
        def __init__(self, resource_ids_list):
            self.resource_ids_list = list(resource_ids_list)

    class WCs:
        def __init__(self):
            # single workcenter with resource ids 0..9
            self.workcenters_list = [WC(list(range(10)))]

    return WCs()


def test_generate_ops_are_unique_within_job():
    # Deterministic seed for repeatability
    random.seed(12345)
    tg = TaskGenerator(max_retries=50)

    # populate jobs registry and workcenters so generation can succeed
    from utils.job import Job
    tg.jobs = [Job(i, f'J{i}', f'Job {i}') for i in range(10)]
    tg.workcenters = _make_stub_workcenters()

    seq = tg.generate_constrained_task(num_ops=5)
    ids = [getattr(j, 'index_id', None) for j in seq]
    # ensure all op index_ids in the generated sequence are unique
    assert len(ids) == len(set(ids)), f"Found duplicate op ids in generated job: {ids}"
