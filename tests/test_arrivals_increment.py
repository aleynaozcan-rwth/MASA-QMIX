import time
from environment import MASAEnv
from utils.task_generator import TaskGenerator


def _make_wc_meta():
    class WCMeta:
        pass

    wc = WCMeta()
    # create a small machine registry covering ops 0..9 so arrival generator
    # can resolve allowed workcenters for generated operations
    reg = {}
    machine_list = []
    for i in range(10):
        name = f'M{i}'
        reg[name] = {'workcenter': i, 'capabilities': [i]}
        machine_list.append(name)
    wc.machine_registry = reg
    wc.machine_list = machine_list
    wc.machine_index = {m: i for i, m in enumerate(wc.machine_list)}
    wc.eligible_operator_groups_by_wc = {i: [] for i in range(10)}
    return wc


def test_arrivals_add_jobs_over_time():
    wc_meta = _make_wc_meta()
    env = MASAEnv(auto_build=False, auto_start_arrivals=False, use_yaml_config=False, num_wcs=2, num_ops=3, num_jobs=0, workcenters_meta=wc_meta)

    tg = TaskGenerator()
    # populate jobs registry so TaskGenerator can map resource ids
    from utils.job import Job
    tg.jobs = [Job(i, f'J{i}', f'Job {i}') for i in range(10)]
    # provide a minimal workcenters stub so generate_constrained_task can pick ops
    class WC:
        def __init__(self, resource_ids_list):
            self.resource_ids_list = list(resource_ids_list)

    class WCs:
        def __init__(self):
            self.workcenters_list = [WC(list(range(10)))]

    tg.workcenters = WCs()
    setattr(tg, '_owner_env', env)
    # Ensure env has processing_time_means entries for all generated ops/machines
    proc = {}
    for i in range(10):
        proc[f'Op{i+1}'] = {f'M{i}': 1.0}
    # TaskGenerator expects env.config to be a mapping of OpName -> {machine: dur}
    env.config = proc
    # schedule arrivals on the environment with a relatively high lambda
    tg.start(env, arrival_lambda=10.0)

    # run the sim for a short duration and assert we have received jobs
    env.env.run(until=1.0)
    assert len(env.jobs) > 0, "No jobs were added by TaskGenerator arrival loop"
