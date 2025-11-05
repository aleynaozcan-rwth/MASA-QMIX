import simpy
from environment import MASAEnv


def test_create_decision_item_basic():
    env = MASAEnv(num_jobs=1, num_operators=1, num_wcs=1, seed=0, strict_mode=False)
    # pick the first job and its current op
    job = env.jobs[0]
    op = job.current_op()
    # create decision_item via WorkCenters helper
    di = env.workcenters_meta.create_decision_item(env, job, op)

    # basic keys
    assert isinstance(di, dict)
    for k in ["job_id", "obs", "avail_row", "allowed_machines", "allowed_machine_indices", "per_machine_durations", "base_duration"]:
        assert k in di

    # obs should be a numpy array-like of length >= 1
    obs = di['obs']
    assert getattr(obs, 'shape', None) is not None

    # The environment is responsible for creating a resume Event. Create
    # one here, attach it and then succeed it to simulate the policy.
    evt = simpy.Event(env.env)
    di['resume_evt'] = evt
    allowed = di.get('allowed_machine_indices', [])
    if allowed:
        choice = allowed[0]
        evt.succeed(int(choice))
        assert evt.triggered
        assert evt.value == int(choice)
    else:
        evt.succeed(None)
        assert evt.triggered
        assert evt.value is None
