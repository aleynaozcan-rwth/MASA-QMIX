import simpy
from environment import MASAEnv


def test_create_decision_item_basic():
    env = MASAEnv(num_jobs=1, num_operators=1, num_wcs=1, seed=0, strict_mode=False)
    # pick the first job and its current op
    job = env.jobs[0]
    op = job.current_op()
    # create decision_item via WorkCenters helper
    di, evt = env.workcenters_meta.create_decision_item(env, job, op)

    # basic keys
    assert isinstance(di, dict)
    for k in ["job_id", "obs", "avail_row", "allowed_machines", "allowed_machine_indices", "per_machine_durations", "base_duration", "resume"]:
        assert k in di

    # obs should be a numpy array-like of length >= 1
    obs = di['obs']
    assert getattr(obs, 'shape', None) is not None

    # resume should trigger the returned event when called with a valid choice
    resume = di['resume']
    # if there is an allowed machine index, use it; else pass None
    allowed = di.get('allowed_machine_indices', [])
    if allowed:
        choice = allowed[0]
        resume(choice)
        assert evt.triggered
        assert evt.value == int(choice)
    else:
        resume(None)
        assert evt.triggered
        assert evt.value is None
