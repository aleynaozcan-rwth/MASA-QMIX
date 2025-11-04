import pytest
from environment import MASAEnv


def test_resume_with_machine_name():
    env = MASAEnv(num_jobs=0, num_operators=1, seed=0, strict_mode=False)
    # create a single job with an op allowed on WC 0
    ops = [(0, [0], {0: 1.0})]
    job = env.add_job(ops)

    di, evt = env.workcenters_meta.create_decision_item(env, job, job.current_op())
    assert 'resume' in di
    # pick a valid machine name from the workcenters
    mlist = getattr(env.workcenters_meta, 'machine_list', [])
    assert len(mlist) > 0
    mname = mlist[0]
    expected_idx = env.workcenters_meta.machine_index.get(mname)
    # call resume with machine name
    di['resume'](mname)
    assert evt.triggered
    assert evt.value == int(expected_idx)


def test_resume_with_workcenter_index():
    env = MASAEnv(num_jobs=0, num_operators=1, seed=1, strict_mode=False)
    ops = [(0, [0], {0: 1.0})]
    job = env.add_job(ops)
    di, evt = env.workcenters_meta.create_decision_item(env, job, job.current_op())
    # resume with legacy workcenter index (0)
    di['resume'](0)
    assert evt.triggered
    # value should be an integer machine index (one of allowed_machine_indices)
    assert isinstance(evt.value, int) or evt.value is None
    if evt.value is not None:
        assert int(evt.value) in di.get('allowed_machine_indices', [])


def test_resume_with_invalid_choice_returns_none():
    env = MASAEnv(num_jobs=0, num_operators=1, seed=2, strict_mode=False)
    ops = [(0, [0], {0: 1.0})]
    job = env.add_job(ops)
    di, evt = env.workcenters_meta.create_decision_item(env, job, job.current_op())
    # invalid string
    di['resume']('NON_EXISTENT_MACHINE')
    assert evt.triggered
    assert evt.value is None

    # create another decision to test invalid numeric
    di2, evt2 = env.workcenters_meta.create_decision_item(env, job, job.current_op())
    di2['resume'](9999)
    assert evt2.triggered
    assert evt2.value is None
