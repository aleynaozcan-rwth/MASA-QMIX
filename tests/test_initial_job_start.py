import pytest

from environment import MASAEnv


def test_initial_jobs_started_on_reset():
    """MASAEnv.reset() should schedule initial jobs immediately at t=0.

    This verifies that after reset() there are active jobs, the simpy
    event queue contains scheduled events, and the corresponding Job
    objects are marked active and not finished.
    """
    env = MASAEnv()
    obs, info = env.reset()

    # There should be at least one active job recorded by the env
    active_jobs = getattr(env, 'active_jobs', [])
    assert len(active_jobs) > 0, "No active jobs after reset()"

    # The internal simpy queue should contain scheduled processes
    q = getattr(env.env, '_queue', None)
    assert q is not None and len(q) > 0, "SimPy event queue is empty after reset()"

    # Ensure the first N jobs (N == number of active jobs) are active and not finished
    n_active = len(active_jobs)
    jobs = getattr(env, 'jobs', [])
    assert len(jobs) >= n_active
    assert all(getattr(j, 'is_active', False) and not getattr(j, 'finished', False) for j in jobs[:n_active])
