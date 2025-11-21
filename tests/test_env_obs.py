import numpy as np
from environment import MASAEnv
from utils import env_obs


def test_build_agent_obs_and_state_shape():
    env = MASAEnv(num_jobs=4, episode_limit=200, seed=123, config_path='configs/env_no_arrival.yaml')
    obs, info = env.reset()
    # pick the first non-finished job
    job = env.jobs[0]
    aobs = env._build_agent_obs(job)
    aobs2 = env_obs.build_agent_obs(env, job, job_index=0)
    assert isinstance(aobs, np.ndarray)
    assert aobs.shape == (7,), f"Expected shape (7,), got {aobs.shape}"
    assert np.all(aobs >= 0.0)  # All elements should be non-negative integers
    # the delegated function should return same-shape output
    assert isinstance(aobs2, np.ndarray)
    assert aobs2.shape == aobs.shape


def test_build_state_vector_shape():
    env = MASAEnv(num_jobs=4, episode_limit=200, seed=123, config_path='configs/env_no_arrival.yaml')
    _ = env.reset()
    s = env._build_state_vector()
    s2 = env_obs.build_state_vector(env)
    assert isinstance(s, np.ndarray)
    assert s.shape[0] == env.state_dim
    assert s.shape == (10,), f"Expected state shape (10,), got {s.shape}"
    assert isinstance(s2, np.ndarray)
    assert s2.shape == s.shape
