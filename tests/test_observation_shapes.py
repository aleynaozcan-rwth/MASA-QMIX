import numpy as np
from environment import MASAEnv


def test_observation_and_state_shapes():
    # Minimal deterministic environment
    env = MASAEnv(num_jobs=1, num_operators=1, num_wcs=1, seed=0, auto_build=True)
    # Ensure at least one job exists
    assert len(env.jobs) >= 1
    obs = env._build_agent_obs(env.jobs[0])
    state = env._build_state_vector()
    assert isinstance(obs, np.ndarray)
    assert isinstance(state, np.ndarray)
    assert obs.shape == (7,), f"Expected shape (7,), got {obs.shape}"
    assert state.shape == (env.state_dim,)
