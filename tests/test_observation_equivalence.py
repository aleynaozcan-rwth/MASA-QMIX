import numpy as np
from environment import MASAEnv
from utils.env_obs import build_state_vector


def test_env_and_helper_state_equivalence():
    # Minimal env: do not auto_build arrivals; use default args via module
    env = MASAEnv(auto_build=False, auto_load_config=False, auto_start_arrivals=False)
    # ensure environment has canonical state_dim set
    assert hasattr(env, 'state_dim')
    s_env = env._build_state_vector()
    s_helper = build_state_vector(env)
    assert np.allclose(np.asarray(s_env), np.asarray(s_helper))


def test_env_and_helper_agent_obs_equivalence():
    import numpy as _np
    from utils.env_obs import build_agent_obs

    env = MASAEnv(auto_build=False, auto_load_config=False, auto_start_arrivals=False)
    # Add a minimal job so build_agent_obs has something to read
    # Use a simple operation tuple: (op_type, allowed_wcs, per_wc_durations)
    ops = [(0, [0], {0: 1.0})]
    job = env.add_job(ops)

    # Ensure job was added
    assert job is not None

    a_env = env._build_agent_obs(job)
    a_helper = build_agent_obs(env, job)
    assert _np.allclose(_np.asarray(a_env), _np.asarray(a_helper))
