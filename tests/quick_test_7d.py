"""Quick standalone test for 7D observation."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from environment import MASAEnv
from MARL.common.arguments import get_mutable_args

# Get mutable args and set minimal required fields
args = get_mutable_args()
args.n_operation_types = 4
args.max_wait_time = 200
args.avg_wait_scale = 1.0
args.reward_decay = 0.99
args.interarrival_time = 10.0
env = MASAEnv(args, auto_start_arrivals=False)

# Check obs_dim_agent
print(f"obs_dim_agent: {env.obs_dim_agent}")
assert env.obs_dim_agent == 7, f"Expected 7, got {env.obs_dim_agent}"

# Reset and get first observation
obs, info = env.reset()
print(f"Reset observation shape: {obs.shape}")

# Get agent observation for first job
if len(env.jobs) > 0:
    job = env.jobs[0]
    aobs = env._build_agent_obs(job)
    print(f"Agent obs shape: {aobs.shape}")
    print(f"Agent obs dtype: {aobs.dtype}")
    print(f"Agent obs values: {aobs}")
    
    assert aobs.shape == (7,), f"Expected (7,), got {aobs.shape}"
    assert aobs.dtype == np.float32, f"Expected float32, got {aobs.dtype}"
    assert np.all(aobs >= 0), "Expected all non-negative values"
    
    print("\n✅ All checks passed!")
else:
    print("No jobs available")
