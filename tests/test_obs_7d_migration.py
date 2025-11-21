"""Quick validation test for 7D observation migration."""
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from environment import MASAEnv
from utils.env_obs import build_agent_obs
from MARL.common.arguments import get_common_args

# Load mutable args with config
from MARL.common.arguments import get_mutable_args
args = get_mutable_args()
args.n_operation_types = 4
args.max_wait_time = 200
args.n_agents = 10
args.job_max_ops = 4
args.avg_wait_scale = 1.0
args.reward_decay = 0.99

# Create environment
env = MASAEnv(args=args)

# Verify obs_dim_agent is 7
print(f"✓ obs_dim_agent = {env.obs_dim_agent} (expected 7)")
assert env.obs_dim_agent == 7, f"Expected obs_dim_agent=7, got {env.obs_dim_agent}"

# Reset environment
env.reset()

# Build observation for first job
if len(env.jobs) > 0:
    job = env.jobs[0]
    obs = build_agent_obs(env, job, job_index=0)
    
    print(f"✓ Observation shape: {obs.shape} (expected (7,))")
    assert obs.shape == (7,), f"Expected shape (7,), got {obs.shape}"
    
    print(f"✓ Observation dtype: {obs.dtype} (expected float32)")
    assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
    
    print(f"✓ All elements non-negative: {np.all(obs >= 0)}")
    assert np.all(obs >= 0), "Expected all non-negative values"
    
    print(f"\nObservation elements:")
    print(f"  [0] current_op_type: {obs[0]}")
    print(f"  [1] total_operations: {obs[1]}")
    print(f"  [2] remaining_operations: {obs[2]}")
    print(f"  [3] wait_time: {obs[3]}")
    print(f"  [4] theoretical_machine_count: {obs[4]}")
    print(f"  [5] free_machine_count: {obs[5]}")
    print(f"  [6] n_jobs_active: {obs[6]}")
    
    print("\n✅ All checks passed! 7D observation migration successful.")
else:
    print("⚠ No jobs available to test observation")
