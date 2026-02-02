#!/usr/bin/env python3
"""
Network'e giden observation'ların gerçekten normalize olup olmadığını test et.
"""

import sys
import numpy as np
sys.path.append('.')

from MARL.common.arguments import get_common_args
from environment import MASAEnv

# Create environment
args = get_common_args()
env = MASAEnv(args=args, seed=42, auto_start_arrivals=False)
env.reset()

# Add some jobs
for i in range(5):
    env.add_job()

# Advance simulation a bit
for _ in range(10):
    env.env.run(until=env.env.now + 1.0)

# Get observations
obs_list = env._build_all_agent_obs()

print("=" * 80)
print("OBSERVATION NORMALIZATION TEST")
print("=" * 80)
print(f"\nNumber of observations: {len(obs_list)}")
print(f"Observation shape: {obs_list[0].shape if len(obs_list) > 0 else 'N/A'}")

if len(obs_list) > 0:
    print("\nChecking normalization (all values should be in [0, 1]):\n")
    
    for idx, obs in enumerate(obs_list[:3]):  # Check first 3
        print(f"Job {idx} observation:")
        print(f"  {obs}")
        
        # Check range
        min_val = obs.min()
        max_val = obs.max()
        
        if min_val < 0 or max_val > 1.0:
            print(f"  ❌ OUT OF RANGE! min={min_val:.4f}, max={max_val:.4f}")
        else:
            print(f"  ✅ In range [0, 1]: min={min_val:.4f}, max={max_val:.4f}")
        
        # Check for NaN/Inf
        if np.isnan(obs).any():
            print(f"  ❌ CONTAINS NaN!")
        if np.isinf(obs).any():
            print(f"  ❌ CONTAINS Inf!")
        
        print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
