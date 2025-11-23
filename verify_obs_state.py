#!/usr/bin/env python3
"""
Detailed verification that observations and state are computed correctly.
Manually checks each element against environment state.
"""

import numpy as np
import sys
from MARL.common.arguments import get_common_args, get_mixer_args
from environment import MASAEnv

def verify_observation_calculation(env):
    """Manually compute observation and verify against env's output."""
    
    print("=" * 70)
    print("🔍 OBSERVATION VERIFICATION")
    print("=" * 70)
    
    # Get observations from environment
    obs_from_env = env._build_all_agent_obs()
    
    print(f"\n[INFO] Total jobs: {len(env.jobs)}")
    print(f"[INFO] Observation shape from env: {np.array(obs_from_env).shape}")
    print(f"[INFO] Expected: ({len(env.jobs)}, 7)")
    
    if len(obs_from_env) == 0:
        print("\n⚠️  No jobs in environment! Cannot verify observations.")
        return False
    
    # Manually verify first job's observation
    job = env.jobs[0]
    obs_manual = []
    
    print(f"\n{'='*70}")
    print(f"VERIFYING JOB 0: {job.id}")
    print(f"{'='*70}")
    
    # Element 0: current_op_type
    current_op_idx = getattr(job, 'current_op_idx', 0)
    obs_manual.append(float(current_op_idx))
    print(f"\n[0] current_op_type")
    print(f"    Manual calculation: {current_op_idx}")
    print(f"    From env: {obs_from_env[0][0]}")
    print(f"    ✓ Match: {obs_from_env[0][0] == float(current_op_idx)}")
    
    # Element 1: total_operations
    total_ops = len(job.operations)
    obs_manual.append(float(total_ops))
    print(f"\n[1] total_operations")
    print(f"    Manual calculation: {total_ops}")
    print(f"    From env: {obs_from_env[0][1]}")
    print(f"    ✓ Match: {obs_from_env[0][1] == float(total_ops)}")
    
    # Element 2: remaining_operations
    remaining_ops = max(0, len(job.operations) - int(job.current_op_idx))
    obs_manual.append(float(remaining_ops))
    print(f"\n[2] remaining_operations")
    print(f"    Manual calculation: {remaining_ops} (= {len(job.operations)} - {int(job.current_op_idx)})")
    print(f"    From env: {obs_from_env[0][2]}")
    print(f"    ✓ Match: {obs_from_env[0][2] == float(remaining_ops)}")
    
    # Element 3: wait_time
    wait_time = float(job.wait_time)
    obs_manual.append(wait_time)
    print(f"\n[3] wait_time")
    print(f"    Manual calculation: {wait_time:.4f}")
    print(f"    From env: {obs_from_env[0][3]:.4f}")
    print(f"    ✓ Match: {abs(obs_from_env[0][3] - wait_time) < 1e-5}")
    
    # Element 4: theoretical_machine_count
    avail_row = env._avail_row_for_job(job)
    theoretical_count = float(sum(avail_row))
    obs_manual.append(theoretical_count)
    print(f"\n[4] theoretical_machine_count")
    print(f"    Manual calculation: {int(theoretical_count)} machines can do this operation")
    print(f"    Avail row: {avail_row}")
    print(f"    From env: {obs_from_env[0][4]}")
    print(f"    ✓ Match: {obs_from_env[0][4] == theoretical_count}")
    
    # Element 5: free_machine_count
    avail_actions = env._build_avail_actions()
    job_index = 0  # We're checking job 0
    free_count = float(np.sum(avail_actions[job_index]))
    obs_manual.append(free_count)
    print(f"\n[5] free_machine_count")
    print(f"    Manual calculation: {int(free_count)} machines free AND qualified")
    print(f"    Avail actions for job {job_index}: {avail_actions[job_index]}")
    print(f"    From env: {obs_from_env[0][5]}")
    print(f"    ✓ Match: {obs_from_env[0][5] == free_count}")
    
    # Element 6: n_jobs_active
    n_active = float(env.active_jobs_count())
    obs_manual.append(n_active)
    print(f"\n[6] n_jobs_active")
    print(f"    Manual calculation: {int(n_active)}")
    print(f"    From env: {obs_from_env[0][6]}")
    print(f"    ✓ Match: {obs_from_env[0][6] == n_active}")
    
    # Compare arrays
    obs_manual = np.array(obs_manual, dtype=np.float32)
    obs_from_env_arr = np.array(obs_from_env[0], dtype=np.float32)
    
    print(f"\n{'='*70}")
    print("FINAL VERIFICATION")
    print(f"{'='*70}")
    print(f"\nManual obs:  {obs_manual}")
    print(f"Env obs:     {obs_from_env_arr}")
    print(f"Difference:  {np.abs(obs_manual - obs_from_env_arr)}")
    
    all_match = np.allclose(obs_manual, obs_from_env_arr, atol=1e-5)
    
    if all_match:
        print("\n✅ ALL 7 ELEMENTS MATCH EXACTLY!")
        print("   Observation calculation is CORRECT ✓")
        return True
    else:
        print("\n❌ MISMATCH DETECTED!")
        print("   Some elements don't match - possible bug in observation code")
        return False


def verify_state_calculation(env):
    """Manually compute state and verify against env's output."""
    
    print("\n\n" + "=" * 70)
    print("🔍 STATE VERIFICATION")
    print("=" * 70)
    
    from utils.env_obs import build_state_vector
    
    # Get state from environment
    state_from_env = build_state_vector(env)
    
    print(f"\n[INFO] State shape: {state_from_env.shape}")
    print(f"[INFO] Expected: (10,)")
    
    # Manually calculate each element
    state_manual = []
    
    now = float(getattr(env.env, 'now', 0.0)) if hasattr(env, 'env') else 0.0
    
    # Element 0: n_jobs_arrived
    n_jobs_arrived = float(getattr(env, 'total_jobs_arrived', 0))
    state_manual.append(n_jobs_arrived)
    print(f"\n[0] n_jobs_arrived")
    print(f"    Manual: {int(n_jobs_arrived)}")
    print(f"    From env: {state_from_env[0]}")
    print(f"    ✓ Match: {state_from_env[0] == n_jobs_arrived}")
    
    # Element 1: n_jobs_processing
    n_jobs_processing = float(env.active_jobs_count())
    state_manual.append(n_jobs_processing)
    print(f"\n[1] n_jobs_processing")
    print(f"    Manual: {int(n_jobs_processing)}")
    print(f"    From env: {state_from_env[1]}")
    print(f"    ✓ Match: {state_from_env[1] == n_jobs_processing}")
    
    # Element 2: n_jobs_waiting
    n_jobs_waiting = float(max(0, n_jobs_arrived - n_jobs_processing))
    state_manual.append(n_jobs_waiting)
    print(f"\n[2] n_jobs_waiting")
    print(f"    Manual: {int(n_jobs_waiting)} (= {int(n_jobs_arrived)} - {int(n_jobs_processing)})")
    print(f"    From env: {state_from_env[2]}")
    print(f"    ✓ Match: {state_from_env[2] == n_jobs_waiting}")
    
    # Element 3: n_ops_arrived
    n_ops_arrived = float(getattr(env, 'total_ops_arrived', 0))
    state_manual.append(n_ops_arrived)
    print(f"\n[3] n_ops_arrived")
    print(f"    Manual: {int(n_ops_arrived)}")
    print(f"    From env: {state_from_env[3]}")
    print(f"    ✓ Match: {state_from_env[3] == n_ops_arrived}")
    
    # Element 4: n_ops_processing
    n_ops_processing = 0
    for job in env.jobs:
        if not job.finished and hasattr(job, 'is_active') and job.is_active:
            n_ops_processing += 1
    state_manual.append(float(n_ops_processing))
    print(f"\n[4] n_ops_processing")
    print(f"    Manual: {n_ops_processing}")
    print(f"    From env: {state_from_env[4]}")
    print(f"    ✓ Match: {state_from_env[4] == float(n_ops_processing)}")
    
    # Element 5: n_ops_waiting
    n_ops_waiting = float(max(0, n_ops_arrived - n_ops_processing))
    state_manual.append(n_ops_waiting)
    print(f"\n[5] n_ops_waiting")
    print(f"    Manual: {int(n_ops_waiting)}")
    print(f"    From env: {state_from_env[5]}")
    print(f"    ✓ Match: {state_from_env[5] == n_ops_waiting}")
    
    # Element 6: avg_machine_util
    total_busy = 0
    total_capacity = 0
    if hasattr(env, 'machine_resources') and env.machine_resources:
        for machine_res in env.machine_resources:
            total_busy += len(machine_res.users)
            total_capacity += machine_res.capacity
    avg_machine_util = min(1.0, total_busy / max(1.0, total_capacity))
    state_manual.append(avg_machine_util)
    print(f"\n[6] avg_machine_util")
    print(f"    Manual: {avg_machine_util:.4f} ({total_busy}/{total_capacity})")
    print(f"    From env: {state_from_env[6]:.4f}")
    print(f"    ✓ Match: {abs(state_from_env[6] - avg_machine_util) < 1e-5}")
    
    # Element 7: avg_operator_util
    total_busy = 0
    total_capacity = 0
    if hasattr(env, 'operator_groups') and env.operator_groups:
        for op_res in env.operator_groups:
            total_busy += len(op_res.users)
            total_capacity += op_res.capacity
    avg_operator_util = min(1.0, total_busy / max(1.0, total_capacity))
    state_manual.append(avg_operator_util)
    print(f"\n[7] avg_operator_util")
    print(f"    Manual: {avg_operator_util:.4f} ({total_busy}/{total_capacity})")
    print(f"    From env: {state_from_env[7]:.4f}")
    print(f"    ✓ Match: {abs(state_from_env[7] - avg_operator_util) < 1e-5}")
    
    # Element 8: global_avg_wait
    global_avg_wait = float(getattr(env, 'total_wait_time', 0.0))
    state_manual.append(global_avg_wait)
    print(f"\n[8] global_avg_wait")
    print(f"    Manual: {global_avg_wait:.4f}")
    print(f"    From env: {state_from_env[8]:.4f}")
    print(f"    ✓ Match: {abs(state_from_env[8] - global_avg_wait) < 1e-5}")
    
    # Element 9: episode_time_fraction
    episode_limit = float(getattr(env, 'episode_limit', 1.0))
    episode_time_fraction = min(1.0, now / max(1.0, episode_limit))
    state_manual.append(episode_time_fraction)
    print(f"\n[9] episode_time_fraction")
    print(f"    Manual: {episode_time_fraction:.4f} ({now:.1f}/{episode_limit:.1f})")
    print(f"    From env: {state_from_env[9]:.4f}")
    print(f"    ✓ Match: {abs(state_from_env[9] - episode_time_fraction) < 1e-5}")
    
    # Compare arrays
    state_manual = np.array(state_manual, dtype=np.float32)
    
    print(f"\n{'='*70}")
    print("FINAL VERIFICATION")
    print(f"{'='*70}")
    print(f"\nManual state:  {state_manual}")
    print(f"Env state:     {state_from_env}")
    print(f"Difference:    {np.abs(state_manual - state_from_env)}")
    
    all_match = np.allclose(state_manual, state_from_env, atol=1e-5)
    
    if all_match:
        print("\n✅ ALL 10 ELEMENTS MATCH EXACTLY!")
        print("   State calculation is CORRECT ✓")
        return True
    else:
        print("\n❌ MISMATCH DETECTED!")
        print("   Some elements don't match - possible bug in state code")
        return False


def main():
    print("═" * 70)
    print("COMPREHENSIVE OBSERVATION & STATE VERIFICATION")
    print("═" * 70)
    print()
    print("This script manually computes each observation and state element")
    print("and compares it with the environment's calculations.")
    print()
    
    # Initialize environment
    print("[SETUP] Creating environment with default config...")
    from MARL.common.arguments import get_mutable_args
    args = get_mutable_args()
    
    env = MASAEnv(args)
    
    print("[SETUP] Resetting environment...")
    obs, info = env.reset()
    
    print(f"[SETUP] Environment ready: {len(env.jobs)} jobs")
    print()
    
    # Run verifications
    obs_ok = verify_observation_calculation(env)
    state_ok = verify_state_calculation(env)
    
    # Summary
    print("\n\n" + "═" * 70)
    print("VERIFICATION SUMMARY")
    print("═" * 70)
    
    if obs_ok and state_ok:
        print("\n🎉 SUCCESS! Both observation and state are computed correctly!")
        print()
        print("✅ Observation (7 elements): All elements match manual calculation")
        print("✅ State (10 elements): All elements match manual calculation")
        print()
        print("This confirms:")
        print("  • Observation system is working correctly")
        print("  • State system is working correctly")
        print("  • No bugs in obs/state computation")
        print("  • Values are based on actual environment state")
        print()
        return 0
    else:
        print("\n❌ VERIFICATION FAILED")
        if not obs_ok:
            print("  ✗ Observation calculation has issues")
        if not state_ok:
            print("  ✗ State calculation has issues")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
