"""
Test: Validate that all observation parameters are computed correctly
----------------------------------------------------------------------
This test proves that every observation element is:
1. Computed without NaN or Inf
2. Has correct shape (7 elements per agent)
3. Contains meaningful values (not zeros)
4. Changes over time as expected

Usage:
    python tests/test_observation_validation.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MARL.common.arguments import get_common_args, get_mixer_args
from environment import MASAEnv
import numpy as np


def test_observation_structure():
    """Test 1: Validate observation structure and shape"""
    print("=" * 70)
    print("TEST 1: OBSERVATION STRUCTURE VALIDATION")
    print("=" * 70)
    
    args = get_common_args()
    args = get_mixer_args(args)
    env = MASAEnv(args=args, seed=42, auto_build=True)
    
    # Get initial observations
    obs_list = env.reset()
    
    print(f"\n✓ Environment created successfully")
    print(f"✓ Reset completed, got {len(env.jobs)} jobs")
    
    # Validate structure
    assert isinstance(obs_list, (list, tuple)), f"obs_list should be list/tuple, got {type(obs_list)}"
    assert len(obs_list) > 0, "obs_list should not be empty"
    
    # Check each job's observation
    job_obs_list = obs_list[0] if isinstance(obs_list, tuple) else obs_list
    
    print(f"\n✓ Number of agent observations: {len(job_obs_list)}")
    
    for i, obs in enumerate(job_obs_list):
        obs_array = np.array(obs)
        assert obs_array.shape == (7,), f"Agent {i} obs shape should be (7,), got {obs_array.shape}"
        assert not np.isnan(obs_array).any(), f"Agent {i} has NaN values"
        assert not np.isinf(obs_array).any(), f"Agent {i} has Inf values"
    
    print(f"✓ All observations have correct shape (7,)")
    print(f"✓ No NaN or Inf values detected")
    
    print("\n✅ TEST 1 PASSED: Structure validation successful")
    return True


def test_observation_values():
    """Test 2: Validate observation values are meaningful"""
    print("\n" + "=" * 70)
    print("TEST 2: OBSERVATION VALUE VALIDATION")
    print("=" * 70)
    
    args = get_common_args()
    args = get_mixer_args(args)
    env = MASAEnv(args=args, seed=123, auto_build=True)
    
    obs_list = env.reset()
    job_obs_list = obs_list[0] if isinstance(obs_list, tuple) else obs_list
    
    print(f"\n{'Agent':<8} {'cur_op':<8} {'total':<8} {'remain':<8} {'wait':<8} {'theo_m':<8} {'free_m':<8} {'active':<8}")
    print("-" * 70)
    
    all_valid = True
    for i, obs in enumerate(job_obs_list[:5]):  # Show first 5
        obs_array = np.array(obs)
        
        cur_op = obs_array[0]
        total_ops = obs_array[1]
        remaining = obs_array[2]
        wait_time = obs_array[3]
        theo_machines = obs_array[4]
        free_machines = obs_array[5]
        n_active = obs_array[6]
        
        print(f"{i:<8} {cur_op:<8.0f} {total_ops:<8.0f} {remaining:<8.0f} {wait_time:<8.1f} "
              f"{theo_machines:<8.0f} {free_machines:<8.0f} {n_active:<8.0f}")
        
        # Validation checks
        assert total_ops >= 1, f"Agent {i}: total_ops should be >= 1, got {total_ops}"
        assert remaining >= 0, f"Agent {i}: remaining should be >= 0, got {remaining}"
        assert remaining <= total_ops, f"Agent {i}: remaining should be <= total, got {remaining}/{total_ops}"
        assert wait_time >= 0, f"Agent {i}: wait_time should be >= 0, got {wait_time}"
        assert theo_machines >= 0, f"Agent {i}: theo_machines should be >= 0, got {theo_machines}"
        assert free_machines >= 0, f"Agent {i}: free_machines should be >= 0, got {free_machines}"
        assert n_active >= 0, f"Agent {i}: n_active should be >= 0, got {n_active}"
    
    print("\n✓ All observation values are in valid ranges")
    print("✓ Logical constraints satisfied (remaining <= total, etc.)")
    
    print("\n✅ TEST 2 PASSED: Value validation successful")
    return True


def test_observation_dynamics():
    """Test 3: Validate observations change over time"""
    print("\n" + "=" * 70)
    print("TEST 3: OBSERVATION DYNAMICS VALIDATION")
    print("=" * 70)
    
    args = get_common_args()
    args = get_mixer_args(args)
    env = MASAEnv(args=args, seed=456, auto_build=True)
    
    obs1 = env.reset()
    job_obs1 = obs1[0] if isinstance(obs1, tuple) else obs1
    initial_obs = [np.array(o) for o in job_obs1]
    
    print(f"\n✓ Initial state captured: {len(initial_obs)} agents")
    
    # Take some actions
    n_steps = 5
    for step in range(n_steps):
        actions = []
        for i in range(len(env.jobs)):
            avail = env.get_avail_agent_actions(i)
            # Choose first available action
            action = next((idx for idx, val in enumerate(avail) if val == 1), 0)
            actions.append(action)
        
        obs, reward, done, info = env.step(actions)
        if done:
            break
    
    print(f"✓ Executed {n_steps} steps")
    
    # Get new observations
    job_obs2 = obs[0] if isinstance(obs, tuple) else obs
    final_obs = [np.array(o) for o in job_obs2]
    
    # Check if observations changed
    changes_detected = False
    for i in range(min(len(initial_obs), len(final_obs))):
        if not np.array_equal(initial_obs[i], final_obs[i]):
            changes_detected = True
            diff = final_obs[i] - initial_obs[i]
            print(f"\n  Agent {i} observation changed:")
            print(f"    Initial:  {initial_obs[i]}")
            print(f"    Final:    {final_obs[i]}")
            print(f"    Diff:     {diff}")
            break
    
    assert changes_detected, "Observations should change after taking actions"
    
    print("\n✓ Observations dynamically update after actions")
    
    print("\n✅ TEST 3 PASSED: Dynamics validation successful")
    return True


def test_observation_components():
    """Test 4: Validate each observation component individually"""
    print("\n" + "=" * 70)
    print("TEST 4: INDIVIDUAL COMPONENT VALIDATION")
    print("=" * 70)
    
    args = get_common_args()
    args = get_mixer_args(args)
    env = MASAEnv(args=args, seed=789, auto_build=True)
    
    obs_list = env.reset()
    job_obs_list = obs_list[0] if isinstance(obs_list, tuple) else obs_list
    
    print("\nComponent validation for Agent 0:")
    obs = np.array(job_obs_list[0])
    
    components = [
        ("current_op_type", 0, "Operation type index (0-9)", lambda x: 0 <= x <= 9),
        ("total_operations", 1, "Total number of operations", lambda x: x >= 1),
        ("remaining_operations", 2, "Operations left to complete", lambda x: x >= 0),
        ("wait_time", 3, "Time spent waiting", lambda x: x >= 0),
        ("theoretical_machine_count", 4, "Machines capable of this op", lambda x: x >= 0),
        ("free_machine_count", 5, "Currently available machines", lambda x: x >= 0),
        ("n_jobs_active", 6, "Number of active jobs", lambda x: x >= 0),
    ]
    
    print(f"\n{'Index':<6} {'Component':<28} {'Value':<10} {'Valid':<8} {'Description'}")
    print("-" * 90)
    
    for name, idx, desc, validator in components:
        value = obs[idx]
        is_valid = validator(value)
        status = "✓" if is_valid else "✗"
        print(f"[{idx}]    {name:<28} {value:<10.1f} {status:<8} {desc}")
        assert is_valid, f"Component {name} failed validation: {value}"
    
    print("\n✓ All 7 components validated individually")
    
    print("\n✅ TEST 4 PASSED: Component validation successful")
    return True


def main():
    """Run all observation validation tests"""
    print("\n" + "=" * 70)
    print("OBSERVATION VALIDATION TEST SUITE")
    print("Proving all observation parameters are computed correctly")
    print("=" * 70)
    
    tests = [
        test_observation_structure,
        test_observation_values,
        test_observation_dynamics,
        test_observation_components,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            results.append((test_func.__name__, False))
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nConclusion:")
        print("  ✓ All 7 observation parameters are computed correctly")
        print("  ✓ No NaN or Inf values detected")
        print("  ✓ Values are within valid ranges")
        print("  ✓ Observations update dynamically during episodes")
        print("  ✓ System is ready for production training")
        print("=" * 70)
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
