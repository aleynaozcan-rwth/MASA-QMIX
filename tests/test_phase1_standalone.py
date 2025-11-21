#!/usr/bin/env python3
"""
Standalone Test for Phase 1 Fail-Fast Fixes (A6, A4, A2)
No external dependencies - runs with standard library only.
"""
import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test counters
tests_passed = 0
tests_failed = 0
tests_total = 0


def test_case(name):
    """Decorator to mark test cases"""
    def decorator(func):
        def wrapper():
            global tests_passed, tests_failed, tests_total
            tests_total += 1
            print(f"\n{'='*70}")
            print(f"TEST {tests_total}: {name}")
            print('='*70)
            try:
                func()
                tests_passed += 1
                print(f"✅ PASSED: {name}")
                return True
            except AssertionError as e:
                tests_failed += 1
                print(f"❌ FAILED: {name}")
                print(f"   Assertion: {e}")
                traceback.print_exc()
                return False
            except Exception as e:
                tests_failed += 1
                print(f"❌ ERROR: {name}")
                print(f"   Exception: {e}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator


# ============================================================================
# A6 Tests: RolloutWorker._select_actions must crash on missing avail info
# ============================================================================

@test_case("A6.1: RolloutWorker raises ValueError when avail info missing")
def test_a6_missing_avail_raises():
    """When observation lacks both 'allowed_machine_indices' and 'avail_row', must raise ValueError"""
    from MARL.common.rollout import RolloutWorker
    from MARL.common.arguments import get_common_args
    
    args = get_common_args()
    args.n_agents = 3
    args.n_actions = 5
    args.obs_shape = 7
    args.state_shape = 10
    args.rnn_hidden_dim = 64
    args.epsilon_start = 1.0
    args.epsilon_end = 0.05
    args.epsilon_anneal_steps = 50000
    args.seed = 42
    args.device = 'cpu'
    args.episode_limit = 100
    
    worker = RolloutWorker(None, args, device='cpu')
    
    # Create obs_batch with MISSING availability info
    obs_batch = [
        {"some_other_key": [1, 2, 3]},
        {"another_key": [4, 5, 6]},
    ]
    
    # This should raise ValueError
    raised_correct_error = False
    try:
        worker._select_actions(obs_batch, None, evaluate=False)
    except ValueError as e:
        if "missing availability info" in str(e):
            raised_correct_error = True
            print(f"   ✓ Correctly raised ValueError: {e}")
    
    assert raised_correct_error, "Should have raised ValueError with 'missing availability info'"


@test_case("A6.2: RolloutWorker succeeds with valid avail_row")
def test_a6_valid_avail_succeeds():
    """When observation has valid avail_row, should work normally"""
    from MARL.common.rollout import RolloutWorker
    from MARL.common.arguments import get_common_args
    
    args = get_common_args()
    args.n_agents = 3
    args.n_actions = 5
    args.obs_shape = 7
    args.state_shape = 10
    args.rnn_hidden_dim = 64
    args.epsilon_start = 1.0
    args.epsilon_end = 0.05
    args.epsilon_anneal_steps = 50000
    args.seed = 42
    args.device = 'cpu'
    args.episode_limit = 100
    
    worker = RolloutWorker(None, args, device='cpu')
    
    # Create obs_batch WITH valid availability info
    obs_batch = [
        {"avail_row": [1, 1, 0, 1, 0]},
        {"allowed_machine_indices": [0, 2, 3]},
    ]
    
    # This should work without raising
    actions, _ = worker._select_actions(obs_batch, None, evaluate=True)
    assert len(actions) == 2, f"Expected 2 actions, got {len(actions)}"
    print(f"   ✓ Returned valid actions: {actions}")


# ============================================================================
# A4 Tests: Environment.pop_decision_reward must crash on infeasibility failures
# ============================================================================

@test_case("A4.1: Reward computation raises on malformed avail_row")
def test_a4_malformed_avail_raises():
    """When avail_row is malformed, infeasibility check must raise RuntimeError"""
    import numpy as np
    from environment import MASAEnv
    from MARL.common.arguments import get_common_args
    
    args = get_common_args()
    args.n_agents = 2
    args.initial_jobs = 2
    args.n_actions = 5
    args.obs_shape = 7
    args.state_shape = 10
    args.num_operators = 1
    args.num_wcs = 5
    args.n_operation_types = 3
    args.seed = 42
    
    env = MASAEnv(args=args, auto_build=False, auto_start_arrivals=False)
    
    # Inject malformed decision info
    env._last_decision_info = [
        {
            'job_completed': False,
            'wait_time_norm': 0.5,
            'avail_row': "invalid_string_not_array",  # MALFORMED
            'chosen_action': 2,
        }
    ]
    
    # This should raise RuntimeError
    raised_correct_error = False
    try:
        env.pop_decision_reward()
    except RuntimeError as e:
        if "Infeasibility check failed" in str(e):
            raised_correct_error = True
            print(f"   ✓ Correctly raised RuntimeError: {e}")
    
    assert raised_correct_error, "Should have raised RuntimeError with 'Infeasibility check failed'"


@test_case("A4.2: Reward computation raises when action chosen without avail_row")
def test_a4_action_without_avail_raises():
    """When action chosen but avail_row is None, must raise RuntimeError"""
    from environment import MASAEnv
    from MARL.common.arguments import get_common_args
    
    args = get_common_args()
    args.n_agents = 2
    args.initial_jobs = 2
    args.n_actions = 5
    args.obs_shape = 7
    args.state_shape = 10
    args.num_operators = 1
    args.num_wcs = 5
    args.n_operation_types = 3
    args.seed = 42
    
    env = MASAEnv(args=args, auto_build=False, auto_start_arrivals=False)
    
    # Inject decision info with action but NO avail_row
    env._last_decision_info = [
        {
            'job_completed': False,
            'wait_time_norm': 0.5,
            'avail_row': None,  # MISSING
            'chosen_action': 2,  # But action WAS chosen
        }
    ]
    
    # This should raise RuntimeError
    raised_correct_error = False
    try:
        env.pop_decision_reward()
    except RuntimeError as e:
        if "avail_row is None" in str(e):
            raised_correct_error = True
            print(f"   ✓ Correctly raised RuntimeError: {e}")
    
    assert raised_correct_error, "Should have raised RuntimeError with 'avail_row is None'"


@test_case("A4.3: Reward computation succeeds with valid data")
def test_a4_valid_feasibility_succeeds():
    """When infeasibility check has valid data, should work normally"""
    import numpy as np
    from environment import MASAEnv
    from MARL.common.arguments import get_common_args
    
    args = get_common_args()
    args.n_agents = 2
    args.initial_jobs = 2
    args.n_actions = 5
    args.obs_shape = 7
    args.state_shape = 10
    args.num_operators = 1
    args.num_wcs = 5
    args.n_operation_types = 3
    args.seed = 42
    
    env = MASAEnv(args=args, auto_build=False, auto_start_arrivals=False)
    
    # Inject valid decision info
    env._last_decision_info = [
        {
            'job_completed': False,
            'wait_time_norm': 0.5,
            'avail_row': [1, 0, 1, 1, 0],  # Valid
            'chosen_action': 2,  # Valid
        }
    ]
    
    # This should work
    reward = env.pop_decision_reward()
    assert isinstance(reward, float), f"Expected float reward, got {type(reward)}"
    print(f"   ✓ Returned valid reward: {reward}")


# ============================================================================
# A2 Tests: QMIX.select_actions must crash on malformed avail_batch
# ============================================================================

@test_case("A2.1: QMIX raises IndexError on misaligned avail_batch")
def test_a2_misaligned_avail_raises():
    """When avail_batch length doesn't match n_agents, must raise IndexError"""
    import numpy as np
    from MARL.policy.qmix import QMIX
    from MARL.common.arguments import get_mixer_args
    
    args = get_mixer_args()
    args.n_agents = 3
    args.n_actions = 5
    args.obs_shape = 7
    args.state_shape = 10
    args.rnn_hidden_dim = 64
    args.lr = 0.0005
    args.cuda = False
    args.last_action = True
    args.reuse_network = True
    args.epsilon = 0.0
    
    policy = QMIX(args)
    
    # Create obs_batch with 3 agents
    obs_batch = np.random.rand(3, 6).astype(np.float32)
    
    # Create avail_batch with WRONG length (only 2 agents)
    avail_batch = [
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
        # Missing 3rd agent's mask
    ]
    
    # This should raise IndexError
    raised_correct_error = False
    try:
        policy.select_actions(obs_batch, avail_batch, evaluate=True)
    except IndexError as e:
        raised_correct_error = True
        print(f"   ✓ Correctly raised IndexError: {e}")
    
    assert raised_correct_error, "Should have raised IndexError on misaligned avail_batch"


@test_case("A2.2: QMIX raises on malformed avail_batch entry")
def test_a2_malformed_entry_raises():
    """When avail_batch entry cannot be converted, must raise"""
    import numpy as np
    from MARL.policy.qmix import QMIX
    from MARL.common.arguments import get_mixer_args
    
    args = get_mixer_args()
    args.n_agents = 2
    args.n_actions = 5
    args.obs_shape = 7
    args.state_shape = 10
    args.rnn_hidden_dim = 64
    args.lr = 0.0005
    args.cuda = False
    args.last_action = True
    args.reuse_network = True
    args.epsilon = 0.0
    
    policy = QMIX(args)
    
    obs_batch = np.random.rand(2, 6).astype(np.float32)
    
    # Create avail_batch with malformed entry
    avail_batch = [
        [1, 1, 0, 1, 0],
        {"bad": "dict"},  # MALFORMED
    ]
    
    # This should raise ValueError or TypeError
    raised_correct_error = False
    try:
        policy.select_actions(obs_batch, avail_batch, evaluate=True)
    except (ValueError, TypeError) as e:
        raised_correct_error = True
        print(f"   ✓ Correctly raised {type(e).__name__}: {e}")
    
    assert raised_correct_error, "Should have raised ValueError or TypeError on malformed entry"


@test_case("A2.3: QMIX succeeds with valid avail_batch")
def test_a2_valid_avail_succeeds():
    """When avail_batch is properly aligned, should work normally"""
    import numpy as np
    from MARL.policy.qmix import QMIX
    from MARL.common.arguments import get_mixer_args
    
    args = get_mixer_args()
    args.n_agents = 3
    args.n_actions = 5
    args.obs_shape = 7
    args.state_shape = 10
    args.rnn_hidden_dim = 64
    args.lr = 0.0005
    args.cuda = False
    args.last_action = True
    args.reuse_network = True
    args.epsilon = 0.0
    
    policy = QMIX(args)
    
    obs_batch = np.random.rand(3, 6).astype(np.float32)
    
    # Create valid avail_batch
    avail_batch = [
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1],
    ]
    
    # This should work
    actions = policy.select_actions(obs_batch, avail_batch, evaluate=True)
    assert len(actions) == 3, f"Expected 3 actions, got {len(actions)}"
    assert all(isinstance(a, int) for a in actions), f"All actions should be int, got {[type(a) for a in actions]}"
    print(f"   ✓ Returned valid actions: {actions}")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("PHASE 1 FAIL-FAST VALIDATION TEST SUITE")
    print("Testing: A6 (RolloutWorker), A4 (Reward), A2 (QMIX)")
    print("="*70)
    
    # Run A6 tests
    test_a6_missing_avail_raises()
    test_a6_valid_avail_succeeds()
    
    # Run A4 tests
    test_a4_malformed_avail_raises()
    test_a4_action_without_avail_raises()
    test_a4_valid_feasibility_succeeds()
    
    # Run A2 tests
    test_a2_misaligned_avail_raises()
    test_a2_malformed_entry_raises()
    test_a2_valid_avail_succeeds()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {tests_total}")
    print(f"✅ Passed:    {tests_passed}")
    print(f"❌ Failed:    {tests_failed}")
    print("="*70)
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 1 fixes are working correctly.")
        print("\nNext steps:")
        print("  1. Run your main training script to verify integration")
        print("  2. Check that errors surface with clear messages")
        print("  3. Proceed to Phase 2 (A1, A3, A5) when ready")
        sys.exit(0)
    else:
        print(f"\n⚠️  {tests_failed} test(s) failed. Review the output above.")
        sys.exit(1)
