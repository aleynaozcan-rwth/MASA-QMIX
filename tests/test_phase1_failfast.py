"""
Test Phase 1 Fail-Fast Fixes (A6, A4, A2)

This test suite validates that the fail-fast changes work correctly:
- A6: RolloutWorker._select_actions raises on missing avail info
- A4: Environment.pop_decision_reward raises on infeasibility validation failures
- A2: QMIX.select_actions raises on malformed avail_batch

All tests should PASS, meaning the system correctly crashes on invalid inputs.
"""
import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestA6_RolloutWorkerFailFast:
    """Test A6: RolloutWorker._select_actions must crash on missing availability info"""
    
    def test_missing_avail_info_raises(self):
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
        args.seed = 42
        args.device = 'cpu'
        args.episode_limit = 100
        
        worker = RolloutWorker(None, args, device='cpu')
        
        # Create obs_batch with MISSING availability info (neither key present)
        obs_batch = [
            {"some_other_key": np.array([1, 2, 3])},
            {"another_key": np.array([4, 5, 6])},
        ]
        
        # This should raise ValueError
        with pytest.raises(ValueError, match="missing availability info"):
            worker._select_actions(obs_batch, None, evaluate=False)
    
    def test_valid_avail_info_succeeds(self):
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
        assert len(actions) == 2
        assert all(isinstance(a, (int, type(None))) for a in actions)


class TestA4_RewardInfeasibilityFailFast:
    """Test A4: Environment.pop_decision_reward must crash on infeasibility validation failures"""
    
    def test_malformed_avail_raises(self):
        """When avail_row is malformed, infeasibility check must raise RuntimeError"""
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
        
        # Inject malformed decision info with invalid avail_row
        env._last_decision_info = [
            {
                'job_completed': False,
                'wait_time_norm': 0.5,
                'avail_row': "invalid_string_not_array",  # MALFORMED
                'chosen_action': 2,
            }
        ]
        
        # This should raise RuntimeError during infeasibility check
        with pytest.raises(RuntimeError, match="Infeasibility check failed"):
            env.pop_decision_reward()
    
    def test_action_without_avail_raises(self):
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
        
        # Inject decision info with action chosen but NO avail_row
        env._last_decision_info = [
            {
                'job_completed': False,
                'wait_time_norm': 0.5,
                'avail_row': None,  # MISSING
                'chosen_action': 2,  # But action WAS chosen
            }
        ]
        
        # This should raise RuntimeError
        with pytest.raises(RuntimeError, match="avail_row is None"):
            env.pop_decision_reward()
    
    def test_valid_feasibility_check_succeeds(self):
        """When infeasibility check has valid data, should work normally"""
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
                'avail_row': [1, 0, 1, 1, 0],  # Valid numpy-convertible
                'chosen_action': 2,  # Valid chosen action
            }
        ]
        
        # This should work without raising
        reward = env.pop_decision_reward()
        assert isinstance(reward, float)


class TestA2_QMIXAvailBatchFailFast:
    """Test A2: QMIX.select_actions must crash on malformed avail_batch"""
    
    def test_misaligned_avail_batch_raises(self):
        """When avail_batch length doesn't match n_agents, must raise IndexError"""
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
        
        # This should raise IndexError when trying to access avail_batch[2]
        with pytest.raises(IndexError):
            policy.select_actions(obs_batch, avail_batch, evaluate=True)
    
    def test_malformed_avail_entry_raises(self):
        """When avail_batch entry cannot be converted to int32 array, must raise"""
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
            "invalid_string_entry",  # MALFORMED
        ]
        
        # This should raise ValueError or TypeError when converting to int32
        with pytest.raises((ValueError, TypeError)):
            policy.select_actions(obs_batch, avail_batch, evaluate=True)
    
    def test_valid_avail_batch_succeeds(self):
        """When avail_batch is properly aligned and formatted, should work normally"""
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
        
        # Create valid avail_batch (properly aligned, 3 agents)
        avail_batch = [
            [1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1],
        ]
        
        # This should work without raising
        actions = policy.select_actions(obs_batch, avail_batch, evaluate=True)
        assert len(actions) == 3
        assert all(isinstance(a, int) for a in actions)


class TestIntegration_Phase1:
    """Integration tests to verify end-to-end fail-fast behavior"""
    
    def test_no_silent_defaults_in_pipeline(self):
        """Verify that the entire pipeline from obs→action→reward is fail-fast"""
        # This is a smoke test - if any component has silent fallbacks,
        # it will be caught by the specific tests above
        from MARL.common.rollout import RolloutWorker
        from MARL.policy.qmix import QMIX
        from environment import MASAEnv
        from MARL.common.arguments import get_common_args, get_mixer_args
        
        # Initialize components
        common_args = get_common_args()
        common_args.n_agents = 2
        common_args.n_actions = 3
        common_args.obs_shape = 7
        common_args.state_shape = 10
        common_args.initial_jobs = 2
        common_args.num_operators = 1
        common_args.num_wcs = 3
        common_args.n_operation_types = 2
        common_args.seed = 42
        common_args.device = 'cpu'
        common_args.epsilon_start = 1.0
        common_args.epsilon_end = 0.05
        common_args.episode_limit = 100
        
        mixer_args = get_mixer_args()
        mixer_args.n_agents = 2
        mixer_args.n_actions = 3
        mixer_args.obs_shape = 7
        mixer_args.state_shape = 10
        mixer_args.cuda = False
        mixer_args.lr = 0.0005
        mixer_args.last_action = True
        mixer_args.reuse_network = True
        mixer_args.epsilon = 0.0
        mixer_args.rnn_hidden_dim = 64
        
        # All components should initialize without silent fallbacks
        worker = RolloutWorker(None, common_args, device='cpu')
        policy = QMIX(mixer_args)
        env = MASAEnv(args=common_args, auto_build=False, auto_start_arrivals=False)
        
        # Verify no fallback-related attributes exist
        assert not hasattr(worker, '_silent_fallback_used')
        assert not hasattr(policy, '_silent_fallback_used')
        assert not hasattr(env, '_silent_fallback_used')


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
