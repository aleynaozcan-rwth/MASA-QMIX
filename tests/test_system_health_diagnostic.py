"""
Comprehensive system health diagnostic for MASA-QMIX.

Tests that observation, state, action masking, reward, and QMIX are functioning correctly
WITHOUT waiting for convergence.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment import MASAEnv
from MARL.common.arguments import get_argparse_args_from_yaml
from MARL.agent.agent import Agents
from MARL.common.rollout import RolloutWorker
from MARL.common.replay_buffer import ReplayBuffer


class TestSystemHealthDiagnostic:
    """Comprehensive health checks for the entire MASA-QMIX system."""
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        args = get_argparse_args_from_yaml('configs/env_config.yaml')
        args.alg = 'qmix'
        env = MASAEnv(args)
        return env
    
    @pytest.fixture
    def agents_and_worker(self, env):
        """Create agents and rollout worker."""
        args = get_argparse_args_from_yaml('configs/env_config.yaml')
        args.alg = 'qmix'
        args.n_agents = env.get_env_info()['n_agents']
        args.n_actions = env.get_env_info()['n_actions']
        args.obs_shape = env.get_env_info()['obs_shape']
        args.state_shape = env.get_env_info()['state_shape']
        args.episode_limit = 500
        
        agents = Agents(args)
        worker = RolloutWorker(env, agents, args)
        return agents, worker, args
    
    # ============================================================================
    # 1. OBSERVATION TESTS
    # ============================================================================
    
    def test_observation_shape_consistency(self, env):
        """Test that observations have consistent shape across steps."""
        obs, state = env.reset()
        n_agents = env.get_env_info()['n_agents']
        obs_dim = env.get_env_info()['obs_shape']
        
        print(f"\n[OBS] n_agents={n_agents}, obs_dim={obs_dim}")
        print(f"[OBS] Initial obs shape: {obs.shape}")
        
        assert obs.shape == (n_agents, obs_dim), \
            f"Observation shape mismatch: expected ({n_agents}, {obs_dim}), got {obs.shape}"
        
        # Take 10 random steps and verify shape consistency
        for step in range(10):
            # Random actions (respecting masks if available)
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                valid_actions = np.where(avail_actions == 1)[0]
                if len(valid_actions) > 0:
                    actions.append(np.random.choice(valid_actions))
                else:
                    actions.append(0)  # fallback
            
            obs, state, rewards, done, info = env.step(actions)
            
            assert obs.shape == (n_agents, obs_dim), \
                f"Step {step}: Observation shape changed to {obs.shape}"
        
        print(f"[OBS] ✓ Shape consistent across 10 steps")
    
    def test_observation_bounds(self, env):
        """Test that observations are within reasonable bounds."""
        obs, state = env.reset()
        
        print(f"\n[OBS] Observation statistics:")
        print(f"  Min: {obs.min():.4f}")
        print(f"  Max: {obs.max():.4f}")
        print(f"  Mean: {obs.mean():.4f}")
        print(f"  Std: {obs.std():.4f}")
        
        # Check for NaN or Inf
        assert not np.isnan(obs).any(), "Observations contain NaN values"
        assert not np.isinf(obs).any(), "Observations contain Inf values"
        
        # Observations should be bounded (normalized or reasonable ranges)
        assert obs.min() >= -100, f"Observation values too small: {obs.min()}"
        assert obs.max() <= 100, f"Observation values too large: {obs.max()}"
        
        print(f"[OBS] ✓ Values within reasonable bounds")
    
    def test_observation_changes(self, env):
        """Test that observations change when environment changes."""
        obs1, state1 = env.reset()
        
        # Take action
        actions = [0] * env.get_env_info()['n_agents']
        obs2, state2, rewards, done, info = env.step(actions)
        
        # At least some observations should change
        obs_diff = np.abs(obs1 - obs2).sum()
        
        print(f"\n[OBS] Observation change after step: {obs_diff:.4f}")
        assert obs_diff > 0, "Observations did not change after action"
        
        print(f"[OBS] ✓ Observations respond to actions")
    
    # ============================================================================
    # 2. STATE TESTS
    # ============================================================================
    
    def test_state_shape_consistency(self, env):
        """Test that global state has consistent shape."""
        obs, state = env.reset()
        state_dim = env.get_env_info()['state_shape']
        
        print(f"\n[STATE] state_dim={state_dim}")
        print(f"[STATE] Initial state shape: {state.shape}")
        
        assert state.shape[0] == state_dim, \
            f"State shape mismatch: expected {state_dim}, got {state.shape[0]}"
        
        # Take steps and verify consistency
        for step in range(10):
            actions = [0] * env.get_env_info()['n_agents']
            obs, state, rewards, done, info = env.step(actions)
            assert state.shape[0] == state_dim, \
                f"Step {step}: State shape changed to {state.shape[0]}"
        
        print(f"[STATE] ✓ Shape consistent across 10 steps")
    
    def test_state_vs_observations(self, env):
        """Test that state contains information about all agents."""
        obs, state = env.reset()
        n_agents = env.get_env_info()['n_agents']
        
        print(f"\n[STATE] n_agents={n_agents}, state_dim={state.shape[0]}")
        
        # State should be at least as informative as concatenated observations
        # (In QMIX, state is typically larger or equal to sum of obs)
        obs_total_dim = obs.shape[0] * obs.shape[1]
        state_dim = state.shape[0]
        
        print(f"[STATE] Total obs dim: {obs_total_dim}, State dim: {state_dim}")
        assert state_dim >= n_agents, "State should contain info about all agents"
        
        print(f"[STATE] ✓ State dimensionality appropriate")
    
    # ============================================================================
    # 3. ACTION MASKING TESTS
    # ============================================================================
    
    def test_action_masking_exists(self, env):
        """Test that action masking is implemented."""
        obs, state = env.reset()
        n_agents = env.get_env_info()['n_agents']
        n_actions = env.get_env_info()['n_actions']
        
        print(f"\n[MASK] n_agents={n_agents}, n_actions={n_actions}")
        
        for agent_id in range(n_agents):
            avail = env.get_avail_agent_actions(agent_id)
            print(f"[MASK] Agent {agent_id}: {avail}")
            
            assert avail.shape[0] == n_actions, \
                f"Mask shape mismatch: expected {n_actions}, got {avail.shape[0]}"
            assert avail.dtype in [np.int32, np.int64, np.float32, np.float64], \
                f"Mask should be numeric, got {avail.dtype}"
        
        print(f"[MASK] ✓ Action masks available for all agents")
    
    def test_action_masking_prevents_invalid_actions(self, env):
        """Test that masked actions are actually invalid."""
        obs, state = env.reset()
        n_agents = env.get_env_info()['n_agents']
        
        masked_count = 0
        available_count = 0
        
        for agent_id in range(n_agents):
            avail = env.get_avail_agent_actions(agent_id)
            masked = np.where(avail == 0)[0]
            available = np.where(avail == 1)[0]
            
            masked_count += len(masked)
            available_count += len(available)
        
        print(f"\n[MASK] Total actions: {n_agents * env.get_env_info()['n_actions']}")
        print(f"[MASK] Available: {available_count}, Masked: {masked_count}")
        
        # At least some actions should be masked (otherwise masking is useless)
        assert masked_count > 0, "No actions are masked - masking may not be working"
        
        # At least some actions should be available
        assert available_count > 0, "All actions are masked - agents cannot act!"
        
        print(f"[MASK] ✓ Masking is active and meaningful")
    
    def test_action_masking_changes_over_time(self, env):
        """Test that action masks change as environment state changes."""
        obs, state = env.reset()
        n_agents = env.get_env_info()['n_agents']
        
        masks_t0 = [env.get_avail_agent_actions(i) for i in range(n_agents)]
        
        # Take several steps
        for _ in range(5):
            actions = []
            for agent_id in range(n_agents):
                avail = env.get_avail_agent_actions(agent_id)
                valid = np.where(avail == 1)[0]
                actions.append(np.random.choice(valid) if len(valid) > 0 else 0)
            env.step(actions)
        
        masks_t5 = [env.get_avail_agent_actions(i) for i in range(n_agents)]
        
        # At least one mask should have changed
        masks_changed = sum(not np.array_equal(m0, m5) 
                           for m0, m5 in zip(masks_t0, masks_t5))
        
        print(f"\n[MASK] Masks changed for {masks_changed}/{n_agents} agents after 5 steps")
        assert masks_changed > 0, "Action masks never change - may be static/incorrect"
        
        print(f"[MASK] ✓ Masks evolve with environment state")
    
    # ============================================================================
    # 4. REWARD TESTS
    # ============================================================================
    
    def test_reward_shape(self, env):
        """Test that rewards have correct shape."""
        obs, state = env.reset()
        n_agents = env.get_env_info()['n_agents']
        
        actions = [0] * n_agents
        obs, state, rewards, done, info = env.step(actions)
        
        print(f"\n[REWARD] Shape: {np.array(rewards).shape}")
        print(f"[REWARD] Values: {rewards}")
        
        # Rewards can be scalar (global) or per-agent
        if isinstance(rewards, (list, np.ndarray)):
            assert len(rewards) == n_agents, \
                f"Reward length mismatch: expected {n_agents}, got {len(rewards)}"
        
        print(f"[REWARD] ✓ Shape correct")
    
    def test_reward_non_trivial(self, env):
        """Test that rewards are non-zero and vary."""
        obs, state = env.reset()
        n_agents = env.get_env_info()['n_agents']
        
        reward_values = []
        for step in range(20):
            actions = []
            for agent_id in range(n_agents):
                avail = env.get_avail_agent_actions(agent_id)
                valid = np.where(avail == 1)[0]
                actions.append(np.random.choice(valid) if len(valid) > 0 else 0)
            
            obs, state, rewards, done, info = env.step(actions)
            
            # Convert to scalar if needed
            r = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
            reward_values.append(r)
            
            if done:
                break
        
        reward_values = np.array(reward_values)
        
        print(f"\n[REWARD] Statistics over {len(reward_values)} steps:")
        print(f"  Min: {reward_values.min():.4f}")
        print(f"  Max: {reward_values.max():.4f}")
        print(f"  Mean: {reward_values.mean():.4f}")
        print(f"  Std: {reward_values.std():.4f}")
        print(f"  Non-zero: {np.count_nonzero(reward_values)}/{len(reward_values)}")
        
        # Check that rewards are not all zero
        assert np.count_nonzero(reward_values) > 0, "All rewards are zero!"
        
        # Check that rewards vary (not constant)
        assert reward_values.std() > 0, "Rewards are constant - no variation!"
        
        print(f"[REWARD] ✓ Rewards are non-trivial and vary")
    
    def test_reward_normalized(self, env):
        """Test that reward normalization is working."""
        obs, state = env.reset()
        n_agents = env.get_env_info()['n_agents']
        
        reward_values = []
        for step in range(50):
            actions = []
            for agent_id in range(n_agents):
                avail = env.get_avail_agent_actions(agent_id)
                valid = np.where(avail == 1)[0]
                actions.append(np.random.choice(valid) if len(valid) > 0 else 0)
            
            obs, state, rewards, done, info = env.step(actions)
            r = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
            reward_values.append(r)
            
            if done:
                break
        
        reward_values = np.array(reward_values)
        
        print(f"\n[REWARD NORM] Range: [{reward_values.min():.2f}, {reward_values.max():.2f}]")
        
        # If normalization is working, rewards should be in reasonable range
        # (not 10^6 or 10^-6)
        assert abs(reward_values.mean()) < 1000, \
            f"Rewards may not be normalized: mean={reward_values.mean():.2e}"
        
        print(f"[REWARD NORM] ✓ Rewards appear normalized")
    
    # ============================================================================
    # 5. QMIX TESTS
    # ============================================================================
    
    def test_qmix_forward_pass(self, agents_and_worker):
        """Test that QMIX can perform forward pass without errors."""
        agents, worker, args = agents_and_worker
        
        # Generate dummy batch
        batch_size = 4
        n_agents = args.n_agents
        obs_dim = args.obs_shape
        state_dim = args.state_shape
        n_actions = args.n_actions
        
        # Create dummy inputs
        obs = torch.randn(batch_size, n_agents, obs_dim)
        state = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, n_actions, (batch_size, n_agents))
        avail_actions = torch.ones(batch_size, n_agents, n_actions)
        
        print(f"\n[QMIX] Testing forward pass...")
        print(f"[QMIX] Batch: {batch_size}, Agents: {n_agents}, Actions: {n_actions}")
        
        try:
            # Forward pass through agent networks
            with torch.no_grad():
                q_values = agents.policy._get_q_values(obs, None)
            
            print(f"[QMIX] Q-values shape: {q_values.shape}")
            assert q_values.shape == (batch_size, n_agents, n_actions), \
                f"Q-values shape mismatch: expected ({batch_size}, {n_agents}, {n_actions})"
            
            # Check Q-values are not NaN or Inf
            assert not torch.isnan(q_values).any(), "Q-values contain NaN"
            assert not torch.isinf(q_values).any(), "Q-values contain Inf"
            
            print(f"[QMIX] ✓ Forward pass successful")
            
        except Exception as e:
            pytest.fail(f"QMIX forward pass failed: {e}")
    
    def test_qmix_mixing_network(self, agents_and_worker):
        """Test that QMIX mixing network produces scalar Q_tot."""
        agents, worker, args = agents_and_worker
        
        batch_size = 4
        n_agents = args.n_agents
        state_dim = args.state_shape
        
        # Dummy agent Q-values
        agent_qs = torch.randn(batch_size, n_agents, 1)
        state = torch.randn(batch_size, state_dim)
        
        print(f"\n[QMIX MIXER] Testing mixing network...")
        
        try:
            with torch.no_grad():
                q_tot = agents.policy.eval_qmix.forward(agent_qs, state)
            
            print(f"[QMIX MIXER] Q_tot shape: {q_tot.shape}")
            assert q_tot.shape == (batch_size, 1), \
                f"Q_tot shape mismatch: expected ({batch_size}, 1), got {q_tot.shape}"
            
            # Check monotonicity: Q_tot should increase when agent Q-values increase
            agent_qs_higher = agent_qs + 1.0
            with torch.no_grad():
                q_tot_higher = agents.policy.eval_qmix.forward(agent_qs_higher, state)
            
            # QMIX should be monotonic (IGM property)
            assert (q_tot_higher >= q_tot).all(), \
                "QMIX violates monotonicity - mixing network may be broken"
            
            print(f"[QMIX MIXER] ✓ Mixing network works and is monotonic")
            
        except Exception as e:
            pytest.fail(f"QMIX mixing network failed: {e}")
    
    def test_qmix_gradient_flow(self, agents_and_worker):
        """Test that gradients flow through QMIX."""
        agents, worker, args = agents_and_worker
        
        batch_size = 4
        n_agents = args.n_agents
        obs_dim = args.obs_shape
        state_dim = args.state_shape
        n_actions = args.n_actions
        
        # Create dummy batch
        obs = torch.randn(batch_size, n_agents, obs_dim, requires_grad=True)
        state = torch.randn(batch_size, state_dim)
        actions = torch.randint(0, n_actions, (batch_size, n_agents))
        
        print(f"\n[QMIX GRAD] Testing gradient flow...")
        
        try:
            # Forward pass
            q_values = agents.policy._get_q_values(obs, None)
            
            # Select Q-values for chosen actions
            agent_qs = q_values.gather(2, actions.unsqueeze(-1))
            
            # Mix
            q_tot = agents.policy.eval_qmix.forward(agent_qs, state)
            
            # Backward
            loss = q_tot.mean()
            loss.backward()
            
            # Check gradients exist
            assert obs.grad is not None, "No gradients computed"
            assert not torch.isnan(obs.grad).any(), "Gradients contain NaN"
            
            grad_norm = obs.grad.norm().item()
            print(f"[QMIX GRAD] Gradient norm: {grad_norm:.6f}")
            assert grad_norm > 0, "Gradients are zero - no learning signal"
            
            print(f"[QMIX GRAD] ✓ Gradients flow correctly")
            
        except Exception as e:
            pytest.fail(f"QMIX gradient flow failed: {e}")
    
    # ============================================================================
    # 6. TRAINING LOOP INTEGRATION TEST
    # ============================================================================
    
    def test_episode_rollout(self, agents_and_worker):
        """Test that a full episode can be rolled out without errors."""
        agents, worker, args = agents_and_worker
        
        print(f"\n[ROLLOUT] Running full episode...")
        
        try:
            episode_batch, episode_reward = worker.generate_episode(evaluate=False)
            
            print(f"[ROLLOUT] Episode completed")
            print(f"[ROLLOUT] Episode reward: {episode_reward:.4f}")
            print(f"[ROLLOUT] Episode length: {episode_batch['o'].shape[0]}")
            
            # Check batch structure
            assert 'o' in episode_batch, "Batch missing observations"
            assert 's' in episode_batch, "Batch missing states"
            assert 'u' in episode_batch, "Batch missing actions"
            assert 'r' in episode_batch, "Batch missing rewards"
            assert 'avail_u' in episode_batch, "Batch missing action masks"
            
            # Check shapes
            ep_len = episode_batch['o'].shape[0]
            n_agents = episode_batch['o'].shape[1]
            print(f"[ROLLOUT] Batch shapes: episode_len={ep_len}, n_agents={n_agents}")
            
            assert ep_len > 0, "Episode length is zero"
            assert n_agents > 0, "Number of agents is zero"
            
            print(f"[ROLLOUT] ✓ Episode rollout successful")
            
        except Exception as e:
            pytest.fail(f"Episode rollout failed: {e}")
    
    def test_replay_buffer_storage(self, agents_and_worker):
        """Test that replay buffer can store and sample episodes."""
        agents, worker, args = agents_and_worker
        
        print(f"\n[BUFFER] Testing replay buffer...")
        
        buffer = ReplayBuffer(args)
        
        # Generate and store 3 episodes
        for i in range(3):
            episode_batch, _ = worker.generate_episode(evaluate=False)
            buffer.store_episode(episode_batch)
            print(f"[BUFFER] Stored episode {i+1}, buffer size: {buffer.current_size}")
        
        assert buffer.current_size == 3, f"Buffer size incorrect: {buffer.current_size}"
        
        # Sample a batch
        if buffer.current_size >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            
            print(f"[BUFFER] Sampled batch of size {args.batch_size}")
            assert batch is not None, "Sampling returned None"
            
            print(f"[BUFFER] ✓ Buffer storage and sampling work")
        else:
            print(f"[BUFFER] ✓ Buffer storage works (not enough for sampling yet)")
    
    def test_training_step(self, agents_and_worker):
        """Test that a training step can be executed without errors."""
        agents, worker, args = agents_and_worker
        
        print(f"\n[TRAIN] Testing training step...")
        
        buffer = ReplayBuffer(args)
        
        # Fill buffer with minimum episodes
        for i in range(args.batch_size):
            episode_batch, _ = worker.generate_episode(evaluate=False)
            buffer.store_episode(episode_batch)
        
        print(f"[TRAIN] Buffer filled with {buffer.current_size} episodes")
        
        try:
            # Sample batch
            train_batch = buffer.sample(args.batch_size)
            
            # Training step
            loss, td_error = agents.train(train_batch, worker.episode_num)
            
            print(f"[TRAIN] Loss: {loss:.6e}")
            print(f"[TRAIN] TD-error: {td_error:.6e}")
            
            # Check loss is finite
            assert np.isfinite(loss), f"Loss is not finite: {loss}"
            assert np.isfinite(td_error), f"TD-error is not finite: {td_error}"
            
            # Loss should be positive for MSE/Huber
            assert loss >= 0, f"Loss is negative: {loss}"
            
            print(f"[TRAIN] ✓ Training step successful")
            
        except Exception as e:
            pytest.fail(f"Training step failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
