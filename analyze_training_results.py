#!/usr/bin/env python3
"""
Analyze training results to identify convergence issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
print("Loading data files...")
loss_df = pd.read_csv('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/loss.txt', sep=' ')
td_error_df = pd.read_csv('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/td_error.txt', sep=' ')
training_df = pd.read_csv('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/training_metrics.csv')
episode_df = pd.read_csv('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/episode_metrics.csv')

print("\n" + "="*80)
print("TRAINING DIAGNOSTICS - CONVERGENCE ANALYSIS")
print("="*80)

# 1. Check data size
print(f"\n1. DATA VOLUME:")
print(f"   - Total training steps: {len(loss_df):,}")
print(f"   - Total episodes: {len(episode_df):,}")
print(f"   - Last episode: {episode_df['episode'].iloc[-1]}")
print(f"   - Last epoch: {episode_df['epoch'].iloc[-1]}")

# 2. Loss Analysis
print(f"\n2. LOSS ANALYSIS:")
print(f"   - Initial loss: {loss_df['last_loss'].iloc[0]:.2f}")
print(f"   - Final loss: {loss_df['last_loss'].iloc[-1]:.2e}")
print(f"   - Loss increase: {loss_df['last_loss'].iloc[-1] / loss_df['last_loss'].iloc[0]:.2e}x")
print(f"   - Minimum loss: {loss_df['last_loss'].min():.2f} at step {loss_df['last_loss'].idxmin()}")
print(f"   - Maximum loss: {loss_df['last_loss'].max():.2e} at step {loss_df['last_loss'].idxmax()}")

# Check when loss started exploding
loss_threshold = 1000
explode_idx = loss_df[loss_df['last_loss'] > loss_threshold].index
if len(explode_idx) > 0:
    print(f"   - Loss exceeded {loss_threshold} at step {explode_idx[0]} (episode {loss_df['episode'].iloc[explode_idx[0]]})")

# 3. TD Error Analysis
print(f"\n3. TD ERROR ANALYSIS:")
print(f"   - Initial TD error: {td_error_df['last_td'].iloc[0]:.2f}")
print(f"   - Final TD error: {td_error_df['last_td'].iloc[-1]:.2e}")
print(f"   - TD error increase: {td_error_df['last_td'].iloc[-1] / td_error_df['last_td'].iloc[0]:.2e}x")
print(f"   - Minimum TD error: {td_error_df['last_td'].min():.2f} at step {td_error_df['last_td'].idxmin()}")
print(f"   - Maximum TD error: {td_error_df['last_td'].max():.2e} at step {td_error_df['last_td'].idxmax()}")

# 4. Reward Analysis
print(f"\n4. REWARD ANALYSIS:")
print(f"   - Initial reward: {episode_df['episode_reward'].iloc[0]:.2f}")
print(f"   - Final reward: {episode_df['episode_reward'].iloc[-1]:.2f}")
print(f"   - Mean reward (all): {episode_df['episode_reward'].mean():.2f} ± {episode_df['episode_reward'].std():.2f}")
print(f"   - Mean reward (first 100): {episode_df['episode_reward'].iloc[:100].mean():.2f}")
print(f"   - Mean reward (last 100): {episode_df['episode_reward'].iloc[-100:].mean():.2f}")
print(f"   - Best reward: {episode_df['episode_reward'].max():.2f} at episode {episode_df['episode_reward'].idxmax()}")
print(f"   - Worst reward: {episode_df['episode_reward'].min():.2f} at episode {episode_df['episode_reward'].idxmin()}")

# 5. Q-value Analysis
print(f"\n5. Q-VALUE ANALYSIS:")
print(f"   - Initial Q-value: {training_df['avg_q_value'].iloc[0]:.2f}")
print(f"   - Final Q-value: {training_df['avg_q_value'].iloc[-1]:.2f}")
print(f"   - Q-value trend: {'DECREASING (BAD)' if training_df['avg_q_value'].iloc[-1] < training_df['avg_q_value'].iloc[0] else 'INCREASING'}")
print(f"   - Minimum Q-value: {training_df['avg_q_value'].min():.2f} at step {training_df['avg_q_value'].idxmin()}")

# 6. Epsilon Analysis
print(f"\n6. EXPLORATION ANALYSIS:")
print(f"   - Initial epsilon: {episode_df['epsilon'].iloc[0]:.4f}")
print(f"   - Final epsilon: {episode_df['epsilon'].iloc[-1]:.4f}")
print(f"   - Reached minimum epsilon at episode: {episode_df[episode_df['epsilon'] <= 0.1].index[0] if len(episode_df[episode_df['epsilon'] <= 0.1]) > 0 else 'NOT REACHED'}")

# 7. Stability Analysis (last 100 vs previous 100 episodes)
if len(episode_df) >= 200:
    last_100_reward = episode_df['episode_reward'].iloc[-100:].mean()
    prev_100_reward = episode_df['episode_reward'].iloc[-200:-100].mean()
    print(f"\n7. STABILITY CHECK:")
    print(f"   - Reward (episodes -200:-100): {prev_100_reward:.2f}")
    print(f"   - Reward (episodes -100:end): {last_100_reward:.2f}")
    print(f"   - Change: {last_100_reward - prev_100_reward:.2f} ({((last_100_reward/prev_100_reward - 1)*100):.1f}%)")

# 8. Check for NaN or Inf values
print(f"\n8. DATA QUALITY:")
print(f"   - NaN in loss: {loss_df['last_loss'].isna().sum()}")
print(f"   - Inf in loss: {np.isinf(loss_df['last_loss']).sum()}")
print(f"   - NaN in TD error: {td_error_df['last_td'].isna().sum()}")
print(f"   - NaN in rewards: {episode_df['episode_reward'].isna().sum()}")

# 9. Identify the problem
print(f"\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

problems = []

if loss_df['last_loss'].iloc[-1] > 1e6:
    problems.append("❌ CRITICAL: Loss exploded to astronomical values (>1M)")
    problems.append("   → This indicates numerical instability in the Q-network")

if td_error_df['last_td'].iloc[-1] > 1000:
    problems.append("❌ CRITICAL: TD error exploded (>1000)")
    problems.append("   → Target Q-values are diverging from predicted Q-values")

if training_df['avg_q_value'].iloc[-1] < -1000:
    problems.append("❌ CRITICAL: Q-values became extremely negative")
    problems.append("   → Network is predicting unrealistically bad values")

if episode_df['episode_reward'].iloc[-100:].std() > 10:
    problems.append("⚠️  WARNING: High reward variance in late training")
    problems.append("   → Policy is not stabilizing")

if len(episode_df) < 500:
    problems.append("⚠️  WARNING: Training stopped early (< 500 episodes)")
    problems.append("   → Insufficient training time")

if problems:
    for problem in problems:
        print(problem)
else:
    print("✓ No critical issues detected")

print(f"\n" + "="*80)
print("LIKELY ROOT CAUSES:")
print("="*80)
print("""
Based on the data:

1. GRADIENT EXPLOSION: Loss increased from ~35 to >1e10 (billions!)
   - Learning rate too high
   - Gradient clipping not working or too loose
   - Target network update too infrequent

2. Q-VALUE DIVERGENCE: Q-values went from +0.47 to very negative
   - Overestimation bias in QMIX mixer network
   - Reward scale mismatch
   - Bootstrap targets becoming unstable

3. NO CONVERGENCE: Rewards oscillating around 25-35 without improvement
   - Exploration-exploitation balance off (epsilon decay too slow/fast)
   - Network capacity insufficient for complex state space
   - Replay buffer issues (old experiences dominating)

RECOMMENDATIONS:
→ Reduce learning rate (try 0.0001 instead of 0.0005)
→ Tighten gradient clipping (max_grad_norm = 1.0 or 0.5)
→ Update target network more frequently (target_update_interval = 100)
→ Check reward scaling (current scale = 2.0, might need normalization)
→ Verify observation/state normalization
→ Implement gradient monitoring and early stopping
""")

print("="*80)
