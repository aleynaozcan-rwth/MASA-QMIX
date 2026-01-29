#!/usr/bin/env python3
"""
Distinguish between:
1. "Agents learned optimal uniform distribution" (GOOD)
2. "Agents still random, no learning" (BAD)

Method: Check if wait time improves WHILE LoadBalance stays high
- If wait time decreases BUT LoadBalance stable → LEARNED optimal uniform
- If wait time stable AND LoadBalance stable → NO LEARNING (random policy)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("LOAD BALANCE vs WAIT TIME: LEARNING DIAGNOSTIC")
print("="*70)

# Load data
df_episode = pd.read_csv("/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/episode_metrics.csv")
df_reward = pd.read_csv("/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/reward_components.csv", comment='#')

# Aggregate by episode
df_reward['episode'] = (df_reward['step'] // 50) + 1
lb_by_episode = df_reward.groupby('episode')['LoadBalance'].mean().reset_index()

# Merge
df = df_episode.merge(lb_by_episode, on='episode', how='left')

# Aggregate by epoch for cleaner signal
df_epoch = df.groupby('epoch').agg({
    'wait_time': 'mean',
    'LoadBalance': 'mean',
    'epsilon': 'mean',
    'episode_reward': 'mean'
}).reset_index()

print(f"\nTotal epochs: {len(df_epoch)}")

# Split into phases
n_epochs = len(df_epoch)
phase1 = df_epoch.iloc[:n_epochs//3]  # First third (high exploration)
phase2 = df_epoch.iloc[n_epochs//3:2*n_epochs//3]  # Middle third
phase3 = df_epoch.iloc[2*n_epochs//3:]  # Last third (low exploration)

print("\n" + "="*70)
print("PHASE ANALYSIS")
print("="*70)

print(f"\n📊 PHASE 1 (High Exploration): Epochs 0-{len(phase1)-1}")
print(f"   Epsilon: {phase1['epsilon'].mean():.3f}")
print(f"   Wait Time: {phase1['wait_time'].mean():.2f}")
print(f"   LoadBalance: {phase1['LoadBalance'].mean():.4f}")
print(f"   Episode Reward: {phase1['episode_reward'].mean():.2f}")

print(f"\n📊 PHASE 2 (Medium Exploration): Epochs {len(phase1)}-{len(phase1)+len(phase2)-1}")
print(f"   Epsilon: {phase2['epsilon'].mean():.3f}")
print(f"   Wait Time: {phase2['wait_time'].mean():.2f}")
print(f"   LoadBalance: {phase2['LoadBalance'].mean():.4f}")
print(f"   Episode Reward: {phase2['episode_reward'].mean():.2f}")

print(f"\n📊 PHASE 3 (Low Exploration): Epochs {len(phase1)+len(phase2)}-{n_epochs-1}")
print(f"   Epsilon: {phase3['epsilon'].mean():.3f}")
print(f"   Wait Time: {phase3['wait_time'].mean():.2f}")
print(f"   LoadBalance: {phase3['LoadBalance'].mean():.4f}")
print(f"   Episode Reward: {phase3['episode_reward'].mean():.2f}")

# Calculate changes
wait_change = ((phase3['wait_time'].mean() - phase1['wait_time'].mean()) / phase1['wait_time'].mean()) * 100
lb_change = ((phase3['LoadBalance'].mean() - phase1['LoadBalance'].mean()) / phase1['LoadBalance'].mean()) * 100
reward_change = ((phase3['episode_reward'].mean() - phase1['episode_reward'].mean()) / abs(phase1['episode_reward'].mean())) * 100

print("\n" + "="*70)
print("CHANGES FROM PHASE 1 → PHASE 3")
print("="*70)
print(f"Wait Time: {wait_change:+.1f}%")
print(f"LoadBalance: {lb_change:+.1f}%")
print(f"Episode Reward: {reward_change:+.1f}%")

print("\n" + "="*70)
print("DIAGNOSTIC INTERPRETATION")
print("="*70)

# Decision logic
if wait_change < -3 and abs(lb_change) < 2:
    verdict = "✅ LEARNED OPTIMAL UNIFORM DISTRIBUTION"
    explanation = """
Wait time DECREASED while LoadBalance stayed HIGH and STABLE.
This means agents learned to:
  - Choose appropriate machines for each job type
  - Maintain load balance naturally (heterogeneous workload)
  - Improve efficiency without creating bottlenecks

CONCLUSION: The optimal policy for your heterogeneous job shop
naturally produces uniform machine distribution!
"""
elif wait_change > -3 and abs(lb_change) < 2:
    verdict = "❌ NO SIGNIFICANT LEARNING (Random-like behavior)"
    explanation = """
Wait time did NOT decrease and LoadBalance stayed stable.
This suggests:
  - Agents behave similar to random policy
  - No clear learning signal
  - May need higher reward weights or better exploration

PROBLEM: Learning is not effective.
"""
elif wait_change < -3 and lb_change < -5:
    verdict = "⚠️  LEARNED SPECIALIZATION (Check for bottlenecks)"
    explanation = """
Wait time decreased AND LoadBalance decreased significantly.
This means agents learned to specialize on certain machines.
Could be:
  ✅ Good if certain machines are objectively faster
  ❌ Bad if creating bottlenecks

RECOMMENDATION: Check machine utilization variance.
"""
else:
    verdict = "🤔 MIXED SIGNALS (Need deeper analysis)"
    explanation = """
Complex pattern detected. Requires detailed investigation of:
  - Job type heterogeneity
  - Machine capabilities
  - Operator assignment patterns
"""

print(f"\n{verdict}")
print(explanation)

# Check LoadBalance weight impact
print("\n" + "="*70)
print("REWARD WEIGHT ANALYSIS")
print("="*70)

w5 = 0.3  # LoadBalance weight from CSV
w1 = 3.0  # Completion weight
w4 = 4.0  # Throughput weight

# Calculate typical contribution
avg_lb = df_epoch['LoadBalance'].mean()
lb_contribution = w5 * avg_lb

# Compare to other components (estimate)
typical_completed = 0.6  # ~60% completion rate
typical_throughput = 0.1  # normalized delta

completed_contribution = w1 * typical_completed
throughput_contribution = w4 * typical_throughput

print(f"\nTypical reward contributions:")
print(f"  Completion (w1={w1}): {completed_contribution:.2f}")
print(f"  Throughput (w4={w4}): {throughput_contribution:.2f}")
print(f"  LoadBalance (w5={w5}): {lb_contribution:.2f}")

lb_percentage = (lb_contribution / (completed_contribution + throughput_contribution + lb_contribution)) * 100
print(f"\nLoadBalance contribution: {lb_percentage:.1f}% of positive rewards")

if lb_percentage < 10:
    print(f"\n⚠️  LoadBalance weight is TOO LOW ({lb_percentage:.1f}%)")
    print("Agents barely optimize for it! This explains why it doesn't change.")
    print(f"\nTo make agents care about LoadBalance:")
    print(f"  Current w5={w5} → Suggest w5={w5*3:.1f} (3x increase)")
elif lb_percentage < 20:
    print(f"\n⚠️  LoadBalance weight is LOW ({lb_percentage:.1f}%)")
    print("Agents somewhat optimize for it, but other factors dominate.")
else:
    print(f"\n✅ LoadBalance weight is SIGNIFICANT ({lb_percentage:.1f}%)")
    print("Agents actively optimize for load balance.")

print("\n" + "="*70)
