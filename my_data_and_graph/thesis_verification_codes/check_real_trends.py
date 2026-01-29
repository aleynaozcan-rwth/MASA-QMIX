#!/usr/bin/env python3
"""
Quick check of actual utilization trends from decision_state_metrics.csv
"""

import pandas as pd
import numpy as np

# Load decision-level data
print("Loading decision_state_metrics.csv...")
df = pd.read_csv("/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/decision_state_metrics.csv")
print(f"Total decisions: {len(df)}")

# Calculate episode numbers (assuming ~200 decisions per episode on average)
# Better: use decision_time to group episodes
df['episode'] = ((df.index // 200) + 1)  # Rough estimate

# Aggregate by episode
episode_agg = df.groupby('episode').agg({
    'state_avg_machine_util': 'mean',
    'state_avg_operator_util': 'mean'
}).reset_index()

print(f"\nEpisodes in decision data: {len(episode_agg)}")
print(f"Episode range: {episode_agg['episode'].min()} - {episode_agg['episode'].max()}")

# Compare first vs last 30 episodes
initial = episode_agg.head(30)
final = episode_agg.tail(30)

print("\n" + "="*60)
print("MACHINE UTILIZATION TREND")
print("="*60)
print(f"Initial (first 30 eps): {initial['state_avg_machine_util'].mean():.4f}")
print(f"Final (last 30 eps): {final['state_avg_machine_util'].mean():.4f}")
change_m = ((final['state_avg_machine_util'].mean() - initial['state_avg_machine_util'].mean()) / initial['state_avg_machine_util'].mean()) * 100
print(f"Change: {change_m:+.2f}%")

print("\n" + "="*60)
print("OPERATOR UTILIZATION TREND")
print("="*60)
print(f"Initial (first 30 eps): {initial['state_avg_operator_util'].mean():.4f}")
print(f"Final (last 30 eps): {final['state_avg_operator_util'].mean():.4f}")
change_o = ((final['state_avg_operator_util'].mean() - initial['state_avg_operator_util'].mean()) / initial['state_avg_operator_util'].mean()) * 100
print(f"Change: {change_o:+.2f}%")

# Check reward components LoadBalance
print("\n" + "="*60)
print("LOAD BALANCE (from reward_components.csv)")
print("="*60)
df_reward = pd.read_csv("/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/reward_components.csv", comment='#')
df_reward['episode'] = (df_reward['step'] // 50) + 1
lb_agg = df_reward.groupby('episode')['LoadBalance'].mean().reset_index()
print(f"Episodes in reward data: {len(lb_agg)}")
initial_lb = lb_agg.head(30)['LoadBalance'].mean()
final_lb = lb_agg.tail(30)['LoadBalance'].mean()
print(f"Initial: {initial_lb:.4f}")
print(f"Final: {final_lb:.4f}")
change_lb = ((final_lb - initial_lb) / initial_lb) * 100
print(f"Change: {change_lb:+.2f}%")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print(f"LoadBalance metric increased by {change_lb:.1f}%")
print(f"But machine util changed by {change_m:.1f}%")
print(f"And operator util changed by {change_o:.1f}%")
print("\nLoadBalance is ENTROPY-BASED (variance), not raw utilization!")
print("Higher entropy = more balanced distribution across machines/operators")
