#!/usr/bin/env python3
"""
Analyze trends by EPOCH instead of episode for more stable results.
Find which metrics actually improve during training.
"""

import pandas as pd
import numpy as np

print("="*70)
print("EPOCH-BASED TREND ANALYSIS")
print("="*70)

# Load episode metrics
df = pd.read_csv("/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/episode_metrics.csv")
print(f"\nTotal episodes logged: {len(df)}")
print(f"Episode range: {df['episode'].min()} - {df['episode'].max()}")
print(f"Epoch range: {df['epoch'].min()} - {df['epoch'].max()}")

# Aggregate by epoch
epoch_agg = df.groupby('epoch').agg({
    'episode_reward': 'mean',
    'wait_time': 'mean',
    'epsilon': 'mean'
}).reset_index()

print(f"\nTotal epochs: {len(epoch_agg)}")

# Compare early vs late training
n_early = 20
n_late = 20

early_epochs = epoch_agg.head(n_early)
late_epochs = epoch_agg.tail(n_late)

print("\n" + "="*70)
print(f"COMPARISON: First {n_early} epochs vs Last {n_late} epochs")
print("="*70)

# Episode Reward
early_reward = early_epochs['episode_reward'].mean()
late_reward = late_epochs['episode_reward'].mean()
reward_change = ((late_reward - early_reward) / abs(early_reward)) * 100

print(f"\n📈 EPISODE REWARD:")
print(f"   Early (epochs 0-{n_early-1}): {early_reward:.2f}")
print(f"   Late (epochs {len(epoch_agg)-n_late}-{len(epoch_agg)-1}): {late_reward:.2f}")
print(f"   Change: {reward_change:+.1f}%")
if reward_change > 0:
    print(f"   ✅ IMPROVEMENT!")
else:
    print(f"   ❌ No improvement")

# Wait Time
early_wait = early_epochs['wait_time'].mean()
late_wait = late_epochs['wait_time'].mean()
wait_change = ((late_wait - early_wait) / early_wait) * 100

print(f"\n⏱️  WAIT TIME:")
print(f"   Early: {early_wait:.2f}")
print(f"   Late: {late_wait:.2f}")
print(f"   Change: {wait_change:+.1f}%")
if wait_change < 0:
    print(f"   ✅ IMPROVEMENT (lower is better)!")
else:
    print(f"   ❌ No improvement")

# Now check training metrics (loss, TD error)
print("\n" + "="*70)
print("TRAINING QUALITY METRICS")
print("="*70)

df_train = pd.read_csv("/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/training_metrics.csv")
print(f"\nTotal training steps logged: {len(df_train)}")

# Group by episode to get cleaner trends
train_agg = df_train.groupby('episode').agg({
    'avg_loss': 'mean',
    'avg_td_error': 'mean',
    'avg_q_value': 'mean'
}).reset_index()

early_train = train_agg.head(50)
late_train = train_agg.tail(50)

# Loss
early_loss = early_train['avg_loss'].mean()
late_loss = late_train['avg_loss'].mean()
loss_change = ((late_loss - early_loss) / early_loss) * 100

print(f"\n📉 LOSS:")
print(f"   Early: {early_loss:.4f}")
print(f"   Late: {late_loss:.4f}")
print(f"   Change: {loss_change:+.1f}%")
if loss_change < 0:
    print(f"   ✅ IMPROVEMENT (lower is better)!")

# TD Error
early_td = early_train['avg_td_error'].mean()
late_td = late_train['avg_td_error'].mean()
td_change = ((late_td - early_td) / early_td) * 100

print(f"\n📊 TD ERROR:")
print(f"   Early: {early_td:.4f}")
print(f"   Late: {late_td:.4f}")
print(f"   Change: {td_change:+.1f}%")
if td_change < 0:
    print(f"   ✅ IMPROVEMENT (lower is better)!")

# Q-value
early_q = early_train['avg_q_value'].mean()
late_q = late_train['avg_q_value'].mean()
q_change = ((late_q - early_q) / abs(early_q)) * 100

print(f"\n🎯 Q-VALUE:")
print(f"   Early: {early_q:.4f}")
print(f"   Late: {late_q:.4f}")
print(f"   Change: {q_change:+.1f}%")

print("\n" + "="*70)
print("SUMMARY: METRICS WITH CLEAR IMPROVEMENT")
print("="*70)

improvements = []
if reward_change > 0:
    improvements.append(f"Episode Reward: +{reward_change:.1f}%")
if wait_change < 0:
    improvements.append(f"Wait Time: {wait_change:.1f}%")
if loss_change < 0:
    improvements.append(f"Loss: {loss_change:.1f}%")
if td_change < 0:
    improvements.append(f"TD Error: {td_change:.1f}%")

if improvements:
    for imp in improvements:
        print(f"✅ {imp}")
else:
    print("❌ No clear improvements found in standard metrics")

print("\n" + "="*70)
