#!/usr/bin/env python3
"""
Generate diagnostic plots for failed training run
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Load data
print("Loading data...")
loss_df = pd.read_csv('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/loss.txt', sep=' ')
td_error_df = pd.read_csv('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/td_error.txt', sep=' ')
training_df = pd.read_csv('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/training_metrics.csv')
episode_df = pd.read_csv('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/episode_metrics.csv')

# Create output directory
os.makedirs('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots', exist_ok=True)

# ==============================================================================
# Figure 1: Training Explosion Timeline
# ==============================================================================
print("Generating Figure 1: Training Explosion Timeline...")
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Loss over time (log scale)
ax1 = fig.add_subplot(gs[0, :])
ax1.semilogy(loss_df['train_step'], loss_df['last_loss'], linewidth=0.5, alpha=0.7)
ax1.axvline(x=901, color='red', linestyle='--', label='Explosion Start (step 901)', linewidth=2)
ax1.axhline(y=1000, color='orange', linestyle=':', label='Danger Threshold (1000)', linewidth=2)
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('Loss (log scale)', fontsize=12)
ax1.set_title('Loss Explosion Over Time', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.text(901, 1000, 'EXPLOSION\nSTARTS HERE', 
         fontsize=10, color='red', fontweight='bold',
         ha='left', va='bottom')

# Plot 2: TD Error over time (log scale)
ax2 = fig.add_subplot(gs[1, 0])
ax2.semilogy(td_error_df['train_step'], td_error_df['last_td'], linewidth=0.5, alpha=0.7, color='orange')
ax2.axvline(x=901, color='red', linestyle='--', linewidth=2)
ax2.axhline(y=100, color='orange', linestyle=':', linewidth=2)
ax2.set_xlabel('Training Step', fontsize=12)
ax2.set_ylabel('TD Error (log scale)', fontsize=12)
ax2.set_title('TD Error Divergence', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Q-values over time
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(training_df['train_step'], training_df['avg_q_value'], linewidth=0.5, alpha=0.7, color='green')
ax3.axvline(x=901, color='red', linestyle='--', linewidth=2)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax3.set_xlabel('Training Step', fontsize=12)
ax3.set_ylabel('Average Q-value', fontsize=12)
ax3.set_title('Q-value Explosion', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Episode rewards
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(episode_df['episode'], episode_df['episode_reward'], linewidth=1, alpha=0.7, color='blue')
ax4.axhline(y=episode_df['episode_reward'].mean(), color='green', linestyle='--', 
            label=f'Mean: {episode_df["episode_reward"].mean():.2f}', linewidth=2)
ax4.fill_between(episode_df['episode'], 
                  episode_df['episode_reward'].mean() - episode_df['episode_reward'].std(),
                  episode_df['episode_reward'].mean() + episode_df['episode_reward'].std(),
                  alpha=0.2, color='green', label=f'±1 std ({episode_df["episode_reward"].std():.2f})')
ax4.set_xlabel('Episode', fontsize=12)
ax4.set_ylabel('Episode Reward', fontsize=12)
ax4.set_title('Reward: No Convergence (Flat)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: Epsilon decay
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(episode_df['episode'], episode_df['epsilon'], linewidth=1, alpha=0.7, color='purple')
ax5.set_xlabel('Episode', fontsize=12)
ax5.set_ylabel('Epsilon (Exploration)', fontsize=12)
ax5.set_title('Exploration Rate Decay', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

plt.suptitle('MASA-QMIX Training Failure: Gradient Explosion Timeline', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/training_failure_timeline.png', 
            dpi=150, bbox_inches='tight')
print("✅ Saved: training_failure_timeline.png")
plt.close()

# ==============================================================================
# Figure 2: Before vs After Explosion Comparison
# ==============================================================================
print("Generating Figure 2: Before vs After Explosion...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Define explosion point
explosion_step = 901
before_mask = loss_df['train_step'] <= explosion_step
after_mask = loss_df['train_step'] > explosion_step

# Before explosion data
loss_before = loss_df[before_mask]
td_before = td_error_df[before_mask]
train_before = training_df[training_df['train_step'] <= explosion_step]

# After explosion data (first 1000 steps after)
loss_after = loss_df[after_mask].iloc[:1000]
td_after = td_error_df[after_mask].iloc[:1000]
train_after = training_df[(training_df['train_step'] > explosion_step)].iloc[:1000]

# Row 1: Before explosion
axes[0, 0].plot(loss_before['train_step'], loss_before['last_loss'], linewidth=1, color='blue')
axes[0, 0].set_title('Loss: BEFORE Explosion\n(Steps 1-901)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Training Step')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(td_before['train_step'], td_before['last_td'], linewidth=1, color='blue')
axes[0, 1].set_title('TD Error: BEFORE Explosion\n(Steps 1-901)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Training Step')
axes[0, 1].set_ylabel('TD Error')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(train_before['train_step'], train_before['avg_q_value'], linewidth=1, color='blue')
axes[0, 2].set_title('Q-values: BEFORE Explosion\n(Steps 1-901)', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Training Step')
axes[0, 2].set_ylabel('Avg Q-value')
axes[0, 2].grid(True, alpha=0.3)

# Row 2: After explosion
axes[1, 0].semilogy(loss_after['train_step'], loss_after['last_loss'], linewidth=1, color='red')
axes[1, 0].set_title('Loss: AFTER Explosion\n(Steps 902-1902)', fontsize=12, fontweight='bold', color='red')
axes[1, 0].set_xlabel('Training Step')
axes[1, 0].set_ylabel('Loss (log scale)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].semilogy(td_after['train_step'], td_after['last_td'], linewidth=1, color='red')
axes[1, 1].set_title('TD Error: AFTER Explosion\n(Steps 902-1902)', fontsize=12, fontweight='bold', color='red')
axes[1, 1].set_xlabel('Training Step')
axes[1, 1].set_ylabel('TD Error (log scale)')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(train_after['train_step'], train_after['avg_q_value'], linewidth=1, color='red')
axes[1, 2].set_title('Q-values: AFTER Explosion\n(Steps 902-1902)', fontsize=12, fontweight='bold', color='red')
axes[1, 2].set_xlabel('Training Step')
axes[1, 2].set_ylabel('Avg Q-value')
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Training Metrics: Before vs After Explosion', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/before_after_explosion.png', 
            dpi=150, bbox_inches='tight')
print("✅ Saved: before_after_explosion.png")
plt.close()

# ==============================================================================
# Figure 3: Statistical Summary
# ==============================================================================
print("Generating Figure 3: Statistical Summary...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Reward distribution
axes[0, 0].hist(episode_df['episode_reward'], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].axvline(episode_df['episode_reward'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {episode_df["episode_reward"].mean():.2f}')
axes[0, 0].set_xlabel('Episode Reward', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Reward Distribution (No Improvement)', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss distribution (before explosion)
axes[0, 1].hist(loss_df[before_mask]['last_loss'], bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].set_xlabel('Loss (Before Explosion)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Loss Distribution: Healthy Phase', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# TD Error distribution (before explosion)
axes[1, 0].hist(td_error_df[before_mask]['last_td'], bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].set_xlabel('TD Error (Before Explosion)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('TD Error Distribution: Healthy Phase', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Rolling statistics
window = 100
rolling_mean = episode_df['episode_reward'].rolling(window=window).mean()
rolling_std = episode_df['episode_reward'].rolling(window=window).std()
axes[1, 1].plot(episode_df['episode'], rolling_mean, linewidth=2, label='Rolling Mean (100 ep)', color='blue')
axes[1, 1].fill_between(episode_df['episode'], 
                         rolling_mean - rolling_std, 
                         rolling_mean + rolling_std,
                         alpha=0.2, color='blue', label='±1 std')
axes[1, 1].set_xlabel('Episode', fontsize=12)
axes[1, 1].set_ylabel('Reward', fontsize=12)
axes[1, 1].set_title('Reward: No Upward Trend', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Statistical Analysis: No Convergence Detected', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/statistical_summary.png', 
            dpi=150, bbox_inches='tight')
print("✅ Saved: statistical_summary.png")
plt.close()

print("\n" + "="*80)
print("All diagnostic plots generated successfully!")
print("="*80)
print("\nGenerated files:")
print("  1. training_failure_timeline.png - Overall explosion timeline")
print("  2. before_after_explosion.png - Comparison before/after step 901")
print("  3. statistical_summary.png - Statistical distributions")
print("\nLocation: /home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/")
print("="*80)
