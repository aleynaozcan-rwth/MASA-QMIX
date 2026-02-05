#!/usr/bin/env python3
"""Plot training-only episode rewards (evaluation episodes removed)"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Read training-only rewards
rewards_file = "./my_data_and_graph/historydata/episode_rewards_training_only.txt"
episodes = []
rewards = []

with open(rewards_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or ',' not in line:
            continue
        parts = line.split(',')
        try:
            ep = int(parts[0])
            reward = float(parts[1])
            episodes.append(ep)
            rewards.append(reward)
        except:
            continue

# Calculate moving average
window = 50
ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
ma_episodes = episodes[window-1:]

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot raw rewards
ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Rewards')

# Plot moving average
ax.plot(ma_episodes, ma_rewards, color='red', linewidth=2, label=f'Moving Average (window={window})')

ax.set_xlabel('Training Episode', fontsize=12)
ax.set_ylabel('Episode Reward', fontsize=12)
ax.set_title('Training Episode Rewards (Evaluation Episodes Removed)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add statistics text
stats_text = f'Total Episodes: {len(episodes)}\n'
stats_text += f'Epochs: {len(episodes)/4:.1f}\n'
stats_text += f'Avg Reward: {np.mean(rewards):.2f}\n'
stats_text += f'Std Dev: {np.std(rewards):.2f}'

ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10)

plt.tight_layout()

# Save
output_dir = "./my_data_and_graph/historydata"
output_file = os.path.join(output_dir, "training_only_rewards.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {output_file}")

plt.close()

print(f"\n✓ Training episodes: {len(episodes)}")
print(f"✓ Epochs run: {len(episodes)/4:.1f}")
print(f"✓ Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
