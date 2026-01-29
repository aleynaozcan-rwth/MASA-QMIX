#!/usr/bin/env python3
"""
Plot decision points per episode
Shows all episodes on x-axis and number of decision points on y-axis
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend for cluster without display
import matplotlib
matplotlib.use('Agg')

# ==================== Configuration ====================
TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Parse Timeline File ====================
print("Parsing scheduling timeline...")

episodes = []  # List of {episode: int, dp_count: int}
current_episode = None
current_dp_count = 0

with open(TIMELINE_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Match episode start: "=== EPISODE 0 ==="
        episode_match = re.match(r'^=== EPISODE (\d+) ===$', line)
        if episode_match:
            # Save previous episode if exists
            if current_episode is not None:
                episodes.append({
                    'episode': current_episode,
                    'dp_count': current_dp_count
                })
            
            # Start new episode
            current_episode = int(episode_match.group(1))
            current_dp_count = 0
            continue
        
        # Match decision point events:
        # 1. Job arrivals: "[t=X.XX] New job ... arrived"
        if '[t=' in line and 'New job' in line and 'arrived' in line:
            current_dp_count += 1
            continue
        
        # 2. Operation completions: "[t=X.XX] ... finished ... next queued"
        if '[t=' in line and 'finished' in line and 'next queued' in line:
            current_dp_count += 1
            continue

# Save last episode
if current_episode is not None:
    episodes.append({
        'episode': current_episode,
        'dp_count': current_dp_count
    })

print(f"Parsed {len(episodes)} episodes")

# ==================== Extract Data ====================
episode_numbers = [ep['episode'] for ep in episodes]
dp_counts = [ep['dp_count'] for ep in episodes]

# Calculate statistics
mean_dp = np.mean(dp_counts)
std_dp = np.std(dp_counts)
min_dp = np.min(dp_counts)
max_dp = np.max(dp_counts)
total_dp = np.sum(dp_counts)

print(f"\nDecision Points Statistics:")
print(f"  Total DPs: {total_dp:,}")
print(f"  Mean DPs per episode: {mean_dp:.2f}")
print(f"  Std Dev: {std_dp:.2f}")
print(f"  Range: {min_dp} - {max_dp}")

# ==================== Create Plot ====================
fig, ax = plt.subplots(figsize=(14, 6))

# Bar plot for decision points per episode
ax.bar(episode_numbers, dp_counts, width=1.0, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.5)

# Add mean line
ax.axhline(y=mean_dp, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dp:.2f}', alpha=0.8)

# Add std dev bands
ax.axhspan(mean_dp - std_dp, mean_dp + std_dp, alpha=0.15, color='red', 
           label=f'±1 Std Dev: {std_dp:.2f}', zorder=0)

# Labels and title
ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax.set_title('Decision Points Distribution Across Episodes', fontsize=14, fontweight='bold', pad=15)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Legend
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Tight layout
plt.tight_layout()

# Save
output_path = OUTPUT_DIR / 'decision_points_per_episode.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved decision points per episode: {output_path}")

plt.close()
