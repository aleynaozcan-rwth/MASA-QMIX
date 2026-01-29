#!/usr/bin/env python3
"""
Standard Approach - DP Distribution Histogram: Fixed 25 DPs per episode
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard approach parameters
total_episodes = 1295  # Same as in the proposed framework
dps_per_episode = 25   # Fixed for standard approach
total_dps = total_episodes * dps_per_episode

print(f"Standard Approach - DP Distribution Statistics:")
print(f"  Total Episodes: {total_episodes}")
print(f"  DPs per Episode: {dps_per_episode} (fixed)")
print(f"  Total DPs: {total_dps:,}")

# All episodes have exactly 25 DPs
dp_counts = np.full(total_episodes, dps_per_episode)

# Calculate statistics
mean_dp = np.mean(dp_counts)
median_dp = np.median(dp_counts)
min_dp = np.min(dp_counts)
max_dp = np.max(dp_counts)

print(f"  Mean: {mean_dp:.1f}")
print(f"  Median: {median_dp:.0f}")
print(f"  Range: {min_dp} - {max_dp}")

# Create histogram
fig, ax = plt.subplots(figsize=(12, 5))

# Since all values are 25, create a single bar
bins = [24.5, 25.5]  # Bar centered at 25
ax.hist(dp_counts, bins=bins, color='cornflowerblue', edgecolor='black', 
         alpha=0.7, linewidth=1.2)

# Add mean line (which is also the median, mode, etc. since all values are the same)
ax.axvline(x=mean_dp, color='green', linestyle='--', linewidth=2.5, 
            alpha=0.8, label=f'Mean: {mean_dp:.1f}')

# Add legend info
from matplotlib.patches import Patch
legend_info_dps = Patch(facecolor='none', edgecolor='none', label=f'Total DPs: {total_dps:,}')
legend_info_episodes = Patch(facecolor='none', edgecolor='none', label=f'Total Episodes: {total_episodes:,}')
legend_info_sequential = Patch(facecolor='none', edgecolor='none', label='Sequential Decision Making: 10 Agents')

ax.set_xlabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Total Decision Points (Standard Approach)', 
              fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

# Set x-axis limits to show context around 25
ax.set_xlim(20, 30)

# Add legend with handles
handles, labels = ax.get_legend_handles_labels()
handles.extend([legend_info_dps, legend_info_episodes, legend_info_sequential])
ax.legend(handles=handles, loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()

output_path = OUTPUT_DIR / 'standard_dp_distribution_histogram.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved standard approach histogram: {output_path}")
plt.close()
