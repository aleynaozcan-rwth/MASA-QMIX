#!/usr/bin/env python3
"""
Standard Approach - Total Decision Points only: Fixed 25 DPs per episode.
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

print(f"Standard Approach - Total Decision Points Statistics:")
print(f"  Total Episodes: {total_episodes}")
print(f"  DPs per Episode: {dps_per_episode} (fixed)")
print(f"  Total DPs: {total_dps:,}")

# Create episode numbers and constant DPs
episode_numbers = np.arange(total_episodes)
total_dps_array = np.full(total_episodes, dps_per_episode)

# Create simple visualization with only total DPs
fig, ax = plt.subplots(figsize=(20, 10))

# Left y-axis: Total DPs
ax.set_xlabel('Episode', fontsize=18, fontweight='heavy')
ax.set_ylabel('Total Decision Points per Episode', fontsize=14, fontweight='heavy')

# Plot constant line at 25
ax.plot(episode_numbers, total_dps_array, linewidth=3, color='#2E86C1', alpha=0.9, zorder=3,
        label=f'Total DPs (fixed at {dps_per_episode}, total: {total_dps:,})')

ax.set_xlim(0, len(episode_numbers) - 1)
# Y-axis from 15 to 35
ax.set_ylim(15, 35)
ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

# Right legend - episode information
from matplotlib.lines import Line2D
info_lines = [
    Line2D([0], [0], color='none', label=f'Total Episodes: {total_episodes:,}'),
    Line2D([0], [0], color='none', label='Sequential Decision Making: 10 Agents')
]

from matplotlib.legend import Legend
leg2 = Legend(ax, info_lines, [l.get_label() for l in info_lines],
             loc='upper right', fontsize=10, framealpha=0.95)
leg2._legend_box.align = 'left'
ax.add_artist(leg2)

plt.title('Total Decision Points per Episode (Standard Approach)', 
            fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()

output_path = OUTPUT_DIR / 'standard_dps_only.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved standard approach DPs only plot: {output_path}")
plt.close()
