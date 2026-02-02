#!/usr/bin/env python3
"""
Plot Stochastic Job Arrivals Comparison: Standard vs Proposed
Shows the variability in job arrivals per episode
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/thesis_verification_data/arrivals_departures_per_episode.txt')

print("Loading arrival data for stochastic comparison...")

episodes = []
arrivals = []

with open(DATA_FILE, 'r') as f:
    next(f)  # Skip header
    for line in f:
        ep, arr, dep, ratio = line.strip().split(',')
        episodes.append(int(ep))
        arrivals.append(int(arr))

arrivals = np.array(arrivals)
episodes = np.array(episodes)

print(f"Loaded {len(episodes)} episodes")
print(f"Arrivals: min={min(arrivals)}, max={max(arrivals)}, mean={np.mean(arrivals):.2f}")

# Standard baseline: constant 10 jobs per episode
standard_arrivals = np.ones_like(arrivals) * 10

# Calculate moving average for proposed (smoothed line)
def moving_average(data, window_size=20):
    # Use 'same' mode to keep same length
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

ma_window = 20
arrivals_ma = moving_average(arrivals, ma_window)

# Skip first and last 10 episodes for cleaner visualization
skip = 10
episodes_ma = episodes[skip:-skip]
arrivals_ma = arrivals_ma[skip:-skip]

# Re-index episodes to start from 0
episodes_display = episodes_ma - episodes_ma[0]

# =============================================================================
# COMBINED FIGURE: Stochastic Arrivals Comparison
# =============================================================================

fig, ax = plt.subplots(figsize=(16, 6.2))

# Plot standard baseline (constant 10)
ax.axhline(y=10, color='#3498DB', linestyle='--', linewidth=2.5, 
           label='Standard (Constant 10 jobs/episode)', alpha=0.9, zorder=3)

# Fill area for standard (small range to show it's constant)
ax.fill_between(episodes_display, 9.8, 10.2, color='#3498DB', alpha=0.15, zorder=1)

# Plot moving average for proposed
ax.plot(episodes_display, arrivals_ma, color='#E67E22', linewidth=2.5, 
        label=f'Proposed (Moving Avg, window={ma_window}, Mean: {np.mean(arrivals):.2f})', alpha=0.9, zorder=4)

# Fill area between standard and proposed moving average
ax.fill_between(episodes_display, 10, arrivals_ma, 
                where=(arrivals_ma >= 10), color='#27AE60', alpha=0.25,
                label='Above Standard', zorder=1)
ax.fill_between(episodes_display, arrivals_ma, 10,
                where=(arrivals_ma < 10), color='#E74C3C', alpha=0.25,
                label='Below Standard', zorder=1)

# Labels and title
ax.set_xlabel('Episode Number', fontsize=14, fontweight='bold')
ax.set_ylabel('Job Arrivals per Episode', fontsize=14, fontweight='bold')
ax.set_title('Job Arrivals per Episode: Standard vs Proposed', 
             fontsize=16, fontweight='bold', pad=15)

# Set axis limits - start from 0
ax.set_xlim(0, episodes_display[-1])  # Start from 0
ax.set_ylim(8, 13)  # Show from 8 to 13

# Grid
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

plt.tight_layout()
output_file = OUTPUT_DIR / 'stochastic_arrivals_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_file}")
plt.close()

print(f"\n✓ Stochastic arrivals comparison completed!")
