#!/usr/bin/env python3
"""
Plot Job Arrivals vs Departures per Episode
Two visualizations:
1. Dual-line chart showing arrivals and departures
2. Completion ratio chart
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/arrivals_departures_per_episode.txt')

print("Loading arrival/departure data...")

episodes = []
arrivals = []
departures = []
ratios = []

filtered_count = 0
with open(DATA_FILE, 'r') as f:
    next(f)  # Skip header
    for line in f:
        ep, arr, dep, ratio = line.strip().split(',')
        # Filter out training failure episodes (0 departures = no jobs completed)
        if int(dep) > 0:
            episodes.append(int(ep))
            arrivals.append(int(arr))
            departures.append(int(dep))
            ratios.append(float(ratio))
        else:
            filtered_count += 1

print(f"Loaded {len(episodes)} episodes (filtered {filtered_count} training failures with 0 departures)")
print(f"Arrivals: {sum(arrivals)}, Departures: {sum(departures)}")
print(f"Overall Ratio: {sum(departures)/sum(arrivals):.2%}")

# =============================================================================
# CHART 1: DUAL-LINE ARRIVALS vs DEPARTURES
# =============================================================================

fig, ax = plt.subplots(figsize=(20, 10))

# Plot arrivals and departures
ax.plot(episodes, arrivals, linewidth=2.0, color='#8E44AD', alpha=0.3, 
        label=f'Job Arrivals (Total: {sum(arrivals)})', marker='o', markersize=3, markevery=50)
ax.plot(episodes, departures, linewidth=2.0, color='#E67E22', alpha=0.3,
        label=f'Job Departures (Total: {sum(departures)})', marker='s', markersize=3, markevery=50)

# Add moving averages to smooth the signals
window_size = 20
arrivals_ma = np.convolve(arrivals, np.ones(window_size)/window_size, mode='valid')
departures_ma = np.convolve(departures, np.ones(window_size)/window_size, mode='valid')
ma_episodes = episodes[window_size-1:]

ax.plot(ma_episodes, arrivals_ma, linewidth=3.5, color='#5B2C6F', alpha=0.95,
        label=f'Arrivals Moving Avg (window={window_size})', linestyle='-', zorder=6)
ax.plot(ma_episodes, departures_ma, linewidth=3.5, color='#CA6F1E', alpha=0.95,
        label=f'Departures Moving Avg (window={window_size})', linestyle='-', zorder=6)

# Add average lines
avg_arrivals = sum(arrivals) / len(arrivals)
avg_departures = sum(departures) / len(departures)

ax.axhline(y=avg_arrivals, color='#8E44AD', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Avg Arrivals: {avg_arrivals:.2f}')
ax.axhline(y=avg_departures, color='#E67E22', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Avg Departures: {avg_departures:.2f}')

ax.set_xlabel('Episode Number', fontsize=18, fontweight='bold')
ax.set_ylabel('Number of Jobs', fontsize=18, fontweight='bold')
ax.set_title('Job Arrivals vs Departures per Episode\n(Proposed Framework - Event-Driven Scheduling)', 
             fontsize=20, fontweight='bold', pad=20)

ax.set_xlim(0, max(episodes))
ax.set_ylim(0, max(max(arrivals), max(departures)) + 2)

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='upper right', fontsize=14, framealpha=0.95)

ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
output1 = OUTPUT_DIR / 'job_arrivals_vs_departures.png'
plt.savefig(output1, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved Chart 1: {output1}")
plt.close()

# =============================================================================
# CHART 2: COMPLETION RATIO
# =============================================================================

fig, ax = plt.subplots(figsize=(20, 10))

# Add background color zones for completion ranges
ax.axhspan(0.0, 0.3, alpha=0.15, color='#E74C3C', label='Low Completion (<30%)', zorder=0)
ax.axhspan(0.3, 0.7, alpha=0.15, color='#F39C12', label='Medium Completion (30-70%)', zorder=0)
ax.axhspan(0.7, 1.005, alpha=0.15, color='#2E86C1', label='High Completion (≥70%)', zorder=0)

# Plot completion ratio (green line on top of colored zones)
ax.plot(episodes, ratios, linewidth=1.0, color='#27AE60', alpha=0.9,
        label=f'Completion Ratio (Avg: {np.mean(ratios):.2%})', zorder=5)

# Add moving average to smooth the signal
window_size = 20
moving_avg = np.convolve(ratios, np.ones(window_size)/window_size, mode='valid')
moving_avg_episodes = episodes[window_size-1:]
ax.plot(moving_avg_episodes, moving_avg, linewidth=3.5, color='#145A32', alpha=0.95,
        label=f'Moving Average (window={window_size})', linestyle='-', zorder=6)

# Add reference lines
ax.axhline(y=1.0, color='#2E86C1', linestyle='--', linewidth=2.0, alpha=0.8,
           label='100% Completion (All jobs finished)', zorder=3)
ax.axhline(y=0.7, color='#F39C12', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
ax.axhline(y=0.3, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
ax.axhline(y=np.mean(ratios), color='#27AE60', linestyle=':', linewidth=2.0, alpha=0.8,
           label=f'Average: {np.mean(ratios):.2%}', zorder=3)

ax.set_xlabel('Episode Number', fontsize=18, fontweight='bold')
ax.set_ylabel('Completion Ratio (Departures / Arrivals)', fontsize=18, fontweight='bold')
ax.set_title('Job Completion Ratio per Episode\n(Ratio of Departed Jobs to Arrived Jobs)', 
             fontsize=20, fontweight='bold', pad=20)

ax.set_xlim(0, max(episodes))
ax.set_ylim(0, 1.005)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='lower right', fontsize=13, framealpha=0.95, ncol=2)

ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
output2 = OUTPUT_DIR / 'job_completion_ratio.png'
plt.savefig(output2, dpi=150, bbox_inches='tight')
print(f"✓ Saved Chart 2: {output2}")
plt.close()

# =============================================================================
# CHART 3: COMBINED VIEW (DUAL Y-AXIS)
# =============================================================================

fig, ax1 = plt.subplots(figsize=(20, 10))

# Left y-axis: Arrivals and Departures
color1 = '#8E44AD'  # Purple for arrivals
color2 = '#E67E22'  # Orange for departures

# Calculate moving averages (no raw data, only smooth trends)
window_size = 20
arrivals_ma = np.convolve(arrivals, np.ones(window_size)/window_size, mode='valid')
departures_ma = np.convolve(departures, np.ones(window_size)/window_size, mode='valid')
ma_episodes = episodes[window_size-1:]

ax1.plot(ma_episodes, arrivals_ma, linewidth=3.5, color='#5B2C6F', alpha=0.95,
         label=f'Arrivals Moving Avg (window={window_size})', linestyle='-')
ax1.plot(ma_episodes, departures_ma, linewidth=3.5, color='#CA6F1E', alpha=0.95,
         label=f'Departures Moving Avg (window={window_size})', linestyle='-')

ax1.set_xlabel('Episode Number', fontsize=18, fontweight='bold')
ax1.set_ylabel('Number of Jobs', fontsize=18, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelsize=14, labelcolor='black')
ax1.tick_params(axis='x', labelsize=14)

ax1.set_xlim(0, max(episodes))
ax1.set_ylim(0, max(max(arrivals), max(departures)) + 2)

# Right y-axis: Current Jobs in System (Arrivals - Departures)
ax2 = ax1.twinx()
color3 = '#27AE60'

# Calculate current jobs (cumulative arrivals - cumulative departures per episode)
current_jobs = [arr - dep for arr, dep in zip(arrivals, departures)]
current_jobs_ma = np.convolve(current_jobs, np.ones(window_size)/window_size, mode='valid')

ax2.plot(ma_episodes, current_jobs_ma, linewidth=3.0, color=color3, alpha=0.9, linestyle='--',
         label=f'Current Jobs in System (Avg: {np.mean(current_jobs):.2f})', marker='D', markersize=3, markevery=50)

ax2.set_ylabel('Current Jobs in System', fontsize=18, fontweight='bold', color=color3)
ax2.tick_params(axis='y', labelsize=14, labelcolor=color3)
ax2.set_ylim(0, max(current_jobs) + 2)

# Title
ax1.set_title('Job Arrivals, Departures, and Current Jobs in System per Episode\n(Proposed Framework)', 
              fontsize=20, fontweight='bold', pad=20)

# Grid
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14, framealpha=0.95)

plt.tight_layout()
output3 = OUTPUT_DIR / 'job_arrivals_departures_combined.png'
plt.savefig(output3, dpi=150, bbox_inches='tight')
print(f"✓ Saved Chart 3: {output3}")
plt.close()

print(f"\n✓ All visualizations completed!")
print(f"  - Dual-line chart: arrivals vs departures")
print(f"  - Completion ratio chart with color zones")
print(f"  - Combined view with dual y-axis")
