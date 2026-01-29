#!/usr/bin/env python3
"""
Plot Operation Arrivals vs Completions per Episode
More granular than job-level analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/operation_arrivals_completions_per_episode.txt')

print("Loading operation arrival/completion data...")

episodes = []
op_arrivals = []
op_completions = []
ratios = []

filtered_count = 0
with open(DATA_FILE, 'r') as f:
    next(f)  # Skip header
    for line in f:
        ep, arr, comp, ratio = line.strip().split(',')
        # Filter out training failure episodes (0 completions = no scheduling occurred)
        if int(comp) > 0:
            episodes.append(int(ep))
            op_arrivals.append(int(arr))
            op_completions.append(int(comp))
            ratios.append(float(ratio))
        else:
            filtered_count += 1

print(f"Loaded {len(episodes)} episodes (filtered {filtered_count} training failures with 0 completions)")
print(f"Total Op Arrivals: {sum(op_arrivals)}, Total Op Completions: {sum(op_completions)}")
print(f"Overall Ratio: {sum(op_completions)/sum(op_arrivals):.2%}")

# =============================================================================
# CHART 1: DUAL-LINE OPERATION ARRIVALS vs COMPLETIONS
# =============================================================================

fig, ax = plt.subplots(figsize=(20, 10))

# Plot arrivals and completions
ax.plot(episodes, op_arrivals, linewidth=2.0, color='#2E86C1', alpha=0.8, 
        label=f'Operation Arrivals (Total: {sum(op_arrivals)})', marker='o', markersize=3, markevery=50)
ax.plot(episodes, op_completions, linewidth=2.0, color='#E74C3C', alpha=0.8,
        label=f'Operation Completions (Total: {sum(op_completions)})', marker='s', markersize=3, markevery=50)

# Add average lines
avg_arrivals = sum(op_arrivals) / len(op_arrivals)
avg_completions = sum(op_completions) / len(op_completions)

ax.axhline(y=avg_arrivals, color='#2E86C1', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Avg Op Arrivals: {avg_arrivals:.2f}')
ax.axhline(y=avg_completions, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.5,
           label=f'Avg Op Completions: {avg_completions:.2f}')

ax.set_xlabel('Episode Number', fontsize=18, fontweight='bold')
ax.set_ylabel('Number of Operations', fontsize=18, fontweight='bold')
ax.set_title('Operation Arrivals vs Completions per Episode\n(Operation-Level Analysis - More Granular than Job-Level)', 
             fontsize=20, fontweight='bold', pad=20)

ax.set_xlim(0, max(episodes))
ax.set_ylim(0, max(max(op_arrivals), max(op_completions)) + 5)

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='upper right', fontsize=14, framealpha=0.95)

ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
output1 = OUTPUT_DIR / 'operation_arrivals_vs_completions.png'
plt.savefig(output1, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved Chart 1: {output1}")
plt.close()

# =============================================================================
# CHART 2: OPERATION COMPLETION RATIO WITH COLOR ZONES
# =============================================================================

fig, ax = plt.subplots(figsize=(20, 10))

# Add background color zones for completion ranges (DIFFERENT colors from job-level)
ax.axhspan(0.0, 0.3, alpha=0.15, color='#9B59B6', label='Low Completion (<30%)', zorder=0)  # Purple
ax.axhspan(0.3, 0.7, alpha=0.15, color='#F1C40F', label='Medium Completion (30-70%)', zorder=0)  # Yellow
ax.axhspan(0.7, 1.005, alpha=0.15, color='#1ABC9C', label='High Completion (≥70%)', zorder=0)  # Turquoise

# Plot completion ratio (dark blue line on top of colored zones)
ax.plot(episodes, ratios, linewidth=1.0, color='#1F618D', alpha=0.9,
        label=f'Operation Completion Ratio (Avg: {np.mean(ratios):.2%})', zorder=5)

# Add moving average to smooth the signal
window_size = 20
moving_avg = np.convolve(ratios, np.ones(window_size)/window_size, mode='valid')
moving_avg_episodes = episodes[window_size-1:]
ax.plot(moving_avg_episodes, moving_avg, linewidth=3.5, color='#FF6F00', alpha=0.95,
        label=f'Moving Average (window={window_size})', linestyle='-', zorder=6)

# Add reference lines
ax.axhline(y=1.0, color='#1ABC9C', linestyle='--', linewidth=2.0, alpha=0.8,
           label='100% Completion (All operations finished)', zorder=3)
ax.axhline(y=0.7, color='#F1C40F', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
ax.axhline(y=0.3, color='#9B59B6', linestyle='--', linewidth=1.5, alpha=0.7, zorder=3)
ax.axhline(y=np.mean(ratios), color='#1F618D', linestyle=':', linewidth=2.0, alpha=0.8,
           label=f'Average: {np.mean(ratios):.2%}', zorder=3)

ax.set_xlabel('Episode Number', fontsize=18, fontweight='bold')
ax.set_ylabel('Operation Completion Ratio', fontsize=18, fontweight='bold')
ax.set_title('Operation Completion Ratio per Episode\n(Ratio of Completed Operations to Arrived Operations)', 
             fontsize=20, fontweight='bold', pad=20)

ax.set_xlim(0, max(episodes))
ax.set_ylim(0, 1.005)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='lower right', fontsize=13, framealpha=0.95, ncol=2)

ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
output2 = OUTPUT_DIR / 'operation_completion_ratio.png'
plt.savefig(output2, dpi=150, bbox_inches='tight')
print(f"✓ Saved Chart 2: {output2}")
plt.close()

# =============================================================================
# CHART 3: COMBINED VIEW (DUAL Y-AXIS)
# =============================================================================

fig, ax1 = plt.subplots(figsize=(20, 10))

# Left y-axis: Operation Arrivals and Completions
color1 = '#2E86C1'
color2 = '#E74C3C'

ax1.plot(episodes, op_arrivals, linewidth=2.5, color=color1, alpha=0.8, 
         label=f'Op Arrivals (Avg: {avg_arrivals:.2f})', marker='o', markersize=3, markevery=50)
ax1.plot(episodes, op_completions, linewidth=2.5, color=color2, alpha=0.8,
         label=f'Op Completions (Avg: {avg_completions:.2f})', marker='s', markersize=3, markevery=50)

ax1.set_xlabel('Episode Number', fontsize=18, fontweight='bold')
ax1.set_ylabel('Number of Operations', fontsize=18, fontweight='bold', color='black')
ax1.tick_params(axis='y', labelsize=14, labelcolor='black')
ax1.tick_params(axis='x', labelsize=14)

ax1.set_xlim(0, max(episodes))
ax1.set_ylim(0, max(max(op_arrivals), max(op_completions)) + 5)

# Right y-axis: Completion Ratio
ax2 = ax1.twinx()
color3 = '#27AE60'

ax2.plot(episodes, ratios, linewidth=2.0, color=color3, alpha=0.7, linestyle='--',
         label=f'Completion Ratio (Avg: {np.mean(ratios):.2%})', marker='D', markersize=3, markevery=50)

ax2.set_ylabel('Operation Completion Ratio', fontsize=18, fontweight='bold', color=color3)
ax2.tick_params(axis='y', labelsize=14, labelcolor=color3)
ax2.set_ylim(0, 1.005)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Title
ax1.set_title('Operation Arrivals, Completions, and Completion Ratio per Episode\n(Operation-Level Granularity)', 
              fontsize=20, fontweight='bold', pad=20)

# Grid
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14, framealpha=0.95)

plt.tight_layout()
output3 = OUTPUT_DIR / 'operation_arrivals_completions_combined.png'
plt.savefig(output3, dpi=150, bbox_inches='tight')
print(f"✓ Saved Chart 3: {output3}")
plt.close()

print(f"\n✓ All operation-level visualizations completed!")
print(f"  - Operation completion ratio: {np.mean(ratios):.1%} (vs Job ratio: 54.8%)")
print(f"  - Avg operations/episode: {avg_arrivals:.1f} arrivals, {avg_completions:.1f} completions")
