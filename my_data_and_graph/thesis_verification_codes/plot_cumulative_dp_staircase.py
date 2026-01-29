#!/usr/bin/env python3
"""
Cumulative Decision Points Staircase Comparison:
Proposed Framework (Event-Driven) vs Standard Approach (Fixed Intervals)
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_decision_point_timestamps(episode_num=0):
    """Parse timestamps of all decision points in a specific episode."""
    
    timestamps = []
    current_episode = None
    in_lifecycle = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                match = re.search(r'EPISODE (\d+)', line)
                current_episode = int(match.group(1)) if match else None
                in_lifecycle = False
                
                # If we've passed the target episode, stop
                if current_episode is not None and current_episode > episode_num:
                    break
                continue
            
            if 'LIFECYCLE TRACE START' in line:
                in_lifecycle = True
                continue
            if 'LIFECYCLE TRACE END' in line:
                in_lifecycle = False
                continue
            
            # Only process if we're in the target episode
            if current_episode != episode_num:
                continue
            
            # Job arrivals (in lifecycle trace)
            if in_lifecycle and '[t=' in line and 'New job' in line and 'arrived' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                if time_match:
                    timestamps.append(float(time_match.group(1)))
                continue
            
            # Operation completions (outside lifecycle trace)
            if not in_lifecycle and '[t=' in line and 'finished' in line and 'next queued' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                if time_match:
                    timestamps.append(float(time_match.group(1)))
                continue
    
    return sorted(timestamps)

# Use episode 10 timestamps (37 DPs, with 6.00 removed)
print("Loading decision point timestamps...")
episode_to_plot = 10

dp_timestamps = [
    0.00, 0.00, 0.00, 0.00,  # Initial jobs
    2.05, 3.58, 3.62, 4.54, 5.22,  # Early clustering
    6.15,  # 6.00 removed
    7.58, 7.62, 7.78, 8.38, 8.57,  # Mid clustering
    9.93, 9.95, 10.97,
    11.82, 12.28, 12.68,
    13.64, 13.93,
    15.22, 15.40, 15.97,
    16.75, 17.23, 17.40, 17.93,
    19.75, 20.15,
    21.80, 22.39,
    23.40, 23.98, 24.57
]

print(f"Episode {episode_to_plot}: {len(dp_timestamps)} decision points (up to t=25)")
print(f"Time range: {dp_timestamps[0]:.2f} to {dp_timestamps[-1]:.2f}")
print(f"\nProposed Framework Decision Point Timestamps:")
for i, t in enumerate(dp_timestamps, 1):
    print(f"  DP {i:2d}: t = {t:6.2f}")
print(f"\nStandard Approach Decision Point Timestamps:")
for i in range(25):
    print(f"  DP {i+1:2d}: t = {i*1.0:6.2f}")

# Create cumulative counts for proposed framework
proposed_times = [0] + dp_timestamps  # Start at 0
proposed_counts = list(range(len(proposed_times)))  # 0, 1, 2, ..., N

# Standard approach: Fixed 25 DPs at regular intervals (interval = 1.0)
standard_dp_count = 25
standard_interval = 1.0  # Fixed interval
standard_times = [i * standard_interval for i in range(standard_dp_count + 1)]
standard_counts = list(range(standard_dp_count + 1))

print(f"\nStandard approach:")
print(f"  DPs: {standard_dp_count}")
print(f"  Interval: {standard_interval:.2f}")
print(f"  Time range: 0.00 to {standard_times[-1]:.2f}")

# Create the plot
fig, ax = plt.subplots(figsize=(18, 8))

# Proposed framework - event-driven staircase (irregular steps)
ax.step(proposed_times, proposed_counts, where='post', linewidth=2.5, 
        color='#2E86C1', alpha=0.9, label=f'Proposed Framework (Event-Driven, {len(dp_timestamps)} DPs)')

# Standard approach - uniform staircase (regular steps)
ax.step(standard_times, standard_counts, where='post', linewidth=2.5, 
        color='#E74C3C', alpha=0.9, linestyle='--',
        label=f'Standard Approach (Fixed Intervals, {standard_dp_count} DPs)')

# Mark decision points with vertical lines for proposed framework (show clustering)
for t in dp_timestamps:
    ax.axvline(x=t, color='#2E86C1', alpha=0.25, linewidth=1.0, linestyle='--', zorder=0)

# Mark standard intervals with vertical lines
for t in standard_times[1:]:
    ax.axvline(x=t, color='#E74C3C', alpha=0.25, linewidth=1.0, linestyle='--', zorder=0)

ax.set_xlabel('Continuous Time', fontsize=16, fontweight='heavy')
ax.set_ylabel('Cumulative Decision Points', fontsize=16, fontweight='heavy')
ax.set_title(f'Cumulative Decision Points: Event-Driven vs Fixed Intervals\n(Episode {episode_to_plot})', 
             fontsize=16, fontweight='bold', pad=20)

ax.set_xlim(0, 25)
ax.set_ylim(0, max(len(dp_timestamps), standard_dp_count) + 2)

ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

# Add timestamp annotations on vertical lines (select well-spaced timestamps only)
# Proposed framework: Choose timestamps with sufficient spacing between them
well_spaced_timestamps = [
    2.05, 4.54,           # Interval 0-5 (avoid 3.58/3.62 cluster)
    6.15, 8.38, 9.93,     # Interval 5-10 (avoid 7.58/7.62 and 9.95)
    10.97, 12.68, 13.93,  # Interval 10-15
    15.97, 17.40, 17.93,  # Interval 15-20 (added 17.93)
    19.75, 21.80, 23.40, 24.57   # Interval 20-25
]

# Get corresponding y-positions (cumulative count at each timestamp)
for t in well_spaced_timestamps:
    # Find how many DPs occurred before or at this timestamp
    y_pos = sum(1 for dp_t in dp_timestamps if dp_t <= t)
    
    # Add annotation above the vertical line
    ax.annotate(f'{t:.2f}', 
                xy=(t, y_pos), 
                xytext=(0, 8),  # 8 points above
                textcoords='offset points',
                fontsize=9, 
                ha='center', 
                va='bottom',
                color='#2E86C1', 
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#2E86C1', alpha=0.9, linewidth=1))

# Standard approach: Add annotations for well-spaced fixed intervals
# Include standard times that are near proposed timestamps to show the difference
standard_annotated = [
    1.0, 2.0,           # 2.0 near proposed 2.05
    5.0, 6.0,           # 6.0 near proposed 6.15
    8.0,                # 8.0 near proposed 8.38
    10.0, 11.0,         # 10.0 near proposed 9.93, 11.0 near 10.97
    13.0, 14.0,         # 13.0 near proposed 12.68, 14.0 near 13.93
    15.0, 16.0, 17.0,   # 16.0 near proposed 15.97, 17.0 near 17.40
    18.0, 19.0, 20.0,   # 18.0 near proposed 17.93, 19.0 near 19.75, 20.0 after
    21.0, 22.0, 23.0, 24.0    # 21.0 near proposed 21.80, 22.0 after, 23.0 near 23.40
]

for t in standard_annotated:
    # For standard approach, cumulative count at time t is the DP index + 1
    # Since DPs are at 0, 1, 2, 3..., at time t=1.0 we have 2 DPs (0 and 1)
    y_pos = int(t) + 1
    
    # Add annotation above the vertical line
    ax.annotate(f'{t:.1f}', 
                xy=(t, y_pos), 
                xytext=(0, 8),  # 8 points above
                textcoords='offset points',
                fontsize=9, 
                ha='center', 
                va='bottom',
                color='#E74C3C', 
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#E74C3C', alpha=0.9, linewidth=1))

# Legend with additional info
from matplotlib.lines import Line2D
info_lines = [
    Line2D([0], [0], color='none', label=f'Episode Duration: 25.0'),
    Line2D([0], [0], color='none', label=f'Proposed DPs: {len(dp_timestamps)}'),
    Line2D([0], [0], color='none', label=f'Standard DPs: {standard_dp_count}'),
    Line2D([0], [0], color='none', label=f'Standard Interval: {standard_interval:.2f}')
]

handles, labels = ax.get_legend_handles_labels()
handles.extend(info_lines)
ax.legend(handles=handles, loc='upper left', fontsize=12, framealpha=0.95)

plt.tight_layout()

output_path = OUTPUT_DIR / 'cumulative_dp_staircase_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved cumulative DP staircase comparison: {output_path}")
plt.close()
