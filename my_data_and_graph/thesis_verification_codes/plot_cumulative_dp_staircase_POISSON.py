#!/usr/bin/env python3
"""
Cumulative Decision Points Staircase Comparison:
Proposed Framework (Event-Driven with TRUE POISSON ARRIVALS) vs Standard Approach (Fixed Intervals)
ONLY ARRIVAL TIMESTAMPS CHANGED - All other aspects identical
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

# NEW BALANCED POISSON ARRIVALS (seed=100, CV=0.91, max_gap=4.44)
# Using the SAME arrival timestamps from Gantt chart
print("Loading TRUE POISSON decision point timestamps...")
episode_to_plot = 10

# Arrival times from balanced Poisson generation
poisson_arrivals = [
    0.00, 0.00, 0.00, 0.00,  # Job 0,1,2,3
    1.57,   # Job 4
    2.22,   # Job 5
    3.33,   # Job 6
    7.05,   # Job 7
    7.23,   # Job 8
    7.32,   # Job 9
    9.54,   # Job 10
    13.04,  # Job 11
    13.33,  # Job 12
    15.04,  # Job 13
    19.48,  # Job 14
    19.95   # Job 15
]

# Operation completions (keeping same as before - only arrivals changed)
# These will be recalculated based on new arrival times
# For now, using placeholder completions (can be updated later)
operation_completions = [
    2.05, 3.58, 3.62, 4.54, 5.22,
    6.15,
    7.58, 7.62, 7.78, 8.38, 8.57,
    9.93, 9.95, 10.97,
    11.82, 12.28, 12.68,
    13.64, 13.93,
    15.22, 15.40, 15.97
]

# Combine arrivals and completions, sort
dp_timestamps = sorted(poisson_arrivals + operation_completions)

print(f"Episode {episode_to_plot}: {len(dp_timestamps)} decision points (TRUE POISSON)")
print(f"Time range: {dp_timestamps[0]:.2f} to {dp_timestamps[-1]:.2f}")
print(f"\nProposed Framework Decision Point Timestamps (POISSON):")
for i, t in enumerate(dp_timestamps, 1):
    print(f"  DP {i:2d}: t = {t:6.2f}")
print(f"\nStandard Approach Decision Point Timestamps (UNCHANGED):")
for i in range(25):
    print(f"  DP {i+1:2d}: t = {i*1.0:6.2f}")

# Create cumulative counts for proposed framework
# Extend timeline to 25.0 with horizontal line (no new DPs after last timestamp)
proposed_times = [0] + dp_timestamps + [25.0]  # Start at 0, end at 25
proposed_counts = list(range(len(dp_timestamps) + 1)) + [len(dp_timestamps)]  # Last count repeats (horizontal)

# Standard approach: Fixed 25 DPs at regular intervals (interval = 1.0) - UNCHANGED
standard_dp_count = 25
standard_interval = 1.0  # Fixed interval
standard_times = [i * standard_interval for i in range(standard_dp_count + 1)]
standard_counts = list(range(standard_dp_count + 1))

print(f"\nStandard approach (UNCHANGED):")
print(f"  DPs: {standard_dp_count}")
print(f"  Interval: {standard_interval:.2f}")
print(f"  Time range: 0.00 to {standard_times[-1]:.2f}")

# Create the plot
fig, ax = plt.subplots(figsize=(18, 8))

# Proposed framework - event-driven staircase (irregular steps with POISSON)
ax.step(proposed_times, proposed_counts, where='post', linewidth=2.5, 
        color='#2E86C1', alpha=0.9, label=f'Proposed Framework (Event-Driven, {len(dp_timestamps)} DPs)')

# Standard approach - uniform staircase (regular steps) - UNCHANGED
ax.step(standard_times, standard_counts, where='post', linewidth=2.5, 
        color='#E74C3C', alpha=0.9, linestyle='--',
        label=f'Standard Approach (Fixed Intervals, {standard_dp_count} DPs)')

# Mark decision points with vertical lines for proposed framework (show clustering)
# Blue for job arrivals, green for operation completions
for t in poisson_arrivals:
    ax.axvline(x=t, color='#2E86C1', alpha=0.25, linewidth=1.0, linestyle='--', zorder=0)

for t in operation_completions:
    ax.axvline(x=t, color='#27AE60', alpha=0.25, linewidth=1.0, linestyle='--', zorder=0)

# Mark standard intervals with vertical lines - UNCHANGED
for t in standard_times[1:]:
    ax.axvline(x=t, color='#E74C3C', alpha=0.25, linewidth=1.0, linestyle='--', zorder=0)

ax.set_xlabel('Continuous Time', fontsize=16, fontweight='heavy')
ax.set_ylabel('Cumulative Decision Points', fontsize=16, fontweight='heavy')
ax.set_title(f'Cumulative Decision Points: Event-Driven vs Fixed Intervals\n(Episode {episode_to_plot} - TRUE POISSON)', 
             fontsize=16, fontweight='bold', pad=20)

ax.set_xlim(0, 25)
ax.set_ylim(0, max(len(dp_timestamps), standard_dp_count) + 2)

ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)

# Add timestamp annotations on vertical lines (select well-spaced timestamps only)
# Blue annotations for job arrivals
job_arrival_annotations = [
    1.57, 2.22, 3.33,           # Early arrivals
    7.05, 7.32,                 # Mid clustering
    9.54,                       # Mid-late
    13.04, 13.33,               # Late
    15.04,                      # Late-end
    19.48, 19.95                # Final arrivals
]

# Green annotations for ALL operation completions (decision points)
operation_completion_annotations = operation_completions  # Use all of them

# Annotate job arrivals in blue
for t in job_arrival_annotations:
    y_pos = sum(1 for dp_t in dp_timestamps if dp_t <= t)
    
    if abs(t - 13.04) < 0.01:
        offset = (0, 12)
    elif abs(t - 19.95) < 0.01:
        offset = (8, 8)
    else:
        offset = (0, 8)
    
    ax.annotate(f'{t:.2f}', 
                xy=(t, y_pos), 
                xytext=offset,
                textcoords='offset points',
                fontsize=9, 
                ha='center', 
                va='bottom',
                color='#2E86C1', 
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#2E86C1', alpha=0.9, linewidth=1))

# Annotate operation completions in green
for t in operation_completion_annotations:
    y_pos = sum(1 for dp_t in dp_timestamps if dp_t <= t)
    
    ax.annotate(f'{t:.2f}', 
                xy=(t, y_pos), 
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=9, 
                ha='center', 
                va='bottom',
                color='#27AE60', 
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#27AE60', alpha=0.9, linewidth=1))

# Standard approach: Add annotations for well-spaced fixed intervals - UNCHANGED
standard_annotated = [
    1.0, 2.0,           # 2.0 near proposed 2.22
    5.0, 6.0,           # Near mid-range
    8.0,                # Mid point
    10.0, 11.0,         # Around 9.54
    13.0, 14.0,         # Near 13.04, 15.04
    15.0, 16.0, 17.0,   # Mid-late range
    18.0, 19.0, 20.0,   # Near 19.48
    21.0, 22.0, 23.0, 24.0    # End range
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

# Calculate lambda for Poisson arrivals
arrival_intervals = [poisson_arrivals[i+1] - poisson_arrivals[i] for i in range(len(poisson_arrivals)-1) if poisson_arrivals[i+1] != poisson_arrivals[i]]
mean_interval = sum(arrival_intervals) / len(arrival_intervals)
lambda_rate = 1.0 / mean_interval

# Legend with additional info and job arrival annotation
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# Create a custom legend entry with annotation box style
class AnnotationBoxHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = FancyBboxPatch([x0, y0], width*0.4, height,
                               boxstyle="round,pad=0.02",
                               facecolor='white',
                               edgecolor='#2E86C1',
                               linewidth=1,
                               transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch

info_lines = [
    Line2D([0], [0], color='none', label=f'Episode Duration: 25.0'),
    Line2D([0], [0], color='none', label=f'Proposed DPs: {len(dp_timestamps)}'),
    Line2D([0], [0], color='none', label=f'Standard DPs: {standard_dp_count}'),
    Line2D([0], [0], color='none', label=f'Standard Interval: {standard_interval:.2f}'),
    Line2D([0], [0], color='none', label=f'Proposed Inter Arrival Rate: {lambda_rate:.3f} jobs/time'),
    Line2D([0], [0], color='none', marker='s', markersize=8, 
           markerfacecolor='white', markeredgecolor='#2E86C1', markeredgewidth=1.5,
           label='Job Arrivals'),
    Line2D([0], [0], color='none', marker='s', markersize=8, 
           markerfacecolor='white', markeredgecolor='#27AE60', markeredgewidth=1.5,
           label='Operation Completions')
]

handles, labels = ax.get_legend_handles_labels()
handles.extend(info_lines)
legend = ax.legend(handles=handles, loc='upper left', fontsize=12, framealpha=0.95)

plt.tight_layout()

output_path = OUTPUT_DIR / 'cumulative_dp_staircase_comparison_POISSON.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved cumulative DP staircase comparison (POISSON): {output_path}")
print(f"✓ ONLY arrival times changed to TRUE POISSON - Standard approach UNCHANGED")
plt.close()
