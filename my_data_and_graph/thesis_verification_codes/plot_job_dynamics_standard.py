#!/usr/bin/env python3
"""
Plot job dynamics for standard approach: cumulative arrivals, departures, and current agent count over time
All 10 jobs arrive at t=0, no completions
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Creating Job Dynamics Plot for Standard Approach (Episode 10)...")

# Standard approach: all 10 jobs arrive at t=0
arrivals = [
    {'time': 0.00, 'jobs': [f'Job_{i}' for i in range(10)]},  # All 10 jobs at t=0
]

# No completions in standard approach (all jobs still running at end)
completions = []

# Timeline for standard approach - extended to 25
timeline_end = 25.0

print(f"\nTotal arrivals: 10 jobs")
print(f"Completed jobs: 0 jobs")
print(f"Jobs still in progress at timeline end: 10")
print(f"Time range: 0.00 to {timeline_end:.2f}")

# Create timeline for plotting
time_points = np.linspace(0, timeline_end, 1000)

cumulative_arrivals = []
cumulative_departures = []
current_count = []

for t in time_points:
    # Count arrivals up to time t
    arrived = 0
    for arrival in arrivals:
        if arrival['time'] <= t:
            arrived += len(arrival['jobs'])
    
    # Count departures up to time t (always 0)
    departed = 0
    
    # Current jobs in system
    current = arrived - departed
    
    cumulative_arrivals.append(arrived)
    cumulative_departures.append(departed)
    current_count.append(current)

# Create plot
fig, ax = plt.subplots(figsize=(20, 11))

# Plot lines
line1 = ax.plot(time_points, cumulative_arrivals, linewidth=3.5, 
                label='Cumulative Arrivals', color='#7B1FA2', linestyle='-', marker='', alpha=0.9)
line2 = ax.plot(time_points, cumulative_departures, linewidth=3.5,
                label='Cumulative Departures', color='#E65100', linestyle='-.', marker='', alpha=0.9)
line3 = ax.plot(time_points, current_count, linewidth=3.5,
                label='Current Jobs in System', color='#27AE60', linestyle='--', marker='', alpha=0.9)

# Labels and title
ax.set_xlabel('Time', fontsize=20, fontweight='heavy')
ax.set_ylabel('Number of Jobs', fontsize=20, fontweight='heavy')
ax.set_title('Agent Lifecycle\n(Episode 10)', 
             fontsize=24, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Add vertical line for arrival time at t=0 with annotation
unique_arrival_times = [0.00]
print(f"\nAdding arrival time marker at t=0.00...")

ax.axvline(x=0.00, color='#7B1FA2', linestyle=':', 
           linewidth=1.0, alpha=0.4, zorder=2)

# Add annotation for all 10 jobs arriving at t=0
arrival = arrivals[0]
job_text = ', '.join(arrival['jobs'])
arrived_count = len(arrival['jobs'])

# Horizontal annotation below the line
ax.annotate(job_text, 
            xy=(0.00, arrived_count),
            xytext=(5, -8), textcoords='offset points',
            ha='left', va='top', fontsize=10,
            rotation=0, color='#7B1FA2', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='#7B1FA2', alpha=0.8, linewidth=0.5))

# Add horizontal lines for the constant values
print(f"Adding horizontal lines for constant levels...")

# Arrivals level at 10
ax.axhline(y=10, color='#7B1FA2', linestyle=':', 
           linewidth=1.0, alpha=0.4, zorder=2)

# Current jobs level at 10
ax.axhline(y=10, color='#27AE60', linestyle=':', 
           linewidth=1.0, alpha=0.4, zorder=2)

# Departures at 0 (no line needed, at axis)

# Set x-axis ticks - show a few time points
x_ticks = [0, 5, 10, 15, 20, 25]
ax.set_xticks(x_ticks)
ax.set_xticklabels([f'{t:.2f}' for t in x_ticks], fontsize=11, fontweight='bold', rotation=45, ha='right')

# Remove x-axis margins
ax.set_xlim(0, timeline_end)

# Legend
ax.legend(loc='upper left', fontsize=19, framealpha=0.95)

# Set spine properties
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

# Set y-axis to integer ticks
ax.set_ylim(bottom=-0.5, top=20)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(12)

plt.tight_layout()

output_path = OUTPUT_DIR / 'job_dynamics_standard_episode10.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved job dynamics plot: {output_path}")
plt.close()

# Print some statistics
print(f"\n=== Job Dynamics Statistics (Standard Approach) ===")
print(f"Total arrivals: 10 (all at t=0)")
print(f"Total completions: 0")
print(f"Peak concurrent jobs: 10 (constant)")
print(f"Jobs in system throughout: 10 (constant)")

print("\nPlot complete!")
