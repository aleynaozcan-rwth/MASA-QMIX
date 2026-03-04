#!/usr/bin/env python3
"""
Plot job dynamics: cumulative arrivals, departures, and current agent count over time
Based on Episode 10 timeline data
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Creating Job Dynamics Plot (Episode 10)...")

# Job arrivals from Gantt chart (aligned with cumulative staircase)
arrivals = [
    {'time': 0.00, 'jobs': ['Job_0', 'Job_1', 'Job_2', 'Job_3']},  # 4 jobs at t=0
    {'time': 2.05, 'jobs': ['Job_4']},
    {'time': 4.54, 'jobs': ['Job_5']},
    {'time': 6.15, 'jobs': ['Job_6']},
    {'time': 8.38, 'jobs': ['Job_7']},
    {'time': 9.93, 'jobs': ['Job_8']},
    {'time': 10.97, 'jobs': ['Job_9']},
    {'time': 12.68, 'jobs': ['Job_10']},  # New job added after Job_9
    {'time': 15.97, 'jobs': ['Job_11']},
    {'time': 17.40, 'jobs': ['Job_12']},
    {'time': 19.75, 'jobs': ['Job_13']},
    {'time': 21.80, 'jobs': ['Job_14']},
    {'time': 23.40, 'jobs': ['Job_15']},
]

# Job completions from timeline (exact times when "completed all operations")
completions = [
    {'time': 6.97, 'job': 'Job_4'},
    {'time': 9.75, 'job': 'Job_8'},
    {'time': 12.57, 'job': 'Job_5'},
    {'time': 13.49, 'job': 'Job_0'},
    {'time': 14.43, 'job': 'Job_1'},
    {'time': 15.22, 'job': 'Job_2'},
    {'time': 18.52, 'job': 'Job_3'},
    {'time': 20.10, 'job': 'Job_7'},
    {'time': 22.50, 'job': 'Job_9'},
]

# Timeline ends at t=24.93, remaining jobs still in progress:
# Job_6, Job_10, Job_11, Job_12, Job_13, Job_14, Job_15 (7 jobs remaining)
timeline_end = 24.93

print(f"\nTotal arrivals: {sum(len(a['jobs']) for a in arrivals)} jobs")
print(f"Completed jobs: {len(completions)} jobs")
print(f"Jobs still in progress at timeline end: {sum(len(a['jobs']) for a in arrivals) - len(completions)}")
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
    
    # Count departures up to time t
    departed = sum(1 for comp in completions if comp['time'] <= t)
    
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
ax.set_title('', fontsize=24, fontweight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Add vertical lines for arrival times with job annotations
unique_arrival_times = sorted(set(a['time'] for a in arrivals))
print(f"\nAdding arrival time markers for {len(unique_arrival_times)} unique timestamps...")

for idx, arrival in enumerate(arrivals):
    arrival_time = arrival['time']
    jobs = arrival['jobs']
    
    # Draw vertical dotted line (thin and faint)
    ax.axvline(x=arrival_time, color='#7B1FA2', linestyle=':', 
               linewidth=1.0, alpha=0.4, zorder=2)
    
    # Add annotation with job names
    job_text = ', '.join(jobs)
    # Get current cumulative arrivals count at this time
    arrived_count = sum(len(a['jobs']) for a in arrivals if a['time'] <= arrival_time)
    
    # First annotation is vertical, rest are horizontal
    if idx == 0:
        # Horizontal annotation below the line for first arrival (4 jobs)
        ax.annotate(job_text, 
                    xy=(arrival_time, arrived_count),
                    xytext=(10, -8), textcoords='offset points',
                    ha='left', va='top', fontsize=13,
                    rotation=0, color='#7B1FA2', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                             edgecolor='#7B1FA2', alpha=0.8, linewidth=0.5))
    else:
        # Horizontal annotation for other arrivals
        ax.annotate(job_text, 
                    xy=(arrival_time, arrived_count),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=13,
                    rotation=0, color='#7B1FA2', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                             edgecolor='#7B1FA2', alpha=0.8, linewidth=0.5))

# Add vertical lines for departure times with job annotations
departure_times = sorted([c['time'] for c in completions])
print(f"Adding departure time markers for {len(departure_times)} completed jobs...")

for completion in completions:
    departure_time = completion['time']
    job = completion['job']
    
    # Draw vertical dotted line (thin and faint, orange color)
    ax.axvline(x=departure_time, color='#E65100', linestyle=':', 
               linewidth=1.0, alpha=0.4, zorder=2)
    
    # Add annotation with job name - horizontal, on the line
    # Get cumulative departures count at this time
    departed_count = sum(1 for c in completions if c['time'] <= departure_time)
    ax.annotate(job, 
                xy=(departure_time, departed_count),
                xytext=(0, -8), textcoords='offset points',
                ha='center', va='top', fontsize=13,
                rotation=0, color='#E65100', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                         edgecolor='#E65100', alpha=0.8, linewidth=0.5))

# Add horizontal lines for current job count levels
unique_job_counts = sorted(set(current_count))
print(f"Adding horizontal lines for {len(unique_job_counts)} current job levels...")

for job_count in unique_job_counts:
    if job_count > 0:  # Skip 0 line as it's already at the axis
        # Draw horizontal dotted line (thin and faint, green color)
        ax.axhline(y=job_count, color='#27AE60', linestyle=':', 
                   linewidth=1.0, alpha=0.4, zorder=2)

# Add horizontal lines for cumulative arrivals levels
unique_arrival_counts = sorted(set(cumulative_arrivals))
print(f"Adding horizontal lines for {len(unique_arrival_counts)} cumulative arrival levels...")

for arrival_count in unique_arrival_counts:
    if arrival_count > 0:  # Skip 0 line
        # Draw horizontal dotted line (thin and faint, blue color)
        ax.axhline(y=arrival_count, color='#7B1FA2', linestyle=':', 
                   linewidth=1.0, alpha=0.4, zorder=2)

# Add horizontal lines for cumulative departures levels
unique_departure_counts = sorted(set(cumulative_departures))
print(f"Adding horizontal lines for {len(unique_departure_counts)} cumulative departure levels...")

for departure_count in unique_departure_counts:
    if departure_count > 0:  # Skip 0 line
        # Draw horizontal dotted line (thin and faint, red color)
        ax.axhline(y=departure_count, color='#E65100', linestyle=':', 
                   linewidth=1.0, alpha=0.4, zorder=2)

# Set x-axis ticks to show arrival times
ax.set_xticks(unique_arrival_times)
ax.set_xticklabels([f'{t:.2f}' for t in unique_arrival_times], fontsize=11, fontweight='bold', rotation=45, ha='right')

# Remove x-axis margins
ax.set_xlim(0, timeline_end)

# Legend
ax.legend(loc='upper left', fontsize=19, framealpha=0.95)

# Set spine properties
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

# Set y-axis to integer ticks
ax.set_ylim(bottom=-0.5, top=max(cumulative_arrivals) + 1)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(12)

plt.tight_layout()

output_path = OUTPUT_DIR / 'job_dynamics_episode10.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved job dynamics plot: {output_path}")
plt.close()

# Print some statistics
print(f"\n=== Job Dynamics Statistics ===")
print(f"Total arrivals: {sum(len(a['jobs']) for a in arrivals)}")
print(f"Total completions: {len(completions)}")
print(f"Peak concurrent jobs: {max(current_count)}")
print(f"Final departures at t={timeline_end:.2f}: {cumulative_departures[-1]}")
print(f"Jobs still in system at end: {current_count[-1]}")

print("\nPlot complete!")
