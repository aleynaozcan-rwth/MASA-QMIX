#!/usr/bin/env python3
"""
Job Operation Gantt Chart: Shows job arrivals and their operation sequences
in continuous time with color-coded operation types
"""

import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def count_jobs_per_episode(max_episodes=20):
    """Count number of jobs in each episode to find the busiest one."""
    episode_job_counts = {}
    current_episode = None
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                match = re.search(r'EPISODE (\d+)', line)
                current_episode = int(match.group(1)) if match else None
                if current_episode is not None:
                    episode_job_counts[current_episode] = 0
                if current_episode and current_episode >= max_episodes:
                    break
                continue
            
            if current_episode is not None and 'New job Job_' in line and 'arrived' in line:
                episode_job_counts[current_episode] += 1
    
    return episode_job_counts

def parse_job_operations(episode_num=10):
    """Parse job arrivals with their operation sequences for a specific episode."""
    
    jobs = []  # List of (arrival_time, job_id, operations_list)
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
            
            # Job arrivals with operation details (outside lifecycle trace)
            # [t=X.XX] New job Job_<ID> arrived with N ops: [Op1, Op2, ...]
            if 'New job Job_' in line and 'arrived' in line and 'ops:' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'Job_(\d+)', line)
                # Extract operations: [Op1, Op8, Op2] -> extract numbers
                ops_match = re.search(r'ops: \[(.*?)\]', line)
                
                if time_match and job_match and ops_match:
                    arrival_time = float(time_match.group(1))
                    job_id = int(job_match.group(1))
                    ops_str = ops_match.group(1)
                    # Parse "Op1, Op8, Op2" -> [1, 8, 2]
                    operations = []
                    for op in ops_str.split(','):
                        op = op.strip()
                        op_num_match = re.search(r'Op(\d+)', op)
                        if op_num_match:
                            operations.append(int(op_num_match.group(1)))
                    
                    jobs.append({
                        'arrival_time': arrival_time,
                        'job_id': job_id,
                        'operations': operations,
                        'num_ops': len(operations)
                    })
    
    return jobs

# Use custom data from user's Episode 3 run
print("Using custom Episode 3 data (frequent job arrivals)...")
episode_num = 3

# Hard-coded job data from user's run (duplicates removed within each job)
# Arrival times aligned with Episode 10 cumulative staircase timestamps
jobs_raw = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [1, 8, 2], 'num_ops': 3},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 2, 1, 4], 'num_ops': 4},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6, 2, 1], 'num_ops': 3},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2, 3, 3, 1], 'num_ops': 4},  # Op3 duplicate
    {'arrival_time': 2.05, 'job_id': 4, 'operations': [8], 'num_ops': 1},
    {'arrival_time': 4.54, 'job_id': 5, 'operations': [4, 1, 3], 'num_ops': 3},
    {'arrival_time': 6.15, 'job_id': 6, 'operations': [4, 5, 3, 1, 8], 'num_ops': 5},
    {'arrival_time': 8.38, 'job_id': 7, 'operations': [8, 2, 1, 4, 6], 'num_ops': 5},
    {'arrival_time': 9.93, 'job_id': 8, 'operations': [2], 'num_ops': 1},
    {'arrival_time': 10.97, 'job_id': 9, 'operations': [1, 9], 'num_ops': 2},
    {'arrival_time': 12.68, 'job_id': 15, 'operations': [9, 7, 2, 6, 3, 8], 'num_ops': 6},
    {'arrival_time': 15.97, 'job_id': 10, 'operations': [6, 9, 6, 1, 9], 'num_ops': 5},  # Op6, Op9 duplicates
    {'arrival_time': 17.40, 'job_id': 11, 'operations': [5, 7, 4, 1], 'num_ops': 4},
    {'arrival_time': 19.75, 'job_id': 12, 'operations': [8, 7, 6, 7, 8], 'num_ops': 5},  # Op7, Op8 duplicates
    {'arrival_time': 21.80, 'job_id': 13, 'operations': [3], 'num_ops': 1},
    {'arrival_time': 23.40, 'job_id': 14, 'operations': [6], 'num_ops': 1},
]

# Remove duplicates within each job (keep first occurrence)
import random
random.seed(42)  # For reproducibility

jobs = []
duplicates_removed = {}  # Track how many duplicates removed per job

for job in jobs_raw:
    unique_ops = []
    seen = set()
    removed_count = 0
    
    for op in job['operations']:
        if op not in seen:
            unique_ops.append(op)
            seen.add(op)
        else:
            removed_count += 1
    
    if removed_count > 0:
        duplicates_removed[job['job_id']] = removed_count
    
    jobs.append({
        'arrival_time': job['arrival_time'],
        'job_id': job['job_id'],
        'operations': unique_ops,
        'num_ops': len(unique_ops),
        'original_ops': job['operations']
    })

# For jobs with removed duplicates, add least-used operation types
# Least used operations based on original analysis: Op5 (2), Op7 (3), Op9 (3)
least_used_ops = [5, 7, 9, 4, 3]  # Ordered from least to most common

print(f"\nAdding operations to jobs with removed duplicates:")
for job_id, removed_count in duplicates_removed.items():
    job = next(j for j in jobs if j['job_id'] == job_id)
    current_ops_set = set(job['operations'])
    
    # Find least-used ops that are NOT in this job
    available_ops = [op for op in least_used_ops if op not in current_ops_set]
    
    # Add random operations from available least-used ops
    ops_to_add = random.sample(available_ops, min(removed_count, len(available_ops)))
    
    print(f"  Job {job_id}: removed {removed_count} duplicates, adding {ops_to_add}")
    job['operations'].extend(ops_to_add)
    job['num_ops'] = len(job['operations'])

if not jobs:
    print(f"No jobs found for episode {episode_num}")
    exit(1)

print(f"Found {len(jobs)} jobs in episode {episode_num} (16 total with new Job 15)")
print(f"Time range: {jobs[0]['arrival_time']:.2f} to {jobs[-1]['arrival_time']:.2f}")
print(f"\nFirst 5 jobs:")
for i, job in enumerate(jobs[:5]):
    print(f"  Job {job['job_id']}: t={job['arrival_time']:.2f}, {job['num_ops']} ops: {job['operations']}")

# Define color palette for 9 operation types (Op1-Op9) - Tab20 dark colors only
op_colors = {
    1: '#1f77b4',  # Dark Blue - Op1 (most frequent)
    2: '#ff7f0e',  # Dark Orange - Op2
    3: '#2ca02c',  # Dark Green - Op3
    4: '#d62728',  # Dark Red - Op4
    5: '#9467bd',  # Dark Purple - Op5
    6: '#8c564b',  # Dark Brown - Op6
    7: '#e377c2',  # Dark Pink - Op7
    8: '#7f7f7f',  # Dark Gray - Op8 (second most frequent)
    9: '#17becf',  # Dark Cyan - Op9
}

# Create Gantt chart
fig, ax = plt.subplots(figsize=(24, 8))

# Each job gets a horizontal row
# X-axis: continuous time (arrival time + operation sequence)
# Y-axis: job index
bar_height = 0.8
segment_width = 0.5  # Width per operation in time units (for visualization)

for idx, job in enumerate(jobs):
    y_pos = idx
    x_start = job['arrival_time']
    
    # Draw each operation as a colored segment (without edges first)
    for op_idx, op_type in enumerate(job['operations']):
        x_position = x_start + (op_idx * segment_width)
        color = op_colors.get(op_type, '#95A5A6')  # Default gray if not found
        
        # Draw operation segment without edge
        rect = mpatches.Rectangle((x_position, y_pos - bar_height/2), 
                                   segment_width, bar_height,
                                   facecolor=color, edgecolor='none', 
                                   linewidth=0, alpha=0.85)
        ax.add_patch(rect)
    
    # Draw outer border for the entire job
    total_width = len(job['operations']) * segment_width
    border = mpatches.Rectangle((x_start, y_pos - bar_height/2), 
                                total_width, bar_height,
                                facecolor='none', edgecolor='black', 
                                linewidth=1.2, alpha=1.0, zorder=5)
    ax.add_patch(border)
    
    # Draw internal vertical lines between operations
    for op_idx in range(1, len(job['operations'])):
        x_divider = x_start + (op_idx * segment_width)
        ax.plot([x_divider, x_divider], 
               [y_pos - bar_height/2, y_pos + bar_height/2],
               color='black', linewidth=0.8, alpha=0.7, zorder=5)

# Add vertical lines and annotations for arrival times
unique_arrival_times = sorted(set(job['arrival_time'] for job in jobs))
print(f"\nAdding arrival time annotations for {len(unique_arrival_times)} unique timestamps...")

for arrival_time in unique_arrival_times:
    if arrival_time > 0:  # Skip t=0 to avoid clutter
        # Draw vertical dashed line
        ax.axvline(x=arrival_time, color='#2E86C1', linestyle='--', 
                   linewidth=2.0, alpha=0.7, zorder=1)
        
        # Find the highest job at this arrival time
        jobs_at_time = [i for i, job in enumerate(jobs) if job['arrival_time'] == arrival_time]
        if jobs_at_time:
            max_y_pos = max(jobs_at_time)
            
            # Special handling for 23.40 - place below to avoid overflow
            if arrival_time >= 23.0:
                ax.annotate(f'{arrival_time:.2f}', 
                           xy=(arrival_time, max_y_pos - 0.5),
                           xytext=(0, -25),
                           textcoords='offset points',
                           ha='center', va='top',
                           fontsize=13, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='#2E86C1', 
                                    edgecolor='black', linewidth=1.0, alpha=0.9),
                           color='white', zorder=10)
            else:
                # Add annotation box above the highest job
                ax.annotate(f'{arrival_time:.2f}', 
                           xy=(arrival_time, max_y_pos + 0.5),
                           xytext=(0, 15),
                           textcoords='offset points',
                           ha='center', va='bottom',
                           fontsize=13, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='#2E86C1', 
                                    edgecolor='black', linewidth=1.0, alpha=0.9),
                           color='white', zorder=10)

# Set axis properties
ax.set_xlabel('Continuous Time (Job Arrival)', fontsize=22, fontweight='heavy')
ax.set_ylabel('Job Index', fontsize=22, fontweight='heavy')
ax.set_title('Job Operation Sequences in Continuous Time (Episode 10)', 
             fontsize=28, fontweight='bold', pad=20)

# X-axis: show arrival times
max_time = max(job['arrival_time'] + len(job['operations']) * segment_width for job in jobs)
ax.set_xlim(0, max_time)

# Y-axis: job indices
ax.set_ylim(-1, len(jobs))
ax.set_yticks(range(0, len(jobs), max(1, len(jobs)//20)))  # Show ~20 ticks max

# Grid
ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, axis='x')

# Set spine (frame) properties to black
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

# Legend for operation types
legend_elements = [mpatches.Patch(facecolor=op_colors[op], edgecolor='black', 
                                   label=f'Operation Type {op}')
                   for op in sorted(op_colors.keys())]
ax.legend(handles=legend_elements, loc='lower right', fontsize=15, 
          title='Operation Types', title_fontsize=16, ncol=3, framealpha=0.95)

plt.tight_layout()

output_path = OUTPUT_DIR / 'job_operation_gantt_chart.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved job operation Gantt chart: {output_path}")
plt.close()
