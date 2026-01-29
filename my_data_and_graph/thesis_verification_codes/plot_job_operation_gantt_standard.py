#!/usr/bin/env python3
"""
Job Operation Gantt Chart - Standard Approach (Fixed Initial Jobs):
10 initial jobs, all with same operation sequence [6, 5, 3, 4, 9, 7, 8, 2, 1]
All arrive at t=0, no new jobs arrive
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Creating Standard Approach Gantt Chart...")

# Standard approach: 10 initial jobs, all with same operation sequence
# All arrive at t=0
jobs = []
for job_id in range(10):
    jobs.append({
        'arrival_time': 0.00,
        'job_id': job_id,
        'operations': [6, 5, 3, 4, 9, 7, 8, 2, 1],
        'num_ops': 9
    })

print(f"Created {len(jobs)} jobs")
print(f"All jobs arrive at t=0.00")
print(f"All jobs have 9 operations: [6, 5, 3, 4, 9, 7, 8, 2, 1]")

# Define color palette for 9 operation types (Op1-Op9) - Tab20 dark colors only
op_colors = {
    1: '#1f77b4',  # Dark Blue
    2: '#ff7f0e',  # Dark Orange
    3: '#2ca02c',  # Dark Green
    4: '#d62728',  # Dark Red
    5: '#9467bd',  # Dark Purple
    6: '#8c564b',  # Dark Brown
    7: '#e377c2',  # Dark Pink
    8: '#7f7f7f',  # Dark Gray
    9: '#17becf',  # Dark Cyan
}

# Create Gantt chart
fig, ax = plt.subplots(figsize=(14, 7.4))  # Wider figure

# Each job gets a horizontal row
# X-axis: continuous time (arrival time + operation sequence)
# Y-axis: job index
bar_height = 0.8
segment_width = 0.04  # Very narrow for compact boxes

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

# Set axis properties
ax.set_xlabel('Continuous Time (Job Arrival)', fontsize=14, fontweight='heavy')
ax.set_ylabel('Job Index', fontsize=14, fontweight='heavy')
ax.set_title('Job Operation Sequences in Continuous Time\n(Episode 10)', 
             fontsize=16, fontweight='bold', pad=20)

# X-axis: compact view with minimal ticks (only t=0 matters since all jobs arrive there)
max_time = max(job['arrival_time'] + len(job['operations']) * segment_width for job in jobs)
ax.set_xlim(0, max_time + 0.5)
ax.set_xticks([0])  # Only show t=0
ax.set_xticklabels(['0'])

# Y-axis: job indices (tight fit for 10 jobs)
ax.set_ylim(-0.5, len(jobs))  # Tight boundary after last job
ax.set_yticks(range(len(jobs)))

# Grid
ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, axis='x')

# Legend for operation types
legend_elements = [mpatches.Patch(facecolor=op_colors[op], edgecolor='black', 
                                   label=f'Operation Type {op}')
                   for op in sorted(op_colors.keys())]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, 
          title='Operation Types', title_fontsize=11, ncol=3, framealpha=0.95)

plt.tight_layout()

output_path = OUTPUT_DIR / 'job_operation_gantt_chart_standard.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved standard approach Gantt chart: {output_path}")
