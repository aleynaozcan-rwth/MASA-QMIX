#!/usr/bin/env python3
"""
Analyze operation type frequency and operations per job for proposed framework
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from pathlib import Path
import random

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
random.seed(42)

print("Analyzing Proposed Framework Operations...")

# Episode 10 (Episode 3 in original data) - Custom heterogeneous jobs
jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [1, 8, 2], 'num_ops': 3},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 2, 1, 4], 'num_ops': 4},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6, 2, 1], 'num_ops': 3},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2, 3, 1], 'num_ops': 3},
    {'arrival_time': 2.05, 'job_id': 4, 'operations': [8], 'num_ops': 1},
    {'arrival_time': 4.54, 'job_id': 5, 'operations': [4, 1, 3], 'num_ops': 3},
    {'arrival_time': 6.15, 'job_id': 6, 'operations': [4, 5, 3, 1, 8], 'num_ops': 5},
    {'arrival_time': 8.38, 'job_id': 7, 'operations': [8, 2, 1, 4, 6], 'num_ops': 5},
    {'arrival_time': 9.93, 'job_id': 8, 'operations': [2], 'num_ops': 1},
    {'arrival_time': 10.97, 'job_id': 9, 'operations': [1, 9], 'num_ops': 2},
    {'arrival_time': 15.97, 'job_id': 10, 'operations': [6, 9, 1], 'num_ops': 3},
    {'arrival_time': 17.40, 'job_id': 11, 'operations': [5, 7, 4, 1], 'num_ops': 4},
    {'arrival_time': 19.75, 'job_id': 12, 'operations': [8, 7, 6], 'num_ops': 3},
    {'arrival_time': 21.80, 'job_id': 13, 'operations': [3], 'num_ops': 1},
    {'arrival_time': 23.40, 'job_id': 14, 'operations': [6], 'num_ops': 1},
]

# Add Job 15
jobs.append({
    'arrival_time': 12.68,
    'job_id': 15,
    'operations': [9, 7, 2, 6, 3, 8],
    'num_ops': 6
})

# Sort by arrival time
jobs.sort(key=lambda x: x['arrival_time'])

# Remove duplicates within each job
duplicates_removed = {}
for job in jobs:
    original_ops = job['operations'][:]
    unique_ops = []
    seen = set()
    removed = 0
    for op in original_ops:
        if op not in seen:
            unique_ops.append(op)
            seen.add(op)
        else:
            removed += 1
    
    if removed > 0:
        duplicates_removed[job['job_id']] = removed
    
    job['operations'] = unique_ops
    job['num_ops'] = len(unique_ops)

# Add least-used operations to jobs with removed duplicates
least_used_ops = [5, 7, 9, 4, 3]

for job_id, removed_count in duplicates_removed.items():
    job = next(j for j in jobs if j['job_id'] == job_id)
    current_ops_set = set(job['operations'])
    available_ops = [op for op in least_used_ops if op not in current_ops_set]
    ops_to_add = random.sample(available_ops, min(removed_count, len(available_ops)))
    job['operations'].extend(ops_to_add)
    job['num_ops'] = len(job['operations'])

print(f"\nTotal jobs: {len(jobs)}")

# Analysis 1: Operation Type Frequency
all_operations = []
for job in jobs:
    all_operations.extend(job['operations'])

op_frequency = Counter(all_operations)
print(f"\n=== Operation Type Frequency ===")
for op_type in sorted(op_frequency.keys()):
    print(f"Operation Type {op_type}: {op_frequency[op_type]} times")

# Analysis 2: Operations per Job Frequency
ops_per_job = [job['num_ops'] for job in jobs]
ops_per_job_frequency = Counter(ops_per_job)
print(f"\n=== Operations per Job Frequency ===")
for num_ops in sorted(ops_per_job_frequency.keys()):
    print(f"{num_ops} operations: {ops_per_job_frequency[num_ops]} jobs")

# Color palette
op_colors = {
    1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd',
    6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f', 9: '#17becf'
}

# Create visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Operation Type Frequency
op_types = sorted(op_frequency.keys())
frequencies = [op_frequency[op] for op in op_types]
colors = [op_colors[op] for op in op_types]

bars1 = ax1.bar([f'Op{op}' for op in op_types], frequencies, color=colors, 
                edgecolor='black', linewidth=1.5, alpha=0.85)
ax1.set_xlabel('Operation Type', fontsize=14, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax1.set_title('Operation Type Frequency\n(Episode 10)', 
              fontsize=16, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle=':', axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Set spine properties
for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

# Plot 2: Operations per Job Frequency
num_ops_list = sorted(ops_per_job_frequency.keys())
job_frequencies = [ops_per_job_frequency[num] for num in num_ops_list]

bars2 = ax2.bar([str(num) for num in num_ops_list], job_frequencies, 
                color='#34495e', edgecolor='black', linewidth=1.5, alpha=0.85)
ax2.set_xlabel('Number of Operations per Job', fontsize=14, fontweight='bold')
ax2.set_ylabel('Number of Jobs', fontsize=14, fontweight='bold')
ax2.set_title('Operations per Job Distribution\n(Episode 10)', 
              fontsize=16, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle=':', axis='y')

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Set spine properties
for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

plt.tight_layout()

output_path = OUTPUT_DIR / 'proposed_operations_frequency_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved frequency analysis: {output_path}")
plt.close()

print("\nAnalysis complete!")
