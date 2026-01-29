#!/usr/bin/env python3
"""
Analyze operation type frequency and operations per job for standard approach
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Analyzing Standard Approach Operations...")

# Standard approach: 10 initial jobs, all with same operation sequence
jobs = []
for job_id in range(10):
    jobs.append({
        'arrival_time': 0.00,
        'job_id': job_id,
        'operations': [6, 5, 3, 4, 9, 7, 8, 2, 1],
        'num_ops': 9
    })

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

# Center the single bar - use position 2.5 to center in range 0-5
x_position = 2.5
bars2 = ax2.bar([x_position], job_frequencies, 
                width=0.85, color='#34495e', edgecolor='black', linewidth=1.5, alpha=0.85)
ax2.set_xticks([x_position])
ax2.set_xticklabels([str(num_ops_list[0])])
ax2.set_xlim(-0.5, 5.5)  # Set wider x-axis to match proposed framework scale
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

output_path = OUTPUT_DIR / 'standard_operations_frequency_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved frequency analysis: {output_path}")
plt.close()

print("\nAnalysis complete!")
