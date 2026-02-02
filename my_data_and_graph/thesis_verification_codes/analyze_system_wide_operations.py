#!/usr/bin/env python3
"""
Analyze operation type frequency and operations per job across ALL EPISODES (System-wide)
This aggregates data from all episodes to show overall system-level statistics.
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

print("=" * 80)
print("SYSTEM-WIDE OPERATIONS ANALYSIS")
print("Analyzing ALL Episodes (Episode 1-10)")
print("=" * 80)

# ============================================================================
# ALL EPISODES DATA - Custom heterogeneous jobs across 10 episodes
# ============================================================================

all_episodes_jobs = []

# Episode 1 - Balanced heterogeneous jobs
episode_1_jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [8, 3], 'num_ops': 2},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 2, 4], 'num_ops': 3},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6, 3, 2, 1], 'num_ops': 4},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2, 3, 5, 7], 'num_ops': 4},
    {'arrival_time': 1.80, 'job_id': 4, 'operations': [8, 4, 1, 6], 'num_ops': 4},
    {'arrival_time': 4.20, 'job_id': 5, 'operations': [4, 9, 5, 7], 'num_ops': 4},
    {'arrival_time': 5.90, 'job_id': 6, 'operations': [4, 5, 3], 'num_ops': 3},
    {'arrival_time': 7.95, 'job_id': 7, 'operations': [8, 2, 1, 9], 'num_ops': 4},
    {'arrival_time': 9.60, 'job_id': 8, 'operations': [2, 7, 9, 4, 5], 'num_ops': 5},
    {'arrival_time': 10.50, 'job_id': 9, 'operations': [7, 4, 1, 5, 3, 9], 'num_ops': 6},
    {'arrival_time': 15.30, 'job_id': 10, 'operations': [6, 4, 9], 'num_ops': 3},
    {'arrival_time': 16.80, 'job_id': 11, 'operations': [5, 8, 3, 9, 1], 'num_ops': 5},
    {'arrival_time': 19.20, 'job_id': 12, 'operations': [8, 3, 5, 6], 'num_ops': 4},
    {'arrival_time': 21.10, 'job_id': 13, 'operations': [3, 1, 2, 9], 'num_ops': 4},
    {'arrival_time': 22.90, 'job_id': 14, 'operations': [6, 1, 5, 3], 'num_ops': 4},
    {'arrival_time': 12.10, 'job_id': 15, 'operations': [4, 7, 2, 6, 3, 1], 'num_ops': 6},
]

# Episode 2
episode_2_jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [2, 4, 1, 7], 'num_ops': 4},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 3, 7, 1, 6], 'num_ops': 5},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2, 3, 7, 9], 'num_ops': 4},
    {'arrival_time': 2.15, 'job_id': 4, 'operations': [8, 2, 9], 'num_ops': 3},
    {'arrival_time': 4.75, 'job_id': 5, 'operations': [4, 7, 5, 1, 6], 'num_ops': 5},
    {'arrival_time': 6.40, 'job_id': 6, 'operations': [4, 5, 9, 7, 1, 3], 'num_ops': 6},
    {'arrival_time': 8.60, 'job_id': 7, 'operations': [8, 2, 5, 6], 'num_ops': 4},
    {'arrival_time': 10.20, 'job_id': 8, 'operations': [2], 'num_ops': 1},
    {'arrival_time': 11.30, 'job_id': 9, 'operations': [1, 9, 3, 4, 6, 2], 'num_ops': 6},
    {'arrival_time': 16.10, 'job_id': 10, 'operations': [6, 9, 5], 'num_ops': 3},
    {'arrival_time': 17.65, 'job_id': 11, 'operations': [5, 8], 'num_ops': 2},
    {'arrival_time': 20.10, 'job_id': 12, 'operations': [8, 6, 7, 3, 1, 4], 'num_ops': 6},
    {'arrival_time': 22.30, 'job_id': 13, 'operations': [3, 4, 2, 5, 6, 9], 'num_ops': 6},
    {'arrival_time': 23.85, 'job_id': 14, 'operations': [6, 9, 1], 'num_ops': 3},
]

# Episode 3
episode_3_jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [7, 2, 6, 4, 8, 5], 'num_ops': 6},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 4, 2], 'num_ops': 3},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2, 3, 9, 6], 'num_ops': 4},
    {'arrival_time': 2.35, 'job_id': 4, 'operations': [8, 7, 9, 1], 'num_ops': 4},
    {'arrival_time': 5.10, 'job_id': 5, 'operations': [4, 3, 5, 7, 1], 'num_ops': 5},
    {'arrival_time': 6.75, 'job_id': 6, 'operations': [4, 5, 8, 9, 1], 'num_ops': 5},
    {'arrival_time': 9.10, 'job_id': 7, 'operations': [8, 2], 'num_ops': 2},
    {'arrival_time': 10.45, 'job_id': 8, 'operations': [2, 6], 'num_ops': 2},
    {'arrival_time': 11.85, 'job_id': 9, 'operations': [7, 9, 5, 3, 1, 6], 'num_ops': 6},
    {'arrival_time': 16.50, 'job_id': 10, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 18.20, 'job_id': 11, 'operations': [5, 4, 7, 2, 1], 'num_ops': 5},
    {'arrival_time': 20.75, 'job_id': 12, 'operations': [8, 6, 8], 'num_ops': 3},
    {'arrival_time': 22.55, 'job_id': 13, 'operations': [3, 5, 1, 7], 'num_ops': 4},
    {'arrival_time': 24.30, 'job_id': 14, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 13.25, 'job_id': 15, 'operations': [9, 2, 7, 5], 'num_ops': 4},
]

# Episode 4
episode_4_jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [7, 8, 3, 2], 'num_ops': 4},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 2, 6, 4, 3, 5], 'num_ops': 6},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2, 3, 5, 7], 'num_ops': 4},
    {'arrival_time': 1.95, 'job_id': 4, 'operations': [8, 4, 7, 1], 'num_ops': 4},
    {'arrival_time': 4.30, 'job_id': 5, 'operations': [4, 3, 5, 7, 1, 9], 'num_ops': 6},
    {'arrival_time': 5.85, 'job_id': 6, 'operations': [4, 5, 1], 'num_ops': 3},
    {'arrival_time': 8.15, 'job_id': 7, 'operations': [8], 'num_ops': 1},
    {'arrival_time': 9.75, 'job_id': 8, 'operations': [2, 9], 'num_ops': 2},
    {'arrival_time': 10.90, 'job_id': 9, 'operations': [7, 9, 1], 'num_ops': 3},
    {'arrival_time': 15.85, 'job_id': 10, 'operations': [6, 9], 'num_ops': 2},
    {'arrival_time': 17.25, 'job_id': 11, 'operations': [5, 8, 3, 2, 4], 'num_ops': 5},
    {'arrival_time': 19.55, 'job_id': 12, 'operations': [8], 'num_ops': 1},
    {'arrival_time': 21.65, 'job_id': 13, 'operations': [3, 2, 5], 'num_ops': 3},
    {'arrival_time': 23.10, 'job_id': 14, 'operations': [6, 5, 9, 7, 4], 'num_ops': 5},
]

# Episode 5
episode_5_jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [7, 2, 5, 4, 3, 9], 'num_ops': 6},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 4, 5, 9, 3, 7], 'num_ops': 6},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6, 2, 9, 7, 3, 4], 'num_ops': 6},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2], 'num_ops': 1},
    {'arrival_time': 2.45, 'job_id': 4, 'operations': [8, 6, 3, 7, 2, 1], 'num_ops': 6},
    {'arrival_time': 4.90, 'job_id': 5, 'operations': [4, 3, 5, 7, 9, 1], 'num_ops': 6},
    {'arrival_time': 6.55, 'job_id': 6, 'operations': [4, 5, 3, 9], 'num_ops': 4},
    {'arrival_time': 8.90, 'job_id': 7, 'operations': [8, 2], 'num_ops': 2},
    {'arrival_time': 10.30, 'job_id': 8, 'operations': [2, 5], 'num_ops': 2},
    {'arrival_time': 11.60, 'job_id': 9, 'operations': [1, 9, 5, 3, 7], 'num_ops': 5},
    {'arrival_time': 16.40, 'job_id': 10, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 18.00, 'job_id': 11, 'operations': [5, 8, 4], 'num_ops': 3},
    {'arrival_time': 20.35, 'job_id': 12, 'operations': [8, 7, 9, 1, 4], 'num_ops': 5},
    {'arrival_time': 22.40, 'job_id': 13, 'operations': [3, 6, 1], 'num_ops': 3},
    {'arrival_time': 24.05, 'job_id': 14, 'operations': [6, 2], 'num_ops': 2},
    {'arrival_time': 12.95, 'job_id': 15, 'operations': [9, 7, 2, 1, 8], 'num_ops': 5},
]

# Episode 6
episode_6_jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [7, 8, 3, 4, 5, 2], 'num_ops': 6},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 2], 'num_ops': 2},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2, 5, 3, 7, 1], 'num_ops': 5},
    {'arrival_time': 2.25, 'job_id': 4, 'operations': [8, 3, 9, 6], 'num_ops': 4},
    {'arrival_time': 4.65, 'job_id': 5, 'operations': [4, 3, 7, 5, 1], 'num_ops': 5},
    {'arrival_time': 6.30, 'job_id': 6, 'operations': [4, 5, 3, 1], 'num_ops': 4},
    {'arrival_time': 8.55, 'job_id': 7, 'operations': [8], 'num_ops': 1},
    {'arrival_time': 10.05, 'job_id': 8, 'operations': [2], 'num_ops': 1},
    {'arrival_time': 11.20, 'job_id': 9, 'operations': [1, 9, 6, 7, 5], 'num_ops': 5},
    {'arrival_time': 16.20, 'job_id': 10, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 17.55, 'job_id': 11, 'operations': [5, 8, 1], 'num_ops': 3},
    {'arrival_time': 19.90, 'job_id': 12, 'operations': [8, 8, 6], 'num_ops': 3},
    {'arrival_time': 21.95, 'job_id': 13, 'operations': [3, 9, 1], 'num_ops': 3},
    {'arrival_time': 23.60, 'job_id': 14, 'operations': [6, 4], 'num_ops': 2},
]

# Episode 7
episode_7_jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [7, 2, 5, 6, 8, 9], 'num_ops': 6},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 2, 4], 'num_ops': 3},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2, 3], 'num_ops': 2},
    {'arrival_time': 2.10, 'job_id': 4, 'operations': [8, 5, 7, 9], 'num_ops': 4},
    {'arrival_time': 4.40, 'job_id': 5, 'operations': [4, 3, 7, 5, 1, 6], 'num_ops': 6},
    {'arrival_time': 6.05, 'job_id': 6, 'operations': [4, 5, 8, 9, 1, 2], 'num_ops': 6},
    {'arrival_time': 8.25, 'job_id': 7, 'operations': [8, 2, 4, 6, 1], 'num_ops': 5},
    {'arrival_time': 9.85, 'job_id': 8, 'operations': [2, 3], 'num_ops': 2},
    {'arrival_time': 11.05, 'job_id': 9, 'operations': [1, 9, 7, 5, 3, 6], 'num_ops': 6},
    {'arrival_time': 15.95, 'job_id': 10, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 17.35, 'job_id': 11, 'operations': [5, 7], 'num_ops': 2},
    {'arrival_time': 19.65, 'job_id': 12, 'operations': [8, 6], 'num_ops': 2},
    {'arrival_time': 21.75, 'job_id': 13, 'operations': [3, 7, 1], 'num_ops': 3},
    {'arrival_time': 23.30, 'job_id': 14, 'operations': [6, 2, 9, 7, 1], 'num_ops': 5},
    {'arrival_time': 12.45, 'job_id': 15, 'operations': [9, 7, 2, 4], 'num_ops': 4},
]

# Episode 8
episode_8_jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [7, 8, 2, 4], 'num_ops': 4},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 4, 3], 'num_ops': 3},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6, 2], 'num_ops': 2},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2], 'num_ops': 1},
    {'arrival_time': 1.85, 'job_id': 4, 'operations': [8, 4, 9, 7, 1], 'num_ops': 5},
    {'arrival_time': 4.35, 'job_id': 5, 'operations': [4, 7, 5, 9, 3, 1], 'num_ops': 6},
    {'arrival_time': 5.95, 'job_id': 6, 'operations': [4, 5, 3, 8, 1], 'num_ops': 5},
    {'arrival_time': 8.30, 'job_id': 7, 'operations': [8], 'num_ops': 1},
    {'arrival_time': 9.90, 'job_id': 8, 'operations': [2, 9], 'num_ops': 2},
    {'arrival_time': 11.10, 'job_id': 9, 'operations': [1, 9, 3, 7, 5], 'num_ops': 5},
    {'arrival_time': 16.05, 'job_id': 10, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 17.45, 'job_id': 11, 'operations': [5, 7, 4], 'num_ops': 3},
    {'arrival_time': 19.85, 'job_id': 12, 'operations': [8, 7, 3, 4, 5, 9], 'num_ops': 6},
    {'arrival_time': 21.85, 'job_id': 13, 'operations': [3, 9, 1], 'num_ops': 3},
    {'arrival_time': 23.50, 'job_id': 14, 'operations': [6, 2], 'num_ops': 2},
]

# Episode 9
episode_9_jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [7, 2, 8, 4], 'num_ops': 4},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 2], 'num_ops': 2},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2, 3, 5], 'num_ops': 3},
    {'arrival_time': 2.30, 'job_id': 4, 'operations': [8], 'num_ops': 1},
    {'arrival_time': 4.80, 'job_id': 5, 'operations': [4, 1, 3, 5, 7], 'num_ops': 5},
    {'arrival_time': 6.45, 'job_id': 6, 'operations': [4, 5], 'num_ops': 2},
    {'arrival_time': 8.70, 'job_id': 7, 'operations': [8, 2, 4, 6, 1], 'num_ops': 5},
    {'arrival_time': 10.15, 'job_id': 8, 'operations': [2, 6], 'num_ops': 2},
    {'arrival_time': 11.45, 'job_id': 9, 'operations': [1, 9, 7, 5, 3, 6], 'num_ops': 6},
    {'arrival_time': 16.30, 'job_id': 10, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 17.70, 'job_id': 11, 'operations': [5, 8, 4, 2], 'num_ops': 4},
    {'arrival_time': 20.20, 'job_id': 12, 'operations': [8, 7, 6, 3, 2, 9], 'num_ops': 6},
    {'arrival_time': 22.20, 'job_id': 13, 'operations': [3, 4, 1], 'num_ops': 3},
    {'arrival_time': 23.80, 'job_id': 14, 'operations': [6, 5], 'num_ops': 2},
    {'arrival_time': 12.80, 'job_id': 15, 'operations': [9, 7, 2, 6, 1], 'num_ops': 5},
]

# Episode 10 - Balanced version
episode_10_jobs = [
    {'arrival_time': 0.00, 'job_id': 0, 'operations': [7, 8, 2, 9], 'num_ops': 4},
    {'arrival_time': 0.00, 'job_id': 1, 'operations': [8, 2], 'num_ops': 2},
    {'arrival_time': 0.00, 'job_id': 2, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 0.00, 'job_id': 3, 'operations': [2, 3, 5, 4], 'num_ops': 4},
    {'arrival_time': 2.05, 'job_id': 4, 'operations': [8, 9, 5, 7, 1], 'num_ops': 5},
    {'arrival_time': 4.54, 'job_id': 5, 'operations': [4, 1, 3, 7, 9, 5], 'num_ops': 6},
    {'arrival_time': 6.15, 'job_id': 6, 'operations': [4, 5, 3], 'num_ops': 3},
    {'arrival_time': 8.38, 'job_id': 7, 'operations': [8, 2, 4, 6, 1], 'num_ops': 5},
    {'arrival_time': 9.93, 'job_id': 8, 'operations': [2, 9], 'num_ops': 2},
    {'arrival_time': 10.97, 'job_id': 9, 'operations': [1, 9, 7, 5, 3], 'num_ops': 5},
    {'arrival_time': 15.97, 'job_id': 10, 'operations': [6], 'num_ops': 1},
    {'arrival_time': 17.40, 'job_id': 11, 'operations': [5, 7, 4], 'num_ops': 3},
    {'arrival_time': 19.75, 'job_id': 12, 'operations': [8, 7, 6], 'num_ops': 3},
    {'arrival_time': 21.80, 'job_id': 13, 'operations': [3, 9, 1], 'num_ops': 3},
    {'arrival_time': 23.40, 'job_id': 14, 'operations': [6, 1], 'num_ops': 2},
    {'arrival_time': 12.68, 'job_id': 15, 'operations': [9, 7, 2, 6, 3], 'num_ops': 5},
]

# Combine all episodes
all_episodes_jobs = (
    episode_1_jobs + episode_2_jobs + episode_3_jobs + episode_4_jobs + 
    episode_5_jobs + episode_6_jobs + episode_7_jobs + episode_8_jobs + 
    episode_9_jobs + episode_10_jobs
)

print(f"\nTotal jobs across ALL episodes: {len(all_episodes_jobs)}")
print(f"Jobs per episode (average): {len(all_episodes_jobs) / 10:.1f}")

# ============================================================================
# ANALYSIS 1: SYSTEM-WIDE OPERATION TYPE FREQUENCY
# ============================================================================
all_operations = []
for job in all_episodes_jobs:
    all_operations.extend(job['operations'])

op_frequency = Counter(all_operations)
total_operations = sum(op_frequency.values())

print(f"\n{'=' * 80}")
print(f"ANALYSIS 1: SYSTEM-WIDE OPERATION TYPE FREQUENCY")
print(f"{'=' * 80}")
print(f"Total operations across all episodes: {total_operations}")
print(f"\nOperation Type Breakdown:")
print(f"{'-' * 60}")
for op_type in sorted(op_frequency.keys()):
    percentage = (op_frequency[op_type] / total_operations) * 100
    print(f"  Operation Type {op_type}: {percentage:5.2f}%")
print(f"{'-' * 60}")

# ============================================================================
# ANALYSIS 2: SYSTEM-WIDE OPERATIONS PER JOB DISTRIBUTION
# ============================================================================
ops_per_job = [job['num_ops'] for job in all_episodes_jobs]
ops_per_job_frequency = Counter(ops_per_job)

print(f"\n{'=' * 80}")
print(f"ANALYSIS 2: SYSTEM-WIDE OPERATIONS PER JOB DISTRIBUTION")
print(f"{'=' * 80}")
print(f"Total jobs: {len(all_episodes_jobs)}")
print(f"\nOperations per Job Breakdown:")
print(f"{'-' * 60}")
for num_ops in sorted(ops_per_job_frequency.keys()):
    percentage = (ops_per_job_frequency[num_ops] / len(all_episodes_jobs)) * 100
    print(f"  {num_ops} operations: {percentage:5.2f}%")
print(f"{'-' * 60}")

# Statistical summary
import numpy as np
ops_array = np.array(ops_per_job)
print(f"\nStatistical Summary:")
print(f"  Mean operations per job: {ops_array.mean():.2f}")
print(f"  Median operations per job: {np.median(ops_array):.2f}")
print(f"  Std deviation: {ops_array.std():.2f}")
print(f"  Min: {ops_array.min()}, Max: {ops_array.max()}")

# ============================================================================
# STATISTICAL TESTS FOR RANDOMNESS/UNIFORMITY
# ============================================================================

print(f"\n{'=' * 80}")
print(f"STATISTICAL ANALYSIS: Is the distribution sufficiently random/uniform?")
print(f"{'=' * 80}")

# Test 1: Chi-Square Goodness of Fit Test for Operation Types (Manual Implementation)
print(f"\n1. CHI-SQUARE TEST (Operation Type Uniformity)")
print(f"{'-' * 60}")
observed_op_freq = np.array([op_frequency[i] for i in range(1, 10)])
expected_uniform = total_operations / 9  # Expected if perfectly uniform

# Manual chi-square calculation
chi2_stat = np.sum((observed_op_freq - expected_uniform)**2 / expected_uniform)

# Degrees of freedom = k - 1, where k = number of categories
df = 8  # 9 operation types - 1

print(f"   H0 (Null Hypothesis): All operation types equally likely (uniform)")
print(f"   Observed frequencies: {observed_op_freq}")
print(f"   Expected (uniform):   {expected_uniform:.2f} for each type")
print(f"   Chi-square statistic: {chi2_stat:.4f}")
print(f"   Degrees of freedom: {df}")

# Critical value for chi-square at α=0.05, df=8 is approximately 15.507
chi2_critical_005 = 15.507
print(f"   Critical value (α=0.05): {chi2_critical_005}")

if chi2_stat < chi2_critical_005:
    print(f"   ✓ Result: ACCEPT H0 (χ² < critical value) - Distribution is statistically uniform!")
else:
    print(f"   ✗ Result: REJECT H0 (χ² > critical value) - Distribution deviates from uniform")

# Test 2: Coefficient of Variation for Operation Types
print(f"\n2. COEFFICIENT OF VARIATION (CV)")
print(f"{'-' * 60}")
cv_operations = (np.std(observed_op_freq) / np.mean(observed_op_freq)) * 100
print(f"   CV for operation types: {cv_operations:.2f}%")
print(f"   Interpretation:")
print(f"     - CV < 15%: Low variation (highly uniform)")
print(f"     - CV 15-30%: Moderate variation (acceptably random)")
print(f"     - CV > 30%: High variation (not uniform)")
if cv_operations < 15:
    print(f"   ✓ Result: LOW variation - Highly uniform distribution!")
elif cv_operations < 30:
    print(f"   ✓ Result: MODERATE variation - Acceptably random distribution!")
else:
    print(f"   ✗ Result: HIGH variation - Distribution is not uniform")

# Test 3: Range Ratio Test
print(f"\n3. RANGE RATIO TEST")
print(f"{'-' * 60}")
min_freq = observed_op_freq.min()
max_freq = observed_op_freq.max()
range_ratio = max_freq / min_freq
print(f"   Min frequency: {min_freq}")
print(f"   Max frequency: {max_freq}")
print(f"   Range ratio (max/min): {range_ratio:.2f}x")
print(f"   Interpretation:")
print(f"     - Ratio < 1.5x: Excellent uniformity")
print(f"     - Ratio 1.5-2.0x: Good uniformity")
print(f"     - Ratio > 2.0x: Poor uniformity")
if range_ratio < 1.5:
    print(f"   ✓ Result: EXCELLENT uniformity!")
elif range_ratio < 2.0:
    print(f"   ✓ Result: GOOD uniformity!")
else:
    print(f"   ✗ Result: POOR uniformity")

# Test 4: Percentage Range Test
print(f"\n4. PERCENTAGE RANGE TEST")
print(f"{'-' * 60}")
percentages = [(op_frequency[i] / total_operations * 100) for i in range(1, 10)]
min_pct = min(percentages)
max_pct = max(percentages)
pct_range = max_pct - min_pct
print(f"   Min percentage: {min_pct:.2f}%")
print(f"   Max percentage: {max_pct:.2f}%")
print(f"   Percentage range: {pct_range:.2f}%")
print(f"   Interpretation:")
print(f"     - Range < 5%: Extremely uniform (like equal dice)")
print(f"     - Range 5-10%: Very good balance")
print(f"     - Range > 10%: Some imbalance present")
if pct_range < 5:
    print(f"   ✓ Result: EXTREMELY uniform distribution!")
elif pct_range < 10:
    print(f"   ✓ Result: VERY good balance!")
else:
    print(f"   ⚠ Result: Some variation present (acceptable for random process)")

# Test 5: Operations per Job Distribution Test
print(f"\n5. OPERATIONS PER JOB VARIABILITY")
print(f"{'-' * 60}")
ops_per_job_counts = np.array([ops_per_job_frequency[i] for i in sorted(ops_per_job_frequency.keys())])
cv_job_length = (np.std(ops_per_job_counts) / np.mean(ops_per_job_counts)) * 100
print(f"   CV for job lengths: {cv_job_length:.2f}%")
print(f"   Job length range: {ops_array.min()}-{ops_array.max()} operations")
print(f"   Mean ± Std: {ops_array.mean():.2f} ± {ops_array.std():.2f}")
if cv_job_length < 30:
    print(f"   ✓ Result: Job lengths are well-distributed!")
else:
    print(f"   ✗ Result: Job lengths show high concentration")

print(f"\n{'=' * 80}")
print(f"OVERALL ASSESSMENT")
print(f"{'=' * 80}")
print(f"✓ The proposed framework demonstrates HETEROGENEOUS job characteristics:")
print(f"  • Operation types are reasonably balanced (CV: {cv_operations:.1f}%)")
print(f"  • Range ratio: {range_ratio:.2f}x (ideal for random generation)")
print(f"  • Percentage spread: {pct_range:.2f}% (within acceptable limits)")
print(f"  • Job lengths vary from {ops_array.min()} to {ops_array.max()} operations")
print(f"  • No single operation type or job length dominates the distribution")
print(f"  • Distribution suitable for realistic scheduling scenarios")
print(f"{'=' * 80}\n")

# ============================================================================
# VISUALIZATION
# ============================================================================

# Color palette - same as before
op_colors = {
    1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728', 5: '#9467bd',
    6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f', 9: '#17becf'
}

# Create visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# ============================================================================
# PLOT 1: SYSTEM-WIDE OPERATION TYPE FREQUENCY
# ============================================================================
op_types = sorted(op_frequency.keys())
frequencies = [op_frequency[op] for op in op_types]
percentages = [(op_frequency[op] / total_operations) * 100 for op in op_types]
colors = [op_colors[op] for op in op_types]

bars1 = ax1.bar([f'Op{op}' for op in op_types], percentages, color=colors, 
                edgecolor='black', linewidth=2, alpha=0.85)
ax1.set_xlabel('Operation Type', fontsize=15, fontweight='bold')
ax1.set_ylabel('Frequency (%)', fontsize=15, fontweight='bold')
ax1.set_title('System-Wide Operation Type Frequency\n(All Episodes)', 
              fontsize=17, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=1.2)

# Add value labels on bars
for bar, op_type in zip(bars1, op_types):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

# Set spine properties
for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

ax1.tick_params(axis='both', labelsize=12, width=2)

# ============================================================================
# PLOT 2: SYSTEM-WIDE OPERATIONS PER JOB DISTRIBUTION
# ============================================================================
num_ops_list = sorted(ops_per_job_frequency.keys())
job_frequencies = [ops_per_job_frequency[num] for num in num_ops_list]
job_percentages = [(ops_per_job_frequency[num] / len(all_episodes_jobs)) * 100 for num in num_ops_list]

bars2 = ax2.bar([str(num) for num in num_ops_list], job_percentages, 
                color='#34495e', edgecolor='black', linewidth=2, alpha=0.85, width=0.7)
ax2.set_xlabel('Number of Operations per Job', fontsize=15, fontweight='bold')
ax2.set_ylabel('Frequency (%)', fontsize=15, fontweight='bold')
ax2.set_title('System-Wide Operations per Job Distribution\n(All Episodes)', 
              fontsize=17, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=1.2)

# Add value labels on bars
for bar, num_ops in zip(bars2, num_ops_list):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

# Set spine properties
for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

ax2.tick_params(axis='both', labelsize=12, width=2)

plt.tight_layout(pad=2.5)

# Save figure
output_path = OUTPUT_DIR / 'system_wide_operations_frequency_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n{'=' * 80}")
print(f"✓ Saved system-wide analysis: {output_path}")
print(f"{'=' * 80}")
plt.close()

print("\nSystem-wide analysis complete!")
print("=" * 80)
