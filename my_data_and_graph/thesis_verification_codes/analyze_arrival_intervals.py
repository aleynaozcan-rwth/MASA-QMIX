#!/usr/bin/env python3
"""
Analyze inter-arrival times for proposed framework
"""

import numpy as np

# Job arrivals from Episode 10
arrivals = [
    {'time': 0.00, 'jobs': ['Job_0', 'Job_1', 'Job_2', 'Job_3']},  # 4 jobs at t=0
    {'time': 2.05, 'jobs': ['Job_4']},
    {'time': 4.54, 'jobs': ['Job_5']},
    {'time': 6.15, 'jobs': ['Job_6']},
    {'time': 8.38, 'jobs': ['Job_7']},
    {'time': 9.93, 'jobs': ['Job_8']},
    {'time': 10.97, 'jobs': ['Job_9']},
    {'time': 12.68, 'jobs': ['Job_10']},
    {'time': 15.97, 'jobs': ['Job_11']},
    {'time': 17.40, 'jobs': ['Job_12']},
    {'time': 19.75, 'jobs': ['Job_13']},
    {'time': 21.80, 'jobs': ['Job_14']},
    {'time': 23.40, 'jobs': ['Job_15']},
]

# Extract arrival times
arrival_times = [a['time'] for a in arrivals]

# Calculate inter-arrival intervals
intervals = []
for i in range(1, len(arrival_times)):
    interval = arrival_times[i] - arrival_times[i-1]
    intervals.append(interval)

print("="*60)
print("INTER-ARRIVAL TIME ANALYSIS (Proposed Framework)")
print("="*60)

print("\n--- Arrival Timeline ---")
for i, arrival in enumerate(arrivals):
    jobs_str = ', '.join(arrival['jobs'])
    print(f"Arrival {i+1}: t={arrival['time']:.2f} -> {jobs_str}")

print(f"\n--- Inter-Arrival Intervals ---")
for i, interval in enumerate(intervals):
    print(f"Interval {i+1}: {arrival_times[i]:.2f} → {arrival_times[i+1]:.2f} = {interval:.2f} time units")

print(f"\n--- Statistical Summary ---")
print(f"Total number of arrival events: {len(arrival_times)}")
print(f"Total number of jobs: {sum(len(a['jobs']) for a in arrivals)}")
print(f"Number of inter-arrival intervals: {len(intervals)}")
print(f"\nInter-Arrival Interval Statistics:")
print(f"  Mean interval: {np.mean(intervals):.2f} time units")
print(f"  Median interval: {np.median(intervals):.2f} time units")
print(f"  Std deviation: {np.std(intervals):.2f} time units")
print(f"  Min interval: {np.min(intervals):.2f} time units")
print(f"  Max interval: {np.max(intervals):.2f} time units")
print(f"  Total time span: {arrival_times[-1] - arrival_times[0]:.2f} time units")

print(f"\n--- Arrival Rate ---")
total_jobs = sum(len(a['jobs']) for a in arrivals)
time_span = arrival_times[-1] - arrival_times[0]
if time_span > 0:
    arrival_rate = total_jobs / time_span
    print(f"Average arrival rate: {arrival_rate:.3f} jobs/time unit")
    print(f"Average time between job arrivals: {time_span / (total_jobs - 1):.2f} time units")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
