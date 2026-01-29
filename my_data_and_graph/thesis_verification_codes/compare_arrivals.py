#!/usr/bin/env python3
"""
Compare OLD vs NEW (Poisson) arrival times
"""

# OLD arrival times (from original Episode 10)
old_arrivals = [
    {'time': 0.00, 'jobs': ['Job_0', 'Job_1', 'Job_2', 'Job_3']},
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

# NEW Poisson arrival times
new_arrivals = [
    {'time': 0.00, 'jobs': ['Job_0', 'Job_1', 'Job_2', 'Job_3']},
    {'time': 0.92, 'jobs': ['Job_4']},
    {'time': 6.78, 'jobs': ['Job_5']},
    {'time': 9.35, 'jobs': ['Job_6']},
    {'time': 11.13, 'jobs': ['Job_7']},
    {'time': 11.46, 'jobs': ['Job_8']},
    {'time': 11.79, 'jobs': ['Job_9']},
    {'time': 11.91, 'jobs': ['Job_10']},
    {'time': 15.83, 'jobs': ['Job_11']},
    {'time': 17.62, 'jobs': ['Job_12']},
    {'time': 20.03, 'jobs': ['Job_13']},
    {'time': 20.07, 'jobs': ['Job_14']},
    {'time': 26.90, 'jobs': ['Job_15']},
]

print("="*80)
print("ARRIVAL TIME COMPARISON: OLD (Non-Poisson) vs NEW (True Poisson)")
print("="*80)

print("\n{:<15} {:<20} {:<20} {:<15}".format("Job ID", "OLD Time", "NEW Time", "Change"))
print("-"*80)

# Create flat list for comparison
old_flat = []
for arr in old_arrivals:
    for job in arr['jobs']:
        old_flat.append((job, arr['time']))

new_flat = []
for arr in new_arrivals:
    for job in arr['jobs']:
        new_flat.append((job, arr['time']))

for (job_old, time_old), (job_new, time_new) in zip(old_flat, new_flat):
    change = time_new - time_old
    change_str = f"{change:+.2f}"
    print("{:<15} {:<20} {:<20} {:<15}".format(
        job_old, 
        f"t={time_old:.2f}",
        f"t={time_new:.2f}",
        change_str
    ))

print("-"*80)
print(f"{'Timeline End':<15} {'t=23.40':<20} {'t=26.90':<20} {'+3.50':<15}")

# Calculate interval changes
print("\n" + "="*80)
print("INTER-ARRIVAL INTERVAL COMPARISON")
print("="*80)

old_times = [a['time'] for a in old_arrivals]
new_times = [a['time'] for a in new_arrivals]

old_intervals = [old_times[i+1] - old_times[i] for i in range(len(old_times)-1)]
new_intervals = [new_times[i+1] - new_times[i] for i in range(len(new_times)-1)]

import numpy as np

print("\n{:<20} {:<15} {:<15}".format("Metric", "OLD", "NEW (Poisson)"))
print("-"*80)
print("{:<20} {:<15} {:<15}".format("Mean Interval", f"{np.mean(old_intervals):.4f}", f"{np.mean(new_intervals):.4f}"))
print("{:<20} {:<15} {:<15}".format("Std Dev", f"{np.std(old_intervals, ddof=1):.4f}", f"{np.std(new_intervals, ddof=1):.4f}"))
print("{:<20} {:<15} {:<15}".format("CV (σ/μ)", f"{np.std(old_intervals, ddof=1)/np.mean(old_intervals):.4f}", f"{np.std(new_intervals, ddof=1)/np.mean(new_intervals):.4f}"))
print("{:<20} {:<15} {:<15}".format("Min Interval", f"{np.min(old_intervals):.4f}", f"{np.min(new_intervals):.4f}"))
print("{:<20} {:<15} {:<15}".format("Max Interval", f"{np.max(old_intervals):.4f}", f"{np.max(new_intervals):.4f}"))

print("\n" + "="*80)
print("KEY OBSERVATIONS")
print("="*80)

print("\n1. INDIVIDUAL JOB CHANGES:")
print("   - Job_4: 2.05 → 0.92 (arrived 1.13 time units EARLIER)")
print("   - Job_5: 4.54 → 6.78 (arrived 2.24 time units LATER)")
print("   - Job_10: 12.68 → 11.91 (arrived 0.77 time units EARLIER)")
print("   - Job_15: 23.40 → 26.90 (arrived 3.50 time units LATER)")

print("\n2. VARIABILITY INCREASE:")
old_cv = np.std(old_intervals, ddof=1)/np.mean(old_intervals)
new_cv = np.std(new_intervals, ddof=1)/np.mean(new_intervals)
print(f"   - OLD CV: {old_cv:.4f} (too regular, not Poisson)")
print(f"   - NEW CV: {new_cv:.4f} (≈1.0, TRUE Poisson!)")
print(f"   - Increase: {(new_cv - old_cv):.4f} ({(new_cv/old_cv - 1)*100:.1f}% more variable)")

print("\n3. EXTREME INTERVALS:")
print(f"   - OLD: Min={np.min(old_intervals):.2f}, Max={np.max(old_intervals):.2f} (range: {np.max(old_intervals)-np.min(old_intervals):.2f})")
print(f"   - NEW: Min={np.min(new_intervals):.2f}, Max={np.max(new_intervals):.2f} (range: {np.max(new_intervals)-np.min(new_intervals):.2f})")
print("   - NEW has more extreme values (characteristic of exponential dist)")

print("\n4. CLUSTERING:")
print("   - OLD: More evenly spaced")
print("   - NEW: Shows clustering (e.g., Job_7,8,9,10 between t=11.13-11.91)")
print("   - This clustering is NORMAL for Poisson process!")

print("\n" + "="*80)
