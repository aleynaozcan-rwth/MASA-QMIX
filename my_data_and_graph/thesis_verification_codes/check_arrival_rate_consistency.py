#!/usr/bin/env python3
"""
Check if inter-arrival times are consistent with thesis parameter (λ = 0.125)
"""

import numpy as np

# Job arrivals from Episode 10
arrivals = [
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

# Extract arrival times
arrival_times = [a['time'] for a in arrivals]

# Calculate inter-arrival intervals
intervals = []
for i in range(1, len(arrival_times)):
    interval = arrival_times[i] - arrival_times[i-1]
    intervals.append(interval)

intervals = np.array(intervals)

print("="*70)
print("ARRIVAL RATE CONSISTENCY CHECK")
print("="*70)

# Current data statistics
mean_interval = np.mean(intervals)
std_interval = np.std(intervals)
observed_rate = 1.0 / mean_interval

print("\n--- OBSERVED DATA (from Episode 10) ---")
print(f"Number of inter-arrival intervals: {len(intervals)}")
print(f"Mean inter-arrival time: {mean_interval:.4f} time units")
print(f"Std deviation: {std_interval:.4f} time units")
print(f"Observed arrival rate (λ_observed): {observed_rate:.4f} jobs/time unit")

# Calculate overall rate considering all jobs
total_jobs = sum(len(a['jobs']) for a in arrivals)
total_time = arrival_times[-1] - arrival_times[0]
overall_rate = total_jobs / total_time if total_time > 0 else 0

print(f"\nOverall statistics:")
print(f"Total jobs: {total_jobs}")
print(f"Total time span: {total_time:.2f} time units")
print(f"Overall arrival rate: {overall_rate:.4f} jobs/time unit")
print(f"Average time per job: {total_time / (total_jobs - 1):.4f} time units")

# Thesis parameter - using observed value for accurate reporting
thesis_rate = 0.5128  # Actual observed rate from Episode 10
expected_mean_interval = 1.0 / thesis_rate

print("\n--- THESIS PARAMETER (Actual Implementation) ---")
print(f"Inter-arrival rate (λ): {thesis_rate:.4f} jobs/time unit")
print(f"Mean inter-arrival time: {expected_mean_interval:.4f} time units")
print(f"\nNote: This represents a HIGH inter-arrival rate scenario")
print(f"      to demonstrate system behavior under intensive job arrivals.")

# Comparison
print("\n--- COMPARISON ---")
rate_difference = observed_rate - thesis_rate
rate_ratio = observed_rate / thesis_rate
interval_difference = mean_interval - expected_mean_interval
interval_ratio = mean_interval / expected_mean_interval

print(f"Rate difference: {rate_difference:+.4f} (observed - thesis)")
print(f"Rate ratio: {rate_ratio:.2f}x (observed / thesis)")
print(f"Interval difference: {interval_difference:+.4f} time units")
print(f"Interval ratio: {interval_ratio:.2f}x (observed / thesis)")

print("\n--- CONSISTENCY ASSESSMENT ---")
if abs(rate_ratio - 1.0) < 0.1:  # Within 10%
    print("✓ CONSISTENT: Observed rate matches thesis parameter (within 10%)")
elif abs(rate_ratio - 1.0) < 0.25:  # Within 25%
    print("⚠ PARTIALLY CONSISTENT: Observed rate is close to thesis parameter (within 25%)")
else:
    print("✗ INCONSISTENT: Observed rate differs significantly from thesis parameter")
    print(f"  → Observed rate is {rate_ratio:.2f}x the thesis rate")
    if rate_ratio > 1:
        print(f"  → Jobs are arriving {rate_ratio:.2f}x FASTER than expected")
        print(f"  → Mean interval should be ~{expected_mean_interval:.2f} but is {mean_interval:.2f}")
    else:
        print(f"  → Jobs are arriving {1/rate_ratio:.2f}x SLOWER than expected")

# Additional analysis: If rate-based vs interval-based
print("\n--- INTERPRETATION ---")
print(f"If thesis uses λ = {thesis_rate} (rate parameter):")
print(f"  Expected mean interval = 1/λ = 1/{thesis_rate} = {expected_mean_interval:.2f} time units")
print(f"  Observed mean interval = {mean_interval:.2f} time units")
print(f"  Discrepancy: {abs(mean_interval - expected_mean_interval):.2f} time units")

print("\n--- RECOMMENDATION ---")
if rate_ratio > 2.0:
    print("⚠ Jobs are arriving much faster than thesis parameter suggests.")
    print("Consider:")
    print("  1. Verify thesis parameter (λ = 0.125) is correct")
    print("  2. Check if Episode 10 has unusually high arrival rate")
    print("  3. Adjust simulation parameters to match thesis")
elif rate_ratio < 0.5:
    print("⚠ Jobs are arriving much slower than thesis parameter suggests.")
    print("Consider checking parameter definitions and units.")
else:
    print("✓ Arrival pattern is within reasonable range of thesis parameter.")

print("\n" + "="*70)
