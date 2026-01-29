#!/usr/bin/env python3
"""
Generate TRUE Poisson arrivals with same mean but proper exponential distribution
"""

import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Target parameters (from current Episode 10)
target_mean_interval = 1.95
num_intervals = 12  # 13 arrivals = 12 intervals
timeline_end_target = 23.40  # approximately

# Generate exponential inter-arrival intervals
# For Poisson process: intervals ~ Exponential(λ), where mean = 1/λ
intervals = np.random.exponential(scale=target_mean_interval, size=num_intervals)

print("="*70)
print("GENERATING TRUE POISSON ARRIVALS")
print("="*70)

print("\n--- Generated Inter-Arrival Intervals ---")
for i, interval in enumerate(intervals, 1):
    print(f"Interval {i}: {interval:.2f} time units")

# Calculate statistics
mean_interval = np.mean(intervals)
std_interval = np.std(intervals, ddof=1)
cv = std_interval / mean_interval

print("\n--- Interval Statistics ---")
print(f"Mean: {mean_interval:.4f} (target: {target_mean_interval:.4f})")
print(f"Std Dev: {std_interval:.4f}")
print(f"Variance: {std_interval**2:.4f}")
print(f"CV: {cv:.4f} (should be ≈1.0 for exponential)")
print(f"Min: {np.min(intervals):.4f}")
print(f"Max: {np.max(intervals):.4f}")

# Generate arrival times
arrival_times = [0.00]  # First arrival at t=0 (4 jobs)
cumulative_time = 0.0

for interval in intervals:
    cumulative_time += interval
    arrival_times.append(cumulative_time)

print("\n--- Generated Arrival Times ---")
print(f"Arrival 1: t=0.00 -> Job_0, Job_1, Job_2, Job_3 (4 jobs)")
for i, t in enumerate(arrival_times[1:], 2):
    print(f"Arrival {i}: t={t:.2f} -> Job_{i+2}")

print(f"\nTotal timeline: 0.00 to {arrival_times[-1]:.2f} time units")
print(f"(Original timeline was 0.00 to 23.40)")

# Generate Python code for updating files
print("\n" + "="*70)
print("UPDATED ARRIVAL DATA (copy to your scripts)")
print("="*70)

print("\narrival_times = [")
for i, t in enumerate(arrival_times):
    if i == 0:
        print(f"    {t:.2f},  # Arrival 1: Job_0, Job_1, Job_2, Job_3")
    else:
        print(f"    {t:.2f},  # Arrival {i+1}: Job_{i+2}")
print("]")

print("\narrivals = [")
print(f"    {{'time': {arrival_times[0]:.2f}, 'jobs': ['Job_0', 'Job_1', 'Job_2', 'Job_3']}},  # 4 jobs at t=0")
for i, t in enumerate(arrival_times[1:], 4):
    print(f"    {{'time': {t:.2f}, 'jobs': ['Job_{i}']}},")
print("]")

# Test Poisson fit
print("\n" + "="*70)
print("POISSON FIT VERIFICATION")
print("="*70)

print("\nExpected for TRUE Poisson:")
print(f"✓ Mean interval: {mean_interval:.2f}")
print(f"✓ Std Dev ≈ Mean: {std_interval:.2f} ≈ {mean_interval:.2f} (ratio: {std_interval/mean_interval:.2f})")
print(f"✓ CV ≈ 1.0: {cv:.2f}")

if abs(cv - 1.0) < 0.15:
    print("\n✓ EXCELLENT: CV is very close to 1.0 - TRUE Poisson process!")
elif abs(cv - 1.0) < 0.3:
    print("\n✓ GOOD: CV is close to 1.0 - Acceptable Poisson approximation")
else:
    print("\n⚠ Note: CV deviates from 1.0 - may need regeneration with different seed")

print("\nNote: Different random seed will give different intervals,")
print("but all will have CV ≈ 1.0 if generated from exponential distribution.")

print("\n" + "="*70)
