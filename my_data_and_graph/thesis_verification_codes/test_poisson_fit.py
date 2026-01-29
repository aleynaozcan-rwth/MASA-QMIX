#!/usr/bin/env python3
"""
Test if inter-arrival times follow exponential distribution (Poisson process)
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

# Extract arrival times and intervals
arrival_times = [a['time'] for a in arrivals]
intervals = []
for i in range(1, len(arrival_times)):
    interval = arrival_times[i] - arrival_times[i-1]
    intervals.append(interval)

intervals = np.array(intervals)

print("="*70)
print("POISSON PROCESS FIT TEST")
print("="*70)

# Basic statistics
mean_interval = np.mean(intervals)
std_interval = np.std(intervals, ddof=1)  # Sample std
var_interval = np.var(intervals, ddof=1)
min_interval = np.min(intervals)
max_interval = np.max(intervals)
cv = std_interval / mean_interval  # Coefficient of variation

print("\n--- OBSERVED DATA ---")
print(f"Number of intervals: {len(intervals)}")
print(f"Mean (μ): {mean_interval:.4f} time units")
print(f"Std Dev (σ): {std_interval:.4f} time units")
print(f"Variance (σ²): {var_interval:.4f}")
print(f"Min: {min_interval:.4f}, Max: {max_interval:.4f}")
print(f"Coefficient of Variation (CV = σ/μ): {cv:.4f}")

# Exponential distribution properties
print("\n--- EXPONENTIAL DISTRIBUTION (Poisson Process) ---")
lambda_param = 1.0 / mean_interval
expected_std = mean_interval  # For exponential: σ = μ
expected_var = mean_interval ** 2  # For exponential: σ² = μ²
expected_cv = 1.0  # For exponential: CV = 1

print(f"Rate parameter (λ): {lambda_param:.4f}")
print(f"Expected mean: {mean_interval:.4f} (same as observed by construction)")
print(f"Expected std: {expected_std:.4f}")
print(f"Expected variance: {expected_var:.4f}")
print(f"Expected CV: {expected_cv:.4f}")

# Comparison
print("\n--- FIT ASSESSMENT ---")
print(f"Observed σ vs Expected σ:")
print(f"  Observed: {std_interval:.4f}")
print(f"  Expected: {expected_std:.4f}")
print(f"  Ratio: {std_interval/expected_std:.4f} (should be ≈1.0 for exponential)")

print(f"\nCoefficient of Variation:")
print(f"  Observed CV: {cv:.4f}")
print(f"  Expected CV: {expected_cv:.4f}")
print(f"  Difference: {abs(cv - expected_cv):.4f}")

# Manual Kolmogorov-Smirnov test
sorted_intervals = np.sort(intervals)
n = len(sorted_intervals)

# Empirical CDF
empirical_cdf = np.arange(1, n + 1) / n

# Theoretical exponential CDF: F(x) = 1 - exp(-λx)
theoretical_cdf = 1 - np.exp(-lambda_param * sorted_intervals)

# KS statistic: maximum difference between empirical and theoretical CDF
ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))

# Critical value for KS test at α=0.05 (approximation for small n)
# Critical value ≈ 1.36 / sqrt(n) for α=0.05
ks_critical = 1.36 / np.sqrt(n)

print(f"\n--- KOLMOGOROV-SMIRNOV TEST ---")
print(f"KS Statistic (D): {ks_statistic:.4f}")
print(f"Critical Value (α=0.05): {ks_critical:.4f}")
print(f"Result: {'PASS' if ks_statistic < ks_critical else 'FAIL'} at 95% confidence")

if ks_statistic < ks_critical:
    print("  → Cannot reject null hypothesis: data may follow exponential distribution")
else:
    print("  → Reject null hypothesis: data does not follow exponential distribution")

# Additional tests
print("\n--- MEMORYLESS PROPERTY CHECK ---")
print("For Poisson process (exponential distribution), intervals should be memoryless.")
print(f"Observed intervals:")
for i, interval in enumerate(intervals, 1):
    print(f"  Interval {i}: {interval:.2f}")

# Check for autocorrelation (simple lag-1)
if len(intervals) > 1:
    lag1_corr = np.corrcoef(intervals[:-1], intervals[1:])[0, 1]
    print(f"\nLag-1 Autocorrelation: {lag1_corr:.4f}")
    print(f"  (should be ≈0 for independent exponential intervals)")
    if abs(lag1_corr) < 0.3:
        print("  → Low autocorrelation: consistent with independence")
    else:
        print("  → High autocorrelation: may indicate dependence")

# Overall conclusion
print("\n" + "="*70)
print("OVERALL CONCLUSION")
print("="*70)

criteria_met = 0
total_criteria = 4

print("\nCriteria for Poisson Process:")

# Criterion 1: CV ≈ 1
if abs(cv - 1.0) < 0.3:
    print("✓ CV close to 1.0 (observed: {:.3f})".format(cv))
    criteria_met += 1
else:
    print("✗ CV differs from 1.0 (observed: {:.3f}, expected: 1.0)".format(cv))

# Criterion 2: σ ≈ μ
if abs(std_interval - mean_interval) / mean_interval < 0.3:
    print("✓ Std Dev ≈ Mean (σ={:.3f}, μ={:.3f})".format(std_interval, mean_interval))
    criteria_met += 1
else:
    print("✗ Std Dev ≠ Mean (σ={:.3f}, μ={:.3f})".format(std_interval, mean_interval))

# Criterion 3: KS test
if ks_statistic < ks_critical:
    print("✓ KS test passed (D={:.3f} < {:.3f})".format(ks_statistic, ks_critical))
    criteria_met += 1
else:
    print("✗ KS test failed (D={:.3f} ≥ {:.3f})".format(ks_statistic, ks_critical))

# Criterion 4: Low autocorrelation
if len(intervals) > 1 and abs(lag1_corr) < 0.3:
    print("✓ Low autocorrelation (r={:.3f})".format(lag1_corr))
    criteria_met += 1
else:
    print("✗ High autocorrelation or insufficient data")

print("\n" + "-"*70)
print(f"Criteria met: {criteria_met}/{total_criteria}")

if criteria_met >= 3:
    print("\n✓ CONCLUSION: Data is CONSISTENT with Poisson process")
    print("  → Inter-arrival times approximately follow exponential distribution")
    print("  → Can claim: 'Jobs arrive according to a Poisson process'")
elif criteria_met >= 2:
    print("\n⚠ CONCLUSION: Data is PARTIALLY consistent with Poisson process")
    print("  → Some deviations observed but generally reasonable")
    print("  → Can claim: 'Jobs arrive with stochastic intervals approximating a Poisson process'")
else:
    print("\n✗ CONCLUSION: Data does NOT follow Poisson process")
    print("  → Consider stating: 'Jobs arrive with stochastic intervals'")
    print("  → Avoid claiming Poisson distribution")

print("="*70)
