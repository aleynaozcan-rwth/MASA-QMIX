#!/usr/bin/env python3
"""
Analyze Job Arrival Stochasticity using CDF
Compare empirical inter-arrival times with theoretical exponential distribution (lambda=0.125)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt')

print("Parsing job arrival times from all episodes...")

all_inter_arrival_times = []
episode_count = 0

with open(DATA_FILE, 'r') as f:
    content = f.read()

# Split by episodes
episodes = content.split('=== EPISODE ')

for episode_text in episodes[1:]:  # Skip first empty split
    lines = episode_text.split('\n')
    
    # Extract arrival times from lifecycle trace
    arrival_times = []
    for line in lines:
        if 'New job' in line and 'arrived with' in line:
            # Extract time from lines like: "[t=0.00] New job 0 arrived with 5 ops"
            if '[t=' in line:
                time_str = line.split('[t=')[1].split(']')[0]
                arrival_times.append(float(time_str))
    
    # Calculate inter-arrival times for this episode
    if len(arrival_times) > 1:
        arrival_times.sort()
        for i in range(1, len(arrival_times)):
            inter_arrival = arrival_times[i] - arrival_times[i-1]
            if inter_arrival > 0:  # Ignore simultaneous arrivals at t=0
                all_inter_arrival_times.append(inter_arrival)
        episode_count += 1

print(f"Analyzed {episode_count} episodes")
print(f"Total inter-arrival times collected: {len(all_inter_arrival_times)}")
print(f"Mean inter-arrival time: {np.mean(all_inter_arrival_times):.4f}")
print(f"Std inter-arrival time: {np.std(all_inter_arrival_times):.4f}")
print(f"Coefficient of Variation (CV): {np.std(all_inter_arrival_times)/np.mean(all_inter_arrival_times):.4f}")

# Theoretical exponential distribution with lambda=0.125
lambda_rate = 0.125
theoretical_mean = 1.0 / lambda_rate
print(f"\nTheoretical (lambda={lambda_rate}):")
print(f"Mean inter-arrival time: {theoretical_mean:.4f}")
print(f"Theoretical CV for exponential: 1.0")

# =============================================================================
# CDF COMPARISON
# =============================================================================

fig, ax = plt.subplots(figsize=(16, 10))

# Empirical CDF
sorted_data = np.sort(all_inter_arrival_times)
empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

ax.plot(sorted_data, empirical_cdf, linewidth=2.5, color='#2E86C1', alpha=0.9,
        label=f'Empirical CDF (Mean={np.mean(all_inter_arrival_times):.2f}, CV={np.std(all_inter_arrival_times)/np.mean(all_inter_arrival_times):.2f})')

# Theoretical exponential CDF: F(x) = 1 - exp(-lambda * x)
x_theoretical = np.linspace(0, max(sorted_data), 1000)
theoretical_cdf = 1 - np.exp(-lambda_rate * x_theoretical)

ax.plot(x_theoretical, theoretical_cdf, linewidth=2.5, color='#E74C3C', alpha=0.9, linestyle='--',
        label=f'Theoretical Exponential CDF (λ={lambda_rate}, Mean={theoretical_mean:.2f}, CV=1.0)')

# Manual Kolmogorov-Smirnov test (max difference between CDFs)
# For each empirical data point, find the theoretical CDF value
theoretical_cdf_at_data = 1 - np.exp(-lambda_rate * sorted_data)
ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf_at_data))

print(f"\nKolmogorov-Smirnov Test:")
print(f"KS Statistic: {ks_statistic:.4f}")

# Add KS test result to plot
textstr = f'Kolmogorov-Smirnov Test:\nKS Statistic = {ks_statistic:.4f}'
ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Inter-Arrival Time', fontsize=16, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=16, fontweight='bold')
ax.set_title('Cumulative Distribution Function of Job Inter-Arrival Times\n(All Episodes Combined)', 
             fontsize=18, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='center right', fontsize=14, framealpha=0.95)

ax.tick_params(axis='both', labelsize=14)
ax.set_xlim(0, np.percentile(sorted_data, 99))  # Show up to 99th percentile for clarity
ax.set_ylim(0, 1.0)

plt.tight_layout()
output_cdf = OUTPUT_DIR / 'job_arrival_cdf_comparison.png'
plt.savefig(output_cdf, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved CDF plot: {output_cdf}")
plt.close()

# =============================================================================
# HISTOGRAM WITH EXPONENTIAL PDF OVERLAY
# =============================================================================

fig, ax = plt.subplots(figsize=(16, 10))

# Histogram of inter-arrival times
n, bins, patches = ax.hist(all_inter_arrival_times, bins=50, density=True, alpha=0.7, 
                            color='#2E86C1', edgecolor='black', linewidth=0.5,
                            label=f'Empirical Distribution (n={len(all_inter_arrival_times)})')

# Theoretical exponential PDF: f(x) = lambda * exp(-lambda * x)
x_pdf = np.linspace(0, max(bins), 1000)
theoretical_pdf = lambda_rate * np.exp(-lambda_rate * x_pdf)

ax.plot(x_pdf, theoretical_pdf, linewidth=3.0, color='#E74C3C', alpha=0.9, linestyle='--',
        label=f'Theoretical Exponential PDF (λ={lambda_rate})')

ax.set_xlabel('Inter-Arrival Time', fontsize=16, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=16, fontweight='bold')
ax.set_title('Probability Distribution of Job Inter-Arrival Times\n(Histogram with Theoretical Exponential Overlay)', 
             fontsize=18, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='upper right', fontsize=14, framealpha=0.95)

ax.tick_params(axis='both', labelsize=14)
ax.set_xlim(0, np.percentile(all_inter_arrival_times, 99))

plt.tight_layout()
output_hist = OUTPUT_DIR / 'job_arrival_histogram_comparison.png'
plt.savefig(output_hist, dpi=150, bbox_inches='tight')
print(f"✓ Saved histogram plot: {output_hist}")
plt.close()

print("\n✓ Stochasticity analysis completed!")
print(f"  - Empirical CV = {np.std(all_inter_arrival_times)/np.mean(all_inter_arrival_times):.4f}")
print(f"  - Theoretical CV = 1.0 (perfect exponential)")
print(f"  - KS Statistic = {ks_statistic:.4f}")
if ks_statistic > 0.1:
    print("  - Result: Considerably different from exponential (KS > 0.1)")
else:
    print("  - Result: Close to exponential distribution (KS <= 0.1)")
