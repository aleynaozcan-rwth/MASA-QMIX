#!/usr/bin/env python3
"""
Scientific visualization of inter-arrival time intervals for proposed framework
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Creating Inter-Arrival Time Analysis Plots...")

# Job arrivals from Episode 10 (all arrivals including t=0)
arrivals = [
    {'time': 0.00, 'jobs': ['Job_0', 'Job_1', 'Job_2', 'Job_3']},  # Reference: Job_3
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

# Calculate statistics
mean_interval = np.mean(intervals)
median_interval = np.median(intervals)
std_interval = np.std(intervals)
min_interval = np.min(intervals)
max_interval = np.max(intervals)

print(f"Mean: {mean_interval:.2f}, Median: {median_interval:.2f}, Std: {std_interval:.2f}")

# Create figure with 4 subplots
fig = plt.figure(figsize=(20, 12))

# Subplot 1: Histogram with KDE
ax1 = plt.subplot(2, 2, 1)
n, bins, patches = ax1.hist(intervals, bins=8, density=True, alpha=0.7, 
                             color='#7B1FA2', edgecolor='black', linewidth=1.5)

# Add simple KDE (manual implementation)
x_range = np.linspace(min_interval - 0.5, max_interval + 0.5, 100)
kde_values = np.zeros_like(x_range)
bandwidth = std_interval * 0.5
for interval in intervals:
    kde_values += np.exp(-0.5 * ((x_range - interval) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
kde_values /= len(intervals)

ax1.plot(x_range, kde_values, '-', linewidth=2.5, label='KDE', color='#E65100')

# Add mean line
ax1.axvline(mean_interval, color='#27AE60', linestyle='--', linewidth=2.5, 
            label=f'Mean = {mean_interval:.2f}')
ax1.axvline(median_interval, color='#2980B9', linestyle='-.', linewidth=2.5, 
            label=f'Median = {median_interval:.2f}')

ax1.set_xlabel('Inter-Arrival Time (time units)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
ax1.set_title('Distribution of Inter-Arrival Times', fontsize=16, fontweight='bold', pad=15)
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax1.tick_params(labelsize=12)

# Subplot 2: Time Series of Intervals
ax2 = plt.subplot(2, 2, 2)
interval_indices = np.arange(1, len(intervals) + 1)
ax2.plot(interval_indices, intervals, marker='o', markersize=8, linewidth=2.5, 
         color='#7B1FA2', alpha=0.8, markerfacecolor='#7B1FA2', markeredgecolor='black', markeredgewidth=1)
ax2.axhline(mean_interval, color='#27AE60', linestyle='--', linewidth=2, 
            label=f'Mean = {mean_interval:.2f}', alpha=0.7)

# Add shaded region for ±1 std
ax2.fill_between(interval_indices, mean_interval - std_interval, 
                  mean_interval + std_interval, alpha=0.2, color='#27AE60', 
                  label=f'Mean ± 1σ')

ax2.set_xlabel('Interval Index', fontsize=14, fontweight='bold')
ax2.set_ylabel('Inter-Arrival Time (time units)', fontsize=14, fontweight='bold')
ax2.set_title('Inter-Arrival Times Over Sequence', fontsize=16, fontweight='bold', pad=15)
ax2.legend(fontsize=12, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax2.tick_params(labelsize=12)
ax2.set_xticks(interval_indices)

# Set detailed y-axis ticks to show precise decimal values (e.g., 1.00, 1.25, 1.50, etc.)
y_min = np.floor(min_interval * 4) / 4  # Round down to nearest 0.25
y_max = np.ceil(max_interval * 4) / 4   # Round up to nearest 0.25
y_ticks = np.arange(y_min, y_max + 0.25, 0.25)
ax2.set_yticks(y_ticks)
ax2.set_yticklabels([f'{y:.2f}' for y in y_ticks])

# Subplot 3: Cumulative Distribution Function (CDF)
ax3 = plt.subplot(2, 2, 3)
sorted_intervals = np.sort(intervals)
cdf = np.arange(1, len(sorted_intervals) + 1) / len(sorted_intervals)
ax3.plot(sorted_intervals, cdf, marker='o', markersize=8, linewidth=2.5, 
         color='#7B1FA2', alpha=0.8, markerfacecolor='#7B1FA2', 
         markeredgecolor='black', markeredgewidth=1, label='Empirical CDF')

# Add exponential CDF for comparison (Poisson process assumption)
lambda_param = 1.0 / mean_interval
x_exp = np.linspace(0, max_interval + 0.5, 100)
exp_cdf = 1 - np.exp(-lambda_param * x_exp)
ax3.plot(x_exp, exp_cdf, '--', linewidth=2.5, color='#E65100', 
         label=f'Exponential (λ={lambda_param:.3f})', alpha=0.8)

ax3.set_xlabel('Inter-Arrival Time (time units)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
ax3.set_title('Cumulative Distribution Function (CDF)', fontsize=16, fontweight='bold', pad=15)
ax3.legend(fontsize=12, loc='lower right')
ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax3.tick_params(labelsize=12)
ax3.set_ylim(-0.05, 1.05)

# Subplot 4: Box Plot with Statistics
ax4 = plt.subplot(2, 2, 4)
bp = ax4.boxplot(intervals, vert=True, patch_artist=True, widths=0.5,
                  boxprops=dict(facecolor='#7B1FA2', alpha=0.7, linewidth=2),
                  whiskerprops=dict(linewidth=2, color='#7B1FA2'),
                  capprops=dict(linewidth=2, color='#7B1FA2'),
                  medianprops=dict(linewidth=2.5, color='#E65100'),
                  flierprops=dict(marker='o', markerfacecolor='#E65100', markersize=8, 
                                 markeredgecolor='black', markeredgewidth=1))

# Add individual points
y_data = intervals
x_data = np.random.normal(1, 0.04, size=len(y_data))
ax4.scatter(x_data, y_data, alpha=0.5, color='#2980B9', s=80, edgecolors='black', linewidths=1, zorder=3)

# Add statistics text box
stats_text = f'''Statistics:
Mean: {mean_interval:.2f}
Median: {median_interval:.2f}
Std Dev: {std_interval:.2f}
Min: {min_interval:.2f}
Max: {max_interval:.2f}
Range: {max_interval - min_interval:.2f}
CV: {(std_interval/mean_interval)*100:.1f}%'''

ax4.text(1.35, np.mean(intervals), stats_text, fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='#7B1FA2', 
                  linewidth=2, alpha=0.9),
         verticalalignment='center')

ax4.set_ylabel('Inter-Arrival Time (time units)', fontsize=14, fontweight='bold')
ax4.set_title('Box Plot with Statistical Summary', fontsize=16, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, axis='y')
ax4.tick_params(labelsize=12)
ax4.set_xticks([])
ax4.set_xlim(0.5, 2)

# Main title
fig.suptitle('Inter-Arrival Time Analysis\n(Proposed Framework - Episode 10, Reference: Job_3 at t=0)', 
             fontsize=20, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = OUTPUT_DIR / 'arrival_interval_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved inter-arrival analysis plot: {output_path}")
plt.close()

print("\nPlot complete!")
print(f"Generated 4-panel scientific visualization:")
print(f"  1. Histogram with KDE and mean/median lines")
print(f"  2. Time series showing interval variability")
print(f"  3. Empirical CDF vs theoretical exponential")
print(f"  4. Box plot with statistical summary")
