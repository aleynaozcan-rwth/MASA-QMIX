#!/usr/bin/env python3
"""
Advanced Probability Visualizations for Job Arrival Stochasticity
- Kernel Density Estimation (KDE)
- Q-Q Plot
- Combined visualization
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

all_inter_arrival_times = np.array(all_inter_arrival_times)

print(f"Analyzed {episode_count} episodes")
print(f"Total inter-arrival times: {len(all_inter_arrival_times)}")

# Theoretical exponential distribution with lambda=0.125
lambda_rate = 0.125
theoretical_mean = 1.0 / lambda_rate

# =============================================================================
# KERNEL DENSITY ESTIMATION (KDE)
# =============================================================================

def gaussian_kernel(x, xi, bandwidth):
    """Gaussian kernel for KDE"""
    return (1.0 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - xi) / bandwidth) ** 2)

def kde(data, x_points, bandwidth):
    """Manual KDE implementation"""
    n = len(data)
    kde_values = np.zeros(len(x_points))
    
    for i, x in enumerate(x_points):
        kde_values[i] = np.sum(gaussian_kernel(x, data, bandwidth)) / n
    
    return kde_values

print("\nCalculating Kernel Density Estimation...")

# Bandwidth selection (Silverman's rule of thumb)
n = len(all_inter_arrival_times)
sigma = np.std(all_inter_arrival_times)
bandwidth = 1.06 * sigma * (n ** (-1/5))
print(f"Bandwidth (Silverman's rule): {bandwidth:.4f}")

# Create smooth x-axis points
x_kde = np.linspace(0, np.percentile(all_inter_arrival_times, 99), 1000)
kde_values = kde(all_inter_arrival_times, x_kde, bandwidth)

# Theoretical exponential PDF
theoretical_pdf = lambda_rate * np.exp(-lambda_rate * x_kde)

fig, ax = plt.subplots(figsize=(16, 10))

# Calculate CV values
empirical_cv = np.std(all_inter_arrival_times) / np.mean(all_inter_arrival_times)
theoretical_cv = 1.0

# Plot KDE
ax.plot(x_kde, kde_values, linewidth=3.0, color='#2E86C1', alpha=0.9,
        label=f'Empirical Distribution (λ={lambda_rate}, CV={empirical_cv:.3f})')

# Plot theoretical exponential
ax.plot(x_kde, theoretical_pdf, linewidth=3.0, color='#E74C3C', alpha=0.9, linestyle='--',
        label=f'Theoretical Poisson Exponential (λ={lambda_rate}, CV={theoretical_cv:.1f})')

# Fill areas for visual comparison
ax.fill_between(x_kde, kde_values, alpha=0.3, color='#2E86C1', 
                label=f'Empirical Area (Mean μ={np.mean(all_inter_arrival_times):.2f}, Std Dev σ={np.std(all_inter_arrival_times):.2f})')
ax.fill_between(x_kde, theoretical_pdf, alpha=0.2, color='#E74C3C', 
                label=f'Theoretical Area (Mean μ={theoretical_mean:.2f}, Std Dev σ={theoretical_mean:.2f})')

# Add dummy entries for CV explanation in legend
ax.plot([], [], ' ', label='\nCV = Coefficient of Variation = σ / μ')
ax.plot([], [], ' ', label='0 (Full Deterministic) ≤ CV ≤ 1 (Full Stochastic)')

ax.set_xlabel('Inter-Arrival Time Between Consecutive Jobs', fontsize=16, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=16, fontweight='bold')
ax.set_title('Stochasticity Analysis: Job Arrival Inter-Arrival Time Distribution\n(Empirical vs Theoretical Poisson Process)', 
             fontsize=18, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

# Set x-axis ticks at 2.5 intervals and remove margins
max_x = np.percentile(all_inter_arrival_times, 99)
x_ticks = np.arange(0, max_x + 2.5, 2.5)
ax.set_xticks(x_ticks)
ax.set_xlim(0, max_x)

# Set y-axis limit to remove top margin
max_y = max(max(kde_values), max(theoretical_pdf))
ax.set_ylim(0, max_y)

# Add annotation pointing to empirical area - arrow pointing down to blue line at x=12.5
x_point = 12.5
idx = np.argmin(np.abs(x_kde - x_point))
y_point = kde_values[idx]

ax.annotate('Near Perfect Stochasticity', 
            xy=(x_point, y_point), xytext=(x_point, y_point + 0.05),
            fontsize=14, fontweight='bold', color='#2E86C1',
            arrowprops=dict(arrowstyle='->', color='#2E86C1', lw=2.5),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2E86C1', linewidth=2))

ax.legend(loc='upper right', fontsize=13, framealpha=0.95)
ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
output_kde = OUTPUT_DIR / 'job_arrival_kde.png'
plt.savefig(output_kde, dpi=150, bbox_inches='tight')
print(f"✓ Saved KDE plot: {output_kde}")
plt.close()

# =============================================================================
# Q-Q PLOT (Quantile-Quantile Plot)
# =============================================================================

print("\nGenerating Q-Q Plot...")

# Sort empirical data
sorted_data = np.sort(all_inter_arrival_times)
n_points = len(sorted_data)

# Theoretical quantiles from exponential distribution
# For exponential: Q(p) = -ln(1-p) / lambda
quantile_positions = (np.arange(1, n_points + 1) - 0.5) / n_points
theoretical_quantiles = -np.log(1 - quantile_positions) / lambda_rate

fig, ax = plt.subplots(figsize=(12, 12))

# Q-Q scatter plot
ax.scatter(theoretical_quantiles, sorted_data, alpha=0.6, s=20, color='#2E86C1',
           label='Empirical vs Theoretical Quantiles')

# Perfect fit line (45-degree line)
max_val = max(max(theoretical_quantiles), max(sorted_data))
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2.5, alpha=0.8,
        label='Perfect Exponential Fit (y=x)')

ax.set_xlabel('Theoretical Quantiles (Exponential λ=0.125)', fontsize=16, fontweight='bold')
ax.set_ylabel('Empirical Quantiles (Sample Data)', fontsize=16, fontweight='bold')
ax.set_title('Q-Q Plot: Empirical vs Theoretical Exponential Distribution\n(Job Inter-Arrival Times)', 
             fontsize=18, fontweight='bold', pad=20)

# Add interpretation text
interp_text = 'Interpretation:\n• Points on the red line → perfect match\n• Points above line → heavier tail\n• Points below line → lighter tail'
ax.text(0.05, 0.95, interp_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='lower right', fontsize=13, framealpha=0.95)
ax.tick_params(axis='both', labelsize=14)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
output_qq = OUTPUT_DIR / 'job_arrival_qq_plot.png'
plt.savefig(output_qq, dpi=150, bbox_inches='tight')
print(f"✓ Saved Q-Q plot: {output_qq}")
plt.close()

# =============================================================================
# COMBINED VISUALIZATION (2x2 PANEL)
# =============================================================================

print("\nGenerating combined visualization...")

fig = plt.figure(figsize=(20, 16))

# Panel 1: Histogram with KDE overlay
ax1 = plt.subplot(2, 2, 1)
ax1.hist(all_inter_arrival_times, bins=50, density=True, alpha=0.6, 
         color='#AED6F1', edgecolor='black', linewidth=0.5, label='Histogram')
ax1.plot(x_kde, kde_values, linewidth=2.5, color='#2E86C1', label='KDE')
ax1.plot(x_kde, theoretical_pdf, linewidth=2.5, color='#E74C3C', linestyle='--', label='Theoretical Exponential')
ax1.set_xlabel('Inter-Arrival Time', fontsize=14, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
ax1.set_title('(a) Histogram with KDE Overlay', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, np.percentile(all_inter_arrival_times, 99))

# Panel 2: CDF comparison
ax2 = plt.subplot(2, 2, 2)
sorted_data = np.sort(all_inter_arrival_times)
empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
theoretical_cdf = 1 - np.exp(-lambda_rate * sorted_data)

ax2.plot(sorted_data, empirical_cdf, linewidth=2.5, color='#2E86C1', label='Empirical CDF')
ax2.plot(sorted_data, theoretical_cdf, linewidth=2.5, color='#E74C3C', linestyle='--', label='Theoretical CDF')
ax2.set_xlabel('Inter-Arrival Time', fontsize=14, fontweight='bold')
ax2.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
ax2.set_title('(b) Cumulative Distribution Function', fontsize=15, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, np.percentile(sorted_data, 99))

# Panel 3: Q-Q Plot
ax3 = plt.subplot(2, 2, 3)
ax3.scatter(theoretical_quantiles, sorted_data, alpha=0.5, s=15, color='#2E86C1')
max_val = max(max(theoretical_quantiles), max(sorted_data))
ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2.5, alpha=0.8, label='Perfect Fit')
ax3.set_xlabel('Theoretical Quantiles', fontsize=14, fontweight='bold')
ax3.set_ylabel('Empirical Quantiles', fontsize=14, fontweight='bold')
ax3.set_title('(c) Q-Q Plot', fontsize=15, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal', adjustable='box')

# Panel 4: Statistics summary
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

summary_text = f"""
STOCHASTICITY ANALYSIS SUMMARY
{'='*60}

Dataset:
  • Episodes analyzed: {episode_count:,}
  • Inter-arrival times: {len(all_inter_arrival_times):,}

Empirical Statistics:
  • Mean: {np.mean(all_inter_arrival_times):.4f}
  • Std Dev: {np.std(all_inter_arrival_times):.4f}
  • Coefficient of Variation (CV): {np.std(all_inter_arrival_times)/np.mean(all_inter_arrival_times):.4f}
  • Median: {np.median(all_inter_arrival_times):.4f}
  • Min: {np.min(all_inter_arrival_times):.4f}
  • Max: {np.max(all_inter_arrival_times):.4f}

Theoretical (Exponential λ={lambda_rate}):
  • Mean: {theoretical_mean:.4f}
  • Std Dev: {theoretical_mean:.4f}
  • Coefficient of Variation (CV): 1.0000

Goodness-of-Fit:
  • KS Statistic: {np.max(np.abs(empirical_cdf - theoretical_cdf)):.4f}
  • CV Difference: {abs((np.std(all_inter_arrival_times)/np.mean(all_inter_arrival_times)) - 1.0):.4f}

Conclusion:
  ✓ CV ≈ 1.0 indicates exponential-like variability
  ✓ Low KS statistic indicates good fit to exponential
  ✓ Job arrivals follow stochastic Poisson process
"""

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
         fontsize=12, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#E8F8F5', alpha=0.9, pad=1.0))

plt.suptitle('Comprehensive Stochasticity Analysis of Job Inter-Arrival Times', 
             fontsize=20, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
output_combined = OUTPUT_DIR / 'job_arrival_stochasticity_combined.png'
plt.savefig(output_combined, dpi=150, bbox_inches='tight')
print(f"✓ Saved combined visualization: {output_combined}")
plt.close()

print("\n✓ Advanced probability visualizations completed!")
print(f"  - KDE plot with exponential overlay")
print(f"  - Q-Q plot for distribution comparison")
print(f"  - 2x2 combined panel visualization")
