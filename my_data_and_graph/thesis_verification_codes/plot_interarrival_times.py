#!/usr/bin/env python3
"""
Job Interarrival Times Visualization
Shows the time intervals between consecutive job arrivals
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/thesis_verification_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Job arrival times from Episode 10
# These are the timestamps when new jobs arrived in the system
job_arrivals = [
    0.00,   # Job_0, Job_1, Job_2, Job_3 (4 jobs at start)
    2.05,   # Job_4
    4.54,   # Job_5
    6.15,   # Job_6
    8.38,   # Job_7
    9.93,   # Job_8
    10.97,  # Job_9
    12.68,  # Job_10
    15.97,  # Job_11
    17.40,  # Job_12
    19.75,  # Job_13
    21.80,  # Job_14
    23.40,  # Job_15
]

print("="*70)
print("JOB INTERARRIVAL TIMES ANALYSIS (Episode 10)")
print("="*70)

# Calculate interarrival intervals (time between consecutive arrivals)
interarrival_times = []
for i in range(1, len(job_arrivals)):
    interval = job_arrivals[i] - job_arrivals[i-1]
    interarrival_times.append(interval)

print(f"\nTotal job arrival events: {len(job_arrivals)}")
print(f"Total interarrival intervals: {len(interarrival_times)}")

print("\n--- Interarrival Intervals ---")
for i in range(len(interarrival_times)):
    print(f"Interval {i+1}: t={job_arrivals[i]:.2f} to t={job_arrivals[i+1]:.2f} → Δt = {interarrival_times[i]:.2f}")

print(f"\n--- Statistics ---")
print(f"Mean interarrival time: {np.mean(interarrival_times):.2f} time units")
print(f"Median interarrival time: {np.median(interarrival_times):.2f} time units")
print(f"Std deviation: {np.std(interarrival_times):.2f} time units")
print(f"Min interarrival: {np.min(interarrival_times):.2f} time units")
print(f"Max interarrival: {np.max(interarrival_times):.2f} time units")

# Create figure
fig, ax = plt.subplots(figsize=(18, 8))

# X-axis: continuous time (midpoint of each interval)
# Y-axis: interarrival time (duration of that interval)
x_positions = []
for i in range(len(interarrival_times)):
    midpoint = (job_arrivals[i] + job_arrivals[i+1]) / 2
    x_positions.append(midpoint)

# Plot interarrival times as a step function
# Each bar represents the time gap until the next arrival
for i in range(len(interarrival_times)):
    start_time = job_arrivals[i]
    end_time = job_arrivals[i+1]
    interval = interarrival_times[i]
    
    # Draw horizontal line from start to end at height = interval
    ax.hlines(y=interval, xmin=start_time, xmax=end_time, 
             colors='#2E86C1', linewidth=3, alpha=0.8)
    
    # Draw vertical lines at start and end
    if i == 0:
        ax.vlines(x=start_time, ymin=0, ymax=interval, 
                 colors='#2E86C1', linewidth=2, alpha=0.5, linestyles='dashed')
    ax.vlines(x=end_time, ymin=0, ymax=interval, 
             colors='#2E86C1', linewidth=2, alpha=0.5, linestyles='dashed')

# Add scatter points at midpoints for clarity
ax.scatter(x_positions, interarrival_times, 
          color='#E74C3C', s=100, zorder=5, 
          edgecolors='white', linewidths=2, alpha=0.9)

# Add value labels on significant points
for i in range(len(interarrival_times)):
    # Annotate every interval
    if i < len(x_positions):
        ax.annotate(f'{interarrival_times[i]:.2f}', 
                   xy=(x_positions[i], interarrival_times[i]), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   fontsize=9, 
                   ha='center', 
                   va='bottom',
                   color='#E74C3C', 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#E74C3C', alpha=0.9, linewidth=1))

# Add mean line
mean_interarrival = np.mean(interarrival_times)
std_interarrival = np.std(interarrival_times)
ax.axhline(y=mean_interarrival, color='green', linestyle='--', 
          linewidth=2, alpha=0.7, 
          label=f'Observed Mean Δt = {mean_interarrival:.2f}')

# Add theoretical Poisson process line (exponential interarrival)
# For Poisson process: λ = 1/μ where μ = mean interarrival time
# Exponential distribution: E[Δt] = μ, Var[Δt] = μ²
theoretical_mean = mean_interarrival
theoretical_std = mean_interarrival  # For exponential distribution, std = mean
ax.axhline(y=theoretical_mean, color='purple', linestyle=':', 
          linewidth=2.5, alpha=0.6, 
          label=f'Poisson Process (λ={1/mean_interarrival:.3f}, E[Δt]={theoretical_mean:.2f})')

# Styling
ax.set_xlabel('Continuous Time (Time Units)', fontsize=16, fontweight='heavy')
ax.set_ylabel('Interarrival Time (Δt)', fontsize=16, fontweight='heavy')
ax.set_title('Job Interarrival Times Over Episode Duration\n(Episode 10 - Poisson Process Validation)', 
            fontsize=16, fontweight='bold', pad=20)

ax.set_xlim(-0.5, 25)
ax.set_ylim(0, max(interarrival_times) * 1.2)

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

# Enhanced legend with Poisson comparison
legend_labels = [
    f'Observed Interarrivals (n={len(interarrival_times)})',
    f'Observed Mean: μ = {mean_interarrival:.2f}, σ = {std_interarrival:.2f}',
    f'Poisson Theory: λ = {1/mean_interarrival:.3f}, E[Δt] = {theoretical_mean:.2f}, σ_theory = {theoretical_std:.2f}'
]

# Create custom legend
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='#2E86C1', linewidth=3, alpha=0.8),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2, alpha=0.7),
    Line2D([0], [0], color='purple', linestyle=':', linewidth=2.5, alpha=0.6)
]

ax.legend(custom_lines, legend_labels, fontsize=11, loc='upper right', 
         framealpha=0.95, fancybox=True, shadow=True)

plt.tight_layout()

# Save figure
output_path = OUTPUT_DIR / 'job_interarrival_times_episode_10.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved interarrival times plot: {output_path}")
print("="*70)

plt.close()
