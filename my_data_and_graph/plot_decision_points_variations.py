#!/usr/bin/env python3
"""
Plot decision points per episode - Multiple Visualization Styles
Shows the same data in different visual formats
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d

# Use Agg backend for cluster without display
import matplotlib
matplotlib.use('Agg')

# ==================== Configuration ====================
TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Parse Timeline File ====================
print("Parsing scheduling timeline...")

episodes = []  # List of {episode: int, dp_count: int, arrivals: int, completions: int}
current_episode = None
current_dp_count = 0
current_arrivals = 0
current_completions = 0

with open(TIMELINE_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Match episode start: "=== EPISODE 0 ==="
        episode_match = re.match(r'^=== EPISODE (\d+) ===$', line)
        if episode_match:
            # Save previous episode if exists
            if current_episode is not None:
                episodes.append({
                    'episode': current_episode,
                    'dp_count': current_dp_count,
                    'arrivals': current_arrivals,
                    'completions': current_completions
                })
            
            # Start new episode
            current_episode = int(episode_match.group(1))
            current_dp_count = 0
            current_arrivals = 0
            current_completions = 0
            continue
        
        # Match decision point events:
        # 1. Job arrivals: "[t=X.XX] New job ... arrived"
        if '[t=' in line and 'New job' in line and 'arrived' in line:
            current_dp_count += 1
            current_arrivals += 1
            continue
        
        # 2. Operation completions: "[t=X.XX] ... finished ... next queued"
        if '[t=' in line and 'finished' in line and 'next queued' in line:
            current_dp_count += 1
            current_completions += 1
            continue

# Save last episode
if current_episode is not None:
    episodes.append({
        'episode': current_episode,
        'dp_count': current_dp_count,
        'arrivals': current_arrivals,
        'completions': current_completions
    })

print(f"Parsed {len(episodes)} episodes")

# ==================== Extract Data ====================
episode_numbers = np.array([ep['episode'] for ep in episodes])
dp_counts = np.array([ep['dp_count'] for ep in episodes])
arrivals = np.array([ep['arrivals'] for ep in episodes])
completions = np.array([ep['completions'] for ep in episodes])

# Calculate statistics
mean_dp = np.mean(dp_counts)
std_dp = np.std(dp_counts)
min_dp = np.min(dp_counts)
max_dp = np.max(dp_counts)
total_dp = np.sum(dp_counts)

print(f"\nDecision Points Statistics:")
print(f"  Total DPs: {total_dp:,}")
print(f"  Mean DPs per episode: {mean_dp:.2f}")
print(f"  Std Dev: {std_dp:.2f}")
print(f"  Range: {min_dp} - {max_dp}")

# ==================== Create Multiple Plots ====================

# ------------------------- Version 1: Signal/Line Plot with MA (Stem Style) -------------------------
fig1, ax1 = plt.subplots(figsize=(16, 7))

# Vertical lines (stem style) - thicker and denser for more visible appearance
ax1.vlines(episode_numbers, mean_dp, dp_counts, colors='lightsteelblue', linewidth=2.5, alpha=0.8)

# Moving average (smoothed) - using Gaussian filter for smoother appearance
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D

window_size = 5  # 5-episode window
sigma = 2.0  # Gaussian smoothing parameter
if len(dp_counts) >= window_size:
    dp_smoothed = gaussian_filter1d(dp_counts, sigma=sigma)
    ax1.plot(episode_numbers, dp_smoothed, linewidth=2.5, color='red', alpha=0.95, zorder=5)

# Add polynomial trend line (degree 4 for more dramatic curve)
# z = np.polyfit(episode_numbers, dp_counts, 4)
# p = np.poly1d(z)
# ax1.plot(episode_numbers, p(episode_numbers), linewidth=3.5, color='purple', 
#          linestyle='-.', alpha=0.9, zorder=6)

# Add mean line - green, thick, and on top
ax1.axhline(y=mean_dp, color='green', linestyle='--', linewidth=3.5, alpha=0.95, zorder=10)

# Create custom legend with proper line alignment
legend_line = Line2D([0], [0], color='red', linewidth=2.5, label=f'{window_size}-ep MA (Gaussian)')
# legend_trend = Line2D([0], [0], color='purple', linewidth=3.5, linestyle='-.', label='Polynomial Trend')
legend_mean = Line2D([0], [0], color='green', linewidth=3.5, linestyle='--', label=f'Mean: {mean_dp:.2f}')
legend_text = Line2D([0], [0], color='none', label=f'Std: {std_dp:.2f}')
legend_total_dps = Line2D([0], [0], color='none', label=f'Total DPs: {total_dp:,}')
legend_total_episodes = Line2D([0], [0], color='none', label=f'Total Episodes: {len(episodes):,}')
ax1.legend(handles=[legend_line, legend_mean, legend_text, legend_total_dps, legend_total_episodes], 
           loc='upper right', fontsize=10, framealpha=0.9)

ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax1.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax1.set_title('Decision Point Density Evolution', 
              fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, axis='y')

# Set y-axis to show only data range for dramatic variation display
ax1.set_ylim(36, 52)

# Increase spacing between y-axis tick labels
from matplotlib.ticker import MaxNLocator
ax1.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

# Set x-axis to start from episode 0 without extra space
ax1.set_xlim(0, len(episode_numbers) - 1)

plt.tight_layout()

output_path1 = OUTPUT_DIR / 'dp_per_episode_signal.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved signal view: {output_path1}")
plt.close()

# ------------------------- Version 1b: Signal + Breakdown (Arrivals & Completions) -------------------------
fig1b, ax1b = plt.subplots(figsize=(16, 7))

# Vertical lines (stem style) for total DPs
ax1b.vlines(episode_numbers, mean_dp, dp_counts, colors='lightsteelblue', linewidth=2.5, alpha=0.8)

# Smooth arrivals and completions
arrivals_smooth = gaussian_filter1d(arrivals, sigma=2.0)
completions_smooth = gaussian_filter1d(completions, sigma=2.0)

# Plot arrivals and completions as separate lines
ax1b.plot(episode_numbers, arrivals_smooth, linewidth=3, color='#FF6B6B', 
         alpha=0.9, label='Job Arrivals (5-ep MA)', zorder=6)
ax1b.plot(episode_numbers, completions_smooth, linewidth=3, color='#4ECDC4', 
         alpha=0.9, label='Operation Completions (5-ep MA)', zorder=6)

# Moving average for total (same as before)
window_size = 5
sigma = 2.0
if len(dp_counts) >= window_size:
    dp_smoothed = gaussian_filter1d(dp_counts, sigma=sigma)
    ax1b.plot(episode_numbers, dp_smoothed, linewidth=2.5, color='purple', 
             alpha=0.8, linestyle='--', zorder=5, label='Total DPs (5-ep MA)')

# Add mean line for total DPs
ax1b.axhline(y=mean_dp, color='green', linestyle='--', linewidth=3.5, alpha=0.95, zorder=10)

# Create legend
from matplotlib.lines import Line2D
legend_arrivals = Line2D([0], [0], color='#FF6B6B', linewidth=3, label='Job Arrivals MA')
legend_completions = Line2D([0], [0], color='#4ECDC4', linewidth=3, label='Operation Completions MA')
legend_total = Line2D([0], [0], color='purple', linewidth=2.5, linestyle='--', label='Total DPs MA')
legend_mean = Line2D([0], [0], color='green', linewidth=3.5, linestyle='--', label=f'Mean: {mean_dp:.2f}')
legend_info = Line2D([0], [0], color='none', label=f'Total: {np.sum(arrivals):,} arrivals + {np.sum(completions):,} completions')

ax1b.legend(handles=[legend_arrivals, legend_completions, legend_total, legend_mean, legend_info], 
           loc='upper right', fontsize=10, framealpha=0.9)

ax1b.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax1b.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax1b.set_title('Decision Point Breakdown: Arrivals vs Completions Over Episodes', 
              fontsize=14, fontweight='bold', pad=15)
ax1b.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, axis='y')

# Set y-axis range
ax1b.set_ylim(10, 52)
ax1b.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
ax1b.set_xlim(0, len(episode_numbers) - 1)

plt.tight_layout()

output_path1b = OUTPUT_DIR / 'dp_breakdown_with_signal.png'
plt.savefig(output_path1b, dpi=150, bbox_inches='tight')
print(f"✓ Saved signal+breakdown view: {output_path1b}")
plt.close()

# ------------------------- Version 2: Scatter Plot with Polynomial Trend -------------------------
fig2, ax2 = plt.subplots(figsize=(16, 7))

# Scatter plot with color mapping (gradient from blue to red based on DP count)
scatter = ax2.scatter(episode_numbers, dp_counts, c=dp_counts, cmap='coolwarm', 
                     s=50, alpha=0.6, edgecolors='none')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax2, pad=0.01)
cbar.set_label('Decision Points', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

# Add polynomial trend line (degree 4 for more dramatic curve)
z = np.polyfit(episode_numbers, dp_counts, 4)
p = np.poly1d(z)
ax2.plot(episode_numbers, p(episode_numbers), linewidth=3, color='darkred', 
         linestyle='--', alpha=0.8, zorder=5, label=f'Trend (poly 4)')

# Add mean line
ax2.axhline(y=mean_dp, color='green', linestyle='--', linewidth=3, 
            alpha=0.8, zorder=4, label=f'Mean: {mean_dp:.2f}')

# Add total DPs and episodes info to legend
from matplotlib.patches import Patch
legend_info_dps = Patch(facecolor='none', edgecolor='none', label=f'Total DPs: {total_dp:,}')
legend_info_episodes = Patch(facecolor='none', edgecolor='none', label=f'Total Episodes: {len(episodes):,}')

ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax2.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax2.set_title('Decision Points Scatter Plot with Polynomial Trend', 
              fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

# Add legend with handles
handles, labels = ax2.get_legend_handles_labels()
handles.extend([legend_info_dps, legend_info_episodes])
ax2.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)

# Set y-axis to show only data range
ax2.set_ylim(36, 52)
ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

# Set x-axis to start from episode 0
ax2.set_xlim(0, len(episode_numbers) - 1)

plt.tight_layout()

output_path2 = OUTPUT_DIR / 'dp_per_episode_scatter.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Saved scatter view: {output_path2}")
plt.close()

# ------------------------- Version 3: Histogram - DP Distribution -------------------------
fig3, ax3 = plt.subplots(figsize=(12, 5))

# Calculate median
median_dp = np.median(dp_counts)

# Create histogram
bins = range(int(min_dp), int(max_dp) + 2)  # +2 to include the max value
ax3.hist(dp_counts, bins=bins, color='cornflowerblue', edgecolor='black', 
         alpha=0.7, linewidth=1.2)

# Add mean and median lines
ax3.axvline(x=mean_dp, color='green', linestyle='--', linewidth=2.5, 
            alpha=0.8, label=f'Mean: {mean_dp:.1f}')
ax3.axvline(x=median_dp, color='red', linestyle=':', linewidth=3.5, 
            alpha=0.8, label=f'Median: {median_dp:.0f}')

# Add total DPs and episodes info to legend
from matplotlib.patches import Patch
legend_info_dps = Patch(facecolor='none', edgecolor='none', label=f'Total DPs: {total_dp:,}')
legend_info_episodes = Patch(facecolor='none', edgecolor='none', label=f'Total Episodes: {len(episodes):,}')

ax3.set_xlabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
# ax3.set_title('DP Distribution: Shows Execution Variability', 
#               fontsize=14, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

# Add legend with handles
handles, labels = ax3.get_legend_handles_labels()
handles.extend([legend_info_dps, legend_info_episodes])
ax3.legend(handles=handles, loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()

output_path3 = OUTPUT_DIR / 'dp_distribution_histogram.png'
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"✓ Saved histogram view: {output_path3}")
plt.close()
