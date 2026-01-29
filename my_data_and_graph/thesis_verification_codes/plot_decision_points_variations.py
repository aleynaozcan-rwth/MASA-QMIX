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
in_lifecycle_trace = False

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
            in_lifecycle_trace = False
            continue
        
        # Track lifecycle trace section (only count arrivals here)
        if 'JOB AGENT LIFECYCLE TRACE START' in line:
            in_lifecycle_trace = True
            continue
        
        if 'JOB AGENT LIFECYCLE TRACE END' in line:
            in_lifecycle_trace = False
            continue
        
        # Match decision point events:
        # 1. Job arrivals: Only count in lifecycle trace (to avoid double counting)
        if in_lifecycle_trace and '[t=' in line and 'New job' in line and 'arrived' in line:
            current_dp_count += 1
            current_arrivals += 1
            continue
        
        # 2. Operation completions: Count everywhere (only in TIMELINE section)
        if not in_lifecycle_trace and '[t=' in line and 'finished' in line and 'next queued' in line:
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

# Vertical lines (stem style) - from 0 to dp_counts
ax1.vlines(episode_numbers, 0, dp_counts, colors='lightsteelblue', linewidth=1.5, alpha=0.6, zorder=1)

# Moving average (smoothed) - using Gaussian filter for smoother appearance
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D

window_size = 5  # 5-episode window
sigma_arrivals = 0.5
sigma_completions = 1.5
sigma_total = 1.0

if len(dp_counts) >= window_size:
    # Smooth each component
    arrivals_smooth = gaussian_filter1d(arrivals, sigma=sigma_arrivals)
    completions_smooth = gaussian_filter1d(completions, sigma=sigma_completions)
    dp_smoothed = gaussian_filter1d(dp_counts, sigma=sigma_total)
    
    # Plot arrivals, completions, and total
    ax1.plot(episode_numbers, arrivals_smooth, linewidth=2, color='#FF7F0E', alpha=0.8, zorder=6, 
            label=f'Job Arrivals MA (σ={sigma_arrivals}, total: {np.sum(arrivals):,})')
    ax1.plot(episode_numbers, completions_smooth, linewidth=2, color='purple', alpha=0.8, zorder=5, 
            label=f'Operation Completions MA (σ={sigma_completions}, total: {np.sum(completions):,})')
    ax1.plot(episode_numbers, dp_smoothed, linewidth=2, color='blue', alpha=0.8, zorder=7, 
            label=f'Total DPs MA (σ={sigma_total}, total: {total_dp:,})')

# Add mean line - green, thick, and on top
ax1.axhline(y=mean_dp, color='green', linestyle='--', linewidth=1.5, alpha=0.8, zorder=10, label=f'Mean: {mean_dp:.2f}')

# Add std range shading
ax1.axhspan(mean_dp - std_dp, mean_dp + std_dp, color='green', alpha=0.1, label=f'±1σ: [{mean_dp-std_dp:.1f}, {mean_dp+std_dp:.1f}]')

# Add total info and arrival rate to legend
ax1.plot([], [], ' ', label=f'Total Episodes: {len(episodes):,}')
ax1.plot([], [], ' ', label='Job Arrival Rate λ = 0.125')

ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax1.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax1.set_title('Decision Point Density Evolution', 
              fontsize=13, fontweight='bold', pad=15)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax1.set_xlim(0, len(episode_numbers) - 1)  # Start X-axis from 0, no extra space
ax1.set_ylim(0, None)  # Start Y-axis from 0
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

plt.tight_layout()

output_path1 = OUTPUT_DIR / 'dp_per_episode_signal.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved signal view: {output_path1}")
plt.close()

# ------------------------- Version 1b: Signal + Breakdown (Arrivals & Completions) -------------------------
fig1b, ax1b = plt.subplots(figsize=(16, 7))

# Vertical lines (stem style) for total DPs
ax1b.vlines(episode_numbers, 0, dp_counts, colors='lightsteelblue', linewidth=1.0, alpha=0.4, zorder=1)

# Use different smoothing for each to show their unique patterns and contributions
# Job arrivals: light smoothing to show variability but not too noisy
arrivals_smooth = gaussian_filter1d(arrivals, sigma=0.5)  # Light - show variability
# Operation completions: more smoothing to create distinct pattern
completions_smooth = gaussian_filter1d(completions, sigma=1.5)  # Moderate - distinct pattern
# Total: medium smoothing - shows it's the sum but different from both
dp_smoothed = gaussian_filter1d(dp_counts, sigma=1.0)  # Medium - unique pattern

# Plot arrivals, completions, and total
ax1b.plot(episode_numbers, arrivals_smooth, linewidth=2, color='#FF7F0E', 
         alpha=0.8, label=f'Job Arrivals MA (mean: {np.mean(arrivals):.1f}, total: {np.sum(arrivals):,})', zorder=5)
ax1b.plot(episode_numbers, completions_smooth, linewidth=2, color='purple', 
         alpha=0.8, label=f'Operation Completions MA (mean: {np.mean(completions):.1f}, total: {np.sum(completions):,})', zorder=4)
ax1b.plot(episode_numbers, dp_smoothed, linewidth=2, color='blue', 
         alpha=0.8, label=f'Total DPs MA (mean: {mean_dp:.2f}, total: {total_dp:,})', zorder=3)

# Add mean line for total DPs
ax1b.axhline(y=mean_dp, color='green', linestyle='--', linewidth=2, alpha=0.8, zorder=10, 
            label=f'Overall Mean: {mean_dp:.2f}')

# Add subtle mean indicators for components
ax1b.axhline(y=np.mean(arrivals), color='#FF7F0E', linestyle=':', linewidth=1, alpha=0.4, zorder=4)
ax1b.axhline(y=np.mean(completions), color='#00CED1', linestyle=':', linewidth=1, alpha=0.4, zorder=4)

# Add arrival rate info to legend
ax1b.plot([], [], ' ', label='Job Arrival Rate λ = 0.125')

ax1b.legend(loc='upper right', fontsize=10, framealpha=0.9)

ax1b.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax1b.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax1b.set_title(f'Decision Point Breakdown: Arrivals vs Completions Over Episodes', 
              fontsize=13, fontweight='bold', pad=15)
ax1b.set_ylim(bottom=0)  # Start Y-axis from 0
ax1b.set_xlim(0, len(episode_numbers) - 1)  # Start X-axis from 0, no extra space
ax1b.set_yticks(np.arange(0, 75, 5))  # Y-axis ticks every 5 units
ax1b.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

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
ax2.set_title(f'Decision Points Scatter Plot with Polynomial Trend\n{len(episodes)} episodes | Mean: {mean_dp:.2f} | Range: [{min_dp}, {max_dp}] | Total: {total_dp:,}', 
              fontsize=13, fontweight='bold', pad=15)
ax2.set_xlim(0, len(episode_numbers) - 1)  # Start X-axis from 0, no extra space
ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Add legend with handles
handles, labels = ax2.get_legend_handles_labels()
handles.extend([legend_info_dps, legend_info_episodes])
# Add arrival rate info
from matplotlib.lines import Line2D
arrival_rate_handle = Line2D([], [], color='none', label='Job Arrival Rate λ = 0.125')
handles.append(arrival_rate_handle)
ax2.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)

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
ax3.set_title('Distribution of Total Decision Points', 
              fontsize=13, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

# Add legend with handles
handles, labels = ax3.get_legend_handles_labels()
handles.extend([legend_info_dps, legend_info_episodes])
# Add arrival rate info
from matplotlib.lines import Line2D
arrival_rate_handle = Line2D([], [], color='none', label='Job Arrival Rate λ = 0.125')
handles.append(arrival_rate_handle)
ax3.legend(handles=handles, loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()

output_path3 = OUTPUT_DIR / 'dp_distribution_histogram.png'
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"✓ Saved histogram view: {output_path3}")
plt.close()
# ------------------------- Version 4: Job Arrivals Distribution -------------------------
fig4, ax4 = plt.subplots(figsize=(12, 5))

# Calculate statistics for arrivals
mean_arrivals = np.mean(arrivals)
median_arrivals = np.median(arrivals)
total_arrivals = np.sum(arrivals)

# Create histogram
bins_arrivals = range(int(np.min(arrivals)), int(np.max(arrivals)) + 2)
ax4.hist(arrivals, bins=bins_arrivals, color='#FF7F0E', edgecolor='black', 
         alpha=0.7, linewidth=1.2)

# Add mean and median lines
ax4.axvline(x=mean_arrivals, color='green', linestyle='--', linewidth=2.5, 
            alpha=0.8, label=f'Mean: {mean_arrivals:.1f}')
ax4.axvline(x=median_arrivals, color='red', linestyle=':', linewidth=3.5, 
            alpha=0.8, label=f'Median: {median_arrivals:.0f}')

# Add legend info
legend_info_total_arrivals = Patch(facecolor='none', edgecolor='none', label=f'Total Job Arrival DPs: {total_arrivals:,}')
legend_info_total_dps_arr = Patch(facecolor='none', edgecolor='none', label=f'Total DPs: {total_dp:,}')
legend_info_episodes_arr = Patch(facecolor='none', edgecolor='none', label=f'Total Episodes: {len(episodes):,}')

ax4.set_xlabel('Job Arrivals per Episode', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.set_title('Distribution of Stochastic Job Arrivals', 
              fontsize=13, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

# Add legend with handles
handles4, labels4 = ax4.get_legend_handles_labels()
handles4.extend([legend_info_total_arrivals, legend_info_total_dps_arr, legend_info_episodes_arr])
arrival_rate_handle4 = Line2D([], [], color='none', label='Job Arrival Rate λ = 0.125')
handles4.append(arrival_rate_handle4)
ax4.legend(handles=handles4, loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()

output_path4 = OUTPUT_DIR / 'job_arrivals_distribution.png'
plt.savefig(output_path4, dpi=150, bbox_inches='tight')
print(f"✓ Saved job arrivals histogram: {output_path4}")
plt.close()

# ------------------------- Version 5: Operation Completions Distribution -------------------------
fig5, ax5 = plt.subplots(figsize=(12, 5))

# Calculate statistics for completions
mean_completions = np.mean(completions)
median_completions = np.median(completions)
total_completions_dps = np.sum(completions)

# Create histogram
bins_completions = range(int(np.min(completions)), int(np.max(completions)) + 2)
ax5.hist(completions, bins=bins_completions, color='purple', edgecolor='black', 
         alpha=0.7, linewidth=1.2)

# Add mean and median lines
ax5.axvline(x=mean_completions, color='green', linestyle='--', linewidth=2.5, 
            alpha=0.8, label=f'Mean: {mean_completions:.1f}')
ax5.axvline(x=median_completions, color='red', linestyle=':', linewidth=3.5, 
            alpha=0.8, label=f'Median: {median_completions:.0f}')

# Add legend info
legend_info_total_completions = Patch(facecolor='none', edgecolor='none', label=f'Total Completion DPs: {total_completions_dps:,}')
legend_info_total_dps_comp = Patch(facecolor='none', edgecolor='none', label=f'Total DPs: {total_dp:,}')
legend_info_episodes_comp = Patch(facecolor='none', edgecolor='none', label=f'Total Episodes: {len(episodes):,}')

ax5.set_xlabel('Operation Completions per Episode', fontsize=12, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax5.set_title('Distribution of Operation Completions', 
              fontsize=13, fontweight='bold', pad=15)
ax5.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')

# Add legend with handles
handles5, labels5 = ax5.get_legend_handles_labels()
handles5.extend([legend_info_total_completions, legend_info_total_dps_comp, legend_info_episodes_comp])
arrival_rate_handle5 = Line2D([], [], color='none', label='Job Arrival Rate λ = 0.125')
handles5.append(arrival_rate_handle5)
ax5.legend(handles=handles5, loc='upper right', fontsize=11, framealpha=0.9)

plt.tight_layout()

output_path5 = OUTPUT_DIR / 'operation_completions_distribution.png'
plt.savefig(output_path5, dpi=150, bbox_inches='tight')
print(f"✓ Saved operation completions histogram: {output_path5}")
plt.close()