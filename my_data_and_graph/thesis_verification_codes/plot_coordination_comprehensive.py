#!/usr/bin/env python3
"""
Create an enhanced visualization emphasizing the near-parallel decision making ratio.
Shows multiple perspectives on multi-agent coordination.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

# Use Agg backend for cluster without display
import matplotlib
matplotlib.use('Agg')

# ==================== Configuration ====================
TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Near-parallel threshold: decisions within this time window are considered coordinated
TIME_THRESHOLD = 0.1  # time units (Δt ≤ 0.1)

# ==================== Parse Timeline File ====================
print(f"Parsing scheduling timeline for near-parallel analysis (Δt ≤ {TIME_THRESHOLD})...")

episodes = []
current_episode = None
all_timestamps = []
in_lifecycle_trace = False

with open(TIMELINE_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Match episode start
        episode_match = re.match(r'^=== EPISODE (\d+) ===$', line)
        if episode_match:
            # Save previous episode if exists
            if current_episode is not None and all_timestamps:
                all_timestamps.sort()
                parallel_dp_count = 0
                processed = set()
                
                for i, t1 in enumerate(all_timestamps):
                    if i in processed:
                        continue
                    
                    cluster = [t1]
                    processed.add(i)
                    
                    for j in range(i + 1, len(all_timestamps)):
                        if j in processed:
                            continue
                        if all_timestamps[j] - t1 <= TIME_THRESHOLD:
                            cluster.append(all_timestamps[j])
                            processed.add(j)
                        else:
                            break
                    
                    if len(cluster) >= 2:
                        parallel_dp_count += len(cluster)
                
                episodes.append({
                    'episode': current_episode,
                    'parallel_dps': parallel_dp_count,
                    'total_dps': len(all_timestamps)
                })
            
            current_episode = int(episode_match.group(1))
            all_timestamps = []
            in_lifecycle_trace = False
            continue
        
        if 'LIFECYCLE TRACE START' in line:
            in_lifecycle_trace = True
            continue
        
        if 'LIFECYCLE TRACE END' in line:
            in_lifecycle_trace = False
            continue
        
        # Job arrivals
        if in_lifecycle_trace and '[t=' in line and 'New job' in line and 'arrived' in line:
            timestamp_match = re.search(r'\[t=(\d+\.\d+)\]', line)
            if timestamp_match:
                timestamp = float(timestamp_match.group(1))
                all_timestamps.append(timestamp)
            continue
        
        # Operation completions
        if not in_lifecycle_trace and '[t=' in line and 'finished' in line and 'next queued' in line:
            timestamp_match = re.search(r'\[t=(\d+\.\d+)\]', line)
            if timestamp_match:
                timestamp = float(timestamp_match.group(1))
                all_timestamps.append(timestamp)
            continue

# Save last episode
if current_episode is not None and all_timestamps:
    all_timestamps.sort()
    parallel_dp_count = 0
    processed = set()
    
    for i, t1 in enumerate(all_timestamps):
        if i in processed:
            continue
        
        cluster = [t1]
        processed.add(i)
        
        for j in range(i + 1, len(all_timestamps)):
            if j in processed:
                continue
            if all_timestamps[j] - t1 <= TIME_THRESHOLD:
                cluster.append(all_timestamps[j])
                processed.add(j)
            else:
                break
        
        if len(cluster) >= 2:
            parallel_dp_count += len(cluster)
    
    episodes.append({
        'episode': current_episode,
        'parallel_dps': parallel_dp_count,
        'total_dps': len(all_timestamps)
    })

print(f"Parsed {len(episodes)} episodes")

# ==================== Extract Data ====================
parallel_dps = np.array([ep['parallel_dps'] for ep in episodes])
total_dps = np.array([ep['total_dps'] for ep in episodes])
sequential_dps = total_dps - parallel_dps  # Non-coordinated decisions

total_parallel = np.sum(parallel_dps)
total_sequential = np.sum(sequential_dps)
total_all = np.sum(total_dps)

print(f"\nStatistics:")
print(f"  Near-Parallel DPs: {total_parallel:,} ({100*total_parallel/total_all:.1f}%)")
print(f"  Sequential DPs: {total_sequential:,} ({100*total_sequential/total_all:.1f}%)")
print(f"  Total DPs: {total_all:,}")

# ==================== Create Enhanced Visualization ====================
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1], hspace=0.3, wspace=0.3)

# ===== 1. Main Time Series (Top Left) =====
ax1 = fig.add_subplot(gs[0, 0])
episode_numbers = np.array([ep['episode'] for ep in episodes])

# Area chart showing composition
ax1.fill_between(episode_numbers, 0, sequential_dps, color='lightgray', alpha=0.6, label='Sequential DPs')
ax1.fill_between(episode_numbers, sequential_dps, total_dps, color='#DC143C', alpha=0.7, label='Near-Parallel DPs (Δt ≤ 0.1)')

# Smooth overlay
sigma = 1.0
if len(parallel_dps) >= 5:
    total_smooth = gaussian_filter1d(total_dps, sigma=sigma)
    parallel_smooth = gaussian_filter1d(parallel_dps, sigma=sigma)
    
    ax1.plot(episode_numbers, total_smooth, linewidth=2.5, color='black', alpha=0.8, zorder=10, 
            label=f'Total DPs MA (σ={sigma})')
    ax1.plot(episode_numbers, parallel_smooth, linewidth=2.5, color='darkred', alpha=0.9, zorder=11, 
            label=f'Near-Parallel MA (σ={sigma})')

ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax1.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax1.set_title('Multi-Agent Coordination: Temporal Distribution', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax1.set_xlim(0, len(episode_numbers) - 1)
ax1.set_ylim(0, None)
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# ===== 2. Ratio Comparison Bar (Top Right) =====
ax2 = fig.add_subplot(gs[0, 1])

categories = ['Sequential\nDecisions', 'Near-Parallel\nCoordination']
values = [total_sequential, total_parallel]
percentages = [100*total_sequential/total_all, 100*total_parallel/total_all]
colors = ['lightgray', '#DC143C']

bars = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (bar, val, pct) in enumerate(zip(bars, values, percentages)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{val:,}\n({pct:.1f}%)',
            ha='center', va='center', fontsize=13, fontweight='bold', color='white')

ax2.set_ylabel('Total Decision Points', fontsize=12, fontweight='bold')
ax2.set_title('Decision Type Distribution', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(values) * 1.15)
ax2.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

# ===== 3. Per-Episode Histogram (Bottom Left) =====
ax3 = fig.add_subplot(gs[1, 0])

parallel_ratios = (parallel_dps / total_dps) * 100  # percentage per episode

ax3.hist(parallel_ratios, bins=30, color='#DC143C', alpha=0.7, edgecolor='black', linewidth=1.2)

# Add statistics
mean_ratio = np.mean(parallel_ratios)
median_ratio = np.median(parallel_ratios)

ax3.axvline(mean_ratio, color='green', linestyle='--', linewidth=2.5, 
           label=f'Mean: {mean_ratio:.1f}%')
ax3.axvline(median_ratio, color='blue', linestyle='--', linewidth=2.5, 
           label=f'Median: {median_ratio:.1f}%')

ax3.set_xlabel('Near-Parallel Ratio per Episode (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency (Episodes)', fontsize=12, fontweight='bold')
ax3.set_title('Distribution of Multi-Agent Coordination Ratios', fontsize=13, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax3.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

# ===== 4. Summary Statistics Panel (Bottom Right) =====
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

# Create summary text
summary_text = f"""
NEAR-PARALLEL COORDINATION ANALYSIS
{'='*45}

Time Threshold: Δt ≤ {TIME_THRESHOLD} time units

OVERALL STATISTICS:
  Total Decision Points: {total_all:,}
  Near-Parallel DPs: {total_parallel:,}
  Sequential DPs: {total_sequential:,}
  
  Coordination Ratio: {100*total_parallel/total_all:.1f}%
  
EPISODE STATISTICS:
  Total Episodes: {len(episodes):,}
  Mean DPs per Episode: {np.mean(total_dps):.2f}
  Mean Coordinated DPs: {np.mean(parallel_dps):.2f}
  
PER-EPISODE RATIO:
  Mean: {mean_ratio:.1f}%
  Median: {median_ratio:.1f}%
  Std Dev: {np.std(parallel_ratios):.1f}%
  Range: {np.min(parallel_ratios):.1f}% - {np.max(parallel_ratios):.1f}%

ENVIRONMENT:
  Job Arrival Rate: λ = 0.125
  Mean Interarrival: 8.0 time units
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1))

# ===== Save Figure =====
output_path = OUTPUT_DIR / 'coordination_analysis_comprehensive.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved comprehensive coordination analysis: {output_path}")
plt.close()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"\nKey Finding: {100*total_parallel/total_all:.1f}% of decisions occur within")
print(f"Δt ≤ {TIME_THRESHOLD} time units, demonstrating significant multi-agent coordination.")
