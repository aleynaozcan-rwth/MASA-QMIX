#!/usr/bin/env python3
"""
Comprehensive visualization showing the breakdown of parallel decision points:
- Total DPs, Parallel DPs (combined), Simultaneity DPs, Conflict DPs
Uses 4 y-axes to show all metrics simultaneously.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_with_conflicts():
    """Parse timeline to identify simultaneity and conflicts."""
    
    episodes = []
    current_episode = None
    timestamp_groups = defaultdict(int)
    decision_points = []
    in_lifecycle = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                if current_episode is not None:
                    parallel_count = sum(count for count in timestamp_groups.values() if count > 1)
                    total_count = sum(timestamp_groups.values())
                    
                    job_op_timestamps = defaultdict(list)
                    for timestamp, job_id, operation in decision_points:
                        job_op_timestamps[(job_id, operation)].append(float(timestamp))
                    
                    conflict_dps = set()
                    for (job_id, operation), timestamps in job_op_timestamps.items():
                        if len(timestamps) > 1:
                            for ts in timestamps[1:]:
                                conflict_dps.add((str(ts), job_id, operation))
                    
                    parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
                    conflict_timestamps = {ts for ts, _, _ in conflict_dps}
                    all_coordination_timestamps = parallel_timestamps | conflict_timestamps
                    coordination_dp_count = sum(timestamp_groups[ts] for ts in all_coordination_timestamps)
                    
                    episodes.append({
                        'episode': current_episode,
                        'pure_parallel_dps': parallel_count,
                        'conflict_dps': len(conflict_dps),
                        'total_coordination_dps': coordination_dp_count,
                        'total_dps': total_count
                    })
                
                match = re.search(r'EPISODE (\d+)', line)
                current_episode = int(match.group(1)) if match else None
                timestamp_groups = defaultdict(int)
                decision_points = []
                in_lifecycle = False
                continue
            
            if 'LIFECYCLE TRACE START' in line:
                in_lifecycle = True
                continue
            if 'LIFECYCLE TRACE END' in line:
                in_lifecycle = False
                continue
            
            if in_lifecycle and '[t=' in line and 'New job' in line and 'arrived' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'job (\d+)', line)
                if time_match and job_match:
                    ts = time_match.group(1)
                    job_id = int(job_match.group(1))
                    timestamp_groups[ts] += 1
                    decision_points.append((ts, job_id, 'Op1'))
                continue
            
            if not in_lifecycle and '[t=' in line and 'finished' in line and 'next queued' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'Job_(\d+)\.Op(\d+) finished', line)
                if time_match and job_match:
                    ts = time_match.group(1)
                    job_id = int(job_match.group(1))
                    completed_op = int(job_match.group(2))
                    next_op = completed_op + 1
                    timestamp_groups[ts] += 1
                    decision_points.append((ts, job_id, f'Op{next_op}'))
                continue
    
    # Last episode
    if current_episode is not None:
        parallel_count = sum(count for count in timestamp_groups.values() if count > 1)
        total_count = sum(timestamp_groups.values())
        
        job_op_timestamps = defaultdict(list)
        for timestamp, job_id, operation in decision_points:
            job_op_timestamps[(job_id, operation)].append(float(timestamp))
        
        conflict_dps = set()
        for (job_id, operation), timestamps in job_op_timestamps.items():
            if len(timestamps) > 1:
                for ts in timestamps[1:]:
                    conflict_dps.add((str(ts), job_id, operation))
        
        parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
        conflict_timestamps = {ts for ts, _, _ in conflict_dps}
        all_coordination_timestamps = parallel_timestamps | conflict_timestamps
        coordination_dp_count = sum(timestamp_groups[ts] for ts in all_coordination_timestamps)
        
        episodes.append({
            'episode': current_episode,
            'pure_parallel_dps': parallel_count,
            'conflict_dps': len(conflict_dps),
            'total_coordination_dps': coordination_dp_count,
            'total_dps': total_count
        })
    
    return episodes

print("Parsing timeline for conflict breakdown analysis...")
episodes = parse_with_conflicts()
print(f"Parsed {len(episodes)} episodes")

episode_numbers = np.array([ep['episode'] for ep in episodes])
pure_parallel_dps = np.array([ep['pure_parallel_dps'] for ep in episodes])
conflict_dps = np.array([ep['conflict_dps'] for ep in episodes])
total_coordination_dps = np.array([ep['total_coordination_dps'] for ep in episodes])
total_dps = np.array([ep['total_dps'] for ep in episodes])

total_pure_parallel = np.sum(pure_parallel_dps)
total_conflicts = np.sum(conflict_dps)
total_coordination = np.sum(total_coordination_dps)
total_dps_sum = np.sum(total_dps)

print(f"\nStatistics:")
print(f"  Total DPs: {total_dps_sum:,}")
print(f"  Parallel DPs (combined): {total_coordination:,} ({100*total_coordination/total_dps_sum:.1f}%)")
print(f"  - Simultaneity: {total_pure_parallel:,} ({100*total_pure_parallel/total_dps_sum:.1f}%)")
print(f"  - Conflicts: {total_conflicts:,} ({100*total_conflicts/total_dps_sum:.1f}%)")

# Create comprehensive dual-axis plot with stacked areas
fig = plt.figure(figsize=(18, 8))
host = host_subplot(111, axes_class=AA.Axes, figure=fig)

par1 = host.twinx()

# Labels
host.set_xlabel('Episode', fontsize=12, fontweight='bold')
host.set_ylabel('Total DPs', fontsize=11, fontweight='bold', color='steelblue')
par1.set_ylabel('Parallel DPs', fontsize=11, fontweight='bold', color='#DC143C')

# Plot data
sigma = 1.0

# Total DPs (blue line)
if len(total_dps) >= 5:
    total_smooth = gaussian_filter1d(total_dps, sigma=sigma)
    p1, = host.plot(episode_numbers, total_smooth, linewidth=2, color='steelblue', alpha=0.8,
                    label=f'Total DPs (σ={sigma}, total: {total_dps_sum:,})')

# Stacked area chart for simultaneity + conflicts
if len(pure_parallel_dps) >= 5 and len(conflict_dps) >= 5:
    simul_smooth = gaussian_filter1d(pure_parallel_dps, sigma=sigma)
    conflict_smooth = gaussian_filter1d(conflict_dps, sigma=sigma)
    
    # Area 1: Simultaneity (orange, bottom)
    par1.fill_between(episode_numbers, 0, simul_smooth, color='#FF7F0E', alpha=0.6,
                      label=f'Simultaneity (σ={sigma}, total: {total_pure_parallel:,})')
    
    # Area 2: Conflicts (purple, stacked on top)
    par1.fill_between(episode_numbers, simul_smooth, simul_smooth + conflict_smooth, 
                      color='purple', alpha=0.6,
                      label=f'Conflicts (σ={sigma}, total: {total_conflicts:,})')
    
    # Outline: Total parallel (red dashed line)
    coord_smooth = gaussian_filter1d(total_coordination_dps, sigma=sigma)
    p2, = par1.plot(episode_numbers, coord_smooth, linewidth=2.5, color='#DC143C', alpha=0.9, linestyle='--',
                    label=f'Parallel Total (σ={sigma}, total: {total_coordination:,})')

# Styling
host.set_xlim(0, len(episode_numbers) - 1)
host.set_ylim(20, max(total_dps) * 1.05)
par1.set_ylim(0, max(total_coordination_dps) * 1.2)

host.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# Color the y-axis labels
host.axis["left"].label.set_color('steelblue')
par1.axis["right"].label.set_color('#DC143C')

# Legend
host.legend(loc='upper left', fontsize=10, framealpha=0.95)

# Title
plt.title('Parallel Decision Points: Complete Breakdown', 
         fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()

output_path = OUTPUT_DIR / 'conflict_in_parallel_decision_making.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved conflict breakdown plot: {output_path}")
plt.close()

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print("\nTwo visualizations available:")
print("  1. parallel_decision_points.png - Clean dual-axis view")
print("  2. conflict_in_parallel_decision_making.png - Complete 4-axis breakdown")
