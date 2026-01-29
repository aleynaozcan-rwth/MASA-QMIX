#!/usr/bin/env python3
"""
Enhanced Parallel Decision Points with filled areas: 
Total DPs (light), Parallel DPs (darker), Conflicts (darkest)
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d
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
    """Parse timeline to identify both simultaneity and conflicts."""
    
    episodes = []
    current_episode = None
    timestamp_groups = defaultdict(int)  # timestamp -> count
    decision_points = []  # (timestamp, job_id, operation)
    in_lifecycle = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                if current_episode is not None:
                    # Calculate parallel DPs and conflicts
                    parallel_count = sum(count for count in timestamp_groups.values() if count > 1)
                    total_count = sum(timestamp_groups.values())
                    
                    # Detect conflicts (job-op appearing multiple times)
                    job_op_timestamps = defaultdict(list)
                    for timestamp, job_id, operation in decision_points:
                        job_op_timestamps[(job_id, operation)].append(float(timestamp))
                    
                    # Count conflict DPs (retry attempts)
                    conflict_dps = set()
                    for (job_id, operation), timestamps in job_op_timestamps.items():
                        if len(timestamps) > 1:
                            # All except the first one are retries (conflicts)
                            for ts in timestamps[1:]:
                                conflict_dps.add((str(ts), job_id, operation))
                    
                    # Combine: pure parallel + conflict retries
                    parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
                    conflict_timestamps = {ts for ts, _, _ in conflict_dps}
                    
                    # Total coordination-requiring DPs
                    all_coordination_timestamps = parallel_timestamps | conflict_timestamps
                    
                    # Count DPs at these timestamps
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
            
            # Job arrivals
            if in_lifecycle and '[t=' in line and 'New job' in line and 'arrived' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'job (\d+)', line)
                if time_match and job_match:
                    ts = time_match.group(1)
                    job_id = int(job_match.group(1))
                    timestamp_groups[ts] += 1
                    decision_points.append((ts, job_id, 'Op1'))
                continue
            
            # Completions
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

print("Parsing timeline for filled area visualization...")
episodes = parse_with_conflicts()
print(f"Parsed {len(episodes)} episodes")

# Extract data
episode_numbers = np.array([ep['episode'] for ep in episodes])
pure_parallel_dps = np.array([ep['pure_parallel_dps'] for ep in episodes])
conflict_dps = np.array([ep['conflict_dps'] for ep in episodes])
total_coordination_dps = np.array([ep['total_coordination_dps'] for ep in episodes])
total_dps = np.array([ep['total_dps'] for ep in episodes])

# Calculate statistics
total_pure_parallel = np.sum(pure_parallel_dps)
total_conflicts = np.sum(conflict_dps)
total_coordination = np.sum(total_coordination_dps)
total_dps_sum = np.sum(total_dps)

# Smooth data
window_size = 5
total_smooth = uniform_filter1d(total_dps, size=window_size) if len(total_dps) >= window_size else total_dps
coord_smooth = uniform_filter1d(total_coordination_dps, size=window_size) if len(total_coordination_dps) >= window_size else total_coordination_dps
conflict_smooth = uniform_filter1d(conflict_dps, size=window_size) if len(conflict_dps) >= window_size else conflict_dps

print(f"\nStatistics:")
print(f"  Total DPs: {total_dps_sum:,}")
print(f"  Total Parallel: {total_coordination:,} ({100*total_coordination/total_dps_sum:.1f}%)")
print(f"  Total Conflicts: {total_conflicts:,} ({100*total_conflicts/total_dps_sum:.1f}%)")

# Create visualization with 3 y-axes (like parallel_with_conflicts)
fig = plt.figure(figsize=(20, 10))
host = host_subplot(111, axes_class=AA.Axes, figure=fig)

par1 = host.twinx()  # Parallel DPs (normal right)
par2 = host.twinx()  # Conflicts (offset right)

# Offset the conflict axis to the right
par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(35, 0))

# Make both right axes visible
par1.axis["right"].toggle(all=True)
par2.axis["right"].toggle(all=True)

# Left y-axis: Total DPs - light steelblue fill
host.set_xlabel('Episode', fontsize=14, fontweight='black')
host.set_ylabel('Total Decision Points per Episode', fontsize=15, fontweight='black')
host.axis["left"].label.set_color('steelblue')
host.tick_params(axis='y', colors='steelblue')

host.fill_between(episode_numbers, 0, total_smooth, color='steelblue', alpha=0.3, zorder=1)
host.set_xlim(0, len(episode_numbers) - 1)
host.set_ylim(0, 60)
host.set_yticks(np.arange(0, 61, 10))
host.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)

# Right y-axis 1: Parallel DPs - darker crimson fill
par1.set_ylabel('Parallel Decision Points per Episode', fontsize=15, fontweight='black')
par1.axis["right"].label.set_color('#DC143C')
par1.axis["right"].major_ticks.set_color('#DC143C')
par1.axis["right"].major_ticklabels.set_color('#DC143C')

par1.fill_between(episode_numbers, 0, coord_smooth, color='#DC143C', alpha=0.5, zorder=2)
par1.set_ylim(0, 24)  # Scale so that parallel 6 aligns with total 15

# Right y-axis 2: Conflicts - darkest purple fill
par2.set_ylabel('Conflict Points per Episode', fontsize=15, fontweight='black')
par2.axis["right"].label.set_color('purple')
par2.axis["right"].major_ticks.set_color('purple')
par2.axis["right"].major_ticklabels.set_color('purple')

par2.fill_between(episode_numbers, 0, conflict_smooth, color='purple', alpha=0.7, zorder=3)

# Scale conflict axis: use same ratio as parallel
# If parallel 6 = total 15, and parallel 0 = total 0
# Then parallel_max = 24, so conflict should follow same scaling
parallel_max = 24
conflict_max = parallel_max  # Keep same scale
par2.set_ylim(0, conflict_max)

# Custom ticks for conflicts
ticks = list(range(0, int(conflict_max) + 1, 3))  # 0, 3, 6, 9, ...
par2.set_yticks(ticks)

# Left legend - data lines
from matplotlib.lines import Line2D
left_legend_lines = [
    Line2D([0], [0], color='steelblue', linewidth=3, alpha=0.3, 
           label=f'Total DPs MA (window={window_size}, total: {total_dps_sum:,})'),
    Line2D([0], [0], color='#DC143C', linewidth=3, alpha=0.5,
           label=f'Parallel DPs MA (window={window_size}, total: {total_coordination:,})'),
    Line2D([0], [0], color='purple', linewidth=3, alpha=0.7,
           label=f'Conflicts MA (window={window_size}, total: {total_conflicts:,})')
]
host.legend(handles=left_legend_lines, loc='upper left', fontsize=10, framealpha=0.95)

# Right legend - episode and job information
total_parallel_combined = total_pure_parallel + total_conflicts
info_lines = [
    Line2D([0], [0], color='none', label=f'Total Episodes: {len(episodes):,}'),
    Line2D([0], [0], color='none', label=f'Parallel Ratio: {100*total_parallel_combined/total_dps_sum:.1f}%'),
    Line2D([0], [0], color='none', label=f'  - Simultaneity: {100*total_pure_parallel/total_dps_sum:.1f}%'),
    Line2D([0], [0], color='none', label=f'  - Conflicts: {100*total_conflicts/total_dps_sum:.1f}%'),
    Line2D([0], [0], color='none', label=f'Initial Jobs per Episode: 4'),
    Line2D([0], [0], color='none', label='Job Arrival Rate λ = 0.125')
]

from matplotlib.legend import Legend
leg2 = Legend(host, info_lines, [l.get_label() for l in info_lines],
             loc='upper right', fontsize=10, framealpha=0.95)
leg2._legend_box.align = 'left'
host.add_artist(leg2)

plt.title('Decision Points Distribution', 
          fontsize=16, fontweight='bold', pad=20)

plt.subplots_adjust(right=0.90)

output_path = OUTPUT_DIR / 'parallel_filled.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved filled area plot: {output_path}")
plt.close()
