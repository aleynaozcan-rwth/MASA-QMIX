#!/usr/bin/env python3
"""
Enhanced Parallel Decision Points: Include both simultaneous decisions AND conflict retries.
Both represent coordination challenges that MARL must address.
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

print("Parsing timeline for enhanced parallel decision points...")
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

mean_coordination = np.mean(total_coordination_dps)
std_coordination = np.std(total_coordination_dps)

print(f"\nEnhanced Parallel Decision Points Statistics:")
print(f"  Pure Simultaneity DPs: {total_pure_parallel:,} ({100*total_pure_parallel/total_dps_sum:.1f}%)")
print(f"  Conflict Retry DPs: {total_conflicts:,} ({100*total_conflicts/total_dps_sum:.1f}%)")
print(f"  Total Coordination DPs: {total_coordination:,} ({100*total_coordination/total_dps_sum:.1f}%)")
print(f"  Total DPs: {total_dps_sum:,}")
print(f"  Mean Coordination DPs per episode: {mean_coordination:.2f}")
print(f"  Std Dev: {std_coordination:.2f}")
print(f"  Range: {np.min(total_coordination_dps)} - {np.max(total_coordination_dps)}")

# Create visualization with 3 y-axes
fig = plt.figure(figsize=(20, 10))  # Increased width from 18 to 20
host = host_subplot(111, axes_class=AA.Axes, figure=fig)

par1 = host.twinx()  # Parallel DPs (normal right)
par2 = host.twinx()  # Conflicts (offset right)

# Offset the conflict axis to the right - very close to parallel
par2.axis["right"] = par2.new_fixed_axis(loc="right", offset=(35, 0))

# Make both right axes visible
par1.axis["right"].toggle(all=True)
par2.axis["right"].toggle(all=True)

# Left y-axis: Total DPs
host.set_xlabel('Episode', fontsize=18, fontweight='heavy')
host.set_ylabel('Total Decision Points per Episode', fontsize=14, fontweight='heavy')
host.axis["left"].label.set_color('#1E3A8A')  # Dark blue
host.tick_params(axis='y', colors='#1E3A8A', labelsize=11)
host.tick_params(axis='x', labelsize=11)

# Smooth total data
window_size = 5
if len(total_dps) >= window_size:
    total_smooth = uniform_filter1d(total_dps, size=window_size)
    host.plot(episode_numbers, total_smooth, linewidth=2, color='#5DADE2', alpha=0.9, zorder=3,
            label=f'Total DPs MA (window={window_size}, total: {total_dps_sum:,})')

host.set_xlim(0, len(episode_numbers) - 1)
host.set_ylim(0, 50)  # Fixed upper limit at 50
host.set_yticks(np.arange(0, 51, 10))  # 0, 10, 20, 30, 40, 50
host.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)

# Right y-axis 1: Parallel DPs (normal right edge)
par1.set_ylabel('Parallel Decision Points per Episode', fontsize=14, fontweight='heavy')
par1.axis["right"].label.set_color('#FF8C00')
par1.axis["right"].major_ticks.set_color('#FF8C00')
par1.axis["right"].major_ticklabels.set_color('#FF8C00')
par1.axis["right"].major_ticklabels.set_fontsize(11)

# Coordination DPs (parallel + conflicts) - dark orange line
if len(total_coordination_dps) >= window_size:
    coord_smooth = uniform_filter1d(total_coordination_dps, size=window_size)
    par1.plot(episode_numbers, coord_smooth, linewidth=2.5, color='#FF8C00', alpha=0.95, zorder=10,
            label=f'Parallel DPs MA (window={window_size}, total: {total_coordination:,})')


par1.set_ylim(2, max(total_coordination_dps) * 1.2)

# Right y-axis 2: Conflicts (offset right edge)
par2.set_ylabel('Conflict Points per Episode', fontsize=14, fontweight='heavy')
par2.axis["right"].label.set_color('#FF0000')
par2.axis["right"].major_ticks.set_color('#FF0000')
par2.axis["right"].major_ticklabels.set_color('#FF0000')
par2.axis["right"].major_ticklabels.set_fontsize(11)

# Conflicts only - red line with own axis
if len(conflict_dps) >= window_size:
    conflict_smooth = uniform_filter1d(conflict_dps, size=window_size)
    par2.plot(episode_numbers, conflict_smooth, linewidth=2.0, color='#FF0000', alpha=0.85, zorder=8,
            label=f'Conflicts MA (window={window_size}, total: {total_conflicts:,})')

# Scale conflict axis: conflict 0 aligns with parallel 2, conflict 6 aligns with parallel 4
# Parallel range 2->4 = 2 units, Conflict range 0->6 = 6 units
# So conflict_max = 6 * (parallel_max - 2) / (4 - 2) = 3 * (parallel_max - 2)
parallel_max = max(total_coordination_dps) * 1.2
conflict_max = 3 * (parallel_max - 2)
par2.set_ylim(0, conflict_max)

# Custom ticks: 0, 6, then 12, 18, 24, ...
ticks = [0, 6] + list(range(12, int(conflict_max) + 1, 6))
par2.set_yticks(ticks)

# Left legend - original with data lines (unique only)
lines1, labels1 = host.get_legend_handles_labels()
lines2, labels2 = par1.get_legend_handles_labels()
lines3, labels3 = par2.get_legend_handles_labels()

# Combine and remove duplicates
all_lines = lines1 + lines2 + lines3
all_labels = labels1 + labels2 + labels3
unique_lines = []
unique_labels = []
for line, label in zip(all_lines, all_labels):
    if label not in unique_labels:
        unique_lines.append(line)
        unique_labels.append(label)

host.legend(unique_lines, unique_labels,
           loc='upper left', fontsize=10, framealpha=0.95)

# Right legend - episode and job information
from matplotlib.lines import Line2D
# Calculate correct parallel total (simultaneous + conflicts)
total_parallel_combined = total_pure_parallel + total_conflicts
info_lines = [
    Line2D([0], [0], color='none', label=f'Total Episodes: {len(episodes):,}'),
    Line2D([0], [0], color='none', label=f'Parallel Ratio: {100*total_parallel_combined/total_dps_sum:.1f}%'),
    Line2D([0], [0], color='none', label=f'  - Simultaneity: {100*total_pure_parallel/total_dps_sum:.1f}%'),
    Line2D([0], [0], color='none', label=f'  - Conflicts: {100*total_conflicts/total_dps_sum:.1f}%'),
    Line2D([0], [0], color='none', label=f'Initial Jobs per Episode: 4'),
    Line2D([0], [0], color='none', label='Job Arrival Rate λ = 0.125')
]

# Add second legend manually
from matplotlib.legend import Legend
leg2 = Legend(host, info_lines, [l.get_label() for l in info_lines],
             loc='upper right', fontsize=10, framealpha=0.95)
leg2._legend_box.align = 'left'
host.add_artist(leg2)

plt.title('Multi-Agent Coordination: Parallel Decision Points with Conflicts', 
            fontsize=14, fontweight='bold', pad=20)

plt.subplots_adjust(right=0.90)  # Adjust plot area to show conflict axis

output_path = OUTPUT_DIR / 'parallel_with_conflicts.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved parallel with conflicts plot: {output_path}")
plt.close()
