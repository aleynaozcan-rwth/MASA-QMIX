#!/usr/bin/env python3
"""
Total Decision Points only - no parallel or conflict lines.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from collections import defaultdict

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

print("Parsing timeline for total decision points...")
episodes = parse_with_conflicts()
print(f"Parsed {len(episodes)} episodes")

# Extract data
episode_numbers = np.array([ep['episode'] for ep in episodes])
total_dps = np.array([ep['total_dps'] for ep in episodes])

# Calculate statistics
total_dps_sum = np.sum(total_dps)

print(f"\nTotal Decision Points Statistics:")
print(f"  Total DPs: {total_dps_sum:,}")
print(f"  Mean DPs per episode: {np.mean(total_dps):.2f}")
print(f"  Std Dev: {np.std(total_dps):.2f}")
print(f"  Range: {np.min(total_dps)} - {np.max(total_dps)}")

# Create simple visualization with only total DPs
fig, ax = plt.subplots(figsize=(20, 10))

# Left y-axis: Total DPs
ax.set_xlabel('Episode', fontsize=18, fontweight='heavy')
ax.set_ylabel('Total Decision Points per Episode', fontsize=14, fontweight='heavy')

# Smooth total data
window_size = 5
if len(total_dps) >= window_size:
    total_smooth = uniform_filter1d(total_dps, size=window_size)
    ax.plot(episode_numbers, total_smooth, linewidth=2, color='#2E86C1', alpha=0.9, zorder=3,
            label=f'Total DPs MA (window={window_size}, total: {total_dps_sum:,})')

ax.set_xlim(0, len(episode_numbers) - 1)
# Y-axis from 15 to 50
ax.set_ylim(15, 50)
ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

# Right legend - episode information
from matplotlib.lines import Line2D
info_lines = [
    Line2D([0], [0], color='none', label=f'Total Episodes: {len(episodes):,}'),
    Line2D([0], [0], color='none', label=f'Initial Jobs per Episode: 4'),
    Line2D([0], [0], color='none', label='Job Arrival Rate λ = 0.125')
]

from matplotlib.legend import Legend
leg2 = Legend(ax, info_lines, [l.get_label() for l in info_lines],
             loc='upper right', fontsize=10, framealpha=0.95)
leg2._legend_box.align = 'left'
ax.add_artist(leg2)

plt.title('Total Decision Points per Episode', 
            fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()

output_path = OUTPUT_DIR / 'total_dps_only.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved total DPs only plot: {output_path}")
plt.close()
