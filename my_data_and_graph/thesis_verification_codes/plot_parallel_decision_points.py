#!/usr/bin/env python3
"""
Enhanced Parallel Decision Points: Include both simultaneous decisions AND conflict retries.
Both represent coordination challenges that MARL must address.
NO CONFLICT LINE - just the combined total.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
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
    
    # Track simultaneous timestamps with agent counts and episode info
    simultaneous_timestamps_list = []  # [(episode, timestamp, agent_count), ...]
    conflict_timestamps_list = []  # [(episode, timestamp, agent_count), ...]
    
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
                    
                    # Collect conflict timestamps with agent counts
                    for ts, job_id, op in conflict_dps:
                        agent_count = timestamp_groups.get(ts, 1)
                        conflict_timestamps_list.append((current_episode, float(ts), agent_count))
                    
                    # Combine: pure parallel + conflict retries
                    parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
                    
                    # Collect timestamp and agent count for each simultaneous timestamp
                    for ts in parallel_timestamps:
                        agent_count = timestamp_groups[ts]
                        simultaneous_timestamps_list.append((current_episode, float(ts), agent_count))
                    
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
        
        # Collect conflict timestamps for last episode
        for ts, job_id, op in conflict_dps:
            agent_count = timestamp_groups.get(ts, 1)
            conflict_timestamps_list.append((current_episode, float(ts), agent_count))
        
        parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
        
        # Collect timestamp and agent count for last episode
        for ts in parallel_timestamps:
            agent_count = timestamp_groups[ts]
            simultaneous_timestamps_list.append((current_episode, float(ts), agent_count))
        
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
    
    return episodes, simultaneous_timestamps_list, conflict_timestamps_list

print("Parsing timeline for enhanced parallel decision points...")
episodes, simultaneous_timestamps_list, conflict_timestamps_list = parse_with_conflicts()
print(f"Parsed {len(episodes)} episodes")

# Show simultaneous agent count distribution
from collections import Counter

# Extract just agent counts for distribution
simultaneous_agent_counts = [agent_count for _, _, agent_count in simultaneous_timestamps_list]
agent_count_dist = Counter(simultaneous_agent_counts)

print(f"\nSimultaneous Decision Points - Agent Count Distribution:")
print(f"  Total simultaneous timestamps: {len(simultaneous_timestamps_list)}")
for n in sorted(agent_count_dist.keys()):
    print(f"    {n} agents: {agent_count_dist[n]} timestamps")

print(f"\nVerification: {sum(n * count for n, count in agent_count_dist.items())} total DPs")
print(f"  (should match Pure Simultaneity DPs)")

# Show first 20 simultaneous timestamps as examples
print(f"\nFirst 20 simultaneous timestamps (episode, timestamp, agent_count):")
for i, (ep, ts, count) in enumerate(simultaneous_timestamps_list[:20], 1):
    print(f"  {i}. Episode {ep}, t={ts:.2f}, {count} agents")

# Show 3-agent timestamps specifically
three_agent_timestamps = [(ep, ts, count) for ep, ts, count in simultaneous_timestamps_list if count == 3]
print(f"\n3-Agent Simultaneous Timestamps (total: {len(three_agent_timestamps)}):")
for ep, ts, count in three_agent_timestamps:
    print(f"  Episode {ep}, t={ts:.2f}, {count} agents")

# Show conflict timestamps
conflict_agent_counts = [agent_count for _, _, agent_count in conflict_timestamps_list]
conflict_count_dist = Counter(conflict_agent_counts)

print(f"\n{'='*60}")
print(f"Conflict Decision Points - Agent Count Distribution:")
print(f"  Total conflict timestamps: {len(conflict_timestamps_list)}")
for n in sorted(conflict_count_dist.keys()):
    print(f"    {n} agents: {conflict_count_dist[n]} timestamps")

print(f"\nFirst 20 conflict timestamps (episode, timestamp, agent_count):")
for i, (ep, ts, count) in enumerate(conflict_timestamps_list[:20], 1):
    print(f"  {i}. Episode {ep}, t={ts:.2f}, {count} agents")

# Combine all parallel timestamps and sort chronologically
print(f"\n{'='*60}")
print(f"COMBINED ALL PARALLEL TIMESTAMPS (Simultaneous + Conflicts)")
print(f"{'='*60}")

all_parallel_timestamps = []
# Add simultaneous with type marker
for ep, ts, count in simultaneous_timestamps_list:
    all_parallel_timestamps.append((ep, ts, count, 'simultaneous'))
# Add conflicts with type marker
for ep, ts, count in conflict_timestamps_list:
    all_parallel_timestamps.append((ep, ts, count, 'conflict'))

# Sort by episode, then timestamp
all_parallel_timestamps.sort(key=lambda x: (x[0], x[1]))

print(f"\nTotal parallel timestamps: {len(all_parallel_timestamps)}")
print(f"  Simultaneous: {len(simultaneous_timestamps_list)}")
print(f"  Conflicts: {len(conflict_timestamps_list)}")

# Show agent count distribution for ALL parallel timestamps
all_agent_counts = [count for _, _, count, _ in all_parallel_timestamps]
all_count_dist = Counter(all_agent_counts)

print(f"\nAgent Count Distribution (ALL 4,787 parallel timestamps):")
for n in sorted(all_count_dist.keys()):
    print(f"  {n} agents: {all_count_dist[n]:,} timestamps ({100*all_count_dist[n]/len(all_parallel_timestamps):.1f}%)")

# Show first 50 in chronological order
print(f"\nFirst 50 parallel timestamps (chronological order):")
for i, (ep, ts, count, ptype) in enumerate(all_parallel_timestamps[:50], 1):
    type_label = "SIM" if ptype == 'simultaneous' else "CON"
    print(f"  {i}. Ep{ep:4d} t={ts:6.2f} [{type_label}] {count} agents")

print()

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

print(f"\nParallel Decision Points Statistics:")
print(f"  Pure Simultaneity DPs: {total_pure_parallel:,} ({100*total_pure_parallel/total_dps_sum:.1f}%)")
print(f"  Conflict Retry DPs: {total_conflicts:,} ({100*total_conflicts/total_dps_sum:.1f}%)")
print(f"  Total Coordination DPs: {total_coordination:,} ({100*total_coordination/total_dps_sum:.1f}%)")
print(f"  Total DPs: {total_dps_sum:,}")
print(f"  Mean Coordination DPs per episode: {mean_coordination:.2f}")
print(f"  Std Dev: {std_coordination:.2f}")
print(f"  Range: {np.min(total_coordination_dps)} - {np.max(total_coordination_dps)}")

# Create visualization
fig, ax1 = plt.subplots(figsize=(16, 7))

# Left y-axis: Total DPs
ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Decision Points per Episode', fontsize=12, fontweight='bold', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

# Total DPs as area fill
ax1.fill_between(episode_numbers, 0, total_dps, color='steelblue', alpha=0.2, zorder=1)

# Smooth total data
sigma_total = 1.0
if len(total_dps) >= 5:
    total_smooth = gaussian_filter1d(total_dps, sigma=sigma_total)
    ax1.plot(episode_numbers, total_smooth, linewidth=2, color='steelblue', alpha=0.7, zorder=3,
            label=f'Total DPs MA (σ={sigma_total}, total: {total_dps_sum:,})')

ax1.set_xlim(0, len(episode_numbers) - 1)
ax1.set_ylim(0, None)
ax1.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=0)

# Right y-axis: Coordination DPs
ax2 = ax1.twinx()
ax2.set_ylabel('Parallel Decision Points per Episode', fontsize=12, fontweight='bold', color='#DC143C')
ax2.tick_params(axis='y', labelcolor='#DC143C')

# Coordination DPs (parallel + conflicts) - red line
sigma_coord = 1.0
if len(total_coordination_dps) >= 5:
    coord_smooth = gaussian_filter1d(total_coordination_dps, sigma=sigma_coord)
    ax2.plot(episode_numbers, coord_smooth, linewidth=2.5, color='#DC143C', alpha=0.95, zorder=10,
            label=f'Parallel DPs MA (σ={sigma_coord}, total: {total_coordination:,})')

# Scatter
ax2.scatter(episode_numbers, total_coordination_dps, s=3, color='#DC143C', alpha=0.3, zorder=9)

# Mean line
ax2.axhline(y=mean_coordination, color='green', linestyle='--', linewidth=2, alpha=0.9, zorder=11,
          label=f'Parallel Mean: {mean_coordination:.2f}')

# Std shading
ax2.axhspan(mean_coordination - std_coordination, mean_coordination + std_coordination, 
          color='green', alpha=0.15, zorder=8,
          label=f'Parallel ±1σ: [{mean_coordination-std_coordination:.1f}, {mean_coordination+std_coordination:.1f}]')

ax2.set_ylim(2, max(total_coordination_dps) * 1.2)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='none', label=f'Total Episodes: {len(episodes):,}'),
    Line2D([0], [0], color='none', label=f'Parallel Ratio: {100*total_coordination/total_dps_sum:.1f}%'),
    Line2D([0], [0], color='none', label=f'  - Simultaneity: {100*total_pure_parallel/total_dps_sum:.1f}%'),
    Line2D([0], [0], color='none', label=f'  - Conflicts: {100*total_conflicts/total_dps_sum:.1f}%'),
    Line2D([0], [0], color='none', label=f'Initial Jobs per Episode: 4'),
    Line2D([0], [0], color='none', label='Job Arrival Rate λ = 0.125')
]

ax1.legend(lines1 + lines2 + custom_lines, labels1 + labels2 + [l.get_label() for l in custom_lines],
          loc='upper right', fontsize=10, framealpha=0.95)

fig.suptitle('Multi-Agent Coordination: Parallel Decision Points', 
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()

output_path = OUTPUT_DIR / 'parallel_decision_points.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved parallel decision points plot: {output_path}")
plt.close()
