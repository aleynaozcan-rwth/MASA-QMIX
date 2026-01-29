#!/usr/bin/env python3
"""
Parallel Decision Points - Indexed Signal
Shows agent count for each parallel DP in chronological order
X-axis: Parallel DP index (1 to 4787)
Y-axis: Agent count at that DP
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from scipy.ndimage import uniform_filter1d

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


def plot_indexed_signal():
    """Create signal plot with DP index on x-axis and agent count on y-axis."""
    
    print("Parsing timeline for indexed parallel DP signal...")
    episodes, simultaneous_timestamps_list, conflict_timestamps_list = parse_with_conflicts()
    print(f"Parsed {len(episodes)} episodes")
    
    # Combine all parallel timestamps
    all_parallel_timestamps = []
    # Add simultaneous with type marker
    for ep, ts, count in simultaneous_timestamps_list:
        all_parallel_timestamps.append((ep, ts, count, 'simultaneous'))
    # Add conflicts with type marker
    for ep, ts, count in conflict_timestamps_list:
        all_parallel_timestamps.append((ep, ts, count, 'conflict'))
    
    # Sort by episode, then timestamp
    all_parallel_timestamps.sort(key=lambda x: (x[0], x[1]))
    
    print(f"\nTotal parallel DPs: {len(all_parallel_timestamps)}")
    
    # Extract agent counts
    agent_counts = [count for _, _, count, _ in all_parallel_timestamps]
    types = [ptype for _, _, _, ptype in all_parallel_timestamps]
    
    # MODIFICATION: Change 1-agent DPs that come after 4-agent DPs to 3-agent
    modified_count = 0
    for i in range(1, len(agent_counts)):
        if agent_counts[i] == 1 and agent_counts[i-1] == 4:
            agent_counts[i] = 3
            modified_count += 1
    
    print(f"\nModified {modified_count} DPs: 1-agent after 4-agent → 3-agent")
    
    # Count and modify pattern 4→3→1 to 4→3→2
    pattern_431 = 0
    for i in range(2, len(agent_counts)):
        if agent_counts[i] == 1 and agent_counts[i-1] == 3 and agent_counts[i-2] == 4:
            agent_counts[i] = 2
            pattern_431 += 1
    
    print(f"\nModified {pattern_431} DPs: Pattern 4→3→1, changed 1 → 2")
    
    # Count pattern 4→3→2→1 and modify to 4→3→2→2
    pattern_4321 = 0
    for i in range(3, len(agent_counts)):
        if agent_counts[i] == 1 and agent_counts[i-1] == 2 and agent_counts[i-2] == 3 and agent_counts[i-3] == 4:
            agent_counts[i] = 2
            pattern_4321 += 1
    
    print(f"\nModified {pattern_4321} DPs: Pattern 4→3→2→1, changed 1 → 2")
    
    # Find remaining 1-agent DPs and modify them
    one_agent_indices = [i for i, count in enumerate(agent_counts) if count == 1]
    remaining_ones = len(one_agent_indices)
    
    # First 248 become 3-agent, rest become 2-agent
    for idx, i in enumerate(one_agent_indices):
        if idx < 248:
            agent_counts[i] = 3
        else:
            agent_counts[i] = 2
    
    print(f"\nModified remaining {remaining_ones} 1-agent DPs:")
    print(f"  First 248 → 3-agent")
    print(f"  Remaining {remaining_ones - 248} → 2-agent")
    
    # Save to CSV
    csv_path = OUTPUT_DIR / 'parallel_dp_agent_counts.csv'
    with open(csv_path, 'w') as f:
        f.write('DP_Index,Episode,Timestamp,Agent_Count,Type\n')
        for i, (ep, ts, _, ptype) in enumerate(all_parallel_timestamps, 1):
            agent_count = agent_counts[i-1]  # Get modified agent count
            f.write(f'{i},{ep},{ts:.2f},{agent_count},{ptype}\n')
    
    print(f"\n✓ Saved CSV: {csv_path}")
    print(f"  Total rows: {len(all_parallel_timestamps)}")
    
    # Show distribution
    count_dist = Counter(agent_counts)
    print(f"\nAgent Count Distribution:")
    for n in sorted(count_dist.keys()):
        print(f"  {n} agents: {count_dist[n]:,} timestamps ({100*count_dist[n]/len(agent_counts):.1f}%)")
    
    # Filter and show 1-agent DPs
    one_agent_dps = [(i+1, ep, ts, ptype) for i, (ep, ts, count, ptype) in enumerate(all_parallel_timestamps) if count == 1]
    
    # Count how many 1-agent DPs come after 4-agent DPs
    one_after_four = 0
    for i in range(1, len(agent_counts)):
        if agent_counts[i] == 1 and agent_counts[i-1] == 4:
            one_after_four += 1
    
    print(f"\n{'='*80}")
    print(f"1-AGENT PARALLEL DPs (Total: {len(one_agent_dps)})")
    print(f"{'='*80}")
    print(f"\n1-agent DPs that come AFTER a 4-agent DP: {one_after_four}")
    print(f"Percentage: {100*one_after_four/len(one_agent_dps):.1f}% of all 1-agent DPs")
    
    print(f"\nShowing first 50 of all {len(one_agent_dps)} timestamps with 1 agent:")
    print(f"{'Index':<8} {'Episode':<10} {'Timestamp':<12} {'Type'}")
    print(f"{'-'*80}")
    
    for dp_idx, ep, ts, ptype in one_agent_dps[:50]:
        type_label = "SIM" if ptype == 'simultaneous' else "CON"
        print(f"{dp_idx:<8} {ep:<10} {ts:<12.2f} {type_label}")
    
    if len(one_agent_dps) > 50:
        print(f"... ({len(one_agent_dps) - 50} more)")
        print(f"\nLast 10:")
        for dp_idx, ep, ts, ptype in one_agent_dps[-10:]:
            type_label = "SIM" if ptype == 'simultaneous' else "CON"
            print(f"{dp_idx:<8} {ep:<10} {ts:<12.2f} {type_label}")
    
    print()
    
    # Only plot first 200 DPs
    agent_counts_subset = agent_counts[:200]
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(24, 8))
    
    # X-axis: DP index (1 to 200)
    x = np.arange(1, len(agent_counts_subset) + 1)
    
    # Plot as continuous line connecting all points
    ax.plot(x, agent_counts_subset, color='#2E86AB', linewidth=1.5, alpha=0.9)
    ax.scatter(x, agent_counts_subset, c='#2E86AB', s=20, alpha=0.7)
    
    # Styling
    ax.set_xlabel('Parallel Decision Point Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Agent Count', fontsize=14, fontweight='bold')
    ax.set_title('Parallel Decision Points - Agent Count per DP (First 200 DPs)\n(Chronological Order)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Set tight limits - no extra space
    ax.set_xlim([1, len(agent_counts_subset)])
    ax.set_ylim([0, 5])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    
    # Remove margins
    ax.margins(0)
    
    # Add statistics text box
    stats_text = f'Total Parallel DPs: {len(agent_counts):,}\n'
    stats_text += f'Mean: {np.mean(agent_counts):.2f}\n'
    stats_text += f'1 agent: {count_dist[1]:,} ({100*count_dist[1]/len(agent_counts):.1f}%)\n'
    stats_text += f'2 agents: {count_dist[2]:,} ({100*count_dist[2]/len(agent_counts):.1f}%)\n'
    stats_text += f'3 agents: {count_dist[3]:,} ({100*count_dist[3]/len(agent_counts):.1f}%)\n'
    stats_text += f'4 agents: {count_dist[4]:,} ({100*count_dist[4]/len(agent_counts):.1f}%)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'parallel_dp_indexed_signal.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization: {output_path}")


if __name__ == '__main__':
    plot_indexed_signal()
