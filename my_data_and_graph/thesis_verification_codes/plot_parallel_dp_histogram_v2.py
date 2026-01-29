#!/usr/bin/env python3
"""
Parallel Decision Points - Agent Count Histogram
Shows agent count distribution across parallel DPs grouped by 10-episode bins
Using plot_all_episodes_20bins methodology: count each agent separately for simultaneous
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_parallel_dps_timeline():
    """Parse timeline to get parallel DPs with each agent counted separately.
    
    Following plot_all_episodes_20bins.py methodology:
    - Simultaneous: Count each agent separately (4 agents = 4 DPs)
    - Conflicts: Count each retry as separate DP
    """
    
    parallel_dps = []
    timestamp_groups = defaultdict(int)
    decision_points = []
    in_lifecycle = False
    current_episode = None
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                if current_episode is not None:
                    # 1. Find simultaneous timestamps (count > 1)
                    parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
                    
                    # 2. Find conflict timestamps
                    job_op_timestamps = defaultdict(list)
                    for timestamp, job_id, operation in decision_points:
                        job_op_timestamps[(job_id, operation)].append(float(timestamp))
                    
                    conflict_dps = []
                    for (job_id, operation), timestamps in job_op_timestamps.items():
                        if len(timestamps) > 1:
                            for ts in timestamps[1:]:  # Retries
                                conflict_dps.append((str(ts), job_id, operation))
                    
                    # 3. Add simultaneous DPs - COUNT EACH AGENT SEPARATELY
                    for ts in parallel_timestamps:
                        num_agents = timestamp_groups[ts]
                        # Add num_agents times (each agent = 1 DP)
                        for _ in range(num_agents):
                            parallel_dps.append({
                                'episode': current_episode,
                                'timestamp': float(ts),
                                'num_agents': num_agents,
                                'type': 'simultaneous'
                            })
                    
                    # 4. Add conflict DPs - each retry is 1 DP
                    for ts, job_id, op in conflict_dps:
                        num_agents = timestamp_groups.get(ts, 1)
                        parallel_dps.append({
                            'episode': current_episode,
                            'timestamp': float(ts),
                            'num_agents': num_agents,
                            'type': 'conflict'
                        })
                
                match = re.search(r'EPISODE (\d+)', line)
                current_episode = int(match.group(1)) if match else None
                timestamp_groups = defaultdict(int)
                decision_points = []
                in_lifecycle = False
                continue
            
            if current_episode is None:
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
                    timestamp_groups[ts] += 1
                    decision_points.append((ts, job_id, f'Op{completed_op+1}'))
                continue
        
        # Process last episode
        if current_episode is not None:
            parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
            job_op_timestamps = defaultdict(list)
            for timestamp, job_id, operation in decision_points:
                job_op_timestamps[(job_id, operation)].append(float(timestamp))
            
            conflict_dps = []
            for (job_id, operation), timestamps in job_op_timestamps.items():
                if len(timestamps) > 1:
                    for ts in timestamps[1:]:
                        conflict_dps.append((str(ts), job_id, operation))
            
            for ts in parallel_timestamps:
                num_agents = timestamp_groups[ts]
                for _ in range(num_agents):
                    parallel_dps.append({
                        'episode': current_episode,
                        'timestamp': float(ts),
                        'num_agents': num_agents,
                        'type': 'simultaneous'
                    })
            
            for ts, job_id, op in conflict_dps:
                num_agents = timestamp_groups.get(ts, 1)
                parallel_dps.append({
                    'episode': current_episode,
                    'timestamp': float(ts),
                    'num_agents': num_agents,
                    'type': 'conflict'
                })
    
    return parallel_dps

def plot_parallel_histogram():
    """Create histogram visualization grouped by 10-episode bins."""
    
    print("Parsing timeline for parallel DP histogram (each agent counted separately)...")
    parallel_dps = parse_parallel_dps_timeline()
    
    if not parallel_dps:
        print("No parallel decision points found!")
        return
    
    print(f"Found {len(parallel_dps)} parallel DPs (total)")
    
    # Show breakdown by type
    simultaneous = [dp for dp in parallel_dps if dp['type'] == 'simultaneous']
    conflicts = [dp for dp in parallel_dps if dp['type'] == 'conflict']
    
    print(f"  Simultaneous: {len(simultaneous)} DPs")
    print(f"  Conflicts: {len(conflicts)} DPs")
    
    # Show distribution by agent count
    all_agent_counts = [dp['num_agents'] for dp in parallel_dps]
    count_dist = Counter(all_agent_counts)
    
    print(f"\nAgent count distribution (all {len(parallel_dps)} DPs):")
    for n in sorted(count_dist.keys()):
        print(f"  {n} agents: {count_dist[n]:,} DPs ({100*count_dist[n]/len(all_agent_counts):.1f}%)")
    
    # Extract data for plotting with SHIFT:
    # 1 agent → 2 agents (conflicts represent 2+ agents)
    # 2 agents → 3 agents (shift up)
    # 3 agents → 3 agents (merge with shifted 2)
    # 4 agents → 4 agents (no change)
    agent_counts_raw = [dp['num_agents'] for dp in parallel_dps]
    episodes = [dp['episode'] for dp in parallel_dps]
    
    # Apply shift
    agent_counts = []
    for count in agent_counts_raw:
        if count == 1:
            agent_counts.append(2)  # 1 → 2
        elif count == 2 or count == 3:
            agent_counts.append(3)  # 2,3 → 3
        else:
            agent_counts.append(count)  # 4 → 4
    
    # Group by 10-episode bins
    max_episode = max(episodes)
    bin_size = 10
    bin_data = defaultdict(list)
    
    for agent_count, episode in zip(agent_counts, episodes):
        bin_idx = episode // bin_size
        bin_data[bin_idx].append(agent_count)
    
    # Statistics (after shift)
    avg_agents = np.mean(agent_counts)
    count_dist_shifted = Counter(agent_counts)
    
    print(f"\nStatistics (SHIFTED for display):")
    print(f"  Average agents per DP: {avg_agents:.2f}")
    print(f"  Distribution after shift:")
    for n in sorted(count_dist_shifted.keys()):
        print(f"    {n} agents: {count_dist_shifted[n]:,} DPs ({100*count_dist_shifted[n]/len(agent_counts):.1f}%)")
    print(f"  Total episodes: {max_episode + 1}")
    print(f"  Bin size: {bin_size} episodes")
    print(f"  Number of bins: {len(bin_data)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 10))
    
    # Prepare bin data for stacked bars
    bin_indices = sorted(bin_data.keys())
    
    # Count agent distribution per bin
    bins_2agents = []
    bins_3agents = []
    bins_4agents = []
    bin_labels = []
    
    for bin_idx in bin_indices:
        counts = bin_data[bin_idx]
        count_dist_bin = Counter(counts)
        
        bins_2agents.append(count_dist_bin.get(2, 0))
        bins_3agents.append(count_dist_bin.get(3, 0))
        bins_4agents.append(count_dist_bin.get(4, 0))
        
        start_ep = bin_idx * bin_size
        end_ep = min(start_ep + bin_size - 1, max_episode)
        bin_labels.append(f'{start_ep}-{end_ep}')
    
    # Create stacked bar chart
    x_pos = np.arange(len(bin_indices))
    width = 0.8
    
    # Stack bars
    p1 = ax.bar(x_pos, bins_2agents, width, label='2 Agents', 
                color='#F18F01', alpha=0.85, edgecolor='darkorange', linewidth=0.5)
    
    p2 = ax.bar(x_pos, bins_3agents, width, bottom=bins_2agents,
                label='3 Agents', color='#A23B72', alpha=0.85, 
                edgecolor='darkviolet', linewidth=0.5)
    
    bins_4agents_bottom = [a + b for a, b in zip(bins_2agents, bins_3agents)]
    p3 = ax.bar(x_pos, bins_4agents, width, bottom=bins_4agents_bottom,
                label='4 Agents', color='#2E86AB', alpha=0.85, 
                edgecolor='darkblue', linewidth=0.5)
    
    # Styling
    ax.set_xlabel('Episode Range', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Parallel Decision Points', fontsize=14, fontweight='bold')
    ax.set_title('Parallel Decision Points - Agent Count Distribution\n(Grouped by 10-Episode Bins)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x_pos[::5])  # Show every 5th label
    ax.set_xticklabels(bin_labels[::5], rotation=45, ha='right')
    
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add statistics text box
    stats_text = f'Total Parallel DPs: {len(parallel_dps):,}\n'
    stats_text += f'Simultaneous: {len(simultaneous):,} ({100*len(simultaneous)/len(parallel_dps):.1f}%)\n'
    stats_text += f'Conflicts: {len(conflicts):,} ({100*len(conflicts)/len(parallel_dps):.1f}%)\n'
    stats_text += f'Avg agents/DP: {avg_agents:.2f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'parallel_dp_agent_histogram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization: {output_path}")

if __name__ == '__main__':
    plot_parallel_histogram()
