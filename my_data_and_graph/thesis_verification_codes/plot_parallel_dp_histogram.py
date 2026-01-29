#!/usr/bin/env python3
"""
Parallel Decision Points - Agent Count Histogram
Shows agent count distribution across parallel DPs grouped by 10-episode bins
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
    """Parse timeline to get ALL parallel DPs (simultaneous + conflicts)."""
    
    parallel_dps = []
    current_episode = None
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                if current_episode is not None:
                    # 1. Find simultaneous timestamps
                    parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
                    
                    # 2. Find conflict timestamps
                    job_op_timestamps = defaultdict(list)
                    for timestamp, job_id, operation in decision_points:
                        job_op_timestamps[(job_id, operation)].append(float(timestamp))
                    
                    conflict_timestamps = set()
                    for (job_id, operation), timestamps in job_op_timestamps.items():
                        if len(timestamps) > 1:
                            for ts in timestamps[1:]:
                                conflict_timestamps.add(str(ts))
                    
                    # 3. Combine all parallel timestamps
                    all_parallel_timestamps = parallel_timestamps | conflict_timestamps
                    
                    # 4. Add ONE DP per parallel timestamp (not per agent)
                    for ts in sorted(all_parallel_timestamps, key=float):
                        actual_agent_count = timestamp_groups[ts]
                        parallel_dps.append({
                            'episode': current_episode,
                            'timestamp': float(ts),
                            'num_agents': actual_agent_count
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
            
            conflict_timestamps = set()
            for (job_id, operation), timestamps in job_op_timestamps.items():
                if len(timestamps) > 1:
                    for ts in timestamps[1:]:
                        conflict_timestamps.add(str(ts))
            
            all_parallel_timestamps = parallel_timestamps | conflict_timestamps
            for ts in sorted(all_parallel_timestamps, key=float):
                actual_agent_count = timestamp_groups[ts]
                parallel_dps.append({
                    'episode': current_episode,
                    'timestamp': float(ts),
                    'num_agents': actual_agent_count
                })
    
    return parallel_dps

def plot_parallel_histogram():
    """Create histogram visualization grouped by 10-episode bins."""
    
    print("Parsing timeline for parallel DP histogram...")
    parallel_dps = parse_parallel_dps_timeline()
    
    if not parallel_dps:
        print("No parallel decision points found!")
        return
    
    print(f"Found {len(parallel_dps)} parallel decision points")
    
    # Show RAW DATA distribution (before any filtering)
    all_agent_counts = [dp['num_agents'] for dp in parallel_dps]
    count_dist_raw = Counter(all_agent_counts)
    
    print(f"\nRAW DATA (all {len(parallel_dps)} parallel timestamps):")
    for n in sorted(count_dist_raw.keys()):
        print(f"  {n} agents: {count_dist_raw[n]:,} timestamps ({100*count_dist_raw[n]/len(all_agent_counts):.1f}%)")
    
    # Calculate if counting each agent separately
    total_individual_dps = sum(num_agents * count for num_agents, count in count_dist_raw.items())
    print(f"\nIf counting each agent separately: {total_individual_dps:,} DPs")
    
    # Extract data and filter only truly parallel (>1 agent originally)
    parallel_dps_filtered = [dp for dp in parallel_dps if dp['num_agents'] > 1]
    
    agent_counts_raw = [dp['num_agents'] for dp in parallel_dps_filtered]
    episodes = [dp['episode'] for dp in parallel_dps_filtered]
    
    print(f"\nFiltered to {len(parallel_dps_filtered)} truly parallel DPs (>1 agent)")
    
    # NO SHIFT - show actual agent counts
    agent_counts = agent_counts_raw
    
    # Group by 10-episode bins
    max_episode = max(episodes)
    bin_size = 10
    bin_data = defaultdict(list)
    
    for agent_count, episode in zip(agent_counts, episodes):
        bin_idx = episode // bin_size
        bin_data[bin_idx].append(agent_count)
    
    # Statistics
    avg_agents = np.mean(agent_counts)
    count_dist = Counter(agent_counts)
    
    print(f"\nStatistics (Shifted Display):")
    print(f"  Average agents: {avg_agents:.2f}")
    print(f"  Distribution:")
    for n in sorted(count_dist.keys()):
        print(f"    {n} agents: {count_dist[n]} DPs ({100*count_dist[n]/len(agent_counts):.1f}%)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 10))
    
    # Prepare histogram data
    bin_labels = []
    histograms = {2: [], 3: [], 4: []}
    
    for bin_idx in sorted(bin_data.keys()):
        start_ep = bin_idx * bin_size
        end_ep = min((bin_idx + 1) * bin_size - 1, max_episode)
        bin_labels.append(f'{start_ep}-{end_ep}')
        
        bin_counts = Counter(bin_data[bin_idx])
        histograms[2].append(bin_counts.get(2, 0))
        histograms[3].append(bin_counts.get(3, 0))
        histograms[4].append(bin_counts.get(4, 0))
    
    # Plot stacked bars
    x_pos = np.arange(len(bin_labels))
    width = 0.7
    
    ax.bar(x_pos, histograms[2], width, label='2 Agents', 
           color='#F18F01', alpha=0.85, edgecolor='black', linewidth=1)
    ax.bar(x_pos, histograms[3], width, bottom=histograms[2], 
           label='3 Agents', color='#A23B72', alpha=0.85, edgecolor='black', linewidth=1)
    
    bottoms_4 = [h2 + h3 for h2, h3 in zip(histograms[2], histograms[3])]
    ax.bar(x_pos, histograms[4], width, bottom=bottoms_4, 
           label='4 Agents', color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=1)
    
    # Labels
    ax.set_xlabel('Episode Range (10-Episode Bins)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Parallel Decision Points', fontsize=13, fontweight='bold')
    ax.set_title('Agent Count Distribution in Parallel DPs per Episode Range', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x_pos[::2])
    ax.set_xticklabels([bin_labels[i] for i in range(0, len(bin_labels), 2)], 
                       rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, fancybox=True)
    
    # Statistics box
    stats_text = f'''Overall Statistics:
Total Parallel DPs: {len(parallel_dps):,}
Episodes: {min(episodes)} - {max(episodes)}

Agent Count (Displayed):
  Average: {avg_agents:.2f}
  Min: {min(agent_counts)}
  Max: {max(agent_counts)}

Distribution:
  2 agents: {count_dist.get(2, 0):,} ({100*count_dist.get(2, 0)/len(agent_counts):.1f}%)
  3 agents: {count_dist.get(3, 0):,} ({100*count_dist.get(3, 0)/len(agent_counts):.1f}%)
  4 agents: {count_dist.get(4, 0):,} ({100*count_dist.get(4, 0)/len(agent_counts):.1f}%)'''
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, 
                edgecolor='black', linewidth=1.5, pad=0.7)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='left',
            bbox=props, family='monospace', fontweight='normal')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'parallel_dp_agent_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved visualization: {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_parallel_histogram()
