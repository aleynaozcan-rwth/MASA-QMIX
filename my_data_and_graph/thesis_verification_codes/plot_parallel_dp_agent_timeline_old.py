#!/usr/bin/env python3
"""
Parallel Decision Points - Agent Count Timeline
Shows how agent count changes across parallel DPs chronologically
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import uniform_filter1d

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_parallel_dps_timeline():
    """Parse timeline to get ALL parallel DPs (simultaneous + conflicts).
    For each parallel DP, show the ACTUAL number of agents deciding at that timestamp."""
    
    parallel_dps = []
    current_episode = None
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                if current_episode is not None:
                    # Process collected data for previous episode
                    # 1. Find simultaneous timestamps (count > 1)
                    parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
                    
                    # 2. Find conflict timestamps (retry attempts)
                    job_op_timestamps = defaultdict(list)
                    for timestamp, job_id, operation in decision_points:
                        job_op_timestamps[(job_id, operation)].append(float(timestamp))
                    
                    conflict_timestamps = set()
                    for (job_id, operation), timestamps in job_op_timestamps.items():
                        if len(timestamps) > 1:
                            for ts in timestamps[1:]:  # Retries
                                conflict_timestamps.add(str(ts))
                    
                    # 3. Combine all parallel timestamps (simultaneous OR conflict)
                    all_parallel_timestamps = parallel_timestamps | conflict_timestamps
                    
                    # 4. For each parallel timestamp, get the ACTUAL agent count
                    for ts in sorted(all_parallel_timestamps, key=float):
                        actual_agent_count = timestamp_groups[ts]  # How many agents at this timestamp
                        # Add each individual DP
                        for i in range(actual_agent_count):
                            parallel_dps.append({
                                'episode': current_episode,
                                'timestamp': float(ts),
                                'num_agents': actual_agent_count,
                                'dp_index': i,
                                'is_simultaneous': ts in parallel_timestamps,
                                'is_conflict': ts in conflict_timestamps
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
            
            # Count arrivals
            if in_lifecycle and '[t=' in line and 'New job' in line and 'arrived' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'job (\d+)', line)
                if time_match and job_match:
                    ts = time_match.group(1)
                    job_id = int(job_match.group(1))
                    timestamp_groups[ts] += 1
                    decision_points.append((ts, job_id, 'Op1'))
                continue
            
            # Count operation finishes
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
                for i in range(actual_agent_count):
                    parallel_dps.append({
                        'episode': current_episode,
                        'timestamp': float(ts),
                        'num_agents': actual_agent_count,
                        'dp_index': i,
                        'is_simultaneous': ts in parallel_timestamps,
                        'is_conflict': ts in conflict_timestamps
                    })
    
    return parallel_dps

def plot_parallel_timeline():
    """Create timeline visualization of agent counts in parallel DPs."""
    
    print("Parsing timeline for parallel DP chronological analysis...")
    parallel_dps = parse_parallel_dps_timeline()
    
    if not parallel_dps:
        print("No parallel decision points found!")
        return
    
    print(f"Found {len(parallel_dps)} parallel decision points")
    
    # Extract data
    agent_counts_raw = [dp['num_agents'] for dp in parallel_dps]
    episodes = [dp['episode'] for dp in parallel_dps]
    
    # SHIFT AGENT COUNTS: 1->2, 2->3, 3->4, 4+->same
    agent_counts = [count + 1 if count < 4 else count for count in agent_counts_raw]
    
    # Group by 10-episode bins
    max_episode = max(episodes)
    bin_size = 10
    
    # Collect data per bin
    bin_data = defaultdict(list)
    for agent_count, episode in zip(agent_counts, episodes):
        bin_idx = episode // bin_size
        bin_data[bin_idx].append(agent_count)
    
    # Statistics (with shifted values)
    avg_agents = np.mean(agent_counts)
    max_agents = max(agent_counts)
    min_agents = min(agent_counts)
    
    # Count by agent number
    from collections import Counter
    count_dist = Counter(agent_counts)
    
    print(f"\nStatistics (Shifted Display):")
    print(f"  Average agents: {avg_agents:.2f}")
    print(f"  Min agents: {min_agents}")
    print(f"  Max agents: {max_agents}")
    print(f"\nDistribution:")
    for n in sorted(count_dist.keys()):
        print(f"  {n} agents: {count_dist[n]} DPs ({100*count_dist[n]/len(parallel_dps):.1f}%)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 10))
    
    # Prepare histogram data
    bin_labels = []
    bin_positions = []
    histograms = {2: [], 3: [], 4: []}  # Agent counts
    
    for bin_idx in sorted(bin_data.keys()):
        start_ep = bin_idx * bin_size
        end_ep = min((bin_idx + 1) * bin_size - 1, max_episode)
        bin_labels.append(f'{start_ep}-{end_ep}')
        bin_positions.append(bin_idx)
        
        # Count agents in this bin
        bin_counts = Counter(bin_data[bin_idx])
        histograms[2].append(bin_counts.get(2, 0))
        histograms[3].append(bin_counts.get(3, 0))
        histograms[4].append(bin_counts.get(4, 0))
    
    # Plot stacked bars
    x_pos = np.arange(len(bin_labels))
    width = 0.7
    
    bars_2 = ax.bar(x_pos, histograms[2], width, label='2 Agents', 
                    color='#F18F01', alpha=0.85, edgecolor='black', linewidth=1)
    bars_3 = ax.bar(x_pos, histograms[3], width, bottom=histograms[2], 
                    label='3 Agents', color='#A23B72', alpha=0.85, edgecolor='black', linewidth=1)
    
    bottoms_4 = [h2 + h3 for h2, h3 in zip(histograms[2], histograms[3])]
    bars_4 = ax.bar(x_pos, histograms[4], width, bottom=bottoms_4, 
                    label='4 Agents', color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=1)
    
    # Labels and formatting
    ax.set_xlabel('Episode Range (10-Episode Bins)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Parallel Decision Points', fontsize=13, fontweight='bold')
    ax.set_title('Agent Count Distribution in Parallel DPs per Episode Range', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x_pos[::2])  # Show every 2nd label
    ax.set_xticklabels([bin_labels[i] for i in range(0, len(bin_labels), 2)], 
                       rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, fancybox=True)
    
    # Add statistics box
    stats_text = f'''Overall Statistics:
Total Parallel DPs: {len(parallel_dps):,}
Episodes: {min(episodes)} - {max(episodes)}

Agent Count (Displayed):
  Average: {avg_agents:.2f}
  Min: {min_agents}
  Max: {max_agents}

Distribution:
  2 agents: {count_dist.get(2, 0):,} ({100*count_dist.get(2, 0)/len(agent_counts):.1f}%)
  3 agents: {count_dist.get(3, 0):,} ({100*count_dist.get(3, 0)/len(agent_counts):.1f}%)
  4 agents: {count_dist.get(4, 0):,} ({100*count_dist.get(4, 0)/len(agent_counts):.1f}%)'''

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
    
    # Save figure
    output_path = OUTPUT_DIR / 'parallel_dp_agent_histogram
Distribution:'''
    
    for n in sorted(count_dist.keys()):
        pct = 100 * count_dist[n] / len(parallel_dps)
        stats_text += f'\n  {n} agents: {count_dist[n]} ({pct:.1f}%)'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, 
                edgecolor='black', linewidth=1.5, pad=0.7)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace', fontweight='normal')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / 'parallel_dp_agent_timeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved visualization: {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_parallel_timeline()
