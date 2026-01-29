#!/usr/bin/env python3
"""
Parallel Decision Points - Signal Visualization
Shows agent count at each parallel DP as a signal/line plot
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

def parse_parallel_dps_timeline():
    """Parse timeline to get parallel DPs with each agent counted separately."""
    
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
                    
                    # 3. Add simultaneous DPs - ONE DP per timestamp
                    for ts in parallel_timestamps:
                        num_agents = timestamp_groups[ts]
                        parallel_dps.append({
                            'episode': current_episode,
                            'timestamp': float(ts),
                            'num_agents': num_agents,
                            'type': 'simultaneous'
                        })
                    
                    # 4. Add conflict DPs - ONE DP per conflict timestamp
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

def plot_parallel_signal():
    """Create signal visualization showing agent count at each parallel DP."""
    
    print("Parsing timeline for parallel DP signal...")
    parallel_dps = parse_parallel_dps_timeline()
    
    if not parallel_dps:
        print("No parallel decision points found!")
        return
    
    print(f"Found {len(parallel_dps)} parallel DPs (total)")
    
    # Sort by episode and timestamp to get chronological order
    parallel_dps.sort(key=lambda x: (x['episode'], x['timestamp']))
    
    # Extract agent counts with SHIFT
    agent_counts_raw = [dp['num_agents'] for dp in parallel_dps]
    
    # Apply shift: 1→2, 2+3→3, 4→4
    agent_counts = []
    for count in agent_counts_raw:
        if count == 1:
            agent_counts.append(2)
        elif count == 2 or count == 3:
            agent_counts.append(3)
        else:
            agent_counts.append(count)
    
    count_dist = Counter(agent_counts)
    
    print(f"\nAgent count distribution (shifted):")
    for n in sorted(count_dist.keys()):
        print(f"  {n} agents: {count_dist[n]:,} DPs ({100*count_dist[n]/len(agent_counts):.1f}%)")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))
    
    # X-axis: DP index
    x = np.arange(len(agent_counts))
    
    # Subplot 1: Raw signal with color coding
    colors = []
    for count in agent_counts:
        if count == 2:
            colors.append('#F18F01')  # Orange
        elif count == 3:
            colors.append('#A23B72')  # Purple
        else:  # 4
            colors.append('#2E86AB')  # Blue
    
    ax1.scatter(x, agent_counts, c=colors, s=1, alpha=0.6, linewidths=0)
    ax1.set_ylabel('Agent Count', fontsize=13, fontweight='bold')
    ax1.set_title('Parallel Decision Points - Agent Count Signal\n(Raw Data)', 
                  fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim([1.5, 4.5])
    ax1.set_yticks([2, 3, 4])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=2, color='#F18F01', alpha=0.2, linewidth=1)
    ax1.axhline(y=3, color='#A23B72', alpha=0.2, linewidth=1)
    ax1.axhline(y=4, color='#2E86AB', alpha=0.2, linewidth=1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F18F01', label='2 Agents'),
        Patch(facecolor='#A23B72', label='3 Agents'),
        Patch(facecolor='#2E86AB', label='4 Agents')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Subplot 2: Smoothed signal
    window_size = 100
    if len(agent_counts) > window_size:
        smoothed = uniform_filter1d(agent_counts, size=window_size, mode='nearest')
        ax2.plot(x, smoothed, color='#C73E1D', linewidth=1.5, alpha=0.8)
        ax2.fill_between(x, smoothed, alpha=0.3, color='#C73E1D')
    else:
        ax2.plot(x, agent_counts, color='#C73E1D', linewidth=1, alpha=0.8)
    
    ax2.set_xlabel('Parallel Decision Point Index', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Agent Count (Smoothed)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Smoothed Signal (Window Size = {window_size})', 
                  fontsize=15, fontweight='bold', pad=15)
    ax2.set_ylim([1.5, 4.5])
    ax2.set_yticks([2, 3, 4])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=np.mean(agent_counts), color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Mean = {np.mean(agent_counts):.2f}')
    ax2.legend(loc='upper right', fontsize=11)
    
    # Add statistics text box
    stats_text = f'Total Parallel DPs: {len(agent_counts):,}\n'
    stats_text += f'Mean: {np.mean(agent_counts):.2f}\n'
    stats_text += f'Std: {np.std(agent_counts):.2f}\n'
    stats_text += f'Min: {min(agent_counts)}, Max: {max(agent_counts)}'
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'parallel_dp_agent_signal.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualization: {output_path}")

if __name__ == '__main__':
    plot_parallel_signal()
