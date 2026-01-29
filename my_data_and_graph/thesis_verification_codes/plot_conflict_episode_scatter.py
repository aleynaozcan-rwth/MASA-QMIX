#!/usr/bin/env python3
"""
Episode-based scatter plot showing conflict distribution across training episodes.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_conflicts_data():
    """Parse timeline to get parallel DPs and conflicts per episode."""
    
    episodes_data = []
    current_episode = None
    timestamp_groups = defaultdict(int)
    decision_points = []
    in_lifecycle = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                if current_episode is not None:
                    # Calculate parallel DPs (simultaneity)
                    parallel_count = sum(count for count in timestamp_groups.values() if count > 1)
                    
                    # Detect conflicts (job-op appearing multiple times)
                    job_op_timestamps = defaultdict(list)
                    for timestamp, job_id, operation in decision_points:
                        job_op_timestamps[(job_id, operation)].append(float(timestamp))
                    
                    # Count conflict DPs (retry attempts)
                    conflict_dps = set()
                    for (job_id, operation), timestamps in job_op_timestamps.items():
                        if len(timestamps) > 1:
                            for ts in timestamps[1:]:
                                conflict_dps.add((str(ts), job_id, operation))
                    
                    num_conflicts = len(conflict_dps)
                    
                    episodes_data.append({
                        'episode': current_episode,
                        'parallel_dps': parallel_count,
                        'conflicts': num_conflicts
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
    
    # Handle last episode
    if current_episode is not None:
        parallel_count = sum(count for count in timestamp_groups.values() if count > 1)
        
        job_op_timestamps = defaultdict(list)
        for timestamp, job_id, operation in decision_points:
            job_op_timestamps[(job_id, operation)].append(float(timestamp))
        
        conflict_dps = set()
        for (job_id, operation), timestamps in job_op_timestamps.items():
            if len(timestamps) > 1:
                for ts in timestamps[1:]:
                    conflict_dps.add((str(ts), job_id, operation))
        
        num_conflicts = len(conflict_dps)
        
        episodes_data.append({
            'episode': current_episode,
            'parallel_dps': parallel_count,
            'conflicts': num_conflicts
        })
    
    return episodes_data


def plot_episode_scatter():
    """Create episode-based scatter plot for conflicts and parallel DPs."""
    
    print("Parsing timeline for episode-based analysis...")
    episodes_data = parse_conflicts_data()
    print(f"Parsed {len(episodes_data)} episodes")
    
    # Extract data
    episodes = [ep['episode'] for ep in episodes_data]
    parallel_dps = np.array([ep['parallel_dps'] for ep in episodes_data])
    conflicts = np.array([ep['conflicts'] for ep in episodes_data])
    
    # Statistics
    print(f"\nEpisode-based Statistics:")
    print(f"  Parallel DPs - Mean: {np.mean(parallel_dps):.2f}, Range: {min(parallel_dps)}-{max(parallel_dps)}")
    print(f"  Conflicts - Mean: {np.mean(conflicts):.2f}, Range: {min(conflicts)}-{max(conflicts)}")
    
    # Create single figure with dual y-axis
    fig, ax1 = plt.subplots(figsize=(18, 10))
    
    # --- Left axis: Parallel DPs ---
    scatter1 = ax1.scatter(episodes, parallel_dps, c='steelblue', marker='o',
                          s=40, alpha=0.5, edgecolors='darkblue', linewidth=0.5,
                          label='Parallel DPs')
    
    # Polynomial trend for parallel DPs
    z1 = np.polyfit(episodes, parallel_dps, 4)
    p1 = np.poly1d(z1)
    x_trend = np.linspace(min(episodes), max(episodes), 300)
    line1 = ax1.plot(x_trend, p1(x_trend), color='blue', linestyle='--', 
                     linewidth=2.5, alpha=0.8, label='Parallel DPs Trend')
    
    # Mean line for parallel DPs
    mean_parallel = np.mean(parallel_dps)
    ax1.axhline(mean_parallel, color='blue', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Mean Parallel: {mean_parallel:.2f}')
    
    ax1.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Parallel Decision Points per Episode', fontsize=14, fontweight='bold', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # --- Right axis: Conflicts ---
    ax2 = ax1.twinx()
    
    scatter2 = ax2.scatter(episodes, conflicts, c='#DC143C', marker='^',
                          s=40, alpha=0.5, edgecolors='darkred', linewidth=0.5,
                          label='Conflicts')
    
    # Polynomial trend for conflicts
    z2 = np.polyfit(episodes, conflicts, 4)
    p2 = np.poly1d(z2)
    line2 = ax2.plot(x_trend, p2(x_trend), color='darkred', linestyle='--', 
                     linewidth=2.5, alpha=0.8, label='Conflicts Trend')
    
    # Mean line for conflicts
    mean_conflicts = np.mean(conflicts)
    ax2.axhline(mean_conflicts, color='purple', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Mean Conflicts: {mean_conflicts:.2f}')
    
    ax2.set_ylabel('Conflict Points per Episode', fontsize=14, fontweight='bold', color='#DC143C')
    ax2.tick_params(axis='y', labelcolor='#DC143C', labelsize=11)
    
    # Title
    ax1.set_title('Parallel Decision Points and Conflicts Across Training Episodes', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11, framealpha=0.95)
    
    # Info text
    textstr = f'Total Episodes: {len(episodes):,}\n'
    textstr += f'Parallel DPs: {np.mean(parallel_dps):.2f} ± {np.std(parallel_dps):.2f}\n'
    textstr += f'Conflicts: {np.mean(conflicts):.2f} ± {np.std(conflicts):.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'conflict_episode_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved episode scatter plot: {output_path}")


if __name__ == '__main__':
    plot_episode_scatter()
