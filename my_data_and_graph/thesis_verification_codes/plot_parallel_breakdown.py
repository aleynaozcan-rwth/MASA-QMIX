#!/usr/bin/env python3
"""
Episode-based breakdown showing simultaneous vs conflict DPs within parallel decisions.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_breakdown_data():
    """Parse timeline to get simultaneous and conflict DPs separately per episode."""
    
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
                    # Calculate simultaneous DPs (pure parallel)
                    simultaneous_count = sum(count for count in timestamp_groups.values() if count > 1)
                    
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
                    
                    conflict_count = len(conflict_dps)
                    
                    episodes_data.append({
                        'episode': current_episode,
                        'simultaneous': simultaneous_count,
                        'conflicts': conflict_count,
                        'total_parallel': simultaneous_count + conflict_count
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
        simultaneous_count = sum(count for count in timestamp_groups.values() if count > 1)
        
        job_op_timestamps = defaultdict(list)
        for timestamp, job_id, operation in decision_points:
            job_op_timestamps[(job_id, operation)].append(float(timestamp))
        
        conflict_dps = set()
        for (job_id, operation), timestamps in job_op_timestamps.items():
            if len(timestamps) > 1:
                for ts in timestamps[1:]:
                    conflict_dps.add((str(ts), job_id, operation))
        
        conflict_count = len(conflict_dps)
        
        episodes_data.append({
            'episode': current_episode,
            'simultaneous': simultaneous_count,
            'conflicts': conflict_count,
            'total_parallel': simultaneous_count + conflict_count
        })
    
    return episodes_data


def plot_parallel_breakdown():
    """Create stacked area plot showing breakdown of parallel DPs."""
    
    print("Parsing timeline for parallel breakdown analysis...")
    episodes_data = parse_breakdown_data()
    print(f"Parsed {len(episodes_data)} episodes")
    
    # Extract data
    episodes = np.array([ep['episode'] for ep in episodes_data])
    simultaneous = np.array([ep['simultaneous'] for ep in episodes_data])
    conflicts = np.array([ep['conflicts'] for ep in episodes_data])
    total_parallel = np.array([ep['total_parallel'] for ep in episodes_data])
    
    # Apply smoothing
    sigma = 1.0
    simultaneous_smooth = gaussian_filter1d(simultaneous.astype(float), sigma=sigma)
    conflicts_smooth = gaussian_filter1d(conflicts.astype(float), sigma=sigma)
    total_smooth = gaussian_filter1d(total_parallel.astype(float), sigma=sigma)
    
    # Statistics
    print(f"\nParallel DPs Breakdown Statistics:")
    print(f"  Simultaneous DPs - Mean: {np.mean(simultaneous):.2f}, Range: {min(simultaneous)}-{max(simultaneous)}")
    print(f"  Conflict DPs - Mean: {np.mean(conflicts):.2f}, Range: {min(conflicts)}-{max(conflicts)}")
    print(f"  Total Parallel DPs - Mean: {np.mean(total_parallel):.2f}, Range: {min(total_parallel)}-{max(total_parallel)}")
    print(f"  Simultaneous ratio: {100*np.sum(simultaneous)/np.sum(total_parallel):.1f}%")
    print(f"  Conflict ratio: {100*np.sum(conflicts)/np.sum(total_parallel):.1f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Stacked area plot
    ax.fill_between(episodes, 0, simultaneous_smooth, 
                    color='steelblue', alpha=0.7, label='Simultaneous (Pure Parallel)')
    ax.fill_between(episodes, simultaneous_smooth, total_smooth,
                    color='#DC143C', alpha=0.7, label='Conflicts (Retries)')
    
    # Overlay scatter points for actual data
    ax.scatter(episodes, simultaneous, c='darkblue', s=15, alpha=0.3, marker='o', edgecolors='none')
    ax.scatter(episodes, total_parallel, c='darkred', s=15, alpha=0.3, marker='^', edgecolors='none')
    
    # Total line
    ax.plot(episodes, total_smooth, color='black', linewidth=2.5, alpha=0.8, 
           label='Total Parallel DPs', linestyle='-')
    
    # Mean lines
    mean_total = np.mean(total_parallel)
    mean_simultaneous = np.mean(simultaneous)
    mean_conflicts = np.mean(conflicts)
    
    ax.axhline(mean_total, color='black', linestyle=':', linewidth=2, alpha=0.6,
              label=f'Mean Total: {mean_total:.2f}')
    ax.axhline(mean_simultaneous, color='blue', linestyle=':', linewidth=1.5, alpha=0.6,
              label=f'Mean Simultaneous: {mean_simultaneous:.2f}')
    
    # Labels and title
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Decision Points per Episode', fontsize=14, fontweight='bold')
    ax.set_title('Parallel Decision Points Breakdown: Simultaneous vs Conflicts\nAcross Training Episodes', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    # Info text
    textstr = f'Total Episodes: {len(episodes):,}\n'
    textstr += f'Mean Total Parallel: {mean_total:.2f}\n'
    textstr += f'  • Simultaneous: {mean_simultaneous:.2f} ({100*np.sum(simultaneous)/np.sum(total_parallel):.1f}%)\n'
    textstr += f'  • Conflicts: {mean_conflicts:.2f} ({100*np.sum(conflicts)/np.sum(total_parallel):.1f}%)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'parallel_breakdown_stacked.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved parallel breakdown plot: {output_path}")


if __name__ == '__main__':
    plot_parallel_breakdown()
