#!/usr/bin/env python3
"""
Scatter plot showing conflict distribution across parallel decision points.
Shows relationship between parallel DPs and conflicts with trend analysis.
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


def plot_conflict_distribution():
    """Create scatter plot showing conflict distribution vs parallel DPs."""
    
    print("Parsing timeline for conflict distribution analysis...")
    episodes_data = parse_conflicts_data()
    print(f"Parsed {len(episodes_data)} episodes")
    
    # Extract data
    parallel_dps = np.array([ep['parallel_dps'] for ep in episodes_data])
    conflicts = np.array([ep['conflicts'] for ep in episodes_data])
    
    # Statistics
    print(f"\nConflict Distribution Statistics:")
    print(f"  Parallel DPs range: {min(parallel_dps)} - {max(parallel_dps)}")
    print(f"  Mean Parallel DPs: {np.mean(parallel_dps):.2f} (±{np.std(parallel_dps):.2f})")
    print(f"  Conflicts range: {min(conflicts)} - {max(conflicts)}")
    print(f"  Mean Conflicts: {np.mean(conflicts):.2f} (±{np.std(conflicts):.2f})")
    
    # Calculate correlation
    correlation = np.corrcoef(parallel_dps, conflicts)[0, 1]
    print(f"  Correlation (Parallel DPs vs Conflicts): {correlation:.3f}")
    
    # Count frequency for each (parallel_dps, conflicts) combination
    from collections import Counter
    point_counts = Counter(zip(parallel_dps, conflicts))
    
    # Prepare data for plotting
    unique_parallel = []
    unique_conflicts = []
    sizes = []
    
    for (p, c), count in point_counts.items():
        unique_parallel.append(p)
        unique_conflicts.append(c)
        sizes.append(count * 15)  # Scale size by frequency
    
    # Linear regression for trend (manual calculation)
    x_mean = np.mean(parallel_dps)
    y_mean = np.mean(conflicts)
    
    numerator = np.sum((parallel_dps - x_mean) * (conflicts - y_mean))
    denominator = np.sum((parallel_dps - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Calculate R²
    y_pred = slope * parallel_dps + intercept
    ss_res = np.sum((conflicts - y_pred) ** 2)
    ss_tot = np.sum((conflicts - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"  Linear trend: y = {slope:.3f}x + {intercept:.3f}")
    print(f"  R² = {r_squared:.3f}")
    print(f"  Unique combinations: {len(point_counts)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Scatter plot with size proportional to frequency
    scatter = ax.scatter(unique_parallel, unique_conflicts, s=sizes, 
                        c=unique_conflicts, cmap='RdYlBu_r', alpha=0.6, 
                        edgecolors='black', linewidth=0.8)
    
    # Trend line
    x_trend = np.array([min(parallel_dps), max(parallel_dps)])
    y_trend = slope * x_trend + intercept
    ax.plot(x_trend, y_trend, 'darkred', linestyle='--', linewidth=2.5, alpha=0.9,
           label=f'Linear Trend: y = {slope:.3f}x + {intercept:.2f}')
    
    # Mean lines
    ax.axhline(y_mean, color='purple', linestyle=':', linewidth=2, alpha=0.7,
              label=f'Mean Conflicts: {y_mean:.2f}')
    ax.axvline(x_mean, color='green', linestyle=':', linewidth=2, alpha=0.7,
              label=f'Mean Parallel DPs: {x_mean:.2f}')
    
    # Labels and title
    ax.set_xlabel('Parallel Decision Points per Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Conflict Points per Episode', fontsize=14, fontweight='bold')
    ax.set_title('Conflicts Within Parallel Decision Points\n(Bubble Size = Episode Frequency)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    legend1 = ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
    
    # Add text box with statistics
    textstr = f'Total Episodes: {len(episodes_data):,}\n'
    textstr += f'Correlation: {correlation:.3f}\n'
    textstr += f'R²: {r_squared:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label('Conflicts per Episode', fontsize=12, fontweight='bold')
    
    # Set integer ticks
    ax.set_xticks(range(int(min(parallel_dps)), int(max(parallel_dps))+1))
    ax.set_yticks(range(0, int(max(conflicts))+1))
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'conflict_distribution_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved conflict distribution scatter plot: {output_path}")
    
    return episodes_data


if __name__ == '__main__':
    plot_conflict_distribution()
