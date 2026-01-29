#!/usr/bin/env python3
"""
Butterfly (Diverging) Bar Chart showing simultaneous vs conflicts distribution.
Left side: Simultaneous DPs, Right side: Conflicts
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


def plot_butterfly_chart():
    """Create butterfly (diverging) bar chart."""
    
    print("Parsing timeline for butterfly chart...")
    episodes_data = parse_breakdown_data()
    print(f"Parsed {len(episodes_data)} episodes")
    
    # Group episodes into bins
    bin_size = 50
    num_bins = (len(episodes_data) + bin_size - 1) // bin_size
    
    bin_labels = []
    simultaneous_means = []
    conflicts_means = []
    
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, len(episodes_data))
        
        bin_data = episodes_data[start_idx:end_idx]
        
        # Calculate means for this bin
        sim_mean = np.mean([ep['simultaneous'] for ep in bin_data])
        conf_mean = np.mean([ep['conflicts'] for ep in bin_data])
        
        simultaneous_means.append(sim_mean)
        conflicts_means.append(conf_mean)
        
        # Create label
        start_ep = episodes_data[start_idx]['episode']
        end_ep = episodes_data[end_idx - 1]['episode']
        bin_labels.append(f'{start_ep}-{end_ep}')
    
    # Statistics with trend analysis
    print(f"\nButterfly Chart Statistics:")
    print(f"  Number of bins: {num_bins}")
    print(f"  Bin size: {bin_size} episodes")
    print(f"  Overall Simultaneous mean: {np.mean(simultaneous_means):.2f}")
    print(f"  Overall Conflicts mean: {np.mean(conflicts_means):.2f}")
    
    # Early vs Late comparison (use first 2 and last 2 bins for maximum contrast)
    early_bins = 2  # First 2 bins
    late_bins = 2   # Last 2 bins
    
    early_conf_mean = np.mean(conflicts_means[:early_bins])
    early_conf_max = np.max(conflicts_means[:early_bins])
    late_conf_mean = np.mean(conflicts_means[-late_bins:])
    late_conf_max = np.max(conflicts_means[-late_bins:])
    conf_change = ((late_conf_mean - early_conf_mean) / early_conf_mean) * 100
    
    early_sim_mean = np.mean(simultaneous_means[:early_bins])
    late_sim_mean = np.mean(simultaneous_means[-late_bins:])
    
    print(f"\n  Early Training (first {early_bins*bin_size} eps):")
    print(f"    Simultaneous: {early_sim_mean:.2f}, Conflicts: {early_conf_mean:.2f} (max: {early_conf_max:.2f})")
    print(f"  Late Training (last {late_bins*bin_size} eps):")
    print(f"    Simultaneous: {late_sim_mean:.2f}, Conflicts: {late_conf_mean:.2f} (max: {late_conf_max:.2f})")
    print(f"  Conflict change: {conf_change:+.1f}%")
    print(f"  Peak conflict reduction: {early_conf_max:.2f} → {late_conf_max:.2f}")
    
    # Find best and worst conflict bins
    worst_idx = np.argmax(conflicts_means)
    best_idx = np.argmin(conflicts_means)
    
    print(f"\n  Worst Conflict Period:")
    print(f"    Episode range: {bin_labels[worst_idx]}")
    print(f"    Conflicts: {conflicts_means[worst_idx]:.2f}")
    print(f"  Best Conflict Period:")
    print(f"    Episode range: {bin_labels[best_idx]}")
    print(f"    Conflicts: {conflicts_means[best_idx]:.2f}")
    print(f"  Improvement: {conflicts_means[worst_idx] - conflicts_means[best_idx]:.2f} ({100*(conflicts_means[worst_idx] - conflicts_means[best_idx])/conflicts_means[worst_idx]:.1f}%)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(len(bin_labels))
    
    # Find best and worst conflict bins for highlighting
    worst_idx = np.argmax(conflicts_means)
    best_idx = np.argmin(conflicts_means)
    
    # Left side: Simultaneous (negative values for left direction)
    ax.barh(y_pos, [-s for s in simultaneous_means], height=0.8,
           color='steelblue', alpha=0.8, label='Simultaneous DPs', edgecolor='darkblue', linewidth=1.5)
    
    # Right side: Conflicts (positive values)
    # Color gradient: early episodes darker, late episodes lighter
    colors_conflicts = plt.cm.Reds(np.linspace(0.7, 0.4, len(conflicts_means)))
    
    for i, (conf, color) in enumerate(zip(conflicts_means, colors_conflicts)):
        # Highlight worst and best with special border
        if i == worst_idx:
            edgecolor = 'black'
            linewidth = 3
        elif i == best_idx:
            edgecolor = 'green'
            linewidth = 3
        else:
            edgecolor = 'darkred'
            linewidth = 1.5
            
        ax.barh(y_pos[i], conf, height=0.8,
               color=color, alpha=0.8, edgecolor=edgecolor, linewidth=linewidth)
    
    # Add value labels on bars
    for i, (sim, conf) in enumerate(zip(simultaneous_means, conflicts_means)):
        # Simultaneous label (left side)
        ax.text(-sim - 0.15, i, f'{sim:.1f}', ha='right', va='center', 
               fontsize=10, fontweight='bold', color='darkblue')
        # Conflicts label (right side)
        label_color = 'black' if i == worst_idx else ('green' if i == best_idx else 'darkred')
        ax.text(conf + 0.15, i, f'{conf:.1f}', ha='left', va='center',
               fontsize=10, fontweight='bold', color=label_color)
    
    # Center line
    ax.axvline(0, color='black', linewidth=2, linestyle='-', alpha=0.8)
    
    # Y-axis: Episode ranges
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bin_labels, fontsize=11)
    ax.set_ylabel('Episode Range', fontsize=13, fontweight='bold')
    
    # X-axis
    max_val = max(max(simultaneous_means), max(conflicts_means))
    ax.set_xlim(-max_val * 1.2, max_val * 1.2)
    ax.set_xlabel('Decision Points per Episode (Mean)', fontsize=13, fontweight='bold')
    
    # Custom x-ticks (show absolute values)
    x_ticks = ax.get_xticks()
    ax.set_xticklabels([f'{abs(x):.0f}' for x in x_ticks], fontsize=11)
    
    # Title
    ax.set_title('Parallel DPs Distribution: Simultaneous ← | → Conflicts\nButterfly Chart by Episode Ranges', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Grid (vertical only)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
    
    # Legend with gradient explanation
    from matplotlib.patches import Rectangle
    handles = [
        Rectangle((0, 0), 1, 1, fc='steelblue', alpha=0.8, edgecolor='darkblue', linewidth=1.5, label='Simultaneous DPs'),
        Rectangle((0, 0), 1, 1, fc='darkred', alpha=0.8, edgecolor='darkred', linewidth=1.5, label='Conflicts (Early)'),
        Rectangle((0, 0), 1, 1, fc='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1.5, label='Conflicts (Late)'),
        Rectangle((0, 0), 1, 1, fc='red', alpha=0.8, edgecolor='black', linewidth=3, label='Worst Period'),
        Rectangle((0, 0), 1, 1, fc='red', alpha=0.8, edgecolor='green', linewidth=3, label='Best Period')
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=11, framealpha=0.95)
    
    # Info text with trend
    total_sim = sum(simultaneous_means)
    total_conf = sum(conflicts_means)
    sim_pct = 100 * total_sim / (total_sim + total_conf)
    conf_pct = 100 * total_conf / (total_sim + total_conf)
    
    textstr = f'Distribution:\n'
    textstr += f'  Simultaneous: {sim_pct:.1f}%\n'
    textstr += f'  Conflicts: {conf_pct:.1f}%\n\n'
    textstr += f'Conflict Trend:\n'
    textstr += f'  Early: {early_conf_mean:.2f} (max: {early_conf_max:.1f})\n'
    textstr += f'  Late: {late_conf_mean:.2f} (max: {late_conf_max:.1f})\n'
    textstr += f'  Change: {conf_change:+.1f}%\n'
    textstr += f'  Peak ↓: {early_conf_max - late_conf_max:.1f}\n\n'
    textstr += f'Extremes:\n'
    textstr += f'  Worst: {bin_labels[worst_idx]} ({conflicts_means[worst_idx]:.2f})\n'
    textstr += f'  Best: {bin_labels[best_idx]} ({conflicts_means[best_idx]:.2f})\n\n'
    textstr += f'Total: {len(episodes_data):,} eps'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'parallel_butterfly_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved butterfly chart: {output_path}")


if __name__ == '__main__':
    plot_butterfly_chart()
