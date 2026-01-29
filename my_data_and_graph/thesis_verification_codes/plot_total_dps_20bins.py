#!/usr/bin/env python3
"""
Total DPs visualization: All episodes grouped into 20-episode bins
Showing parallel vs non-parallel distribution
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
    """Parse timeline to get parallel and non-parallel DPs per episode."""
    
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
                    # Calculate parallel DPs
                    parallel_count = sum(count for count in timestamp_groups.values() if count > 1)
                    
                    # Detect conflicts
                    job_op_timestamps = defaultdict(list)
                    for timestamp, job_id, operation in decision_points:
                        job_op_timestamps[(job_id, operation)].append(float(timestamp))
                    
                    # Count conflict DPs
                    conflict_dps = set()
                    for (job_id, operation), timestamps in job_op_timestamps.items():
                        if len(timestamps) > 1:
                            for ts in timestamps[1:]:
                                conflict_dps.add((str(ts), job_id, operation))
                    
                    # Total coordination DPs
                    parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
                    conflict_timestamps = {ts for ts, _, _ in conflict_dps}
                    all_coordination_timestamps = parallel_timestamps | conflict_timestamps
                    coordination_dp_count = sum(timestamp_groups[ts] for ts in all_coordination_timestamps)
                    
                    # Total DPs
                    total_count = sum(timestamp_groups.values())
                    
                    episodes_data.append({
                        'episode': current_episode,
                        'parallel_dps': coordination_dp_count,
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
                    next_op = completed_op + 1
                    timestamp_groups[ts] += 1
                    decision_points.append((ts, job_id, f'Op{next_op}'))
                continue
    
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
        
        parallel_timestamps = {ts for ts, count in timestamp_groups.items() if count > 1}
        conflict_timestamps = {ts for ts, _, _ in conflict_dps}
        all_coordination_timestamps = parallel_timestamps | conflict_timestamps
        coordination_dp_count = sum(timestamp_groups[ts] for ts in all_coordination_timestamps)
        
        total_count = sum(timestamp_groups.values())
        
        episodes_data.append({
            'episode': current_episode,
            'parallel_dps': coordination_dp_count,
            'total_dps': total_count
        })
    
    return episodes_data


def plot_all_episodes_binned():
    """Create stacked bar chart for all episodes grouped by 20."""
    
    print("Parsing timeline for all episodes...")
    episodes_data = parse_breakdown_data()
    print(f"Parsed {len(episodes_data)} episodes")
    
    # Group into 20-episode bins
    bin_size = 20
    num_bins = (len(episodes_data) + bin_size - 1) // bin_size
    
    bin_labels = []
    non_parallel_means = []
    parallel_means = []
    
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, len(episodes_data))
        
        bin_data = episodes_data[start_idx:end_idx]
        
        # Calculate means
        total_mean = np.mean([ep['total_dps'] for ep in bin_data])
        parallel_mean = np.mean([ep['parallel_dps'] for ep in bin_data])
        non_parallel_mean = total_mean - parallel_mean
        
        non_parallel_means.append(non_parallel_mean)
        parallel_means.append(parallel_mean)
        
        start_ep = bin_data[0]['episode']
        end_ep = bin_data[-1]['episode']
        bin_labels.append(f'{start_ep}-{end_ep}')
    
    # Statistics
    early_third = len(parallel_means) // 3
    late_start = 2 * early_third
    
    early_parallel = np.mean(parallel_means[:early_third])
    late_parallel = np.mean(parallel_means[late_start:])
    parallel_change = ((late_parallel - early_parallel) / early_parallel) * 100
    
    early_nonparallel = np.mean(non_parallel_means[:early_third])
    late_nonparallel = np.mean(non_parallel_means[late_start:])
    nonparallel_change = ((late_nonparallel - early_nonparallel) / early_nonparallel) * 100
    
    print(f"\nStatistics:")
    print(f"  Total bins: {num_bins}")
    print(f"  Overall Non-Parallel mean: {np.mean(non_parallel_means):.2f}")
    print(f"  Overall Parallel mean: {np.mean(parallel_means):.2f}")
    print(f"\nLearning Progression:")
    print(f"  Early → Late Parallel: {early_parallel:.2f} → {late_parallel:.2f} ({parallel_change:+.1f}% change)")
    print(f"  Early → Late Non-Parallel: {early_nonparallel:.2f} → {late_nonparallel:.2f} ({nonparallel_change:+.1f}% change)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 8))
    
    x_pos = np.arange(len(bin_labels))
    width = 0.8
    
    # Calculate parallel ratios as percentages
    parallel_ratios = [parallel_means[i] / (non_parallel_means[i] + parallel_means[i]) * 100
                      for i in range(len(parallel_means))]
    
    # Define categories based on data range
    min_ratio = min(parallel_ratios)
    max_ratio = max(parallel_ratios)
    range_ratio = max_ratio - min_ratio
    
    low_threshold = min_ratio + range_ratio * 0.33
    high_threshold = min_ratio + range_ratio * 0.67
    
    # Soft, harmonious orange colors
    color_low = '#FFC870'      # Soft light orange - more saturated
    color_medium = '#FF9F3D'   # Soft medium orange - more saturated
    color_high = '#FF7020'     # Soft dark orange - more saturated
    
    # Stacked bars
    # Non-parallel - light blue (slightly darker and more blue)
    for i, nonpar in enumerate(non_parallel_means):
        ax.bar(x_pos[i], nonpar, width, color='#6B9BD1', alpha=0.9)
    
    # Parallel - categorized by ratio
    for i, par in enumerate(parallel_means):
        ratio = parallel_ratios[i]
        if ratio < low_threshold:
            color = color_low
        elif ratio < high_threshold:
            color = color_medium
        else:
            color = color_high
        ax.bar(x_pos[i], par, width, bottom=non_parallel_means[i],
              color=color, alpha=0.9)
    
    # X-axis
    ax.set_xlabel('Episode Range (20-Episode Windows)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos[::2])  # Show every 2nd label
    ax.set_xticklabels([bin_labels[i] for i in range(0, len(bin_labels), 2)], 
                       rotation=45, ha='right', fontsize=9)
    
    # Y-axis
    ax.set_ylabel('Total Decision Points per Episode (Mean)', fontsize=13, fontweight='bold')
    
    # Remove margins - bars start at 0 and end at figure edge
    ax.set_xlim(-0.5, len(bin_labels) - 0.5)
    ax.set_ylim(0, 50)
    ax.margins(0)
    
    # Use known values
    total_dps_all = 39891
    simultaneous_all = 5553
    conflicts_all = 3306
    parallel_dps_all = simultaneous_all + conflicts_all  # 8859
    non_parallel_dps_all = total_dps_all - parallel_dps_all  # 31032
    
    title = 'Parallel Decision Point Distribution'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Add legend in top-left corner with ratio ranges
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#6B9BD1', label='Non-Parallel DPs'),
        Patch(facecolor=color_low, 
              label=f'Parallel - Low ({min_ratio:.1f}-{low_threshold:.1f}%)'),
        Patch(facecolor=color_medium, 
              label=f'Parallel - Medium ({low_threshold:.1f}-{high_threshold:.1f}%)'),
        Patch(facecolor=color_high, 
              label=f'Parallel - High ({high_threshold:.1f}-{max_ratio:.1f}%)')
    ]
    legend = ax.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(0.01, 0.995), fontsize=10,
                      frameon=True, fancybox=True, shadow=True,
                      facecolor='white', edgecolor='black', framealpha=0.95)
    
    # Add statistics box in top-right corner (inside axes)
    stats_text = f'''Total DPs: {total_dps_all:,}
Non-Parallel DPs: {non_parallel_dps_all:,} ({100*non_parallel_dps_all/total_dps_all:.1f}% of Total)
Parallel DPs: {parallel_dps_all:,} ({100*parallel_dps_all/total_dps_all:.1f}% of Total)
  - Simultaneous: {simultaneous_all:,} ({100*simultaneous_all/total_dps_all:.1f}%)
  - Conflicts: {conflicts_all:,} ({100*conflicts_all/total_dps_all:.1f}%)

Initial Jobs per Episode: 4
Job Arrival Rate: λ=0.125'''
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, 
                edgecolor='black', linewidth=1.5, pad=0.7)
    ax.text(0.78, 0.97, stats_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', horizontalalignment='left',
            bbox=props, family='monospace', fontweight='normal')
    
    # Adjust layout to leave space on the right
    plt.subplots_adjust(left=0.05, right=0.70, top=0.92, bottom=0.08)
    
    # Save
    output_path = OUTPUT_DIR / 'total_dps_20bins_stacked.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved total DPs visualization: {output_path}")


if __name__ == '__main__':
    plot_all_episodes_binned()
