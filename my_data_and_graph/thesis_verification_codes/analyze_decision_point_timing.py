#!/usr/bin/env python3
"""
Analyze decision point timing to understand parallel and near-parallel decisions.
This script examines the temporal distribution of decision points to determine
if decisions cluster together even if not at exact same timestamp.
"""

import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Input file
TIMELINE_FILE = 'my_data_and_graph/historydata/scheduling_timeline.txt'

def parse_timeline_for_timing_analysis():
    """Parse the timeline and extract all decision point timestamps per episode."""
    
    episode_timestamps = []  # List of lists: each episode's timestamps
    current_timestamps = []
    current_episode = None
    in_lifecycle = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Episode start marker
            episode_match = re.match(r'^=== EPISODE (\d+) ===$', line)
            if episode_match:
                # Save previous episode
                if current_timestamps:
                    episode_timestamps.append(sorted(current_timestamps))
                current_timestamps = []
                current_episode = int(episode_match.group(1))
                in_lifecycle = False
                continue
            
            # Track lifecycle section
            if 'LIFECYCLE TRACE START' in line:
                in_lifecycle = True
                continue
            
            if 'LIFECYCLE TRACE END' in line:
                in_lifecycle = False
                continue
            
            # Parse decision points
            # Job arrivals (in lifecycle)
            if in_lifecycle and '[t=' in line and 'New job' in line and 'arrived' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                if time_match:
                    timestamp = float(time_match.group(1))
                    current_timestamps.append(timestamp)
                continue
            
            # Operation completions (outside lifecycle, after === TIMELINE ===)
            if not in_lifecycle and current_episode is not None and '[t=' in line and 'finished' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                if time_match:
                    timestamp = float(time_match.group(1))
                    current_timestamps.append(timestamp)
                continue
        
        # Save last episode
        if current_timestamps:
            episode_timestamps.append(sorted(current_timestamps))
    
    return episode_timestamps


def analyze_temporal_clustering(episode_timestamps, time_windows=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0]):
    """
    Analyze how many decision points occur within various time windows.
    This helps understand if decisions cluster together even if not exactly parallel.
    """
    
    print(f"\n{'='*70}")
    print("TEMPORAL CLUSTERING ANALYSIS")
    print(f"{'='*70}\n")
    
    results = {}
    
    for window in time_windows:
        total_clustered = 0
        total_dps = 0
        episodes_with_clusters = 0
        cluster_sizes = []
        
        for timestamps in episode_timestamps:
            total_dps += len(timestamps)
            episode_clustered = 0
            processed = set()
            
            for i, t1 in enumerate(timestamps):
                if i in processed:
                    continue
                
                # Find all DPs within time window
                cluster = [t1]
                processed.add(i)
                
                for j, t2 in enumerate(timestamps[i+1:], start=i+1):
                    if t2 - t1 <= window:
                        if j not in processed:
                            cluster.append(t2)
                            processed.add(j)
                    else:
                        break  # timestamps are sorted
                
                # Count as clustered if 2+ decisions within window
                if len(cluster) >= 2:
                    episode_clustered += len(cluster)
                    cluster_sizes.append(len(cluster))
            
            total_clustered += episode_clustered
            if episode_clustered > 0:
                episodes_with_clusters += 1
        
        ratio = (total_clustered / total_dps * 100) if total_dps > 0 else 0
        mean_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        
        results[window] = {
            'total_clustered': total_clustered,
            'total_dps': total_dps,
            'ratio': ratio,
            'episodes_with_clusters': episodes_with_clusters,
            'total_episodes': len(episode_timestamps),
            'mean_cluster_size': mean_cluster_size,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0
        }
        
        print(f"Time Window: Δt ≤ {window:.2f} time units")
        print(f"  Clustered DPs: {total_clustered:,} / {total_dps:,} ({ratio:.1f}%)")
        print(f"  Episodes with clusters: {episodes_with_clusters} / {len(episode_timestamps)}")
        print(f"  Mean cluster size: {mean_cluster_size:.2f}")
        print(f"  Max cluster size: {max(cluster_sizes) if cluster_sizes else 0}")
        print()
    
    return results


def analyze_inter_arrival_times(episode_timestamps):
    """Analyze the distribution of time gaps between consecutive decision points."""
    
    all_gaps = []
    
    for timestamps in episode_timestamps:
        for i in range(len(timestamps) - 1):
            gap = timestamps[i+1] - timestamps[i]
            all_gaps.append(gap)
    
    all_gaps = np.array(all_gaps)
    
    print(f"\n{'='*70}")
    print("INTER-ARRIVAL TIME DISTRIBUTION")
    print(f"{'='*70}\n")
    
    print(f"Total inter-arrival gaps: {len(all_gaps):,}")
    print(f"Mean gap: {np.mean(all_gaps):.4f} time units")
    print(f"Median gap: {np.median(all_gaps):.4f} time units")
    print(f"Std dev: {np.std(all_gaps):.4f} time units")
    print(f"Min gap: {np.min(all_gaps):.6f} time units")
    print(f"Max gap: {np.max(all_gaps):.4f} time units")
    print()
    
    # Percentiles
    percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    print("Percentiles of inter-arrival times:")
    for p in percentiles:
        value = np.percentile(all_gaps, p)
        print(f"  {p:5.1f}th percentile: {value:.6f} time units")
    
    # Count very small gaps
    for threshold in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        count = np.sum(all_gaps <= threshold)
        pct = count / len(all_gaps) * 100
        print(f"\nGaps ≤ {threshold:.3f}: {count:,} ({pct:.1f}%)")
    
    return all_gaps


def visualize_gap_distribution(all_gaps):
    """Create visualization of inter-arrival time distribution."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Decision Point Inter-Arrival Time Analysis', fontsize=14, fontweight='bold')
    
    # 1. Full histogram (log scale)
    ax1 = axes[0, 0]
    ax1.hist(all_gaps, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Time Gap (time units)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Inter-Arrival Time Distribution (Full Range)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoomed histogram (gaps < 1.0)
    ax2 = axes[0, 1]
    small_gaps = all_gaps[all_gaps <= 1.0]
    ax2.hist(small_gaps, bins=100, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Time Gap (time units)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Inter-Arrival Times ≤ 1.0 ({len(small_gaps):,} gaps)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Very small gaps (< 0.1)
    ax3 = axes[1, 0]
    very_small_gaps = all_gaps[all_gaps <= 0.1]
    ax3.hist(very_small_gaps, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Time Gap (time units)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Inter-Arrival Times ≤ 0.1 ({len(very_small_gaps):,} gaps)')
    ax3.grid(True, alpha=0.3)
    
    # 4. CDF
    ax4 = axes[1, 1]
    sorted_gaps = np.sort(all_gaps)
    cdf = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps)
    ax4.plot(sorted_gaps, cdf, linewidth=2, color='purple')
    ax4.set_xlabel('Time Gap (time units)')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution Function')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, min(10, np.max(all_gaps)))
    
    # Add markers for key thresholds
    for threshold in [0.01, 0.05, 0.1, 0.5, 1.0]:
        if threshold < np.max(sorted_gaps):
            idx = np.searchsorted(sorted_gaps, threshold)
            prob = cdf[idx] if idx < len(cdf) else 1.0
            ax4.axvline(threshold, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax4.text(threshold, prob, f' {prob*100:.1f}%', fontsize=8)
    
    plt.tight_layout()
    output_file = 'my_data_and_graph/decision_point_timing_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved timing analysis visualization: {output_file}")
    plt.close()


def main():
    print("Parsing scheduling timeline for timing analysis...")
    episode_timestamps = parse_timeline_for_timing_analysis()
    print(f"Parsed {len(episode_timestamps)} episodes")
    
    # Analyze temporal clustering with different time windows
    clustering_results = analyze_temporal_clustering(episode_timestamps)
    
    # Analyze inter-arrival times
    all_gaps = analyze_inter_arrival_times(episode_timestamps)
    
    # Visualize
    visualize_gap_distribution(all_gaps)
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION:")
    print(f"{'='*70}")
    
    # Find optimal time window
    best_window = None
    for window in [0.01, 0.05, 0.1]:
        ratio = clustering_results[window]['ratio']
        if ratio > 20:  # Looking for at least 20% clustered
            best_window = window
            break
    
    if best_window:
        print(f"\nConsider using Δt ≤ {best_window:.2f} as 'near-parallel' threshold:")
        print(f"  This captures {clustering_results[best_window]['ratio']:.1f}% of all decisions")
        print(f"  Mean cluster size: {clustering_results[best_window]['mean_cluster_size']:.2f}")
    else:
        print(f"\nExact parallel decisions (Δt = 0.0): {clustering_results[0.0]['ratio']:.1f}%")
        print("Consider alternative visualization approaches to highlight coordination.")


if __name__ == '__main__':
    main()
