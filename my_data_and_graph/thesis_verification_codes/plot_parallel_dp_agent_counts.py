#!/usr/bin/env python3
"""
Agent Count in Parallel Decision Points
Shows distribution and timeline of agent counts ONLY at parallel DPs (where multiple agents decide simultaneously)
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

def parse_agent_counts():
    """Parse timeline to extract decision points with agent counts."""
    
    all_decision_points = []
    current_episode = None
    in_lifecycle = False
    
    job_arrival_times = {}
    job_completion_times = {}
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                match = re.search(r'EPISODE (\d+)', line)
                if match:
                    current_episode = int(match.group(1))
                    job_arrival_times = {}
                    job_completion_times = {}
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
            
            # Track job arrivals
            if in_lifecycle and '[t=' in line and 'New job' in line and 'arrived' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'job (\d+)', line)
                active_match = re.search(r'Active:(\d+)', line)
                
                if time_match and job_match:
                    timestamp = float(time_match.group(1))
                    job_id = int(job_match.group(1))
                    job_arrival_times[job_id] = timestamp
                    
                    if active_match:
                        num_agents = int(active_match.group(1))
                        
                        all_decision_points.append({
                            'episode': current_episode,
                            'timestamp': timestamp,
                            'num_agents_deciding': num_agents,
                            'dp_type': 'arrival'
                        })
                continue
            
            # Track decision points from operations finishing
            if not in_lifecycle and '[t=' in line and 'finished' in line and 'next queued' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                
                if time_match:
                    timestamp = float(time_match.group(1))
                    
                    all_decision_points.append({
                        'episode': current_episode,
                        'timestamp': timestamp,
                        'num_agents_deciding': 1,
                        'dp_type': 'operation'
                    })
                continue
    
    return all_decision_points

def plot_parallel_agent_counts():
    """Create visualization focusing on parallel decision points."""
    
    print("Parsing timeline for parallel DP agent analysis...")
    all_dps = parse_agent_counts()
    
    if not all_dps:
        print("No decision points found!")
        return
    
    # Filter only parallel DPs (more than 1 agent)
    parallel_dps = [dp for dp in all_dps if dp['num_agents_deciding'] > 1]
    
    print(f"Total decision points: {len(all_dps)}")
    print(f"Parallel decision points: {len(parallel_dps)} ({100*len(parallel_dps)/len(all_dps):.1f}%)")
    
    if not parallel_dps:
        print("No parallel decision points found!")
        return
    
    # Extract agent counts
    agent_counts = [dp['num_agents_deciding'] for dp in parallel_dps]
    
    # Count distribution
    count_distribution = Counter(agent_counts)
    
    # Statistics
    avg_agents = np.mean(agent_counts)
    median_agents = np.median(agent_counts)
    max_agents = max(agent_counts)
    min_agents = min(agent_counts)
    
    print(f"\nParallel DP Statistics:")
    print(f"  Average agents: {avg_agents:.2f}")
    print(f"  Median agents: {median_agents:.1f}")
    print(f"  Min agents: {min_agents}")
    print(f"  Max agents: {max_agents}")
    print(f"\nDistribution:")
    for num_agents in sorted(count_distribution.keys()):
        count = count_distribution[num_agents]
        percentage = 100 * count / len(parallel_dps)
        print(f"  {num_agents} agents: {count:,} DPs ({percentage:.1f}%)")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    
    # ==================== Plot 1: Distribution Bar Chart ====================
    agent_nums = sorted(count_distribution.keys())
    counts = [count_distribution[n] for n in agent_nums]
    percentages = [100 * c / len(parallel_dps) for c in counts]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    bar_colors = [colors[i % len(colors)] for i in range(len(agent_nums))]
    
    bars = ax1.bar(agent_nums, counts, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n({counts[i]:,})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Number of Agents Deciding', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Parallel Decision Points', fontsize=13, fontweight='bold')
    ax1.set_title('Agent Count Distribution in Parallel Decision Points', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_xticks(agent_nums)
    
    # ==================== Plot 2: Timeline Signal ====================
    parallel_indices = [i for i, dp in enumerate(all_dps) if dp['num_agents_deciding'] > 1]
    parallel_agent_counts = [all_dps[i]['num_agents_deciding'] for i in parallel_indices]
    
    # Create full signal with zeros for non-parallel
    full_signal = np.zeros(len(all_dps))
    for idx, count in zip(parallel_indices, parallel_agent_counts):
        full_signal[idx] = count
    
    # Smooth for visualization
    window = 50
    smoothed_signal = uniform_filter1d(full_signal, size=window, mode='nearest')
    
    ax2.plot(range(len(all_dps)), smoothed_signal, linewidth=2, color='#2E86AB', alpha=0.8)
    ax2.fill_between(range(len(all_dps)), smoothed_signal, alpha=0.3, color='#2E86AB')
    
    # Add average line
    ax2.axhline(y=avg_agents * len(parallel_dps) / len(all_dps), color='red', 
                linestyle='--', linewidth=2, alpha=0.6, 
                label=f'Weighted Avg: {avg_agents * len(parallel_dps) / len(all_dps):.2f}')
    
    ax2.set_xlabel('Decision Point Index (All DPs)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Agent Count (Smoothed)', fontsize=13, fontweight='bold')
    ax2.set_title('Agent Count Signal in Parallel DPs Over Time', fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.set_xlim(0, len(all_dps))
    
    # Add statistics box
    stats_text = f'''Parallel DP Statistics:
Total DPs: {len(all_dps):,}
Parallel DPs: {len(parallel_dps):,} ({100*len(parallel_dps)/len(all_dps):.1f}%)

Agent Count in Parallel:
  Average: {avg_agents:.2f}
  Median: {median_agents:.1f}
  Min: {min_agents}
  Max: {max_agents}

Smoothing: {window} DPs'''
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, 
                edgecolor='black', linewidth=1.5, pad=0.7)
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace', fontweight='normal')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / 'parallel_dp_agent_counts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved parallel DP agent count visualization: {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_parallel_agent_counts()
