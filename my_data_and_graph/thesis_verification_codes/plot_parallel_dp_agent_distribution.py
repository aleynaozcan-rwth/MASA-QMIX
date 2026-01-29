#!/usr/bin/env python3
"""
Parallel Decision Points - Agent Count Distribution
Shows distribution of agent counts in parallel decision points only
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

def parse_parallel_dps():
    """Parse timeline to get only parallel decision points (multiple agents deciding)."""
    
    episodes_data = []
    current_episode = None
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                if current_episode is not None:
                    episodes_data.append({
                        'episode': current_episode,
                        'timestamp_groups': dict(timestamp_groups)
                    })
                
                match = re.search(r'EPISODE (\d+)', line)
                current_episode = int(match.group(1)) if match else None
                timestamp_groups = defaultdict(int)
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
                if time_match:
                    timestamp_groups[time_match.group(1)] += 1
                continue
            
            # Count operation finishes
            if not in_lifecycle and '[t=' in line and 'finished' in line and 'next queued' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                if time_match:
                    timestamp_groups[time_match.group(1)] += 1
                continue
        
        # Add last episode
        if current_episode is not None:
            episodes_data.append({
                'episode': current_episode,
                'timestamp_groups': dict(timestamp_groups)
            })
    
    return episodes_data

def plot_parallel_agent_distribution():
    """Create distribution plot of agent counts in parallel DPs."""
    
    print("Parsing timeline for parallel DP agent distribution...")
    episodes = parse_parallel_dps()
    
    # Collect all parallel DPs (where count > 1)
    parallel_dp_counts = []
    single_dp_count = 0
    total_dps = 0
    
    for episode in episodes:
        for timestamp, count in episode['timestamp_groups'].items():
            total_dps += 1
            if count > 1:
                parallel_dp_counts.append(count)
            else:
                single_dp_count += 1
    
    # Count distribution
    agent_count_distribution = Counter(parallel_dp_counts)
    
    # Statistics
    total_parallel = len(parallel_dp_counts)
    parallel_percentage = (total_parallel / total_dps) * 100
    
    print(f"\nStatistics:")
    print(f"  Total Decision Points: {total_dps:,}")
    print(f"  Single-Agent DPs: {single_dp_count:,} ({100*single_dp_count/total_dps:.1f}%)")
    print(f"  Parallel DPs: {total_parallel:,} ({parallel_percentage:.1f}%)")
    print(f"  Max agents in parallel DP: {max(parallel_dp_counts) if parallel_dp_counts else 0}")
    print(f"\nParallel DP Distribution:")
    for num_agents in sorted(agent_count_distribution.keys()):
        count = agent_count_distribution[num_agents]
        pct = (count / total_parallel) * 100
        print(f"    {num_agents} agents: {count:,} DPs ({pct:.1f}% of parallel)")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # ==================== Plot 1: Bar Chart of Parallel DP Distribution ====================
    agent_counts = sorted(agent_count_distribution.keys())
    dp_counts = [agent_count_distribution[n] for n in agent_counts]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(agent_counts)))
    
    bars = ax1.bar(agent_counts, dp_counts, width=0.6, color=colors, 
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels on bars
    for bar, count in zip(bars, dp_counts):
        height = bar.get_height()
        pct = (count / total_parallel) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Number of Agents Deciding Simultaneously', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Parallel Decision Points', fontsize=13, fontweight='bold')
    ax1.set_title('Distribution of Agent Counts in Parallel Decision Points', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(agent_counts)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # ==================== Plot 2: Pie Chart ====================
    # Include single-agent for complete picture
    all_labels = ['Single Agent'] + [f'{n} Agents' for n in agent_counts]
    all_counts = [single_dp_count] + dp_counts
    
    # Colors: gray for single, viridis for parallel
    pie_colors = ['#CCCCCC'] + list(colors)
    
    wedges, texts, autotexts = ax2.pie(all_counts, labels=all_labels, autopct='%1.1f%%',
                                        colors=pie_colors, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # Highlight parallel DPs
    for i, autotext in enumerate(autotexts):
        if i > 0:  # Not single agent
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    ax2.set_title('Decision Point Composition\n(Single vs Parallel)', 
                  fontsize=14, fontweight='bold', pad=15)
    
    # Add statistics box
    stats_text = f'''Overall Statistics:
Total DPs: {total_dps:,}
Parallel DPs: {total_parallel:,} ({parallel_percentage:.1f}%)
Single DPs: {single_dp_count:,} ({100*single_dp_count/total_dps:.1f}%)

Max Parallel Agents: {max(parallel_dp_counts) if parallel_dp_counts else 0}
Avg Parallel Agents: {np.mean(parallel_dp_counts):.2f}'''
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, 
                edgecolor='black', linewidth=1.5, pad=0.7)
    fig.text(0.98, 0.02, stats_text, fontsize=10, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, family='monospace', fontweight='normal')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / 'parallel_dp_agent_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved visualization: {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_parallel_agent_distribution()
