#!/usr/bin/env python3
"""
Agent Count Per Decision Point - Signal Visualization
Shows agent counts and system state as continuous signals/lines
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

def parse_agent_counts():
    """
    Parse timeline to extract:
    - Decision points with number of agents deciding
    - System state: total arrived, active, completed
    """
    
    all_decision_points = []
    current_episode = None
    episode_start_dp_idx = 0
    in_lifecycle = False
    
    # For tracking system state
    job_arrival_times = {}  # job_id -> arrival_time
    job_completion_times = {}  # job_id -> completion_time
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                match = re.search(r'EPISODE (\d+)', line)
                if match:
                    current_episode = int(match.group(1))
                    episode_start_dp_idx = len(all_decision_points)
                    # Reset tracking for new episode
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
            
            # Track job arrivals (initial agents)
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
                            'dp_type': 'arrival',
                            'total_arrived': len(job_arrival_times),
                            'total_completed': len(job_completion_times),
                            'active_agents': num_agents
                        })
                continue
            
            # Track decision points from operations finishing
            if not in_lifecycle and '[t=' in line and 'finished' in line and 'next queued' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'Job_(\d+)\.Op(\d+) finished', line)
                
                if time_match and job_match:
                    timestamp = float(time_match.group(1))
                    job_id = int(job_match.group(1))
                    
                    # Count how many agents are active at this timestamp
                    # (arrived but not completed)
                    active_at_time = 0
                    for jid, arr_time in job_arrival_times.items():
                        if arr_time <= timestamp:
                            comp_time = job_completion_times.get(jid, float('inf'))
                            if comp_time > timestamp:
                                active_at_time += 1
                    
                    all_decision_points.append({
                        'episode': current_episode,
                        'timestamp': timestamp,
                        'num_agents_deciding': 1,  # Single agent completing an operation
                        'dp_type': 'operation',
                        'total_arrived': len([t for t in job_arrival_times.values() if t <= timestamp]),
                        'total_completed': len([t for t in job_completion_times.values() if t <= timestamp]),
                        'active_agents': active_at_time
                    })
                continue
            
            # Track job completions
            if not in_lifecycle and '[t=' in line and 'completed all operations' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'Job_(\d+) completed', line)
                
                if time_match and job_match:
                    timestamp = float(time_match.group(1))
                    job_id = int(job_match.group(1))
                    job_completion_times[job_id] = timestamp
                continue
    
    return all_decision_points

def plot_agent_signals():
    """Create signal visualization of agent counts across all decision points."""
    
    print("Parsing timeline for agent signal analysis...")
    all_dps = parse_agent_counts()
    
    if not all_dps:
        print("No decision points found!")
        return
    
    print(f"Found {len(all_dps)} decision points across all episodes")
    
    # Extract data for plotting
    dp_indices = np.array(range(len(all_dps)))
    agents_deciding = np.array([dp['num_agents_deciding'] for dp in all_dps])
    total_arrived = np.array([dp['total_arrived'] for dp in all_dps])
    active_agents = np.array([dp['active_agents'] for dp in all_dps])
    total_completed = np.array([dp['total_completed'] for dp in all_dps])
    
    # Apply smoothing for cleaner signals
    window = 50
    agents_deciding_smooth = uniform_filter1d(agents_deciding, size=window, mode='nearest')
    active_agents_smooth = uniform_filter1d(active_agents, size=window, mode='nearest')
    
    # Statistics
    avg_agents_deciding = np.mean(agents_deciding)
    max_agents_deciding = max(agents_deciding)
    max_active = max(active_agents)
    
    print(f"\nStatistics:")
    print(f"  Average agents per DP: {avg_agents_deciding:.2f}")
    print(f"  Max agents at single DP: {max_agents_deciding}")
    print(f"  Max active agents in system: {max_active}")
    print(f"  Total jobs across all episodes: {max(total_arrived)}")
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(24, 8))
    
    # ==================== Main Signal Plot ====================
    # Agents deciding at each DP (smoothed signal)
    ax.plot(dp_indices, agents_deciding_smooth, linewidth=2.5, color='#2E86AB', 
            alpha=0.9, label='Agents Deciding per DP (smoothed)', zorder=3)
    
    # Active agents in system (smoothed)
    ax.plot(dp_indices, active_agents_smooth, linewidth=2.5, color='#D62246', 
            alpha=0.8, label='Active Agents in System (smoothed)', zorder=2)
    
    # Fill between for visual appeal
    ax.fill_between(dp_indices, agents_deciding_smooth, alpha=0.2, color='#2E86AB', zorder=1)
    ax.fill_between(dp_indices, active_agents_smooth, alpha=0.15, color='#D62246', zorder=0)
    
    # Average line for agents deciding
    ax.axhline(y=avg_agents_deciding, color='#2E86AB', linestyle='--', 
               linewidth=2, alpha=0.5, label=f'Avg Deciding: {avg_agents_deciding:.2f}')
    
    # Labels and formatting
    ax.set_xlabel('Decision Point Index (Chronological Across All Episodes)', 
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Agents', fontsize=13, fontweight='bold')
    ax.set_title('Agent Count Signals Over Decision Points', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True, 
             fancybox=True, framealpha=0.95)
    
    # Set limits
    ax.set_xlim(0, len(dp_indices))
    ax.set_ylim(0, max(max_agents_deciding, max_active) * 1.1)
    
    # Add statistics box
    stats_text = f'''Statistics:
Total Decision Points: {len(all_dps):,}
Avg Agents/DP: {avg_agents_deciding:.2f}
Max Agents/DP: {max_agents_deciding}
Max Active in System: {max_active}
Total Jobs Generated: {max(total_arrived)}

Smoothing Window: {window} DPs'''
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, 
                edgecolor='black', linewidth=1.5, pad=0.7)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace', fontweight='normal')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / 'agent_count_signals.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved signal visualization: {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_agent_signals()
