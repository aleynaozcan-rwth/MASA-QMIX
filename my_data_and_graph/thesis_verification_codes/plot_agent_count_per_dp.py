#!/usr/bin/env python3
"""
Agent Count Per Decision Point Visualization
Shows:
- Number of agents making decisions at each DP
- Total agents in the system over time
- Active agents in the system
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

def plot_agent_counts():
    """Create visualization of agent counts across all decision points."""
    
    print("Parsing timeline for agent count analysis...")
    all_dps = parse_agent_counts()
    
    if not all_dps:
        print("No decision points found!")
        return
    
    print(f"Found {len(all_dps)} decision points across all episodes")
    
    # Extract data for plotting
    dp_indices = list(range(len(all_dps)))
    agents_deciding = [dp['num_agents_deciding'] for dp in all_dps]
    total_arrived = [dp['total_arrived'] for dp in all_dps]
    active_agents = [dp['active_agents'] for dp in all_dps]
    total_completed = [dp['total_completed'] for dp in all_dps]
    
    # Statistics
    avg_agents_deciding = np.mean(agents_deciding)
    max_agents_deciding = max(agents_deciding)
    max_active = max(active_agents)
    
    print(f"\nStatistics:")
    print(f"  Average agents per DP: {avg_agents_deciding:.2f}")
    print(f"  Max agents at single DP: {max_agents_deciding}")
    print(f"  Max active agents in system: {max_active}")
    print(f"  Total jobs across all episodes: {max(total_arrived)}")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))
    fig.subplots_adjust(hspace=0.3)
    
    # ==================== Plot 1: Agents per Decision Point ====================
    ax1.plot(dp_indices, agents_deciding, linewidth=0.8, color='#2E86AB', alpha=0.7, label='Agents Making Decision')
    ax1.fill_between(dp_indices, agents_deciding, alpha=0.3, color='#2E86AB')
    
    ax1.set_xlabel('Decision Point Index (Chronological)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Agents Deciding', fontsize=13, fontweight='bold')
    ax1.set_title('Agent Count Per Decision Point', fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10)
    
    # Add horizontal line for average
    ax1.axhline(y=avg_agents_deciding, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.6, label=f'Average: {avg_agents_deciding:.2f}')
    
    # ==================== Plot 2: System State ====================
    ax2.plot(dp_indices, total_arrived, linewidth=2, color='#06A77D', alpha=0.8, 
             label='Total Arrivals (Cumulative)', marker='', markersize=1)
    ax2.plot(dp_indices, active_agents, linewidth=2, color='#D62246', alpha=0.8, 
             label='Active Agents', marker='', markersize=1)
    ax2.plot(dp_indices, total_completed, linewidth=2, color='#F77F00', alpha=0.8, 
             label='Total Completed (Cumulative)', marker='', markersize=1)
    
    ax2.set_xlabel('Decision Point Index (Chronological)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of Agents', fontsize=13, fontweight='bold')
    ax2.set_title('System Agent Lifecycle Over Decision Points', fontsize=15, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.fill_between(dp_indices, active_agents, alpha=0.2, color='#D62246')
    
    # Save figure
    output_path = OUTPUT_DIR / 'agent_count_per_dp.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved visualization: {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_agent_counts()
