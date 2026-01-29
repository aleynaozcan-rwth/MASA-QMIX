#!/usr/bin/env python3
"""
Single Episode Decision Point Timeline Analysis
Shows how decision points (arrivals vs completions) are distributed within episodes
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend for cluster without display
import matplotlib
matplotlib.use('Agg')

# ==================== Configuration ====================
TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Parse Timeline File ====================
print("Parsing scheduling timeline for single episode analysis...")

def parse_episode_details(episode_num):
    """Parse a specific episode and extract DP timeline"""
    dps = []  # List of {time, type, description}
    in_episode = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check if we're entering the target episode
            episode_match = re.match(r'^=== EPISODE (\d+) ===$', line)
            if episode_match:
                ep_num = int(episode_match.group(1))
                if ep_num == episode_num:
                    in_episode = True
                    continue
                elif in_episode:
                    # We've moved past our target episode
                    break
            
            if not in_episode:
                continue
            
            # Extract timestamp and event type
            # Job arrival: "[t=X.XX] New job ... arrived"
            arrival_match = re.match(r'\[t=([\d.]+)\] New job (.+?) arrived', line)
            if arrival_match:
                time = float(arrival_match.group(1))
                job_desc = arrival_match.group(2)
                dps.append({
                    'time': time,
                    'type': 'arrival',
                    'description': f'Job {job_desc} arrived'
                })
                continue
            
            # Operation completion: "[t=X.XX] ... finished ... next queued"
            completion_match = re.match(r'\[t=([\d.]+)\] (.+?) finished (.+?) next queued', line)
            if completion_match:
                time = float(completion_match.group(1))
                dps.append({
                    'time': time,
                    'type': 'completion',
                    'description': 'Operation completed'
                })
                continue
    
    return dps

# Select episodes to analyze (first 3 and a few scattered ones)
episodes_to_analyze = [0, 1, 2, 100, 500, 1000, 1294]

for ep_num in episodes_to_analyze:
    dps = parse_episode_details(ep_num)
    
    if not dps:
        print(f"Episode {ep_num}: No data found")
        continue
    
    # Separate arrivals and completions
    arrivals = [dp for dp in dps if dp['type'] == 'arrival']
    completions = [dp for dp in dps if dp['type'] == 'completion']
    
    print(f"Episode {ep_num}: {len(arrivals)} arrivals, {len(completions)} completions")
    
    # ==================== Visualization: Continuous Timeline ====================
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Get time range
    all_times = [dp['time'] for dp in dps]
    max_time = max(all_times) if all_times else 40
    
    # Extract times for each type
    arrival_times = [dp['time'] for dp in arrivals]
    completion_times = [dp['time'] for dp in completions]
    
    # Plot vertical lines for each decision point
    for t in arrival_times:
        ax.axvline(x=t, color='#FF6B6B', linewidth=2.5, alpha=0.8)
    
    for t in completion_times:
        ax.axvline(x=t, color='#4ECDC4', linewidth=2.5, alpha=0.8)
    
    # Add legend with manual entries
    from matplotlib.patches import Patch
    legend_arrival = Patch(facecolor='#FF6B6B', edgecolor='#FF6B6B', alpha=0.8, 
                           label=f'Job Arrivals: {len(arrivals)} ({100*len(arrivals)/len(dps):.1f}%)')
    legend_completion = Patch(facecolor='#4ECDC4', edgecolor='#4ECDC4', alpha=0.8,
                             label=f'Operation Completions: {len(completions)} ({100*len(completions)/len(dps):.1f}%)')
    legend_total = Patch(facecolor='none', edgecolor='none',
                        label=f'Total DPs: {len(dps)}')
    
    ax.legend(handles=[legend_arrival, legend_completion, legend_total], 
             loc='upper right', fontsize=11, framealpha=0.9)
    
    ax.set_xlabel('Time (t)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Decision Points', fontsize=13, fontweight='bold')
    ax.set_title(f'Episode {ep_num}: Decision Point Timeline (Continuous Time)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, max_time + 2)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f'episode_{ep_num}_dp_timeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

print("\n✓ All episode timelines created!")
