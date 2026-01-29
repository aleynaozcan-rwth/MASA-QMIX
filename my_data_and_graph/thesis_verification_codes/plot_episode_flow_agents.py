#!/usr/bin/env python3
"""
Plot agent counts for episode flow parallel DPs (chronological order)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

INPUT_FILE = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/episode_flow_parallel_dps.csv')
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_agent_counts():
    """Plot agent counts for all parallel DPs in chronological order."""
    
    print("Reading episode flow data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Sort by episode then timestamp for chronological order
    df = df.sort_values(['Episode', 'Timestamp']).reset_index(drop=True)
    
    print(f"Total parallel DPs: {len(df)}")
    print(f"Agent distribution:")
    for agent in sorted(df['Agent_Count'].unique()):
        count = len(df[df['Agent_Count'] == agent])
        print(f"  {agent}-agent: {count} ({count/len(df)*100:.1f}%)")
    
    # Create figure - show ALL DPs
    fig, ax = plt.subplots(figsize=(30, 6))
    
    plot_range = len(df)  # Show all DPs
    x = np.arange(plot_range)
    y = df['Agent_Count'].values
    
    # Plot line
    ax.plot(x, y, color='#1f77b4', linewidth=0.8, marker='o', markersize=1)
    
    # Highlight every 15th DP with markers and labels
    tick_interval = 15
    highlight_positions = np.arange(0, plot_range, tick_interval)
    for pos in highlight_positions:
        if pos < len(y):
            agent_count = int(y[pos])
            ax.plot(pos, agent_count, 'ro', markersize=5, zorder=5)  # Red marker
            ax.text(pos, agent_count + 0.15, str(agent_count), 
                   fontsize=8, fontweight='bold', ha='center', va='bottom',
                   color='red')
    
    # Styling
    ax.set_xlabel('Parallel Decision Point Index (every 15th DP marked)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Agent Count', fontsize=12, fontweight='bold')
    ax.set_title('Agent Counts per Parallel Decision Point (Chronological Order)', 
                fontsize=13, fontweight='bold', pad=15)
    
    ax.set_xlim(-1, plot_range)
    ax.set_ylim(0, 5)
    
    # Set x-axis ticks every 15 DPs
    tick_positions = np.arange(0, plot_range, tick_interval)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(pos)) for pos in tick_positions], fontsize=7)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics box
    stats_text = f'''Total Parallel DPs: {len(df):,}
Mean: {df['Agent_Count'].mean():.2f}
2 agents: {len(df[df['Agent_Count']==2]):,} ({len(df[df['Agent_Count']==2])/len(df)*100:.0f}%)
3 agents: {len(df[df['Agent_Count']==3]):,} ({len(df[df['Agent_Count']==3])/len(df)*100:.0f}%)
4 agents: {len(df[df['Agent_Count']==4]):,} ({len(df[df['Agent_Count']==4])/len(df)*100:.0f}%)
Episodes: {df['Episode'].nunique():,}'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / 'episode_flow_agent_counts.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Showing all {plot_range} DPs")

if __name__ == '__main__':
    plot_agent_counts()
