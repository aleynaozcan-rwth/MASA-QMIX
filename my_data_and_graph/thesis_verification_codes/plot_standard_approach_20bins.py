#!/usr/bin/env python3
"""
Standard Approach visualization: Fixed 25 DPs per episode
- No conflicts
- No parallel decision making
- All decision points are non-parallel (sequential)
- Total DPs = 25 * number of episodes
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_standard_approach_binned():
    """Create stacked bar chart for standard approach (all non-parallel)."""
    
    # Standard approach parameters
    total_episodes = 1295  # Same as in the proposed framework
    dps_per_episode = 25   # Fixed for standard approach
    total_dps = total_episodes * dps_per_episode
    
    print(f"Standard Approach Statistics:")
    print(f"  Total Episodes: {total_episodes}")
    print(f"  DPs per Episode: {dps_per_episode} (fixed)")
    print(f"  Total DPs: {total_dps:,}")
    print(f"  Non-Parallel DPs: {total_dps:,} (100%)")
    print(f"  Parallel DPs: 0")
    print(f"  Conflicts: 0")
    
    # Group into 15-episode bins (same as proposed framework)
    bin_size = 15
    num_bins = (total_episodes + bin_size - 1) // bin_size
    
    bin_labels = []
    non_parallel_means = []
    
    for i in range(num_bins):
        start_ep = i * bin_size
        end_ep = min((i + 1) * bin_size, total_episodes)
        
        # All DPs are non-parallel, always 25 per episode
        non_parallel_means.append(dps_per_episode)
        
        bin_labels.append(f'{start_ep}-{end_ep - 1}')
    
    print(f"\nBins created: {num_bins}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 8))
    
    x_pos = np.arange(len(bin_labels))
    width = 0.8
    
    # Color - same as non-parallel from proposed framework
    color_nonparallel = '#6B9BD1'  # Light blue
    
    # Single bar for all non-parallel DPs
    bars = ax.bar(x_pos, non_parallel_means, width, 
                  color=color_nonparallel, alpha=0.9)
    
    # X-axis
    ax.set_xlabel('Episode Range (15-Episode Windows)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos[::3])
    ax.set_xticklabels([bin_labels[i] for i in range(0, len(bin_labels), 3)], 
                       rotation=45, ha='right', fontsize=9)
    
    # Y-axis
    ax.set_ylabel('Total Decision Points per Episode (Mean)', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.5, len(bin_labels) - 0.5)
    ax.set_ylim(0, 35)
    ax.margins(0)
    
    title = 'Decision Points Distribution (Standard Approach)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Add legend in top-left corner
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_nonparallel, label='Non-Parallel DPs (Sequential)')
    ]
    legend = ax.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(0.01, 0.995), fontsize=10,
                      frameon=True, fancybox=True, shadow=True,
                      facecolor='white', edgecolor='black', framealpha=0.95)
    
    # Add statistics box in top-right corner
    stats_text = f'''Total DPs: {total_dps:,}
Non-Parallel DPs: {total_dps:,} (100.0% of Total)
Parallel DPs: 0 (0.0% of Total)
  - Simultaneous: 0 (0.0%)
  - Conflicts: 0 (0.0%)

Sequential Decision Making: 10 Agents'''
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, 
                edgecolor='black', linewidth=1.5, pad=0.7)
    ax.text(0.78, 0.97, stats_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', horizontalalignment='left',
            bbox=props, family='monospace', fontweight='normal')
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.70, top=0.92, bottom=0.08)
    
    # Save
    output_path = OUTPUT_DIR / 'standard_approach_20bins.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved standard approach plot: {output_path}")


if __name__ == '__main__':
    plot_standard_approach_binned()
