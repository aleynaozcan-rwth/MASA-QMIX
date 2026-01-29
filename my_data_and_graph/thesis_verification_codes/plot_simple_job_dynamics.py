"""
Simple Job Dynamics Visualization
==================================

Clean and simple visualization of:
A) Total jobs arrived per episode
B) Total jobs completed per episode  
C) Completion rate (%)
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

TIMELINE_PATH = "./my_data_and_graph/historydata/scheduling_timeline.txt"
OUTPUT_DIR = "./my_data_and_graph/historydata"

def parse_episode_data():
    """
    Parse scheduling_timeline.txt and extract simple metrics per episode
    """
    print("\n" + "="*70)
    print("📊 PARSING EPISODE DATA FROM scheduling_timeline.txt")
    print("="*70 + "\n")
    
    if not os.path.exists(TIMELINE_PATH):
        print(f"❌ File not found: {TIMELINE_PATH}")
        return None
    
    episodes = []
    current_episode = None
    
    with open(TIMELINE_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Episode start
            if "=== EPISODE" in line and "===" in line:
                match = re.search(r'EPISODE (\d+)', line)
                if match:
                    ep_num = int(match.group(1))
                    if current_episode is not None:
                        episodes.append(current_episode)
                    current_episode = {
                        'episode': ep_num,
                        'total_jobs': 0,
                        'completed_jobs': 0,
                        'active_counts': []  # Track all active counts during episode
                    }
            
            # Episode end - get final counts
            elif "LIFECYCLE TRACE END" in line and current_episode is not None:
                episodes.append(current_episode)
                current_episode = None
            
            # Extract lifecycle snapshots for active agent counts
            elif "[Lifecycle]" in line and "Active=" in line and current_episode is not None:
                # [Lifecycle] t=46.80 | Active=9 Pending=0 Completed=17 / Total=26
                match = re.search(r'Active=(\d+).*Completed=(\d+).*Total=(\d+)', line)
                if match:
                    active = int(match.group(1))
                    completed = int(match.group(2))
                    total = int(match.group(3))
                    
                    current_episode['active_counts'].append(active)
                    current_episode['completed_jobs'] = max(current_episode['completed_jobs'], completed)
                    current_episode['total_jobs'] = max(current_episode['total_jobs'], total)
    
    # Add last episode if needed
    if current_episode is not None:
        episodes.append(current_episode)
    
    print(f"✅ Parsed {len(episodes)} episodes\n")
    return episodes

def plot_job_dynamics(episodes):
    """
    Create clean, simple line plots showing:
    A) Total jobs arrived per episode
    B) Completion rate
    With zoomed view for better fluctuation visibility
    """
    if not episodes:
        print("❌ No episode data to plot")
        return
    
    # Extract data
    episode_nums = [ep['episode'] for ep in episodes]
    
    # Calculate peak active agents per episode (this shows job arrival dynamics)
    peak_agents = []
    for ep in episodes:
        if ep['active_counts']:
            peak_agents.append(max(ep['active_counts']))
        else:
            peak_agents.append(0)
    
    completed_jobs = [ep['completed_jobs'] for ep in episodes]
    total_jobs = [ep['total_jobs'] for ep in episodes]
    
    completion_rates = [(c/t*100) if t > 0 else 0 
                        for c, t in zip(completed_jobs, total_jobs)]
    
    # ZOOM: Focus on first 250 episodes for better fluctuation visibility
    zoom_limit = min(250, len(episodes))
    ep_zoom = episode_nums[:zoom_limit]
    agents_zoom = peak_agents[:zoom_limit]
    rates_zoom = completion_rates[:zoom_limit]
    
    # Calculate moving averages (window size 20 for zoomed view)
    window = 20
    
    def moving_average(data, window):
        """Calculate moving average"""
        ma = []
        for i in range(len(data)):
            if i < window:
                ma.append(np.mean(data[:i+1]))
            else:
                ma.append(np.mean(data[i-window+1:i+1]))
        return ma
    
    agents_ma = moving_average(agents_zoom, window)
    rates_ma = moving_average(rates_zoom, window)
    
    print(f"📊 Creating plots for {zoom_limit} episodes (zoomed for clarity)...\n")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle('Job Dynamics Analysis - Episode Level (Zoomed View)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # === Plot A: Total Jobs Arrived ===
    ax1 = axes[0]
    # Raw data with markers (more visible fluctuation)
    ax1.plot(ep_zoom, agents_zoom, linewidth=2.5, color='#2E86AB', alpha=0.7, 
             marker='o', markersize=3, markerfacecolor='#2E86AB', markeredgewidth=0)
    ax1.fill_between(ep_zoom, agents_zoom, alpha=0.15, color='#2E86AB')
    # Moving average (subtle)
    ax1.plot(ep_zoom, agents_ma, linewidth=3, color='#1a4d6d', alpha=0.9, 
             linestyle='--', label=f'Trend (MA{window})')
    
    ax1.set_ylabel('Total Jobs Arrived', fontsize=13, fontweight='bold')
    ax1.set_title('A) Total Jobs Arrived per Episode', 
                  fontsize=13, pad=10, loc='left', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Tight Y-axis for better fluctuation visibility
    y_min = min(agents_zoom) - 0.5
    y_max = max(agents_zoom) + 0.5
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc='best', fontsize=10)
    
    # === Plot B: Completion Rate ===
    ax2 = axes[1]
    # Raw data with markers
    ax2.plot(ep_zoom, rates_zoom, linewidth=2.5, color='#F77F00', alpha=0.7,
             marker='o', markersize=3, markerfacecolor='#F77F00', markeredgewidth=0)
    ax2.fill_between(ep_zoom, rates_zoom, alpha=0.15, color='#F77F00')
    # Moving average (subtle)
    ax2.plot(ep_zoom, rates_ma, linewidth=3, color='#a85300', alpha=0.9,
             linestyle='--', label=f'Trend (MA{window})')
    
    ax2.set_xlabel('Episode Number', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Completion Rate (%)', fontsize=13, fontweight='bold')
    ax2.set_title('B) Job Completion Rate per Episode', 
                  fontsize=13, pad=10, loc='left', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Tight Y-axis for better fluctuation visibility
    y_min_rate = max(30, min(rates_zoom) - 2)  # Don't go below 30%
    y_max_rate = min(100, max(rates_zoom) + 2)  # Don't exceed 100%
    ax2.set_ylim(y_min_rate, y_max_rate)
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'job_dynamics_simple.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {output_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("📊 SUMMARY STATISTICS (Zoomed Episodes 0-{})".format(zoom_limit-1))
    print("="*70)
    print(f"\nTotal Jobs Arrived:")
    print(f"  Mean:                      {np.mean(agents_zoom):.1f}")
    print(f"  Range:                     {min(agents_zoom)} - {max(agents_zoom)}")
    print(f"  Std Dev:                   {np.std(agents_zoom):.2f}")
    print(f"  Fluctuation Range:         {max(agents_zoom) - min(agents_zoom)} jobs")
    print(f"\nCompletion Rate:")
    print(f"  Mean:                      {np.mean(rates_zoom):.1f}%")
    print(f"  Range:                     {min(rates_zoom):.1f}% - {max(rates_zoom):.1f}%")
    print(f"  Std Dev:                   {np.std(rates_zoom):.2f}%")
    print(f"  Fluctuation Range:         {max(rates_zoom) - min(rates_zoom):.1f}%")
    print("="*70 + "\n")

def main():
    episodes = parse_episode_data()
    
    if not episodes:
        print("❌ Failed to parse episode data")
        return
    
    plot_job_dynamics(episodes)
    
    print("🎉 ANALYSIS COMPLETE!\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
