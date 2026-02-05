"""
Extract ThroughputDelta from scheduling_timeline.txt - Episode-based analysis
Calculates throughput delta directly from timeline instead of reward_components.csv
"""

import os
import re
import matplotlib.pyplot as plt
import pandas as pd

def parse_scheduling_timeline(timeline_path):
    """Parse scheduling_timeline.txt to extract job completions - continuous across all episodes"""
    
    if not os.path.exists(timeline_path):
        print(f"❌ {timeline_path} not found!")
        return None
    
    all_decision_points = []
    current_episode = None
    prev_completed = 0
    total_jobs_in_episode = 0
    
    with open(timeline_path, 'r') as f:
        for line in f:
            # Detect episode start
            episode_match = re.search(r'=== EPISODE (\d+) ===', line)
            if episode_match:
                current_episode = int(episode_match.group(1))
                prev_completed = 0
                total_jobs_in_episode = 0
                continue
            
            if current_episode is None:
                continue
            
            # Track new job arrivals to know total_jobs
            new_job_match = re.search(r'\[t=([\d.]+)\] New job .+ arrived', line)
            if new_job_match:
                total_jobs_in_episode += 1
            
            # Parse job completion events ONLY
            completion_match = re.search(r'\[t=([\d.]+)\] (.+) completed all operations', line)
            if completion_match:
                sim_time = float(completion_match.group(1))
                completed_now = prev_completed + 1
                
                # Throughput delta: 1 job completed / total jobs in this episode
                throughput_delta = 1.0 / max(1, total_jobs_in_episode)
                
                all_decision_points.append({
                    'episode': current_episode,
                    'sim_time': sim_time,
                    'completed_in_episode': completed_now,
                    'total_jobs': total_jobs_in_episode,
                    'throughput_delta': throughput_delta
                })
                
                prev_completed = completed_now
    
    return all_decision_points


def plot_throughput_from_timeline(history_dir='my_data_and_graph/historydata'):
    """Generate throughput analysis from scheduling_timeline.txt - First & Last 50 episodes"""
    
    timeline_path = os.path.join(history_dir, 'scheduling_timeline.txt')
    
    print(f"📖 Parsing {timeline_path}...")
    all_dps = parse_scheduling_timeline(timeline_path)
    
    if not all_dps:
        print("❌ No decision point data found!")
        return
    
    # Create DataFrame with continuous dp_index
    df = pd.DataFrame(all_dps)
    df['dp_index'] = range(len(df))  # Continuous 0, 1, 2, 3...
    
    print(f"✅ Parsed {len(df)} job completion decision points")
    print(f"   Across {df['episode'].nunique()} episodes")
    
    # Split into first 50 and last 50 episodes
    df_first = df[df['episode'] < 50].copy()
    df_last = df[df['episode'] >= df['episode'].max() - 49].copy()
    
    print(f"   First 50 episodes: {len(df_first)} decision points")
    print(f"   Last 50 episodes: {len(df_last)} decision points")
    
    # Create figure with two subplots (first 200, last 200)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=150)
    
    # ===== TOP PLOT: First 200 episodes =====
    window1 = min(100, len(df_first) // 10) if len(df_first) > 100 else 20
    df_first['td_ma'] = df_first['throughput_delta'].rolling(window=window1, center=True, min_periods=1).mean()
    
    # Use line plot instead of scatter for better visibility
    ax1.plot(df_first['dp_index'], df_first['throughput_delta'], linewidth=0.5, alpha=0.6, color='green', 
             label='ThroughputDelta')
    ax1.plot(df_first['dp_index'], df_first['td_ma'], color='red', linewidth=3.0, 
             label=f'{window1}-point MA', alpha=1.0)
    
    ax1.set_xlabel('Decision Point (continuous)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('ThroughputDelta', fontsize=11, fontweight='bold')
    ax1.set_title(f'ThroughputDelta - First 50 Episodes (Episodes 0-49)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0.015, 0.035])  # Set y-axis range for better visibility
    
    mean_first = df_first['throughput_delta'].mean()
    std_first = df_first['throughput_delta'].std()
    
    textstr1 = f"Mean: {mean_first:.4f}\n"
    textstr1 += f"Std Dev: {std_first:.4f}\n"
    textstr1 += f"Points: {len(df_first)}"
    
    ax1.text(0.02, 0.98, textstr1, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'),
             fontsize=9)
    
    # ===== BOTTOM PLOT: Last 50 episodes =====
    window2 = min(50, len(df_last) // 10) if len(df_last) > 50 else 10
    df_last['td_ma'] = df_last['throughput_delta'].rolling(window=window2, center=True, min_periods=1).mean()
    
    # Use line plot instead of scatter for better visibility
    ax2.plot(df_last['dp_index'], df_last['throughput_delta'], linewidth=0.5, alpha=0.6, color='blue', 
             label='ThroughputDelta')
    ax2.plot(df_last['dp_index'], df_last['td_ma'], color='red', linewidth=3.0, 
             label=f'{window2}-point MA', alpha=1.0)
    
    ax2.set_xlabel('Decision Point (continuous)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('ThroughputDelta', fontsize=11, fontweight='bold')
    ax2.set_title(f'ThroughputDelta - Last 200 Episodes (Episodes {df_last["episode"].min()}-{df_last["episode"].max()})', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0.015, 0.035])  # Set y-axis range for better visibility
    
    mean_last = df_last['throughput_delta'].mean()
    std_last = df_last['throughput_delta'].std()
    
    textstr2 = f"Mean: {mean_last:.4f}\n"
    textstr2 += f"Std Dev: {std_last:.4f}\n"
    textstr2 += f"Points: {len(df_last)}"
    
    ax2.text(0.02, 0.98, textstr2, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'),
             fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(history_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'throughput_from_timeline.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(history_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'throughput_delta_signal.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot saved: {out_path}")
    print(f"\n📊 ThroughputDelta Comparison:")
    print(f"   First 50 episodes - Mean: {mean_first:.4f} ± {std_first:.4f}")
    print(f"   Last 50 episodes  - Mean: {mean_last:.4f} ± {std_last:.4f}")
    print(f"   Improvement: {((mean_last - mean_first) / mean_first * 100):+.2f}%")


if __name__ == '__main__':
    plot_throughput_from_timeline()
