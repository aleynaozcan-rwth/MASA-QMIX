#!/usr/bin/env python3
"""
Throughput Delta Plot - Shows step-to-step completion rate changes
"""

import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_throughput_delta(history_dir='my_data_and_graph/historydata'):
    """Generate throughput delta plot from reward_components.csv - Episode-based analysis"""
    
    csv_path = os.path.join(history_dir, 'reward_components.csv')
    if not os.path.exists(csv_path):
        print(f"❌ {csv_path} not found!")
        return
    
    try:
        # Read CSV, skip comment lines
        df = pd.read_csv(csv_path, comment='#')
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    if 'ThroughputDelta' not in df.columns or 'sim_time' not in df.columns:
        print("❌ Required columns not found!")
        return
    
    print(f"📊 Loaded {len(df)} decision points from reward_components.csv")
    
    # Detect episode boundaries (sim_time resets to near 0)
    df['episode'] = (df['sim_time'] < df['sim_time'].shift(1, fill_value=0)).cumsum()
    
    # Calculate per-episode throughput (total jobs completed in episode)
    episode_stats = []
    for ep, group in df.groupby('episode'):
        total_completions = (group['ThroughputDelta'] > 0).sum()
        episode_stats.append({
            'episode': ep,
            'jobs_completed': total_completions,
            'decision_points': len(group)
        })
    
    episode_df = pd.DataFrame(episode_stats)
    
    print(f"📊 Detected {len(episode_df)} episodes")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=150)
    
    # Top plot: Jobs completed per episode
    ax1.bar(episode_df['episode'], episode_df['jobs_completed'], 
            color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=0.5)
    
    # Add moving average
    window = min(20, len(episode_df) // 5) if len(episode_df) > 10 else 5
    episode_df['jobs_completed_ma'] = episode_df['jobs_completed'].rolling(window=window, center=True, min_periods=1).mean()
    ax1.plot(episode_df['episode'], episode_df['jobs_completed_ma'], 
             color='red', linewidth=2.5, label=f'{window}-episode MA')
    
    ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Jobs Completed', fontsize=11, fontweight='bold')
    ax1.set_title('Jobs Completed per Episode', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Calculate statistics
    mean_completions = episode_df['jobs_completed'].mean()
    std_completions = episode_df['jobs_completed'].std()
    
    textstr = f"Mean: {mean_completions:.1f} jobs/episode\n"
    textstr += f"Std Dev: {std_completions:.1f}\n"
    textstr += f"Min: {episode_df['jobs_completed'].min():.0f}\n"
    textstr += f"Max: {episode_df['jobs_completed'].max():.0f}"
    
    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'),
             fontsize=9)
    
    # Bottom plot: Histogram of jobs completed per episode
    ax2.hist(episode_df['jobs_completed'], bins=range(0, int(episode_df['jobs_completed'].max())+2), 
             color='steelblue', alpha=0.7, edgecolor='black', align='left')
    ax2.axvline(x=mean_completions, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label=f'Mean: {mean_completions:.1f}')
    
    ax2.set_xlabel('Jobs Completed per Episode', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency (Number of Episodes)', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Jobs Completed per Episode', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(history_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'throughput_delta.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Throughput Delta plot saved: {out_path}")
    print(f"\n📈 Episode-based Job Completion Statistics:")
    print(f"   Total episodes: {len(episode_df)}")
    print(f"   Total decision points: {len(df)}")
    print(f"   Mean jobs/episode: {mean_completions:.1f} ± {std_completions:.1f}")
    print(f"   Min-Max: {episode_df['jobs_completed'].min():.0f} - {episode_df['jobs_completed'].max():.0f}")
    print(f"   Total jobs completed: {episode_df['jobs_completed'].sum():.0f}")


if __name__ == '__main__':
    plot_throughput_delta()
