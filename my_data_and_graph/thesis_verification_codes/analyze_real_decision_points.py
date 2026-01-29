"""
Real Decision Point Analysis from Scheduling Timeline
Shows actual event-driven decision points with continuous time tracking
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import os


def analyze_real_decision_points(timeline_file='my_data_and_graph/historydata/scheduling_timeline.txt',
                                  history_dir='my_data_and_graph/historydata'):
    """Analyze real decision points from scheduling timeline."""
    
    # Parse the file
    episodes = defaultdict(list)
    current_episode = None
    
    print('Reading scheduling timeline...')
    with open(timeline_file, 'r') as f:
        for line in f:
            episode_match = re.match(r'=== EPISODE (\d+) ===', line)
            if episode_match:
                current_episode = int(episode_match.group(1))
                continue
            
            if current_episode is not None:
                # Job arrivals
                arrival_match = re.match(r'\[t=([\d.]+)\] New job.*arrived', line)
                if arrival_match:
                    time = float(arrival_match.group(1))
                    episodes[current_episode].append(('arrival', time))
                    continue
                
                # Operation completions
                completion_match = re.match(r'\[t=([\d.]+)\].*finished.*next queued', line)
                if completion_match:
                    time = float(completion_match.group(1))
                    episodes[current_episode].append(('completion', time))
                    continue
    
    print(f'Parsed {len(episodes)} episodes')
    
    # Calculate statistics
    episode_stats = []
    for ep_num in sorted(episodes.keys()):
        dp_list = sorted(episodes[ep_num], key=lambda x: x[1])
        if dp_list:
            duration = dp_list[-1][1] - dp_list[0][1] if len(dp_list) > 1 else 0
            arrivals = sum(1 for t, _ in dp_list if t == 'arrival')
            completions = sum(1 for t, _ in dp_list if t == 'completion')
            
            episode_stats.append({
                'episode': ep_num,
                'dp_count': len(dp_list),
                'duration': duration,
                'arrivals': arrivals,
                'completions': completions,
                'dp_frequency': len(dp_list) / duration if duration > 0 else 0,
                'times': [t for _, t in dp_list]
            })
    
    # Create visualization
    plots_dir = os.path.join(history_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'real_decision_points_timeline.png')
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.2, 1.5], hspace=0.5, wspace=0.3)
    
    # ==================== Plot 1: Decision Points per Episode ====================
    ax1 = fig.add_subplot(gs[0, :])
    
    episodes_arr = [s['episode'] for s in episode_stats]
    dp_counts = [s['dp_count'] for s in episode_stats]
    
    # Color by DP count
    colors = plt.cm.viridis((np.array(dp_counts) - min(dp_counts)) / (max(dp_counts) - min(dp_counts)))
    ax1.bar(episodes_arr, dp_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add moving average
    if len(dp_counts) > 20:
        window = 50
        ma = np.convolve(dp_counts, np.ones(window)/window, mode='valid')
        ma_episodes = episodes_arr[window-1:]
        ax1.plot(ma_episodes, ma, color='red', linewidth=3, linestyle='--', 
                label=f'{window}-episode MA', zorder=10)
    
    mean_dp = np.mean(dp_counts)
    ax1.axhline(y=mean_dp, color='orange', linestyle=':', linewidth=2.5, alpha=0.7,
               label=f'Mean: {mean_dp:.1f} DPs')
    
    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Decision Points', fontsize=12, fontweight='bold')
    ax1.set_title('Real Decision Points per Episode (from Scheduling Timeline)\nVariable DPs reflect TRUE event-driven dynamics!', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ==================== Plot 2: DP Distribution Histogram ====================
    ax2 = fig.add_subplot(gs[1, :])
    
    ax2.hist(dp_counts, bins=range(min(dp_counts), max(dp_counts)+2), 
            alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.2)
    
    ax2.axvline(mean_dp, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_dp:.1f}')
    ax2.axvline(np.median(dp_counts), color='green', linestyle=':', linewidth=2.5, 
               label=f'Median: {np.median(dp_counts):.0f}')
    
    ax2.set_xlabel('Decision Points per Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('DP Distribution: Shows Execution Variability', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ==================== Plot 3: DP Frequency Evolution ====================
    ax5 = fig.add_subplot(gs[2, :])
    
    dp_frequencies = [s['dp_frequency'] for s in episode_stats]
    
    ax5.plot(episodes_arr, dp_frequencies, linewidth=1.5, color='purple', alpha=0.7)
    
    if len(dp_frequencies) > 20:
        window = 20
        freq_ma = np.convolve(dp_frequencies, np.ones(window)/window, mode='valid')
        ma_episodes_short = episodes_arr[window-1:]
        ax5.plot(ma_episodes_short, freq_ma, color='red', linewidth=2.5, linestyle='--',
                label=f'{window}-ep MA')
    
    mean_freq = np.mean(dp_frequencies)
    std_freq = np.std(dp_frequencies)
    ax5.axhline(y=mean_freq, color='orange', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Mean: {mean_freq:.2f} DPs/time')
    
    # Set y-axis limits to emphasize variation (mean +/- 2.5 std deviations)
    y_min = max(0, mean_freq - 2.5 * std_freq)
    y_max = mean_freq + 2.5 * std_freq
    ax5.set_ylim(y_min, y_max)
    
    ax5.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax5.set_ylabel('DP Frequency (DPs / time unit)', fontsize=12, fontweight='bold')
    ax5.set_title('Decision Point Density Evolution (Zoomed to Show Variation)', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # ==================== Plot 4: Detailed Timeline for Sample Episodes ====================
    # Remove this plot since we need more space for other graphs
    # ax6 = fig.add_subplot(gs[2, :])
    
    # ==================== Summary Statistics ====================
    total_dps = sum(dp_counts)
    avg_dps = np.mean(dp_counts)
    std_dps = np.std(dp_counts)
    
    summary_text = f"""Episodes: {len(episodes)} | Total Decision Points: {total_dps:,}
DPs/Episode: {avg_dps:.1f}±{std_dps:.1f} (range: {min(dp_counts)}-{max(dp_counts)})"""
    
    fig.text(0.98, 0.01, summary_text, fontsize=8, 
            verticalalignment='bottom', horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', alpha=0.9, 
                     edgecolor='gray', linewidth=1))
    
    plt.suptitle('Real Event-Driven Decision Points Analysis\n(Extracted from Continuous-Time Scheduling Timeline)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'\n✓ Saved real decision point analysis: {out_path}')
    print(f'\nTotal Decision Points: {total_dps:,}')
    print(f'Average per Episode: {avg_dps:.2f} ± {std_dps:.2f}')
    print(f'Range: {min(dp_counts)}-{max(dp_counts)} DPs/episode')


if __name__ == '__main__':
    analyze_real_decision_points()
