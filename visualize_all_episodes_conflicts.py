import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('my_data_and_graph/historydata/decision_observation_metrics.csv')

# Get true episode count from timeline
import re
with open('my_data_and_graph/historydata/scheduling_timeline.txt', 'r') as f:
    content = f.read()
    true_episodes = len(re.findall(r'=== EPISODE (\d+) ===', content))

# Estimate episode boundaries (CSV has no episode column, mixed order)
estimated_ep_size = int(len(df) / true_episodes)
episode_breaks = list(range(0, len(df), estimated_ep_size))
episode_breaks.append(len(df))

print(f"Analyzing {true_episodes} episodes (estimated boundaries)...")

# Analyze all episodes
episode_stats = []
parallel_decision_sizes = []
conflict_rates_per_episode = []

for ep_idx in range(len(episode_breaks)-1):
    start, end = episode_breaks[ep_idx], episode_breaks[ep_idx+1]
    ep_df = df.iloc[start:end]
    
    total_decisions = len(ep_df)
    total_conflicts = len(ep_df[ep_df['action_idx'] == -1])
    
    # Parallel decision points
    time_groups = ep_df.groupby('decision_time').size()
    parallel_points = time_groups[time_groups > 1]
    
    episode_stats.append({
        'episode': ep_idx,
        'total_decisions': total_decisions,
        'conflicts': total_conflicts,
        'conflict_rate': total_conflicts / total_decisions if total_decisions > 0 else 0,
        'decision_points': len(time_groups),
        'parallel_points': len(parallel_points),
        'max_parallel': time_groups.max() if len(time_groups) > 0 else 0,
        'avg_parallel': time_groups.mean() if len(time_groups) > 0 else 0
    })
    
    # Collect all parallel decision sizes
    parallel_decision_sizes.extend(time_groups.tolist())
    
    if total_conflicts > 0:
        conflict_rates_per_episode.append(total_conflicts / total_decisions * 100)

stats_df = pd.DataFrame(episode_stats)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Conflict Analysis Across All Training Episodes', 
             fontsize=18, fontweight='bold', y=0.995)

# 1. Conflict rate over episodes
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(stats_df['episode'], stats_df['conflict_rate'] * 100, 
           alpha=0.4, s=20, color='coral')
# Moving average
window = 50
if len(stats_df) > window:
    ma = stats_df['conflict_rate'].rolling(window=window, center=True).mean() * 100
    ax1.plot(stats_df['episode'], ma, color='red', linewidth=2, 
            label=f'{window}-episode moving average')
ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax1.set_ylabel('Conflict Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('Conflict Rate Evolution During Training', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend()

# 2. Conflict rate histogram
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(stats_df['conflict_rate'] * 100, bins=30, color='coral', 
        alpha=0.7, edgecolor='black')
ax2.set_xlabel('Conflict Rate (%)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Number of Episodes', fontsize=10, fontweight='bold')
ax2.set_title('Conflict Rate Distribution', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3)

# 3. Parallel decision size distribution
ax3 = fig.add_subplot(gs[1, 0])
counts, bins, patches = ax3.hist(parallel_decision_sizes, bins=range(1, 
                                 max(parallel_decision_sizes)+2), 
                                 color='steelblue', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Parallel Decisions', fontsize=10, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax3.set_title('Parallel Decision Size Distribution', fontsize=11, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(alpha=0.3)

# 4. Max parallel decisions per episode
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(stats_df['episode'], stats_df['max_parallel'], 
           alpha=0.3, s=15, color='steelblue')
if len(stats_df) > window:
    ma_parallel = stats_df['max_parallel'].rolling(window=window, center=True).mean()
    ax4.plot(stats_df['episode'], ma_parallel, color='darkblue', linewidth=2,
            label=f'{window}-episode MA')
ax4.set_xlabel('Episode', fontsize=10, fontweight='bold')
ax4.set_ylabel('Max Parallel Decisions', fontsize=10, fontweight='bold')
ax4.set_title('Maximum Parallelism per Episode', fontsize=11, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.legend()

# 5. Conflicts vs decisions
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(stats_df['total_decisions'], stats_df['conflicts'], 
           alpha=0.4, s=20, color='red')
ax5.set_xlabel('Total Decisions', fontsize=10, fontweight='bold')
ax5.set_ylabel('Conflicts', fontsize=10, fontweight='bold')
ax5.set_title('Conflicts vs Episode Length', fontsize=11, fontweight='bold')
ax5.grid(alpha=0.3)

# 6. Episodes with conflicts
ax6 = fig.add_subplot(gs[2, 0])
episodes_with_conflicts = len(stats_df[stats_df['conflicts'] > 0])
episodes_without = len(stats_df[stats_df['conflicts'] == 0])
ax6.bar(['With Conflicts', 'No Conflicts'], 
       [episodes_with_conflicts, episodes_without],
       color=['red', 'green'], alpha=0.7, edgecolor='black')
ax6.set_ylabel('Number of Episodes', fontsize=10, fontweight='bold')
ax6.set_title('Episodes by Conflict Presence', fontsize=11, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
for i, (label, val) in enumerate(zip(['With Conflicts', 'No Conflicts'], 
                                     [episodes_with_conflicts, episodes_without])):
    pct = val / len(stats_df) * 100
    ax6.text(i, val, f'{val}\n({pct:.1f}%)', 
            ha='center', va='bottom', fontweight='bold')

# 7. Parallel decision points ratio
ax7 = fig.add_subplot(gs[2, 1])
parallel_ratio = stats_df['parallel_points'] / stats_df['decision_points']
ax7.hist(parallel_ratio * 100, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax7.set_xlabel('Parallel Decision Points (%)', fontsize=10, fontweight='bold')
ax7.set_ylabel('Number of Episodes', fontsize=10, fontweight='bold')
ax7.set_title('Parallelism Ratio Distribution', fontsize=11, fontweight='bold')
ax7.grid(alpha=0.3)

# 8. Summary statistics box
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
summary_text = f"""
TRAINING SUMMARY
{'='*30}

Total Episodes: {len(stats_df):,}
Total Decisions: {stats_df['total_decisions'].sum():,}
Total Conflicts: {stats_df['conflicts'].sum():,}

Overall Conflict Rate: {stats_df['conflicts'].sum() / stats_df['total_decisions'].sum() * 100:.2f}%

Episodes with Conflicts: {episodes_with_conflicts:,}
Episodes without: {episodes_without:,}

Avg Decisions/Episode: {stats_df['total_decisions'].mean():.1f}
Avg Conflicts/Episode: {stats_df['conflicts'].mean():.2f}

Max Parallel Decisions: {stats_df['max_parallel'].max():.0f}
Avg Max Parallel: {stats_df['max_parallel'].mean():.1f}

Parallel Decision Points: {stats_df['parallel_points'].sum():,}
Total Decision Points: {stats_df['decision_points'].sum():,}
Parallelism: {stats_df['parallel_points'].sum() / stats_df['decision_points'].sum() * 100:.1f}%
"""
ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('all_episodes_conflict_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('all_episodes_conflict_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: all_episodes_conflict_analysis.pdf")
print("✓ Saved: all_episodes_conflict_analysis.png")

# Print statistics
print("\n" + "="*70)
print("AGGREGATE STATISTICS ACROSS ALL EPISODES")
print("="*70)
print(f"Total episodes: {len(stats_df):,}")
print(f"Total decisions: {stats_df['total_decisions'].sum():,}")
print(f"Total conflicts: {stats_df['conflicts'].sum():,}")
print(f"Overall conflict rate: {stats_df['conflicts'].sum() / stats_df['total_decisions'].sum() * 100:.2f}%")
print(f"\nEpisodes with conflicts: {episodes_with_conflicts:,} ({episodes_with_conflicts/len(stats_df)*100:.1f}%)")
print(f"Episodes without conflicts: {episodes_without:,} ({episodes_without/len(stats_df)*100:.1f}%)")
print(f"\nAverage decisions per episode: {stats_df['total_decisions'].mean():.2f}")
print(f"Average conflicts per episode: {stats_df['conflicts'].mean():.2f}")
print(f"Average conflict rate per episode: {stats_df['conflict_rate'].mean() * 100:.2f}%")
print(f"\nMax parallel decisions observed: {stats_df['max_parallel'].max():.0f}")
print(f"Average max parallel per episode: {stats_df['max_parallel'].mean():.2f}")
print(f"\nTotal parallel decision points: {stats_df['parallel_points'].sum():,}")
print(f"Total decision points: {stats_df['decision_points'].sum():,}")
print(f"Parallelism ratio: {stats_df['parallel_points'].sum() / stats_df['decision_points'].sum() * 100:.1f}%")
