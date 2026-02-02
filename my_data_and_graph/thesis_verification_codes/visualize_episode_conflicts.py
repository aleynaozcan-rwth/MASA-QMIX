import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('my_data_and_graph/historydata/decision_observation_metrics.csv')

# Detect episode boundaries
time_values = df['decision_time'].values
episode_breaks = [0]
for i in range(1, len(time_values)):
    if time_values[i] < time_values[i-1]:
        episode_breaks.append(i)
episode_breaks.append(len(df))

print(f"Total episodes detected: {len(episode_breaks)-1}")

# Extract Episode 0
ep0 = df.iloc[episode_breaks[0]:episode_breaks[1]].copy()

# Analyze each decision point
decision_points = []
for t in sorted(ep0['decision_time'].unique()):
    at_t = ep0[ep0['decision_time'] == t]
    conflicts = len(at_t[at_t['action_idx'] == -1])
    decision_points.append({
        'time': t,
        'num_decisions': len(at_t),
        'conflicts': conflicts,
        'conflict_rate': conflicts / len(at_t) if len(at_t) > 0 else 0,
        'jobs': sorted(at_t['job_id'].unique())
    })

dp_df = pd.DataFrame(decision_points)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Episode 0: Parallel Decision Making and Conflict Analysis', 
             fontsize=16, fontweight='bold')

# 1. Number of parallel decisions per time point
ax1 = axes[0, 0]
bars1 = ax1.bar(dp_df['time'], dp_df['num_decisions'], 
                color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Decision Time', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Parallel Decisions', fontsize=11, fontweight='bold')
ax1.set_title('Parallel Decision Points', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, (t, n) in enumerate(zip(dp_df['time'], dp_df['num_decisions'])):
    ax1.text(t, n + 0.1, str(int(n)), ha='center', va='bottom', fontweight='bold')

# 2. Conflicts per time point
ax2 = axes[0, 1]
colors = ['red' if c > 0 else 'green' for c in dp_df['conflicts']]
bars2 = ax2.bar(dp_df['time'], dp_df['conflicts'], 
                color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Decision Time', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Conflicts', fontsize=11, fontweight='bold')
ax2.set_title('Conflicts per Decision Point', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for i, (t, c) in enumerate(zip(dp_df['time'], dp_df['conflicts'])):
    if c > 0:
        ax2.text(t, c + 0.05, str(int(c)), ha='center', va='bottom', 
                fontweight='bold', color='red')

# 3. Conflict rate
ax3 = axes[1, 0]
line = ax3.plot(dp_df['time'], dp_df['conflict_rate'] * 100, 
                marker='o', linewidth=2, markersize=8, 
                color='orangered', label='Conflict Rate')
ax3.fill_between(dp_df['time'], 0, dp_df['conflict_rate'] * 100, 
                 alpha=0.3, color='orangered')
ax3.set_xlabel('Decision Time', fontsize=11, fontweight='bold')
ax3.set_ylabel('Conflict Rate (%)', fontsize=11, fontweight='bold')
ax3.set_title('Conflict Rate Over Time', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)
ax3.set_ylim(0, max(dp_df['conflict_rate'] * 100) * 1.2 if max(dp_df['conflict_rate']) > 0 else 20)
for t, rate in zip(dp_df['time'], dp_df['conflict_rate'] * 100):
    if rate > 0:
        ax3.text(t, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', 
                fontweight='bold', color='red')

# 4. Decision vs Conflict stacked
ax4 = axes[1, 1]
successful = dp_df['num_decisions'] - dp_df['conflicts']
ax4.bar(dp_df['time'], successful, label='Successful', 
        color='green', alpha=0.7, edgecolor='black')
ax4.bar(dp_df['time'], dp_df['conflicts'], bottom=successful, 
        label='Conflicts', color='red', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Decision Time', fontsize=11, fontweight='bold')
ax4.set_ylabel('Number of Decisions', fontsize=11, fontweight='bold')
ax4.set_title('Successful vs Conflicted Decisions', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('episode_0_conflict_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('episode_0_conflict_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: episode_0_conflict_analysis.pdf")
print("✓ Saved: episode_0_conflict_analysis.png")

# Print detailed statistics
print("\n" + "="*70)
print("EPISODE 0: DETAILED DECISION POINT ANALYSIS")
print("="*70)
for _, row in dp_df.iterrows():
    status = "🔴 CONFLICT" if row['conflicts'] > 0 else "✓ Clean"
    print(f"\nt={row['time']:.1f} | {status}")
    print(f"  Parallel decisions: {int(row['num_decisions'])}")
    print(f"  Conflicts: {int(row['conflicts'])} ({row['conflict_rate']*100:.1f}%)")
    print(f"  Active jobs: {row['jobs']}")

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"Total decision points: {len(dp_df)}")
print(f"Total decisions: {dp_df['num_decisions'].sum():.0f}")
print(f"Total conflicts: {dp_df['conflicts'].sum():.0f}")
print(f"Overall conflict rate: {dp_df['conflicts'].sum() / dp_df['num_decisions'].sum() * 100:.2f}%")
print(f"Average parallel decisions: {dp_df['num_decisions'].mean():.2f}")
print(f"Max parallel decisions: {dp_df['num_decisions'].max():.0f} (at t={dp_df.loc[dp_df['num_decisions'].idxmax(), 'time']:.1f})")
print(f"Decision points with conflicts: {len(dp_df[dp_df['conflicts'] > 0])}/{len(dp_df)}")
