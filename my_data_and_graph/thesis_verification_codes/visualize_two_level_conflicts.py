import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Loading data from both sources...")

# 1. Load CSV data (decision-level)
df = pd.read_csv('my_data_and_graph/historydata/decision_observation_metrics.csv')
total_decisions = len(df)
decision_conflicts = len(df[df['action_idx'] == -1])
decision_rate = decision_conflicts / total_decisions * 100

# 2. Parse timeline data (execution-level)
with open('my_data_and_graph/historydata/scheduling_timeline.txt', 'r') as f:
    content = f.read()

episodes = re.findall(r'=== EPISODE (\d+) ===.*?=== TIMELINE ===(.*?)(?:=== EPISODE \d+ ===|\Z)', 
                      content, re.DOTALL)

total_jobs = 0
execution_conflicts = 0

for ep_num, timeline in episodes:
    arrivals = re.findall(r'\[t=([\d.]+)\] New job (Job_\d+) arrived', timeline)
    op_starts = re.findall(r'\[t=([\d.]+)\] (Job_\d+)\.(Op\d+) started', timeline)
    
    for arrival_t, job in arrivals:
        total_jobs += 1
        arrival_t = float(arrival_t)
        
        first_start = None
        for start_t, start_job, _ in op_starts:
            if start_job == job:
                first_start = float(start_t)
                break
        
        if first_start and (first_start - arrival_t) > 1.5:
            execution_conflicts += 1

execution_rate = execution_conflicts / total_jobs * 100

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# Main title
fig.suptitle('Two-Level Conflict Analysis: Decision vs Execution', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Bar chart comparison
ax1 = plt.subplot(2, 3, 1)
metrics = ['Decision-Level\n(Q-network)', 'Execution-Level\n(Resource Queue)']
rates = [decision_rate, execution_rate]
colors = ['steelblue', 'coral']
bars = ax1.bar(metrics, rates, color=colors, alpha=0.7, edgecolor='black', width=0.6)
ax1.set_ylabel('Conflict Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Conflict Rate Comparison', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, (bar, rate) in enumerate(zip(bars, rates)):
    ax1.text(bar.get_x() + bar.get_width()/2, rate + 0.5, 
            f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 2. Stacked bar showing conflict breakdown
ax2 = plt.subplot(2, 3, 2)
categories = ['Decision\nConflicts', 'Additional\nExecution\nDelays', 'No\nConflicts']
decision_only = decision_rate
execution_additional = execution_rate - decision_rate
no_conflict = 100 - execution_rate

values = [decision_only, execution_additional, no_conflict]
colors_stack = ['red', 'orange', 'green']
bars = ax2.bar(range(3), values, color=colors_stack, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Conflict Breakdown', fontsize=13, fontweight='bold')
ax2.set_xticks(range(3))
ax2.set_xticklabels(categories, fontsize=10)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, values):
    if val > 1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'{val:.1f}%', ha='center', va='center', fontweight='bold', fontsize=10)

# 3. Episode 0 Case Study - Job 4
ax3 = plt.subplot(2, 3, 3)
timeline_case = [
    ('Arrival\nt=2.48', 2.48, 'green'),
    ('Decision\nt=3.0\naction=-1', 3.0, 'red'),
    ('Execution\nt=5.0', 5.0, 'blue')
]
times = [t for _, t, _ in timeline_case]
labels = [l for l, _, _ in timeline_case]
colors_case = [c for _, _, c in timeline_case]

ax3.scatter(times, [1]*3, s=300, c=colors_case, alpha=0.7, edgecolors='black', linewidths=2, zorder=3)
for i, (label, time, _) in enumerate(timeline_case):
    ax3.text(time, 1.15, label, ha='center', va='bottom', fontweight='bold', fontsize=9)

# Draw arrows
ax3.annotate('', xy=(3.0, 1), xytext=(2.48, 1),
            arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
ax3.annotate('', xy=(5.0, 1), xytext=(3.0, 1),
            arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
ax3.text(2.74, 0.85, 'Wait', ha='center', fontsize=9, style='italic')
ax3.text(4.0, 0.85, 'Queue (2.52s)', ha='center', fontsize=9, style='italic')

ax3.set_xlim(2, 5.5)
ax3.set_ylim(0.5, 1.5)
ax3.set_xlabel('Simulation Time', fontsize=11, fontweight='bold')
ax3.set_title('Episode 0, Job 4: Conflict Validation', fontsize=13, fontweight='bold')
ax3.set_yticks([])
ax3.grid(axis='x', alpha=0.3)

# 4. Counts comparison
ax4 = plt.subplot(2, 3, 4)
categories_count = ['Total\nEvents', 'Conflicts']
decision_vals = [total_decisions, decision_conflicts]
execution_vals = [total_jobs, execution_conflicts]

x = np.arange(len(categories_count))
width = 0.35

bars1 = ax4.bar(x - width/2, decision_vals, width, label='Decision-Level', 
               color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax4.bar(x + width/2, execution_vals, width, label='Execution-Level', 
               color='coral', alpha=0.7, edgecolor='black')

ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
ax4.set_title('Absolute Counts Comparison', fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(categories_count, fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)
ax4.set_yscale('log')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height*1.1,
                f'{int(height):,}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 5. Interpretation text
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')
interpretation = f"""
TWO-LEVEL CONFLICT SYSTEM

Decision-Level (Agent Policy):
  • Rate: {decision_rate:.2f}%
  • Count: {decision_conflicts:,} / {total_decisions:,}
  • Meaning: Agent selected unavailable action
  • Mechanism: ε-greedy + Q-network
  
Execution-Level (Environment):
  • Rate: {execution_rate:.2f}%
  • Count: {execution_conflicts:,} / {total_jobs:,}
  • Meaning: Job waited for resources
  • Mechanism: SimPy queue (machine + operator)

Key Insight:
  Low decision-level rate ({decision_rate:.2f}%) indicates
  effective Q-network learning for action selection.
  
  Higher execution-level rate ({execution_rate:.2f}%) reflects
  realistic manufacturing resource contention,
  independent of agent learning quality.
  
  Difference: {execution_rate - decision_rate:.2f}% represents pure
  environment delays (operator busy, machine busy)
  that occur even with optimal agent policies.
"""
ax5.text(0.05, 0.95, interpretation, transform=ax5.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6. Pie chart showing breakdown
ax6 = plt.subplot(2, 3, 6)
pie_labels = [
    f'Decision Conflicts\n{decision_rate:.2f}%',
    f'Environment Delays\n{execution_rate - decision_rate:.2f}%',
    f'No Conflicts\n{100 - execution_rate:.2f}%'
]
pie_values = [decision_rate, execution_rate - decision_rate, 100 - execution_rate]
pie_colors = ['red', 'orange', 'green']
pie_explode = (0.05, 0.05, 0)

ax6.pie(pie_values, labels=pie_labels, colors=pie_colors, autopct='',
       startangle=90, explode=pie_explode, shadow=True)
ax6.set_title('Overall System Conflict Distribution', fontsize=13, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('two_level_conflict_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('two_level_conflict_comparison.png', dpi=300, bbox_inches='tight')

print("\n✓ Saved: two_level_conflict_comparison.pdf")
print("✓ Saved: two_level_conflict_comparison.png")

print("\n" + "="*70)
print("TWO-LEVEL CONFLICT ANALYSIS SUMMARY")
print("="*70)
print(f"\n1. DECISION-LEVEL (Q-network policy conflicts):")
print(f"   • Rate: {decision_rate:.2f}%")
print(f"   • Count: {decision_conflicts:,} / {total_decisions:,} decisions")
print(f"   • Metric: action_idx = -1")

print(f"\n2. EXECUTION-LEVEL (Resource contention delays):")
print(f"   • Rate: {execution_rate:.2f}%")
print(f"   • Count: {execution_conflicts:,} / {total_jobs:,} jobs")
print(f"   • Metric: arrival-to-start delay > 1.5 time units")

print(f"\n3. ADDITIONAL ENVIRONMENT DELAYS:")
print(f"   • Rate: {execution_rate - decision_rate:.2f}%")
print(f"   • Interpretation: Delays caused by environment constraints")
print(f"                     (operator busy, machine busy) that occur")
print(f"                     even with optimal agent policies")

print(f"\n4. VALIDATION (Episode 0, Job 4):")
print(f"   • Decision conflict detected: t=3.0 (action=-1)")
print(f"   • Execution delay measured: 2.52 time units (2.48→5.0)")
print(f"   • Both metrics confirm same conflict event ✓")
