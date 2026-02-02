import re
import matplotlib.pyplot as plt
import numpy as np

# Parse timeline file
with open('my_data_and_graph/historydata/scheduling_timeline.txt', 'r') as f:
    content = f.read()

# Find all episodes
episodes = re.findall(r'=== EPISODE (\d+) ===.*?=== TIMELINE ===(.*?)(?:=== EPISODE \d+ ===|\Z)', 
                      content, re.DOTALL)

print(f"Analyzing {len(episodes)} episodes from timeline...")

# Analyze each episode
episode_stats = []

for ep_num, timeline in episodes:
    ep_num = int(ep_num)
    
    # Find job arrivals and operation starts
    arrivals = re.findall(r'\[t=([\d.]+)\] New job (Job_\d+) arrived', timeline)
    op_starts = re.findall(r'\[t=([\d.]+)\] (Job_\d+)\.(Op\d+) started', timeline)
    
    total_jobs = len(arrivals)
    delayed_jobs = 0
    delays = []
    
    for arrival_t, job in arrivals:
        arrival_t = float(arrival_t)
        
        # Find first op start for this job
        first_start = None
        for start_t, start_job, _ in op_starts:
            if start_job == job:
                first_start = float(start_t)
                break
        
        if first_start:
            delay = first_start - arrival_t
            delays.append(delay)
            if delay > 1.5:  # Conflict threshold
                delayed_jobs += 1
    
    conflict_rate = (delayed_jobs / total_jobs * 100) if total_jobs > 0 else 0
    avg_delay = np.mean(delays) if delays else 0
    max_delay = max(delays) if delays else 0
    
    episode_stats.append({
        'episode': ep_num,
        'total_jobs': total_jobs,
        'delayed_jobs': delayed_jobs,
        'conflict_rate': conflict_rate,
        'avg_delay': avg_delay,
        'max_delay': max_delay
    })

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Timeline-Based Conflict Analysis Across Training Episodes', 
             fontsize=16, fontweight='bold')

episodes_list = [s['episode'] for s in episode_stats]
conflict_rates = [s['conflict_rate'] for s in episode_stats]
avg_delays = [s['avg_delay'] for s in episode_stats]
max_delays = [s['max_delay'] for s in episode_stats]
delayed_jobs = [s['delayed_jobs'] for s in episode_stats]

# 1. Conflict rate over episodes
ax1 = axes[0, 0]
ax1.scatter(episodes_list, conflict_rates, alpha=0.4, s=20, color='coral')
# Moving average
window = 50
if len(conflict_rates) > window:
    ma = np.convolve(conflict_rates, np.ones(window)/window, mode='valid')
    ma_episodes = episodes_list[window-1:]
    ax1.plot(ma_episodes, ma, color='red', linewidth=2, label=f'{window}-episode MA')
ax1.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax1.set_ylabel('Conflict Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('Execution-Level Conflict Rate Evolution', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend()

# 2. Average delay per episode
ax2 = axes[0, 1]
ax2.scatter(episodes_list, avg_delays, alpha=0.4, s=20, color='steelblue')
if len(avg_delays) > window:
    ma_delay = np.convolve(avg_delays, np.ones(window)/window, mode='valid')
    ax2.plot(ma_episodes, ma_delay, color='darkblue', linewidth=2, label=f'{window}-episode MA')
ax2.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax2.set_ylabel('Average Delay (time units)', fontsize=11, fontweight='bold')
ax2.set_title('Average Job Start Delay', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.legend()

# 3. Conflict rate histogram
ax3 = axes[1, 0]
ax3.hist(conflict_rates, bins=30, color='coral', alpha=0.7, edgecolor='black')
ax3.axvline(np.mean(conflict_rates), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {np.mean(conflict_rates):.1f}%')
ax3.set_xlabel('Conflict Rate (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Number of Episodes', fontsize=11, fontweight='bold')
ax3.set_title('Conflict Rate Distribution', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)
ax3.legend()

# 4. Max delay per episode
ax4 = axes[1, 1]
ax4.scatter(episodes_list, max_delays, alpha=0.4, s=20, color='orangered')
if len(max_delays) > window:
    ma_max = np.convolve(max_delays, np.ones(window)/window, mode='valid')
    ax4.plot(ma_episodes, ma_max, color='darkred', linewidth=2, label=f'{window}-episode MA')
ax4.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax4.set_ylabel('Max Delay (time units)', fontsize=11, fontweight='bold')
ax4.set_title('Maximum Job Start Delay per Episode', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('timeline_conflict_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('timeline_conflict_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: timeline_conflict_analysis.pdf")
print("✓ Saved: timeline_conflict_analysis.png")

# Statistics
print("\n" + "="*70)
print("TIMELINE-BASED CONFLICT STATISTICS")
print("="*70)
print(f"Total episodes analyzed: {len(episode_stats)}")
print(f"Total jobs: {sum(s['total_jobs'] for s in episode_stats)}")
print(f"Total delayed jobs (>1.5s): {sum(s['delayed_jobs'] for s in episode_stats)}")
print(f"\nOverall conflict rate: {sum(s['delayed_jobs'] for s in episode_stats) / sum(s['total_jobs'] for s in episode_stats) * 100:.2f}%")
print(f"Mean episode conflict rate: {np.mean(conflict_rates):.2f}%")
print(f"Median episode conflict rate: {np.median(conflict_rates):.2f}%")
print(f"Std deviation: {np.std(conflict_rates):.2f}%")
print(f"\nAverage delay across all jobs: {np.mean([s['avg_delay'] for s in episode_stats]):.2f} time units")
print(f"Average max delay per episode: {np.mean(max_delays):.2f} time units")

# Phase analysis
phases = [
    ('Episodes 0-200', 0, 200),
    ('Episodes 200-400', 200, 400),
    ('Episodes 400-600', 400, 600),
]

print(f"\nConflict rate by training phase:")
for name, start, end in phases:
    phase_stats = [s for s in episode_stats if start <= s['episode'] < end]
    if phase_stats:
        phase_rate = np.mean([s['conflict_rate'] for s in phase_stats])
        print(f"  {name}: {phase_rate:.2f}%")

# Comparison
print(f"\n" + "="*70)
print("COMPARISON: DECISION-LEVEL vs EXECUTION-LEVEL")
print("="*70)
print(f"Decision-level (CSV, action=-1):     0.92%  (Q-network conflicts)")
print(f"Execution-level (Timeline, >1.5s): {sum(s['delayed_jobs'] for s in episode_stats) / sum(s['total_jobs'] for s in episode_stats) * 100:.2f}%  (Resource contention)")
print(f"\nInterpretation:")
print(f"  • Decision conflicts (0.92%) = Agent chose unavailable action")
print(f"  • Execution delays ({sum(s['delayed_jobs'] for s in episode_stats) / sum(s['total_jobs'] for s in episode_stats) * 100:.2f}%) = Realistic resource waiting")
print(f"  • Low decision rate shows good Q-network learning")
print(f"  • Higher execution rate shows realistic manufacturing constraints")
