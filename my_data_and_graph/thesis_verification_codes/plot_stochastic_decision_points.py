#!/usr/bin/env python3
"""
Analyze decision points per episode with stochastic job arrivals.
Shows arrival vs completion breakdown with statistical summary.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Read the scheduling timeline
print("Parsing scheduling timeline...")
episode_stats = []
current_episode = None
current_arrivals = 0
current_completions = 0
in_lifecycle_trace = False

with open('my_data_and_graph/historydata/scheduling_timeline.txt', 'r') as f:
    for line in f:
        line = line.strip()
        
        # Match episode start: "=== EPISODE 0 ==="
        episode_match = re.match(r'^=== EPISODE (\d+) ===$', line)
        if episode_match:
            # Save previous episode if exists
            if current_episode is not None:
                total_dps = current_arrivals + current_completions
                episode_stats.append({
                    'episode': current_episode,
                    'arrivals': current_arrivals,
                    'completions': current_completions,
                    'total': total_dps
                })
            
            # Start new episode
            current_episode = int(episode_match.group(1))
            current_arrivals = 0
            current_completions = 0
            in_lifecycle_trace = False
            continue
        
        # Track lifecycle trace section
        if 'JOB AGENT LIFECYCLE TRACE START' in line:
            in_lifecycle_trace = True
            continue
        
        if 'JOB AGENT LIFECYCLE TRACE END' in line:
            in_lifecycle_trace = False
            continue
        
        # Only count arrivals inside lifecycle trace section
        if not in_lifecycle_trace or current_episode is None:
            # But completions are outside lifecycle trace, in TIMELINE section
            if '[t=' in line and 'finished' in line and 'next queued' in line:
                current_completions += 1
            continue
        
        # Match decision point events (must start with [t=X.XX]):
        # 1. Job arrivals: "[t=X.XX] New job ... arrived"
        if '[t=' in line and 'New job' in line and 'arrived' in line:
            current_arrivals += 1
            continue
        
        # 2. Operation completions: "[t=X.XX] ... finished ... next queued"
        if '[t=' in line and 'finished' in line and 'next queued' in line:
            current_completions += 1
            continue

# Save last episode
if current_episode is not None:
    total_dps = current_arrivals + current_completions
    episode_stats.append({
        'episode': current_episode,
        'arrivals': current_arrivals,
        'completions': current_completions,
        'total': total_dps
    })

print(f"Total episodes found: {len(episode_stats)}")
for idx in [0, 1, 2, 3, 4]:
    if idx < len(episode_stats):
        s = episode_stats[idx]
        print(f"Episode {s['episode']}: {s['arrivals']} arrivals, {s['completions']} completions, {s['total']} total DPs")
for idx in [-2, -1]:
    if len(episode_stats) + idx >= 0:
        s = episode_stats[idx]
        print(f"Episode {s['episode']}: {s['arrivals']} arrivals, {s['completions']} completions, {s['total']} total DPs")

# Extract data for plotting
episodes_list = [s['episode'] for s in episode_stats]
arrivals_list = [s['arrivals'] for s in episode_stats]
completions_list = [s['completions'] for s in episode_stats]
totals_list = [s['total'] for s in episode_stats]

# Statistics
total_arrivals = sum(arrivals_list)
total_completions = sum(completions_list)
total_dps = sum(totals_list)
mean_arrivals = np.mean(arrivals_list)
std_arrivals = np.std(arrivals_list)
min_arrivals = np.min(arrivals_list)
max_arrivals = np.max(arrivals_list)
mean_total = np.mean(totals_list)
std_total = np.std(totals_list)
min_total = np.min(totals_list)
max_total = np.max(totals_list)

print(f"\n{'='*60}")
print(f"DECISION POINT ANALYSIS (Stochastic Job Arrivals)")
print(f"{'='*60}")
print(f"Total Episodes: {len(episode_stats)}")
print(f"\nJob Arrivals:")
print(f"  Total:     {total_arrivals:,}")
print(f"  Mean/Ep:   {mean_arrivals:.2f} ± {std_arrivals:.2f}")
print(f"  Range:     [{min_arrivals}, {max_arrivals}]")
print(f"\nOperation Completions:")
print(f"  Total:     {total_completions:,}")
print(f"  Mean/Ep:   {np.mean(completions_list):.2f} ± {np.std(completions_list):.2f}")
print(f"\nTotal Decision Points:")
print(f"  Total:     {total_dps:,}")
print(f"  Mean/Ep:   {mean_total:.2f} ± {std_total:.2f}")
print(f"  Range:     [{min_total}, {max_total}]")
print(f"  Breakdown: {total_arrivals/total_dps*100:.1f}% arrivals, {total_completions/total_dps*100:.1f}% completions")
print(f"{'='*60}\n")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# 1. Total decision points per episode (top main plot)
ax1 = plt.subplot(3, 2, (1, 2))
x = episodes_list
y = totals_list

# Plot raw data
ax1.vlines(x, 0, y, colors='lightsteelblue', alpha=0.6, linewidth=1.5, label='Decision Points per Episode')
ax1.scatter(x, y, color='steelblue', s=20, alpha=0.7, zorder=3)

# Moving average
window = 10
if len(y) >= window:
    ma = gaussian_filter1d(y, sigma=2.0)
    ax1.plot(x, ma, color='red', linewidth=2, label=f'Moving Average (σ=2)', zorder=5)

# Mean line
ax1.axhline(mean_total, color='green', linestyle='--', linewidth=1.5, 
            label=f'Mean: {mean_total:.2f}', zorder=10, alpha=0.8)

# Fill std range
ax1.axhspan(mean_total - std_total, mean_total + std_total, 
            color='green', alpha=0.1, label=f'±1σ: [{mean_total-std_total:.1f}, {mean_total+std_total:.1f}]')

ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Total Decision Points', fontsize=12)
ax1.set_title(f'Decision Points per Episode (Stochastic Job Arrivals)\n'
              f'{len(episode_stats)} episodes | Mean: {mean_total:.2f} ± {std_total:.2f} | Range: [{min_total}, {max_total}]',
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

# 2. Stacked area: arrivals vs completions
ax2 = plt.subplot(3, 2, 3)
ax2.fill_between(x, 0, arrivals_list, color='orange', alpha=0.6, label=f'Arrivals (μ={mean_arrivals:.1f}±{std_arrivals:.1f})')
ax2.fill_between(x, arrivals_list, totals_list, color='purple', alpha=0.6, 
                 label=f'Completions (μ={np.mean(completions_list):.1f}±{np.std(completions_list):.1f})')
ax2.set_xlabel('Episode', fontsize=10)
ax2.set_ylabel('Decision Points', fontsize=10)
ax2.set_title('Stacked: Arrivals vs Completions', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Dual line plot
ax3 = plt.subplot(3, 2, 4)
ax3.plot(x, arrivals_list, color='orange', linewidth=2, marker='o', markersize=3, 
         alpha=0.7, label=f'Arrivals (range: [{min_arrivals}, {max_arrivals}])')
ax3.plot(x, completions_list, color='purple', linewidth=2, marker='s', markersize=3, 
         alpha=0.7, label=f'Completions (range: [{np.min(completions_list)}, {np.max(completions_list)}])')
ax3.set_xlabel('Episode', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.set_title('Arrival vs Completion Trends', fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Histogram of arrivals per episode
ax4 = plt.subplot(3, 2, 5)
bins = range(min_arrivals, max_arrivals + 2)
ax4.hist(arrivals_list, bins=bins, color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
ax4.axvline(mean_arrivals, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_arrivals:.2f}')
ax4.set_xlabel('Jobs per Episode', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.set_title(f'Distribution of Job Arrivals\n(Stochastic: range {min_arrivals}-{max_arrivals} jobs)', 
              fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Histogram of total decision points
ax5 = plt.subplot(3, 2, 6)
bins_dp = range(min_total, max_total + 2)
ax5.hist(totals_list, bins=bins_dp, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax5.axvline(mean_total, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_total:.2f}')
ax5.set_xlabel('Total Decision Points per Episode', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title(f'Distribution of Total Decision Points\n(Range: {min_total}-{max_total})', 
              fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('my_data_and_graph/stochastic_decision_points_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: my_data_and_graph/stochastic_decision_points_analysis.png")
plt.close()

# Create comparison figure: First 20 vs Last 20 episodes
if len(episode_stats) >= 40:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # First 20 episodes
    first_20_arrivals = arrivals_list[:20]
    first_20_completions = completions_list[:20]
    first_20_x = list(range(20))
    
    ax1.bar(first_20_x, first_20_arrivals, color='orange', alpha=0.7, label='Arrivals')
    ax1.bar(first_20_x, first_20_completions, bottom=first_20_arrivals, color='purple', alpha=0.7, label='Completions')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Decision Points', fontsize=12)
    ax1.set_title(f'First 20 Episodes (Early Training)\nMean Arrivals: {np.mean(first_20_arrivals):.2f} | Mean Total: {np.mean([a+c for a,c in zip(first_20_arrivals, first_20_completions)]):.2f}', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Last 20 episodes
    last_20_arrivals = arrivals_list[-20:]
    last_20_completions = completions_list[-20:]
    last_20_x = list(range(len(episode_stats) - 20, len(episode_stats)))
    
    ax2.bar(last_20_x, last_20_arrivals, color='orange', alpha=0.7, label='Arrivals')
    ax2.bar(last_20_x, last_20_completions, bottom=last_20_arrivals, color='purple', alpha=0.7, label='Completions')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Decision Points', fontsize=12)
    ax2.set_title(f'Last 20 Episodes (Late Training)\nMean Arrivals: {np.mean(last_20_arrivals):.2f} | Mean Total: {np.mean([a+c for a,c in zip(last_20_arrivals, last_20_completions)]):.2f}', 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('my_data_and_graph/stochastic_early_vs_late_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: my_data_and_graph/stochastic_early_vs_late_comparison.png")
    plt.close()

print("\nAnalysis complete! Verify that job arrivals vary across episodes.")
print(f"Episode 0: {arrivals_list[0]} jobs, Episode 1: {arrivals_list[1]} jobs, Episode 2: {arrivals_list[2]} jobs")
