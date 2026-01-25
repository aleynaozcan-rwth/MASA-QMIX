#!/usr/bin/env python3
"""
Job Arrival vs Operation Completion Trends Across All Episodes
X-axis: Episode number
Y-axis: Count of each decision point type per episode
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# Use Agg backend for cluster without display
import matplotlib
matplotlib.use('Agg')

# ==================== Configuration ====================
TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Parse Timeline File ====================
print("Parsing scheduling timeline for arrival/completion trends...")

episodes = []  # List of {episode, arrivals, completions}
current_episode = None
current_arrivals = 0
current_completions = 0

with open(TIMELINE_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Match episode start: "=== EPISODE 0 ==="
        episode_match = re.match(r'^=== EPISODE (\d+) ===$', line)
        if episode_match:
            # Save previous episode if exists
            if current_episode is not None:
                episodes.append({
                    'episode': current_episode,
                    'arrivals': current_arrivals,
                    'completions': current_completions
                })
            
            # Start new episode
            current_episode = int(episode_match.group(1))
            current_arrivals = 0
            current_completions = 0
            continue
        
        # Count job arrivals: "[t=X.XX] New job ... arrived"
        # Only count the first format (numbered jobs), skip "Job_X" format to avoid double counting
        if '[t=' in line and 'New job' in line and 'arrived' in line:
            # Check if it's the numbered format (e.g., "New job 4 arrived")
            # and not the detailed format (e.g., "New job Job_4 arrived")
            if re.search(r'New job \d+ arrived', line):
                current_arrivals += 1
            continue
        
        # Count operation completions: "[t=X.XX] ... finished ... next queued"
        if '[t=' in line and 'finished' in line and 'next queued' in line:
            current_completions += 1
            continue

# Save last episode
if current_episode is not None:
    episodes.append({
        'episode': current_episode,
        'arrivals': current_arrivals,
        'completions': current_completions
    })

print(f"Parsed {len(episodes)} episodes")

# ==================== Extract Data ====================
episode_numbers = np.array([ep['episode'] for ep in episodes])
arrivals = np.array([ep['arrivals'] for ep in episodes])
completions = np.array([ep['completions'] for ep in episodes])

# Calculate statistics
total_arrivals = np.sum(arrivals)
total_completions = np.sum(completions)
mean_arrivals = np.mean(arrivals)
mean_completions = np.mean(completions)

print(f"\nStatistics:")
print(f"  Total Arrivals: {total_arrivals:,} (Mean: {mean_arrivals:.2f})")
print(f"  Total Completions: {total_completions:,} (Mean: {mean_completions:.2f})")

# ==================== Visualization ====================
fig, ax = plt.subplots(figsize=(16, 7))

# Scatter plot for raw data
ax.scatter(episode_numbers, arrivals, c='#FF6B6B', s=30, alpha=0.5, 
          label=f'Job Arrivals', zorder=3)
ax.scatter(episode_numbers, completions, c='#4ECDC4', s=30, alpha=0.5,
          label=f'Operation Completions', zorder=3)

# Add smoothed trend lines (Gaussian MA)
sigma = 2.0
arrivals_smooth = gaussian_filter1d(arrivals, sigma=sigma)
completions_smooth = gaussian_filter1d(completions, sigma=sigma)

ax.plot(episode_numbers, arrivals_smooth, linewidth=3, color='#FF6B6B', 
       alpha=0.9, label='Arrivals Trend (5-ep MA)', zorder=5)
ax.plot(episode_numbers, completions_smooth, linewidth=3, color='#4ECDC4', 
       alpha=0.9, label='Completions Trend (5-ep MA)', zorder=5)

# Add mean lines
ax.axhline(y=mean_arrivals, color='#FF6B6B', linestyle='--', linewidth=2.5, 
          alpha=0.7, label=f'Arrivals Mean: {mean_arrivals:.2f}')
ax.axhline(y=mean_completions, color='#4ECDC4', linestyle='--', linewidth=2.5, 
          alpha=0.7, label=f'Completions Mean: {mean_completions:.2f}')

# Labels and title
ax.set_xlabel('Episode', fontsize=13, fontweight='bold')
ax.set_ylabel('Decision Points per Episode', fontsize=13, fontweight='bold')
ax.set_title('Job Arrivals vs Operation Completions Across All Episodes', 
            fontsize=14, fontweight='bold', pad=15)

# Legend
ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)

# Grid and limits
ax.grid(True, alpha=0.2, axis='y')
ax.set_xlim(0, len(episode_numbers) - 1)

# Set y-axis to show meaningful range
y_min = min(np.min(arrivals), np.min(completions)) - 2
y_max = max(np.max(arrivals), np.max(completions)) + 2
ax.set_ylim(y_min, y_max)

plt.tight_layout()

output_path = OUTPUT_DIR / 'arrival_completion_trends.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")
plt.close()

print("✓ Visualization complete!")
