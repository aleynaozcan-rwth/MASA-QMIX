#!/usr/bin/env python3
"""
Decision Point Breakdown: Job Arrivals vs Operation Completions
Shows how decision points are distributed between arrivals and completions
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
print("Parsing scheduling timeline for DP breakdown...")

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
        if '[t=' in line and 'New job' in line and 'arrived' in line:
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
total_dps = arrivals + completions

# Calculate statistics
total_arrivals = np.sum(arrivals)
total_completions = np.sum(completions)
total_all = total_arrivals + total_completions

print(f"\nDecision Point Breakdown:")
print(f"  Total Arrivals: {total_arrivals:,} ({100*total_arrivals/total_all:.1f}%)")
print(f"  Total Completions: {total_completions:,} ({100*total_completions/total_all:.1f}%)")
print(f"  Total DPs: {total_all:,}")

# ==================== Visualization 1: Pie Chart ====================
fig1, ax1 = plt.subplots(figsize=(8, 6))

colors = ['#FF6B6B', '#4ECDC4']  # Red-ish for arrivals, Teal for completions
explode = (0.05, 0.05)
wedges, texts, autotexts = ax1.pie([total_arrivals, total_completions], 
                                     labels=['Job Arrivals', 'Operation Completions'],
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     colors=colors,
                                     explode=explode,
                                     textprops={'fontsize': 12, 'fontweight': 'bold'})

# Make percentage text more visible
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')

ax1.set_title('Decision Point Type Distribution', fontsize=14, fontweight='bold', pad=20)

# Add total count as text
plt.text(0, -1.3, f'Total DPs: {total_all:,}', 
         ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
output_path1 = OUTPUT_DIR / 'dp_breakdown_pie.png'
plt.savefig(output_path1, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved pie chart: {output_path1}")
plt.close()

# ==================== Visualization 2: Stacked Bar Chart (Sampled) ====================
fig2, ax2 = plt.subplots(figsize=(16, 6))

# Sample every 10th episode for clarity
sample_step = 10
sampled_episodes = episode_numbers[::sample_step]
sampled_arrivals = arrivals[::sample_step]
sampled_completions = completions[::sample_step]

# Create stacked bar chart
bar_width = 8
ax2.bar(sampled_episodes, sampled_arrivals, bar_width, 
        label='Job Arrivals', color='#FF6B6B', alpha=0.8)
ax2.bar(sampled_episodes, sampled_completions, bar_width, 
        bottom=sampled_arrivals, label='Operation Completions', 
        color='#4ECDC4', alpha=0.8)

ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax2.set_ylabel('Decision Points', fontsize=12, fontweight='bold')
ax2.set_title('Decision Point Breakdown per Episode (Sampled every 10th)', 
              fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
output_path2 = OUTPUT_DIR / 'dp_breakdown_stacked_bar.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✓ Saved stacked bar chart: {output_path2}")
plt.close()

# ==================== Visualization 3: Dual Line Plot with MA ====================
fig3, ax3 = plt.subplots(figsize=(16, 7))

# Plot raw data with transparency
ax3.plot(episode_numbers, arrivals, linewidth=1, color='#FF6B6B', 
         alpha=0.3, label='_nolegend_')
ax3.plot(episode_numbers, completions, linewidth=1, color='#4ECDC4', 
         alpha=0.3, label='_nolegend_')

# Apply Gaussian smoothing for moving average
sigma = 2.0
arrivals_smooth = gaussian_filter1d(arrivals, sigma=sigma)
completions_smooth = gaussian_filter1d(completions, sigma=sigma)

# Plot smoothed lines
ax3.plot(episode_numbers, arrivals_smooth, linewidth=3, color='#FF6B6B', 
         alpha=0.9, label='Job Arrivals (5-ep MA)', zorder=5)
ax3.plot(episode_numbers, completions_smooth, linewidth=3, color='#4ECDC4', 
         alpha=0.9, label='Operation Completions (5-ep MA)', zorder=5)

# Add mean lines
mean_arrivals = np.mean(arrivals)
mean_completions = np.mean(completions)
ax3.axhline(y=mean_arrivals, color='#FF6B6B', linestyle='--', 
            linewidth=2, alpha=0.6, label=f'Arrivals Mean: {mean_arrivals:.1f}')
ax3.axhline(y=mean_completions, color='#4ECDC4', linestyle='--', 
            linewidth=2, alpha=0.6, label=f'Completions Mean: {mean_completions:.1f}')

ax3.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax3.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax3.set_title('Decision Point Evolution: Arrivals vs Completions', 
              fontsize=14, fontweight='bold', pad=15)
ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.2, axis='y')
ax3.set_xlim(0, len(episode_numbers) - 1)

plt.tight_layout()
output_path3 = OUTPUT_DIR / 'dp_breakdown_dual_line.png'
plt.savefig(output_path3, dpi=150, bbox_inches='tight')
print(f"✓ Saved dual line plot: {output_path3}")
plt.close()

# ==================== Visualization 4: Stacked Area Chart ====================
fig4, ax4 = plt.subplots(figsize=(16, 7))

# Smooth data for cleaner area chart
arrivals_smooth = gaussian_filter1d(arrivals, sigma=2.0)
completions_smooth = gaussian_filter1d(completions, sigma=2.0)

# Create stacked area chart
ax4.fill_between(episode_numbers, 0, arrivals_smooth, 
                 color='#FF6B6B', alpha=0.6, label='Job Arrivals')
ax4.fill_between(episode_numbers, arrivals_smooth, 
                 arrivals_smooth + completions_smooth,
                 color='#4ECDC4', alpha=0.6, label='Operation Completions')

# Add boundary lines for clarity
ax4.plot(episode_numbers, arrivals_smooth, color='#FF6B6B', 
         linewidth=2, alpha=0.8)
ax4.plot(episode_numbers, arrivals_smooth + completions_smooth, 
         color='#4ECDC4', linewidth=2, alpha=0.8)

ax4.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax4.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
ax4.set_title('Decision Point Composition Over Training', 
              fontsize=14, fontweight='bold', pad=15)
ax4.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax4.grid(True, alpha=0.2, axis='y')
ax4.set_xlim(0, len(episode_numbers) - 1)

plt.tight_layout()
output_path4 = OUTPUT_DIR / 'dp_breakdown_stacked_area.png'
plt.savefig(output_path4, dpi=150, bbox_inches='tight')
print(f"✓ Saved stacked area chart: {output_path4}")
plt.close()

print(f"\n✓ All 4 visualizations created successfully!")
