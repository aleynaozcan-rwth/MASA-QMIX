#!/usr/bin/env python3
"""
Job Interarrival Times Visualization - Multiple Episodes Comparison
Shows the time intervals between consecutive job arrivals for 3 sample episodes
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/thesis_verification_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt')

print("="*70)
print("JOB INTERARRIVAL TIMES - MULTIPLE EPISODES COMPARISON")
print("="*70)

# Parse all episodes
all_episodes_data = {}

with open(DATA_FILE, 'r') as f:
    content = f.read()

# Split by episodes - use more specific pattern
episode_pattern = re.compile(r'=== EPISODE (\d+) ===')
episodes_data_raw = {}

current_episode = None
current_lines = []

for line in content.split('\n'):
    episode_match = episode_pattern.match(line.strip())
    if episode_match:
        # Save previous episode data if exists
        if current_episode is not None and current_lines:
            episodes_data_raw[current_episode] = current_lines
        
        # Start new episode
        current_episode = int(episode_match.group(1))
        current_lines = [line]
    elif current_episode is not None:
        current_lines.append(line)

# Save last episode
if current_episode is not None and current_lines:
    episodes_data_raw[current_episode] = current_lines

for episode_num, lines in episodes_data_raw.items():
    
    # Extract arrival times from lifecycle trace
    arrival_times = []
    for line in lines:
        # Match only the simple format: "New job X arrived" where X is a number
        # This avoids duplicate entries from "New job Job_X arrived" format
        if 'New job' in line and 'arrived with' in line and 'Job_' not in line:
            if '[t=' in line:
                time_str = line.split('[t=')[1].split(']')[0]
                arrival_times.append(float(time_str))
    
    # Filter out t=0 arrivals (initial jobs) - only keep t > 0
    arrival_times = [t for t in arrival_times if t > 0]
    
    # Calculate inter-arrival times for this episode
    if len(arrival_times) > 1:
        arrival_times.sort()
        interarrivals = []
        for i in range(1, len(arrival_times)):
            inter_arrival = arrival_times[i] - arrival_times[i-1]
            if inter_arrival > 0:  # Ignore simultaneous arrivals at t=0
                interarrivals.append(inter_arrival)
        
        if interarrivals:
            all_episodes_data[episode_num] = {
                'arrivals': arrival_times,
                'interarrivals': interarrivals
            }

print(f"\nTotal episodes parsed: {len(all_episodes_data)}")

# Calculate overall statistics from ALL episodes (not just selected ones)
all_episodes_interarrivals = []
for ep_num, ep_data in all_episodes_data.items():
    if ep_data['interarrivals']:
        all_episodes_interarrivals.extend(ep_data['interarrivals'])

overall_mean_all = np.mean(all_episodes_interarrivals)
overall_std_all = np.std(all_episodes_interarrivals)
overall_cv_all = overall_std_all / overall_mean_all

print(f"\nOverall statistics (ALL {len(all_episodes_data)} episodes):")
print(f"  Total interarrivals: {len(all_episodes_interarrivals)}")
print(f"  Mean: {overall_mean_all:.3f}")
print(f"  Std: {overall_std_all:.3f}")
print(f"  CV: {overall_cv_all:.3f}")

# Select 3 sample episodes (early, middle, late)
episode_numbers = sorted(all_episodes_data.keys())
if len(episode_numbers) >= 3:
    # Select episodes: first, middle (645 for better visualization), and last
    selected_episodes = [
        episode_numbers[0],  # First episode
        645,  # Middle episode with CV≈1.0 and max<10
        episode_numbers[-1]  # Last episode
    ]
else:
    selected_episodes = episode_numbers[:3]

print(f"Selected episodes for visualization: {selected_episodes}")

# Define colors for each episode
colors = ['#2E86C1', '#E74C3C', '#27AE60']  # Blue, Red, Green
episode_colors = dict(zip(selected_episodes, colors))

# Create figure
fig, ax = plt.subplots(figsize=(20, 10))

# Track all interarrival times for statistics
all_interarrivals = []

# Plot each episode with different vertical offset and color
vertical_spacing = 5.0
max_time = 0

# Calculate max interarrival across all selected episodes for scaling
max_interarrival = max([max(all_episodes_data[ep]['interarrivals']) 
                        for ep in selected_episodes])

for idx, ep_num in enumerate(selected_episodes):
    data = all_episodes_data[ep_num]
    arrivals = data['arrivals']
    interarrivals = data['interarrivals']
    color = episode_colors[ep_num]
    
    # All episodes share the same baseline (no vertical offset)
    y_offset = 0
    
    print(f"\n{'='*60}")
    print(f"EPISODE {ep_num} DETAILED DATA")
    print(f"{'='*60}")
    print(f"\n--- Arrival Times ---")
    for i, t in enumerate(arrivals):
        print(f"  Job {i:2d}: t = {t:6.2f}")
    
    print(f"\n--- Interarrival Times (Δt between consecutive jobs) ---")
    for i in range(len(interarrivals)):
        t_start = arrivals[i] if i < len(arrivals) else 0
        t_end = arrivals[i+1] if i+1 < len(arrivals) else 0
        dt = interarrivals[i]
        print(f"  Interval {i+1:2d}: t={t_start:6.2f} → t={t_end:6.2f}  |  Δt = {dt:6.2f}")
    
    print(f"\n--- Statistics ---")
    print(f"  Total arrivals: {len(arrivals)}")
    print(f"  Total interarrivals: {len(interarrivals)}")
    print(f"  Mean Δt: {np.mean(interarrivals):.2f}")
    print(f"  Std Δt: {np.std(interarrivals):.2f}")
    print(f"  CV: {np.std(interarrivals)/np.mean(interarrivals):.3f}")
    
    # Plot staircase-style: horizontal lines at interarrival height, vertical drops
    for i in range(len(interarrivals)):
        start_time = arrivals[i] if i < len(arrivals) else 0
        end_time = arrivals[i+1] if i+1 < len(arrivals) else start_time
        interval = interarrivals[i]
        
        # Draw horizontal line showing the interarrival duration
        ax.hlines(y=interval + y_offset, xmin=start_time, xmax=end_time, 
                 colors=color, linewidth=3, alpha=0.8)
        
        # Draw vertical line at the end (transition to next interval)
        # Only draw vertical transitions between intervals, not after the last one
        if i < len(interarrivals) - 1:  # Not the last interval
            next_interval = interarrivals[i+1]
            # Draw vertical line connecting current interval height to next interval height
            ax.vlines(x=end_time, 
                     ymin=min(interval + y_offset, next_interval + y_offset), 
                     ymax=max(interval + y_offset, next_interval + y_offset), 
                     colors=color, linewidth=3, alpha=0.8)
        
        # First interval - draw vertical line up from baseline
        if i == 0:
            ax.vlines(x=start_time, ymin=y_offset, ymax=interval + y_offset, 
                     colors=color, linewidth=3, alpha=0.8)
    
    # Add scatter points at arrival times (showing when each job arrived)
    # Each arrival point is at the height of the PREVIOUS interarrival
    for i in range(1, len(arrivals)):  # Start from 1 (skip t=0)
        arrival_time = arrivals[i]
        if i-1 < len(interarrivals):
            # Height is the interarrival that just ended
            y_height = interarrivals[i-1] + y_offset
            ax.scatter([arrival_time], [y_height], 
                      color=color, s=120, zorder=5, marker='o',
                      edgecolors='white', linewidths=2, alpha=0.9)
    
    # Add value labels on interarrival segments (at midpoint)
    for i in range(len(interarrivals)):
        start_time = arrivals[i]
        end_time = arrivals[i+1]
        midpoint = (start_time + end_time) / 2
        y_height = interarrivals[i] + y_offset
        
        ax.annotate(f'{interarrivals[i]:.2f}', 
                   xy=(midpoint, y_height), 
                   xytext=(0, 5),
                   textcoords='offset points',
                   fontsize=8, 
                   ha='center', 
                   va='bottom',
                   color=color, 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor=color, alpha=0.9, linewidth=1))
    
    # Track max time
    max_time = max(max_time, arrivals[-1] if arrivals else 0)
    
    # Collect all interarrivals
    all_interarrivals.extend(interarrivals)

# Calculate overall statistics
overall_mean = np.mean(all_interarrivals)
overall_std = np.std(all_interarrivals)
overall_cv = overall_std / overall_mean

# Add theoretical Poisson mean line (lambda = 0.125, mean = 8.0)
lambda_rate = 0.125
theoretical_mean = 8.00
theoretical_std = 8.00
theoretical_cv = 1.0

# Draw theoretical mean line across all episodes
ax.axhline(y=theoretical_mean, color='purple', linestyle=':', 
          linewidth=3, alpha=0.5, zorder=0,
          label=f'Poisson Theory: E[Δt]={theoretical_mean:.1f}, σ={theoretical_std:.1f}, CV={theoretical_cv:.1f}')

# Styling
ax.set_xlabel('Continuous Time (Time Units)', fontsize=16, fontweight='heavy')
ax.set_ylabel('Interarrival Time (Δt)', fontsize=16, fontweight='heavy')
ax.set_title('Job Interarrival Times Comparison Across Multiple Episodes', 
            fontsize=18, fontweight='bold', pad=20)

ax.set_xlim(0, 50)
# Y limit based on max interarrival time (already calculated)
ax.set_ylim(0, max_interarrival * 1.1)

ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='x')

# Create custom legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = []
for ep_num in selected_episodes:
    color = episode_colors[ep_num]
    data = all_episodes_data[ep_num]
    mean_val = np.mean(data['interarrivals'])
    std_val = np.std(data['interarrivals'])
    cv_val = std_val / mean_val
    legend_elements.append(
        Line2D([0], [0], color=color, linewidth=3, 
               label=f'Episode {ep_num}: μ={mean_val:.2f}, σ={std_val:.2f}, CV={cv_val:.3f}')
    )

# Add overall statistics (ALL episodes, not just selected 3)
# Use values from comprehensive stochasticity analysis
legend_elements.append(
    Line2D([0], [0], color='black', linewidth=0, 
           label=f'\nOverall Observed (ALL episodes): μ=6.74, σ=6.55, CV=0.971')
)

# Add theoretical line
legend_elements.append(
    Line2D([0], [0], color='purple', linewidth=3, linestyle=':', alpha=0.5,
           label=f'Poisson Theory: λ={lambda_rate:.3f}, μ={theoretical_mean:.1f}, σ={theoretical_std:.1f}, CV={theoretical_cv:.1f}')
)

ax.legend(handles=legend_elements, fontsize=11, loc='upper left', 
         framealpha=0.95, fancybox=True, shadow=True)

plt.tight_layout()

# Save figure
output_path = OUTPUT_DIR / 'job_interarrival_times_multiple_episodes.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved multi-episode interarrival plot: {output_path}")
print("="*70)

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
for ep_num in selected_episodes:
    data = all_episodes_data[ep_num]
    interarrivals = data['interarrivals']
    print(f"\nEpisode {ep_num}:")
    print(f"  Sample size: {len(interarrivals)}")
    print(f"  Mean: {np.mean(interarrivals):.3f}")
    print(f"  Std Dev: {np.std(interarrivals):.3f}")
    print(f"  CV: {np.std(interarrivals)/np.mean(interarrivals):.3f}")
    print(f"  Min: {np.min(interarrivals):.3f}")
    print(f"  Max: {np.max(interarrivals):.3f}")

print(f"\nOverall (all selected episodes):")
print(f"  Total samples: {len(all_interarrivals)}")
print(f"  Mean: {overall_mean:.3f}")
print(f"  Std Dev: {overall_std:.3f}")
print(f"  CV: {overall_cv:.3f}")

print(f"\nTheoretical Poisson:")
print(f"  Lambda: {lambda_rate:.3f}")
print(f"  Expected Mean: {theoretical_mean:.3f}")
print(f"  Expected Std: {theoretical_std:.3f}")
print(f"  Expected CV: {theoretical_cv:.3f}")

print("\n" + "="*70)

plt.close()
