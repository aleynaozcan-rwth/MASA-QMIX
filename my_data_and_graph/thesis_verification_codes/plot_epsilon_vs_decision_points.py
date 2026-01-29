#!/usr/bin/env python3
"""
Plot Epsilon Decay vs Cumulative Decision Points
Shows how epsilon decays based on actual decision point count from execution
Calibrated so that epsilon reaches minimum at episode 774 (training step 14341)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
history_dir = os.path.join(script_dir, 'historydata')

# ==================== Read epsilon parameters from arguments ====================
# Import arguments to get epsilon_anneal_fraction
import sys
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), 'MARL', 'common'))
from arguments import get_common_args

args = get_common_args()
epsilon_anneal_fraction = args.epsilon_anneal_fraction

# ==================== Epsilon Decay Parameters ====================
epsilon_init = 1.0
epsilon_min = 0.05

# Target: epsilon should reach epsilon_min at episode 774 (training step 14341)
TARGET_EPISODE = 774

# ==================== Read actual decision points from timeline ====================
print("Reading scheduling timeline...")
timeline_path = os.path.join(history_dir, 'scheduling_timeline.txt')

episodes = []
current_episode = None
current_arrivals = 0
current_completions = 0
in_lifecycle_trace = False

with open(timeline_path, 'r') as f:
    for line in f:
        line = line.strip()
        
        # Match episode start
        if line.startswith('=== EPISODE'):
            if current_episode is not None:
                episodes.append({
                    'episode': current_episode,
                    'dp_count': current_arrivals + current_completions
                })
            
            ep_num = int(line.split()[2])
            current_episode = ep_num
            current_arrivals = 0
            current_completions = 0
            in_lifecycle_trace = False
        
        # Track lifecycle trace section
        elif 'JOB AGENT LIFECYCLE TRACE START' in line:
            in_lifecycle_trace = True
        
        elif 'JOB AGENT LIFECYCLE TRACE END' in line:
            in_lifecycle_trace = False
        
        # Count arrivals only in lifecycle trace
        elif in_lifecycle_trace and '[t=' in line and 'New job' in line and 'arrived' in line:
            current_arrivals += 1
        
        # Count completions only outside lifecycle trace (in TIMELINE section)
        elif not in_lifecycle_trace and '[t=' in line and 'finished' in line and 'next queued' in line:
            current_completions += 1

# Add last episode
if current_episode is not None:
    episodes.append({
        'episode': current_episode,
        'dp_count': current_arrivals + current_completions
    })

print(f"Parsed {len(episodes)} episodes")

# ==================== Calculate Decision Points at Target Episode ====================
# Find cumulative DPs at episode 774
cumulative_at_target = 0
for ep_data in episodes:
    if ep_data['episode'] < TARGET_EPISODE:
        cumulative_at_target += ep_data['dp_count']
    elif ep_data['episode'] == TARGET_EPISODE:
        cumulative_at_target += ep_data['dp_count']
        break

print(f"\nTarget: Epsilon should reach {epsilon_min} at episode {TARGET_EPISODE}")
print(f"Decision Points at episode {TARGET_EPISODE}: {cumulative_at_target:,}")

# Calculate gamma_epsilon needed: epsilon_min = epsilon_init * gamma^d
# gamma = (epsilon_min / epsilon_init) ^ (1/d)
if cumulative_at_target > 0:
    gamma_epsilon = (epsilon_min / epsilon_init) ** (1.0 / cumulative_at_target)
    print(f"Calculated gamma_epsilon: {gamma_epsilon:.8f}")
else:
    gamma_epsilon = 0.9995
    print("Warning: Could not find target episode, using default gamma")

# ==================== Calculate cumulative decision points and epsilon ====================
cumulative_dps = []
epsilon_values = []
episode_boundaries = []  # Mark where episodes end

cumulative = 0
for ep_data in episodes:
    dp_count = ep_data['dp_count']
    
    # Calculate epsilon at each decision point in this episode
    for i in range(dp_count):
        cumulative += 1
        cumulative_dps.append(cumulative)
        
        # Epsilon decay formula: ε_d = max(ε_min, ε_init × γ^d)
        epsilon = max(epsilon_min, epsilon_init * (gamma_epsilon ** cumulative))
        epsilon_values.append(epsilon)
    
    # Mark episode boundary
    episode_boundaries.append((cumulative, ep_data['episode']))

# Mark the target episode specially
target_dp_position = None
target_epsilon = None
for dp_pos, ep_num in episode_boundaries:
    if ep_num == TARGET_EPISODE:
        target_dp_position = dp_pos
        # Find epsilon at this position
        idx = np.argmin(np.abs(np.array(cumulative_dps) - target_dp_position))
        target_epsilon = epsilon_values[idx]
        break

print(f"\nTotal Decision Points: {cumulative:,}")
print(f"Epsilon range: {min(epsilon_values):.4f} to {max(epsilon_values):.4f}")

# ==================== Create Visualization ====================
plots_dir = os.path.join(history_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
out_path = os.path.join(plots_dir, 'epsilon_vs_decision_points.png')

fig, ax = plt.subplots(figsize=(14, 7))

# Shade regions with similar colors to reference graph
ax.axhspan(0.5, 1.0, alpha=0.15, color='red', zorder=0)
ax.axhspan(0.2, 0.5, alpha=0.15, color='orange', zorder=0)
ax.axhspan(epsilon_min, 0.2, alpha=0.15, color='yellow', zorder=0)

# Plot epsilon decay with thicker line
ax.plot(cumulative_dps, epsilon_values, linewidth=3, color='steelblue', alpha=0.9, zorder=5)

# Add epsilon_min line
ax.axhline(y=epsilon_min, color='green', linestyle='--', linewidth=2.5, alpha=0.8, zorder=4)

# Mark key milestones with yellow boxes like in reference
milestones = [0.5, 0.2, 0.1, epsilon_min]

for eps_threshold in milestones:
    if eps_threshold >= min(epsilon_values):
        # Find where epsilon crosses this threshold
        idx = np.argmax(np.array(epsilon_values) <= eps_threshold)
        if idx > 0:
            dp_at_threshold = cumulative_dps[idx]
            
            # Add vertical line
            ax.axvline(x=dp_at_threshold, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=3)
            
            # Add yellow annotation box
            ax.annotate(f'ε={eps_threshold}\nDP {dp_at_threshold:,}', 
                       xy=(dp_at_threshold, eps_threshold),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1.5),
                       fontsize=9, fontweight='bold', zorder=10,
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=1.5))

# Highlight target episode (episode 774, training step 14341)
if target_dp_position is not None:
    ax.axvline(x=target_dp_position, color='darkred', linestyle='-', linewidth=2.5, alpha=0.7, zorder=3)
    ax.scatter([target_dp_position], [target_epsilon], s=150, color='red', 
              marker='o', edgecolors='darkred', linewidth=2.5, zorder=15)

# Create custom legend elements (right side box)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='steelblue', linewidth=3, label='Epsilon'),
    Patch(facecolor='red', alpha=0.15, label='High Exploration (ε>0.5)'),
    Patch(facecolor='orange', alpha=0.15, label='Medium Exploration (0.2<ε<0.5)'),
    Patch(facecolor='yellow', alpha=0.15, label='Low Exploration (0.05<ε<0.2)'),
    Line2D([0], [0], color='green', linestyle='--', linewidth=2.5, label=f'Min Epsilon ({epsilon_min}) - Exploitation'),
    Line2D([0], [0], linestyle='None', label=f'Total DP: {cumulative:,}'),
    Line2D([0], [0], linestyle='None', label=f'Anneal Epsilon Fraction = {epsilon_anneal_fraction}'),
    Line2D([0], [0], linestyle='None', label='Job Arrival Rate λ = 0.125'),
    Line2D([0], [0], linestyle='None', label=r'$|N_{DP}(e)|$: number of deciding agents at DP e'),
    Line2D([0], [0], linestyle='None', label=r'$|N_{DP}(e)|$ = time-varying'),
    Line2D([0], [0], linestyle='None', label='e: index of event-triggered DPs')
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

ax.set_xlabel(r'Decision Point (1 RL step = $|\mathbf{N_{DP}(e)}|$ experiences)', fontsize=13, fontweight='bold')
ax.set_ylabel('Epsilon (ε)', fontsize=13, fontweight='bold')
ax.set_title('Epsilon Decay: Exploration → Exploitation Transition (Decision-Point-Based)', 
            fontsize=14, fontweight='bold', pad=15)

ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_xlim(0, cumulative)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved epsilon vs decision points: {out_path}")

plt.close()
