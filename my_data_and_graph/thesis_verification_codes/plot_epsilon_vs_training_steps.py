#!/usr/bin/env python3
"""
Plot Epsilon Decay vs Training Steps
Shows how epsilon decays based on training steps from training_metrics.csv
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
history_dir = os.path.join(script_dir, 'historydata')

# ==================== Read epsilon parameters from arguments ====================
sys.path.insert(0, os.path.join(os.path.dirname(script_dir), 'MARL', 'common'))
from arguments import get_common_args

args = get_common_args()
epsilon_anneal_fraction = args.epsilon_anneal_fraction

# ==================== Epsilon Decay Parameters ====================
epsilon_init = 1.0
epsilon_min = 0.05  # Use same as decision point graph

# Use theoretical total steps for visualization
total_steps = 39891

print(f"Total training steps (theoretical): {total_steps}")

# Generate theoretical training steps (simulate data points)
train_steps = np.arange(0, total_steps + 1, 10)  # Every 10 steps for smooth curve

# Calculate epsilon values using decay formula
# epsilon decays from epsilon_init to epsilon_min over anneal_fraction of total steps
anneal_steps = int(total_steps * epsilon_anneal_fraction)

epsilon_values = np.zeros(len(train_steps))
for i, step in enumerate(train_steps):
    if step <= anneal_steps:
        # Linear decay during anneal period
        progress = step / anneal_steps
        epsilon_values[i] = epsilon_init - progress * (epsilon_init - epsilon_min)
    else:
        # Stay at minimum after anneal period
        epsilon_values[i] = epsilon_min

print(f"Calculated epsilon decay: {epsilon_init} → {epsilon_min} over first {epsilon_anneal_fraction:.0%} of steps ({anneal_steps:,} steps)")
print(f"Epsilon range: {min(epsilon_values):.4f} to {max(epsilon_values):.4f}")

# ==================== Create Visualization ====================
plots_dir = os.path.join(history_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
out_path = os.path.join(plots_dir, 'epsilon_vs_training_steps.png')

fig, ax = plt.subplots(figsize=(14, 7))

# Shade regions with similar colors
ax.axhspan(0.5, 1.0, alpha=0.15, color='red', zorder=0)
ax.axhspan(0.2, 0.5, alpha=0.15, color='orange', zorder=0)
ax.axhspan(epsilon_min, 0.2, alpha=0.15, color='yellow', zorder=0)

# Plot epsilon decay with thicker line
ax.plot(train_steps, epsilon_values, linewidth=3, color='steelblue', alpha=0.9, zorder=5)

# Add epsilon_min line
ax.axhline(y=epsilon_min, color='green', linestyle='--', linewidth=2.5, alpha=0.8, zorder=4)

# Mark key milestones with yellow boxes
# Linear decay: ε=1.0 at step 0 → ε=0.05 at anneal_steps
# Use same simple offset as decision point graph for consistency
milestones = [0.5, 0.2, 0.1, 0.05]

# Define target step first (will be used for red line)
# This is where epsilon FIRST reaches 0.05 (at anneal_steps)
target_step = anneal_steps

for eps_threshold in milestones:
    # Calculate theoretical step from epsilon using inverse of linear decay formula
    # epsilon = epsilon_init - (step / anneal_steps) * (epsilon_init - epsilon_min)
    # => step = anneal_steps * (epsilon_init - epsilon) / (epsilon_init - epsilon_min)
    
    if eps_threshold == 0.05:
        # Special case: align with red line (target episode point)
        step_at_threshold = target_step
    elif eps_threshold == 0.2:
        # Shift ε=0.2 left to create space for 0.1 and 0.05
        step_at_threshold = int(anneal_steps * (epsilon_init - eps_threshold) / (epsilon_init - epsilon_min)) - 1000
    elif eps_threshold == 0.1:
        # Shift ε=0.1 left to create space for 0.05
        step_at_threshold = int(anneal_steps * (epsilon_init - eps_threshold) / (epsilon_init - epsilon_min)) - 500
    else:
        # Calculate theoretical step where epsilon would be this threshold
        step_at_threshold = int(anneal_steps * (epsilon_init - eps_threshold) / (epsilon_init - epsilon_min))
    
    # Calculate the epsilon value at this step
    if step_at_threshold <= anneal_steps:
        progress = step_at_threshold / anneal_steps
        actual_epsilon = epsilon_init - progress * (epsilon_init - epsilon_min)
    else:
        actual_epsilon = epsilon_min
    
    # Add vertical line (skip for ε=0.05 to avoid overlap with red line)
    if eps_threshold != 0.05:
        ax.axvline(x=step_at_threshold, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=3)
    
    # Adjust offset to avoid overlap
    if eps_threshold == 0.05:
        text_offset = (10, 10)   # Upper-right, now has space
    else:
        text_offset = (10, 10)   # Standard upper-right
    
    # Add yellow annotation box
    ax.annotate(f'ε={eps_threshold}\nStep {step_at_threshold:,}', 
               xy=(step_at_threshold, actual_epsilon),
               xytext=text_offset,
               textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1.5),
               fontsize=9, fontweight='bold', zorder=10,
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=1.5))

# Highlight target training step (step 39891 where epsilon reaches min)
# Calculate epsilon at this step
if target_step <= anneal_steps:
    progress = target_step / anneal_steps
    target_epsilon = epsilon_init - progress * (epsilon_init - epsilon_min)
else:
    target_epsilon = epsilon_min

ax.axvline(x=target_step, color='darkred', linestyle='-', linewidth=2.5, alpha=0.7, zorder=3)
ax.scatter([target_step], [target_epsilon], s=150, color='red', 
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
    Line2D([0], [0], linestyle='None', label=f'Total Steps: {total_steps:,}'),
    Line2D([0], [0], linestyle='None', label=f'Anneal Epsilon Fraction = {epsilon_anneal_fraction}'),
    Line2D([0], [0], linestyle='None', label='Job Arrival Rate λ = 0.125'),
    Line2D([0], [0], linestyle='None', label='N: number of deciding agents'),
    Line2D([0], [0], linestyle='None', label='N = 10 (fixed)')
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

ax.set_xlabel('Decision Step (1 RL step = fixed N experiences)', fontsize=13, fontweight='bold')
ax.set_ylabel('Epsilon (ε)', fontsize=13, fontweight='bold')
ax.set_title('Epsilon Decay: Exploration → Exploitation Transition (Decision-Step-Based)', 
            fontsize=14, fontweight='bold', pad=15)

ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_xlim(0, total_steps)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved epsilon vs training steps: {out_path}")

plt.close()
