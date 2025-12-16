#!/usr/bin/env python3
"""
Custom diagram renderer with full control over layout
Usage: python render_diagram_custom.py
Output: masa_qmix_episode_custom.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Figure setup - compact 16:9 ratio
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Helper function for boxes
def add_box(ax, x, y, width, height, text, color, border_color, border_width=2, fontsize=9):
    box = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.05", 
                          facecolor=color, 
                          edgecolor=border_color, 
                          linewidth=border_width)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize, 
            wrap=True, multialignment='center')

# Helper function for arrows
def add_arrow(ax, x1, y1, x2, y2, label='', style='solid', color='black'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linestyle=style, color=color, linewidth=1.5)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label, fontsize=7, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Colors
color_arr = '#E0F2FE'
color_simpy = '#FEF9C3'
color_floor = '#FFF4E6'
color_agent = '#FEF3C7'
color_buffer = '#E0E7FF'
color_exit = '#D1FAE5'

# 1. Job Arrivals (top left)
add_box(ax, 0.3, 7, 1.8, 1.5, 
        "📦 Job Arrivals\n\nJ1: OpA→OpB→OpC\nJ2: OpD→OpA→OpE→OpB\nJ3: OpC→OpB\n⋮\nJn: OpF→OpA→OpE",
        color_arr, '#0284C7', fontsize=8)

# 2. SimPy Environment (center - BIG BOX)
simpy_box = FancyBboxPatch((2.3, 2.5), 7, 6, 
                           boxstyle="round,pad=0.1", 
                           facecolor=color_simpy, 
                           edgecolor='#CA8A04', 
                           linewidth=3)
ax.add_patch(simpy_box)
ax.text(5.8, 8.2, "⚙️ SimPy Environment - Discrete Event Simulation", 
        ha='center', fontsize=10, weight='bold')

# 2a. Job Shop Floor State (inside SimPy)
floor_box = FancyBboxPatch((2.5, 6.2), 6.6, 1.8,
                           boxstyle="round,pad=0.05",
                           facecolor=color_floor,
                           edgecolor='#F59E0B',
                           linewidth=2)
ax.add_patch(floor_box)
ax.text(5.8, 7.9, "🏭 Job Shop Floor State", ha='center', fontsize=9, weight='bold')

# Floor sub-boxes (horizontal layout)
add_box(ax, 2.6, 6.8, 1.5, 1,
        "Machines\nM1:BUSY J2-OpB+O1\nM2:BUSY J5-OpA+O3\nM3:IDLE\nM4:BUSY J1-OpC+O2\nM5:IDLE",
        color_floor, '#F59E0B', 1, 7)

add_box(ax, 4.2, 6.8, 1.4, 1,
        "Operators\nO1:BUSY@M1\nO2:BUSY@M4\nO3:BUSY@M2\nO4:IDLE",
        color_floor, '#F59E0B', 1, 7)

add_box(ax, 5.7, 6.8, 1.3, 1,
        "Queue\nJ3:5s→OpC\nJ4:2s→OpA\nJ7:8s→OpE",
        color_floor, '#F59E0B', 1, 7)

add_box(ax, 7.1, 6.8, 1.9, 1,
        "Active Agents\nJ1→A1 | J2→A2 | J3→A3\nJ4→A4 | J5→A5 | J7→A7\nAll jobs in system = agents",
        color_floor, '#F59E0B', 1, 7)

# 2b. Observation Builder
add_box(ax, 2.6, 5, 2.2, 1,
        "📊 Obs Builder\nPer-agent: obs_i dim n_obs\nGlobal: state dim n_state\nNorm [0,1]",
        color_floor, '#F59E0B', 1.5, 7)

# 2c. Reward Calculator
add_box(ax, 5, 5, 2.2, 1,
        "💰 Reward - Every Step\nR_global = w1×Completed - w2×Wait\n+ w4×Throughput∆ + w5×LoadBalance\nR_total = R_global / reward_scale",
        color_floor, '#F59E0B', 1.5, 7)

# 2d. Completion
add_box(ax, 7.4, 5, 1.5, 1,
        "✅ Completion\nWhen done:\nRemove agent\nUpdate metrics",
        color_floor, '#F59E0B', 1.5, 7)

# 2e. Executor
add_box(ax, 2.6, 3.2, 2.2, 1,
        "▶️ Executor\nExecute joint actions in parallel\nAssign to mc+op\nAdvance time\nUpdate floor",
        color_floor, '#F59E0B', 1.5, 7)

# 3. Exit (bottom center)
add_box(ax, 4.5, 0.5, 1.5, 0.8,
        "🎉 Completed Jobs\nJ0, J8, J12, ...\nExit system",
        color_exit, '#059669', 2, 7)

# 4. Multi-Agent Network (top right)
agent_box = FancyBboxPatch((9.6, 5.5), 3.2, 3,
                           boxstyle="round,pad=0.1",
                           facecolor=color_agent,
                           edgecolor='#F59E0B',
                           linewidth=3)
ax.add_patch(agent_box)
ax.text(11.2, 8.2, "🤖 Multi-Agent Network", ha='center', fontsize=9, weight='bold')

# Agent RNNs (vertical stack)
add_box(ax, 9.8, 7.4, 2.8, 0.5,
        "Agent-1 RNN: obs_1 → GRU rnn_hidden → Q_1(o,a)",
        color_agent, '#F59E0B', 1, 7)
add_box(ax, 9.8, 6.8, 2.8, 0.5,
        "Agent-2 RNN: obs_2 → GRU rnn_hidden → Q_2(o,a)",
        color_agent, '#F59E0B', 1, 7)
ax.text(11.2, 6.5, "⋮  more agents", ha='center', fontsize=8)
add_box(ax, 9.8, 6, 2.8, 0.5,
        "Agent-n RNN: obs_n → GRU rnn_hidden → Q_n(o,a)",
        color_agent, '#F59E0B', 1, 7)

# 5. Experience Storage (right bottom)
buffer_box = FancyBboxPatch((9.6, 0.5), 6, 4.5,
                            boxstyle="round,pad=0.1",
                            facecolor=color_buffer,
                            edgecolor='#6366F1',
                            linewidth=3)
ax.add_patch(buffer_box)
ax.text(12.6, 4.7, "💾 Experience Storage", ha='center', fontsize=9, weight='bold')

# Replay Buffer
add_box(ax, 9.8, 2.5, 2.6, 1.8,
        "Replay Buffer\nStore transition tuple:\n\nstate: s_t dim n_state\nobs: [obs_1,...,obs_n] dim n_agents×n_obs\nactions: [a_1,...,a_n]\nreward: r_t global scalar\nnext_state: s_t+1\nnext_obs: [obs_1',...,obs_n']\ndone: episode_end flag\n\nSize: buffer_size episodes\nCurrently: N episodes stored",
        color_buffer, '#6366F1', 1.5, 6.5)

# Training Module
add_box(ax, 12.6, 2.5, 2.8, 1.8,
        "Training Module\nif learn=True:\n\nSample: batch_size episodes\n→ QMIX Mixer\n→ Compute TD loss\n→ Backprop gradient\n\nEvery train_steps steps",
        color_buffer, '#6366F1', 1.5, 7)

# Policy box (below agents)
add_box(ax, 13, 5.8, 2.4, 2.3,
        "ε-greedy Policy\nFor EACH agent i:\n\nif rand < ε:\n  random valid action\nelse:\n  argmax_a Q_i(o,a)\n\nε decays from\nepsilon_start to\nepsilon_finish\n\nOutput: Joint actions\n[a_1, a_2, ..., a_n]",
        color_agent, '#F59E0B', 1.5, 7)

# === ARROWS ===

# Arrivals to Floor
add_arrow(ax, 2.1, 7.5, 2.5, 7.3, 'Arrive')

# Floor to Obs Builder
add_arrow(ax, 5.8, 6.2, 3.7, 5.9, '')

# Obs Builder to RNNs
add_arrow(ax, 4.8, 5.5, 9.8, 7.5, 'obs batch')
add_arrow(ax, 4.8, 5.4, 9.8, 7, '')
add_arrow(ax, 4.8, 5.3, 9.8, 6.2, '')

# RNNs to Policy
add_arrow(ax, 12.6, 7.5, 13, 7.5, '')
add_arrow(ax, 12.6, 7, 13, 7, '')
add_arrow(ax, 12.6, 6.2, 13, 6.5, '')

# Policy to Executor
add_arrow(ax, 13, 5.8, 3.7, 4.1, 'joint actions')

# Executor to Floor (feedback loop)
add_arrow(ax, 3.7, 3.2, 5.8, 6.2, '', 'dashed')

# Floor to Reward
add_arrow(ax, 5.8, 6.2, 6.1, 5.9, '')

# Reward to Completion
add_arrow(ax, 7.2, 5.5, 7.4, 5.5, '')

# Completion to Exit
add_arrow(ax, 8.2, 5, 5.3, 1.3, 'exit')

# Reward to Replay Buffer
add_arrow(ax, 6.1, 5, 9.8, 3.5, 'store tuple')

# Replay Buffer to Training (conditional)
add_arrow(ax, 12.4, 3.4, 12.6, 3.4, 'if learn=True', 'dashed')

# Training back to RNNs (gradients)
add_arrow(ax, 13.5, 4.3, 11.2, 5.5, '∇θ', 'dashed')

# Floor loop back to Obs Builder
add_arrow(ax, 3, 6.2, 3, 5.9, 'next decision', 'dashed')

plt.tight_layout()
plt.savefig('masa_qmix_episode_custom.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Custom diagram saved to: masa_qmix_episode_custom.png")
plt.close()
