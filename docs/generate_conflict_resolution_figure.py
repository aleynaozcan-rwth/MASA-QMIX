"""
Generate conflict resolution flowchart for thesis.

This script creates a visualization of the parallel decision making and 
policy-consistent conflict resolution mechanism.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib

# Use standard fonts
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

# QMIX diagram color palette
colors = {
    'state': '#C8E6C9',      # Light green - state/observation
    'agent': '#F8BBD0',      # Light pink - agents
    'action': '#B2EBF2',     # Light cyan - actions
    'conflict': '#FFCCBC',   # Light orange - conflict
    'resolution': '#E1BEE7', # Light purple - resolution
    'outcome': '#FFF9C4',    # Light yellow - outcomes
}

fig, ax = plt.subplots(figsize=(12, 11))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-1.5, 12)
ax.axis('off')

# Helper function to create fancy boxes
def create_box(ax, x, y, width, height, text, color, fontsize=10, fontweight='normal'):
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='#212121',
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            fontweight=fontweight, color='#212121')

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color='#424242',
        linewidth=2,
        mutation_scale=20
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Title
ax.text(5, 11.5, 'Parallel Decision Making & Conflict Resolution', 
        ha='center', fontsize=14, fontweight='bold')

# Stage 1: State Freezing
create_box(ax, 5, 10.5, 3, 0.6, 'Event triggers decision at t_d', colors['state'], fontsize=11, fontweight='bold')
create_arrow(ax, 5, 10.2, 5, 9.7)

create_box(ax, 5, 9.3, 3.5, 0.7, 'Freeze global state s_t_d', colors['state'], fontsize=11, fontweight='bold')
create_arrow(ax, 5, 8.95, 5, 8.5)

# Stage 2: Parallel Action Selection
ax.text(5, 8.3, 'Stage 1: Parallel Action Selection', ha='center', 
        fontsize=11, fontweight='bold', color='#1565C0')

# Three agents in parallel
agent_positions = [(2.5, 7.5), (5, 7.5), (7.5, 7.5)]
agent_labels = ['Agent J₃', 'Agent J₇', 'Agent J₁₁']

for i, (x, y) in enumerate(agent_positions):
    create_box(ax, x, y, 1.8, 0.6, agent_labels[i], colors['agent'], fontsize=10, fontweight='bold')
    # Observation arrows from frozen state - stop well before box edge
    create_arrow(ax, 5, 8.2, x, y + 0.45, style='->')
    # Action selection
    create_arrow(ax, x, y - 0.3, x, y - 1.0)

# Q-networks and actions
q_values = ['Q=4.2', 'Q=3.8', 'Q=2.5']
actions = ['a=M₂', 'a=M₂', 'a=M₁']

for i, (x, _) in enumerate(agent_positions):
    # Q-value boxes
    create_box(ax, x, 6.3, 1.5, 0.5, f'Q-network\n{q_values[i]}', 
               colors['action'], fontsize=9)
    create_arrow(ax, x, 6.05, x, 5.6)
    
    # Action boxes
    create_box(ax, x, 5.3, 1.5, 0.5, actions[i], 
               colors['action'], fontsize=10, fontweight='bold')

# Collect actions
create_arrow(ax, 2.5, 5.05, 5, 4.5)
create_arrow(ax, 5, 5.05, 5, 4.5)
create_arrow(ax, 7.5, 5.05, 5, 4.5)

# Stage 3: Conflict Detection
ax.text(5, 4.3, 'Stage 2: Conflict Detection', ha='center', 
        fontsize=11, fontweight='bold', color='#E65100')

create_box(ax, 5, 3.7, 3.5, 0.6, 'Collect: {J₃→M₂, J₇→M₂, J₁₁→M₁}', 
           colors['conflict'], fontsize=10)
create_arrow(ax, 5, 3.4, 5, 3.0)

create_box(ax, 5, 2.6, 3.5, 0.7, 'Detect: C_M₂ = {J₃, J₇}\n|C_M₂| = 2 → CONFLICT!', 
           colors['conflict'], fontsize=10, fontweight='bold')

# Stage 4: Resolution
create_arrow(ax, 3.3, 2.6, 2, 1.5, '')
ax.text(2.3, 2.05, 'M₂ conflict', fontsize=9, 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
create_arrow(ax, 6.7, 2.6, 8, 1.7, '')
ax.text(7.35, 2.15, 'M₁ no conflict', fontsize=9, 
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Outcomes for conflict arrows - DRAW FIRST (behind box)
create_arrow(ax, 0.8, 0.65, 0.8, -0.1)
create_arrow(ax, 3.2, 0.65, 3.2, -0.1)

# Conflict branch (left) - DRAW AFTER ARROWS (on top)
ax.text(2, 1.3, 'Stage 3: ε-greedy Resolution', ha='center', 
        fontsize=10, fontweight='bold', color='#6A1B9A')
create_box(ax, 2, 0.8, 2.8, 0.7, 'Exploit: argmax Q → J₃ wins\n(4.2 > 3.8)', 
           colors['resolution'], fontsize=9, fontweight='bold')

# No conflict branch (right)
create_box(ax, 8, 1.4, 2.2, 0.5, 'J₁₁ gets M₁\n(success)', 
           colors['outcome'], fontsize=9, fontweight='bold')

# Winner outcome (left-left) - create box without text
box_winner = FancyBboxPatch(
    (0.8 - 1.0, -0.55 - 0.425), 2.0, 0.85,
    boxstyle="round,pad=0.1",
    facecolor=colors['outcome'],
    edgecolor='#212121',
    linewidth=2
)
ax.add_patch(box_winner)
# Add text with green "Winner"
ax.text(0.8, -0.35, 'Winner', ha='center', va='center', fontsize=10, 
        fontweight='bold', color='#2E7D32')
ax.text(0.8, -0.65, 'J₃ gets M₂', ha='center', va='center', fontsize=10, 
        fontweight='bold', color='#212121')

# Loser outcome (left-right) - create box without text
box_loser = FancyBboxPatch(
    (3.2 - 1.0, -0.55 - 0.425), 2.0, 0.85,
    boxstyle="round,pad=0.1",
    facecolor=colors['conflict'],
    edgecolor='#212121',
    linewidth=2
)
ax.add_patch(box_loser)
# Add text with red "Loser"
ax.text(3.2, -0.25, 'Loser', ha='center', va='center', fontsize=10, 
        fontweight='bold', color='#C62828')
ax.text(3.2, -0.55, 'J₇ waits till next', ha='center', va='center', fontsize=9, 
        fontweight='bold', color='#212121')
ax.text(3.2, -0.70, 'decision point', ha='center', va='center', fontsize=9, 
        fontweight='bold', color='#212121')
ax.text(3.2, -0.85, 'waiting time penalty', ha='center', va='center', fontsize=8, 
        color='#212121', style='italic')

# Learning feedback - moved closer to loser box
create_arrow(ax, 4.2, -0.55, 4.8, -0.55, style='->')
create_box(ax, 5.5, -0.55, 1.6, 0.7, 'TD Update:\nQ(o⁷,M₂) ↓', 
           '#BBDEFB', fontsize=9, fontweight='bold')

# Legend
legend_y = 11.2
legend_x_start = 0.3
legend_items = [
    ('State/Event', colors['state']),
    ('Agent', colors['agent']),
    ('Action/Q-value', colors['action']),
    ('Conflict', colors['conflict']),
    ('Resolution', colors['resolution']),
    ('Outcome', colors['outcome']),
]

for i, (label, color) in enumerate(legend_items):
    x = legend_x_start + i * 1.6
    legend_box = FancyBboxPatch(
        (x, legend_y - 0.15), 0.3, 0.3,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='#212121',
        linewidth=1
    )
    ax.add_patch(legend_box)
    ax.text(x + 0.45, legend_y, label, fontsize=8, va='center')

plt.tight_layout()

# Save
output_dir = '/home/cc253232/MASA-QMIX/docs/'
plt.savefig(output_dir + 'conflict_resolution_flow.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir + 'conflict_resolution_flow.png', dpi=300, bbox_inches='tight')

print(f"✓ Figure saved to:")
print(f"  - {output_dir}conflict_resolution_flow.pdf")
print(f"  - {output_dir}conflict_resolution_flow.png")
print("\nUsage in LaTeX:")
print(r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{docs/conflict_resolution_flow.pdf}
\caption{Parallel decision making and policy-consistent conflict resolution mechanism. 
When multiple agents select the same machine (e.g., J₃ and J₇ both select M₂), the agent 
with the highest Q-value wins the resource, while losers receive negative rewards and 
learn to avoid congested resources in future similar states.}
\label{fig:conflict_resolution_flow}
\end{figure}
""")

plt.show()
