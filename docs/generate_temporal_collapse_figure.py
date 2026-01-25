"""
Generate temporal abstraction figure for thesis.

This script creates a visualization showing how discrete-event simulation
handles temporal collapse in fixed-interval decision making.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use standard fonts (LaTeX fonts require texlive-full)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

# Create figure
fig, ax = plt.subplots(figsize=(12, 4))

# Time axis
ax.plot([0, 3], [0, 0], 'k-', linewidth=2, zorder=1)
ax.set_xlim(-0.1, 3.1)
ax.set_ylim(-0.8, 1.8)

# Color palette matching QMIX diagram with enhanced visibility
colors = {'arrival': '#81C784', 'completion': '#4DD0E1'}  # Medium green and cyan
grid_color = '#F06292'  # Medium pink
arrow_color = '#BA68C8'  # Medium purple
highlight_bg = '#FFF59D'  # Medium yellow
text_color = '#212121'  # Dark gray for text

# Grid decision points (dark gray squares)
grid_times = [0, 1, 2, 3]
for i, t in enumerate(grid_times):
    ax.axvline(t, color='gray', linestyle='--', alpha=0.4, linewidth=1, zorder=1)
    ax.plot(t, 0, 's', color=grid_color, markersize=14, markeredgecolor='#880E4F', 
            markeredgewidth=2, zorder=5, 
            label='Grid decision point' if i == 0 else '')
    ax.text(t, -0.25, f't_{i}', ha='center', fontsize=13, fontweight='bold')

# True event times between t1=1 and t2=2
event_times = [1.10, 1.25, 1.51, 1.74, 1.89]
event_labels = ['e₁', 'e₂', 'e₃', 'e₄', 'e₅']
event_types = ['arrival', 'completion', 'arrival', 'completion', 'arrival']
epsilons = [0.90, 0.75, 0.49, 0.26, 0.11]

for i, (t, label, evt_type, eps) in enumerate(zip(event_times, event_labels, event_types, epsilons)):
    # Event marker (blue circles)
    color = colors[evt_type]
    ax.plot(t, 1.0, 'o', color=color, markersize=12, 
            markeredgecolor='#212121', markeredgewidth=2, zorder=5,
            label=f'{evt_type.capitalize()} event' if i == 0 or (evt_type == 'completion' and i == 1) else '')
    
    # Event label
    ax.text(t, 1.2, label, ha='center', fontsize=12, fontweight='bold')
    
    # Vertical dotted line from event to time axis
    ax.plot([t, t], [1.0, 0], 'k:', linewidth=2, alpha=0.8, zorder=2)
    
    # Exact time label on time axis
    ax.text(t, -0.05, f'{t:.2f}', ha='center', va='top', 
            fontsize=9, color=grid_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                     edgecolor='#1f77b4', alpha=0.9, linewidth=1))
    
    # Collapse arrow (showing delay)
    y_target = 0.15 + i * 0.12
    ax.annotate('', xy=(2, y_target), xytext=(t, 1.0),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2, 
                                alpha=0.7, shrinkA=5, shrinkB=5),
                zorder=3)
    
    # Epsilon label (only for first and last to avoid clutter)
    if i == 0:
        ax.text(t, 0.5, f'ε₁={eps:.2f}', 
                fontsize=9, color=arrow_color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                         edgecolor=arrow_color, alpha=0.9),
                ha='center', va='center')
    if i == 4:
        ax.text(t, 0.5, f'ε₅={eps:.2f}', 
                fontsize=9, color=arrow_color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', 
                         edgecolor=arrow_color, alpha=0.9),
                ha='center', va='center')

# Decision interval annotation
ax.annotate('', xy=(2, -0.6), xytext=(1, -0.6),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2.5))
ax.text(1.5, -0.75, 'Decision interval: Δt = 1', 
        ha='center', fontsize=12, fontweight='bold')

# Shaded region showing where events occur
ax.axvspan(1, 2, alpha=0.12, color='gray', zorder=0)
ax.text(1.5, 1.6, 'Temporal Collapse Region', 
        ha='center', fontsize=14, color='#212121', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor=grid_color, linewidth=2.5, alpha=0.95))

# Add annotation for the problem
ax.annotate('All events observed\nonly at t₂=2', 
            xy=(2, 0.7), xytext=(2.4, 1.2),
            fontsize=11, color='#212121',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=arrow_color, linewidth=2),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5, 
                          connectionstyle='arc3,rad=0.3'))

# Axis labels
ax.set_xlabel('Continuous Time', fontsize=13, fontweight='bold')
ax.set_ylabel('', fontsize=12)
ax.set_yticks([])

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9, 
         edgecolor='black', fancybox=True)

# Grid for readability
ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=0.5)

plt.tight_layout()

# Save in multiple formats
output_dir = '/home/cc253232/MASA-QMIX/docs/'
plt.savefig(output_dir + 'temporal_abstraction.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir + 'temporal_abstraction.png', dpi=300, bbox_inches='tight')

print(f"✓ Figure saved to:")
print(f"  - {output_dir}temporal_abstraction.pdf")
print(f"  - {output_dir}temporal_abstraction.png")
print("\nUsage in LaTeX:")
print(r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{docs/temporal_abstraction.pdf}
\caption{Temporal abstraction error under fixed-interval decision making. 
Events $e_1, \ldots, e_5$ occurring at distinct continuous times within the 
interval $(t_1, t_2)$ are all collapsed to the next grid decision point $t_2$, 
introducing delays $\varepsilon_r = t_2 - t_{e_r}$. This temporal collapse 
destroys event ordering and causes systematic bias in scheduling decisions.}
\label{fig:temporal_abstraction}
\end{figure}
""")

plt.show()
