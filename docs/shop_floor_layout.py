"""
Shop Floor Layout Visualization
Generates a visual diagram of machines, work centers, and operators
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(18, 8))
ax.set_xlim(0, 18)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
wc1_color = '#E8F4F8'  # Light blue
wc2_color = '#F0E8F8'  # Light purple
wc3_color = '#F8F4E8'  # Light yellow
machine_color = '#4A90E2'  # Blue
operator_color = '#FFB6C1'  # Light pastel pink

# Title
ax.text(9, 7.5, 'Shop Floor Layout', fontsize=20, fontweight='bold', ha='center')

# Work Center 1 - Left
wc1_box = FancyBboxPatch((0.5, 3.0), 5, 4, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='#2E5C8A', linewidth=2.5, 
                          facecolor=wc1_color, alpha=0.7)
ax.add_patch(wc1_box)
ax.text(3, 6.7, 'Work Center 1 (WC1)', fontsize=14, fontweight='bold', ha='center')

# Machine M0 in WC1 (centered)
m0_box = FancyBboxPatch((0.75, 4.5), 2, 1.8, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='#2E5C8A', linewidth=2, 
                         facecolor=machine_color, alpha=0.8)
ax.add_patch(m0_box)
ax.text(1.75, 5.8, 'M0', fontsize=20, fontweight='bold', ha='center', va='center', color='white')
ax.text(1.75, 5.3, '5 ops', fontsize=12, ha='center', va='center', color='white')
ax.text(1.75, 4.85, 'Op1,2,3,5,9', fontsize=9, ha='center', va='center', color='white')

# Machine M1 in WC1 (centered)
m1_box = FancyBboxPatch((3.25, 4.5), 2, 1.8, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='#2E5C8A', linewidth=2, 
                         facecolor=machine_color, alpha=0.8)
ax.add_patch(m1_box)
ax.text(4.25, 5.8, 'M1', fontsize=20, fontweight='bold', ha='center', va='center', color='white')
ax.text(4.25, 5.3, '3 ops', fontsize=12, ha='center', va='center', color='white')
ax.text(4.25, 4.85, 'Op4,5,8', fontsize=9, ha='center', va='center', color='white')

# Work Center 2 - Center
wc2_box = FancyBboxPatch((6.5, 3.0), 5, 4, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='#6B4C8A', linewidth=2.5, 
                          facecolor=wc2_color, alpha=0.7)
ax.add_patch(wc2_box)
ax.text(9, 6.7, 'Work Center 2 (WC2)', fontsize=14, fontweight='bold', ha='center')

# Machine M2 in WC2 (centered)
m2_box = FancyBboxPatch((6.75, 4.5), 2, 1.8, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='#6B4C8A', linewidth=2, 
                         facecolor=machine_color, alpha=0.8)
ax.add_patch(m2_box)
ax.text(7.75, 5.8, 'M2', fontsize=20, fontweight='bold', ha='center', va='center', color='white')
ax.text(7.75, 5.3, '5 ops', fontsize=12, ha='center', va='center', color='white')
ax.text(7.75, 4.85, 'Op1,2,4,6,9', fontsize=9, ha='center', va='center', color='white')

# Machine M3 in WC2 (centered)
m3_box = FancyBboxPatch((9.25, 4.5), 2, 1.8, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='#6B4C8A', linewidth=2, 
                         facecolor=machine_color, alpha=0.8)
ax.add_patch(m3_box)
ax.text(10.25, 5.8, 'M3', fontsize=20, fontweight='bold', ha='center', va='center', color='white')
ax.text(10.25, 5.3, '4 ops', fontsize=12, ha='center', va='center', color='white')
ax.text(10.25, 4.85, 'Op1,3,7,8', fontsize=9, ha='center', va='center', color='white')

# Work Center 3 - Right
wc3_box = FancyBboxPatch((12.5, 3.0), 5, 4, 
                          boxstyle="round,pad=0.1", 
                          edgecolor='#8A7B4C', linewidth=2.5, 
                          facecolor=wc3_color, alpha=0.7)
ax.add_patch(wc3_box)
ax.text(15, 6.7, 'Work Center 3 (WC3)', fontsize=14, fontweight='bold', ha='center')

# Machine M4 in WC3 (most versatile)
m4_box = FancyBboxPatch((13.5, 4.3), 3, 2, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='#8A7B4C', linewidth=2.5, 
                         facecolor=machine_color, alpha=0.9)
ax.add_patch(m4_box)
# Star for most versatile
ax.text(13.8, 6.0, '★', fontsize=24, ha='center', va='center', color='gold')
ax.text(15, 5.6, 'M4', fontsize=20, fontweight='bold', ha='center', va='center', color='white')
ax.text(15, 5.1, '6 ops (Most Versatile)', fontsize=12, ha='center', va='center', color='white')
ax.text(15, 4.6, 'Op1,3,4,6,8,9', fontsize=9, ha='center', va='center', color='white')

# Operator O1 (bigger, pastel green)
o1_circle = Circle((3, 1.5), 0.35, facecolor=operator_color, edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(o1_circle)
ax.text(3, 1.5, 'O1', fontsize=16, fontweight='bold', ha='center', va='center', color='white')
ax.text(3, 0.9, 'M0,M3,M4', fontsize=10, ha='center', va='top')

# Operator O2 (bigger, pastel green)
o2_circle = Circle((7, 1.5), 0.35, facecolor=operator_color, edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(o2_circle)
ax.text(7, 1.5, 'O2', fontsize=16, fontweight='bold', ha='center', va='center', color='white')
ax.text(7, 0.9, 'M1,M2,M4', fontsize=10, ha='center', va='top')

# Operator O3 (bigger, pastel green)
o3_circle = Circle((11, 1.5), 0.35, facecolor=operator_color, edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(o3_circle)
ax.text(11, 1.5, 'O3', fontsize=16, fontweight='bold', ha='center', va='center', color='white')
ax.text(11, 0.9, 'M0,M3,M4', fontsize=10, ha='center', va='top')

# Operator O4 (bigger, pastel green)
o4_circle = Circle((15, 1.5), 0.35, facecolor=operator_color, edgecolor='black', linewidth=2, alpha=0.9)
ax.add_patch(o4_circle)
ax.text(15, 1.5, 'O4', fontsize=16, fontweight='bold', ha='center', va='center', color='white')
ax.text(15, 0.9, 'M1,M2,M4', fontsize=10, ha='center', va='top')

plt.tight_layout()
plt.savefig('/home/cc253232/MASA-QMIX/docs/shop_floor_layout.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/cc253232/MASA-QMIX/docs/shop_floor_layout.pdf', bbox_inches='tight')
print("Shop floor layout saved as PNG and PDF!")
plt.show()
