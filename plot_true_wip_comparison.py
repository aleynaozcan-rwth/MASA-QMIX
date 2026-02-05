#!/usr/bin/env python3
"""
Gerçek WIP vs Expected WIP karşılaştırması
Expected = 100 - completion_ratio (tüm bitmemişler)
True WIP = Başlamış ama bitmemişler (queue dahil değil)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load data
true_wip_df = pd.read_csv('/tmp/true_wip_ratios.csv')

# Sadece training episodes (0-798)
df = true_wip_df[true_wip_df['episode'] <= 798].copy()

exploitation_start = 479
window = 20

# Moving averages
df['true_job_wip_ma'] = df['job_wip_ratio'].rolling(window=window, center=True, min_periods=1).mean()
df['true_op_wip_ma'] = df['op_wip_ratio'].rolling(window=window, center=True, min_periods=1).mean()

# Before/after stats
df_before = df[df['episode'] <= exploitation_start]
df_after = df[df['episode'] > exploitation_start]

# === TRUE JOB WIP ===
fig, ax = plt.subplots(figsize=(14, 4.5), dpi=150)

ax.plot(df['episode'], df['job_wip_ratio'], alpha=0.4, color='gray', linewidth=0.8, label='Raw True WIP')
ax.plot(df['episode'], df['true_job_wip_ma'], color='#A06CD5', linewidth=2.5, label=f'{window}-Episode Moving Average')

# Zone colors - Yellow-Blue-Lilac pastel palette for True WIP
ax.axhspan(0, 15, color='#FFF9C4', alpha=0.35)      # Pastel Yellow
ax.axhspan(15, 30, color='#BBDEFB', alpha=0.35)     # Pastel Blue
ax.axhspan(30, 100, color='#E1BEE7', alpha=0.35)    # Pastel Lilac

# Reference lines
mean_true_job_wip = df['job_wip_ratio'].mean()
ax.axhline(y=mean_true_job_wip, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=exploitation_start, color='orange', linestyle='--', linewidth=2, alpha=0.8)

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='#A06CD5', linewidth=2.5, label=f'{window}-Episode Moving Average'),
    plt.Line2D([0], [0], color='gray', alpha=0.4, linewidth=0.8, label='Raw True WIP (λ=0.4)'),
    plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'Exploitation Start (ep {exploitation_start})'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_true_job_wip:.1f}%'),
    plt.Rectangle((0,0),1,1, fc='#FFF9C4', alpha=0.5, label='Low WIP (<15%)'),
    plt.Rectangle((0,0),1,1, fc='#BBDEFB', alpha=0.5, label='Medium WIP (15-30%)'),
    plt.Rectangle((0,0),1,1, fc='#E1BEE7', alpha=0.5, label='High WIP (≥30%)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95, ncol=2)

# Text box
before_mean = df_before['job_wip_ratio'].mean()
after_mean = df_after['job_wip_ratio'].mean()
avg_wip_jobs = df['wip_jobs'].mean()
avg_jobs_queued = df['jobs_queued'].mean()

textstr = f'Avg WIP jobs/episode: 6 (started but not completed)\n'
textstr += f'Avg queued jobs/episode: {avg_jobs_queued:.0f} (never started)\n'
textstr += f'Before exploitation (ep≤{exploitation_start}): {before_mean:.1f}%\n'
textstr += f'After exploitation (ep>{exploitation_start}): {after_mean:.1f}%'

ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'), fontsize=10)

ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax.set_ylabel('Job WIP Ratio (%)', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(50, df['job_wip_ratio'].max() * 1.1))
ax.set_xlim(df['episode'].min(), df['episode'].max())
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig('my_data_and_graph/historydata/plots/true_job_wip_ratio.png', dpi=150, bbox_inches='tight')
print(f"✅ TRUE JOB WIP saved")

# === TRUE OPERATION WIP ===
fig, ax = plt.subplots(figsize=(14, 4.5), dpi=150)

ax.plot(df['episode'], df['op_wip_ratio'], alpha=0.4, color='gray', linewidth=0.8, label='Raw True WIP')
ax.plot(df['episode'], df['true_op_wip_ma'], color='#2E86AB', linewidth=2.5, label=f'{window}-Episode Moving Average')

# Zone colors - Yellow-Blue-Lilac pastel palette for True WIP
ax.axhspan(0, 15, color='#FFF9C4', alpha=0.35)      # Pastel Yellow
ax.axhspan(15, 30, color='#BBDEFB', alpha=0.35)     # Pastel Blue
ax.axhspan(30, 100, color='#E1BEE7', alpha=0.35)    # Pastel Lilac

# Reference lines
mean_true_op_wip = df['op_wip_ratio'].mean()
ax.axhline(y=mean_true_op_wip, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=exploitation_start, color='orange', linestyle='--', linewidth=2, alpha=0.8)

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='#2E86AB', linewidth=2.5, label=f'{window}-Episode Moving Average'),
    plt.Line2D([0], [0], color='gray', alpha=0.4, linewidth=0.8, label='Raw True WIP (λ=0.4)'),
    plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'Exploitation Start (ep {exploitation_start})'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_true_op_wip:.1f}%'),
    plt.Rectangle((0,0),1,1, fc='#FFF9C4', alpha=0.5, label='Low WIP (<15%)'),
    plt.Rectangle((0,0),1,1, fc='#BBDEFB', alpha=0.5, label='Medium WIP (15-30%)'),
    plt.Rectangle((0,0),1,1, fc='#E1BEE7', alpha=0.5, label='High WIP (≥30%)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95, ncol=2)

# Text box
before_mean_op = df_before['op_wip_ratio'].mean()
after_mean_op = df_after['op_wip_ratio'].mean()
avg_wip_ops = df['wip_operations'].mean()

textstr = f'Avg WIP operations/episode: {avg_wip_ops:.0f} (started but not completed)\n'
textstr += f'Before exploitation (ep≤{exploitation_start}): {before_mean_op:.1f}%\n'
textstr += f'After exploitation (ep>{exploitation_start}): {after_mean_op:.1f}%'

ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'), fontsize=10)

ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax.set_ylabel('Operation WIP Ratio (%)', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(50, df['op_wip_ratio'].max() * 1.1))
ax.set_xlim(df['episode'].min(), df['episode'].max())
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig('my_data_and_graph/historydata/plots/true_operation_wip_ratio.png', dpi=150, bbox_inches='tight')
print(f"✅ TRUE OPERATION WIP saved")

# === COMPARISON STATISTICS ===
print(f"\n📊 GERÇEK WIP ANALİZİ (gelen işlere göre):")
print(f"\n{'='*60}")
print(f"JOB WIP:")
print(f"  True WIP (started but not completed): {df['job_wip_ratio'].mean():.2f}%")
print(f"  Completion ratio:                     {df['job_completion_ratio'].mean():.2f}%")
print(f"  Queue ratio (never started):          {df['job_queued_ratio'].mean():.2f}%")
print(f"  Toplam: {df['job_wip_ratio'].mean() + df['job_completion_ratio'].mean() + df['job_queued_ratio'].mean():.2f}%")

print(f"\n{'='*60}")
print(f"OPERATION WIP:")
print(f"  True WIP (started but not completed): {df['op_wip_ratio'].mean():.2f}%")
print(f"  Completion ratio:                     {df['op_completion_ratio'].mean():.2f}%")

print(f"\n{'='*60}")
print(f"BEFORE EXPLOITATION (ep ≤ {exploitation_start}):")
print(f"  True Job WIP:     {df_before['job_wip_ratio'].mean():.2f}%")
print(f"  True Op WIP:      {df_before['op_wip_ratio'].mean():.2f}%")

print(f"\n{'='*60}")
print(f"AFTER EXPLOITATION (ep > {exploitation_start}):")
print(f"  True Job WIP:     {df_after['job_wip_ratio'].mean():.2f}%")
print(f"  True Op WIP:      {df_after['op_wip_ratio'].mean():.2f}%")

print(f"\n{'='*60}")
print(f"WIP CHANGE:")
print(f"  Job WIP:          {df_before['job_wip_ratio'].mean():.2f}% → {df_after['job_wip_ratio'].mean():.2f}% ({df_after['job_wip_ratio'].mean() - df_before['job_wip_ratio'].mean():+.2f}%)")
print(f"  Operation WIP:    {df_before['op_wip_ratio'].mean():.2f}% → {df_after['op_wip_ratio'].mean():.2f}% ({df_after['op_wip_ratio'].mean() - df_before['op_wip_ratio'].mean():+.2f}%)")

print(f"\n✅ Grafikler kaydedildi!")
print(f"   - true_job_wip_ratio.png")
print(f"   - true_operation_wip_ratio.png")
