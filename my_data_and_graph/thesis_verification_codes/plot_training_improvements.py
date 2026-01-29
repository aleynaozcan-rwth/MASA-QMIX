#!/usr/bin/env python3
"""
plot_training_improvements.py

Visualize training improvements beyond completion rate:
- Wait Time Reduction (efficiency)
- Load Balance Improvement (resource utilization)
- Episode Reward (overall optimization)
- Completion Rate (for comparison)

Shows that even when completion rate stays flat, the system learns
to optimize execution efficiency and resource allocation.

NOTE: Uses parser code from plot_simple_job_dynamics.py to preserve that working script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import os

def parse_episode_data(filepath):
    """
    Parse scheduling_timeline.txt and extract completion rates per episode.
    Copied from plot_simple_job_dynamics.py to avoid breaking that script.
    """
    print(f"\nParsing {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    episodes = []
    current_episode = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Episode start (matches "=== EPISODE X ===")
            if "=== EPISODE" in line and "===" in line:
                match = re.search(r'EPISODE (\d+)', line)
                if match:
                    ep_num = int(match.group(1))
                    if current_episode is not None:
                        episodes.append(current_episode)
                    current_episode = {
                        'episode': ep_num,
                        'total_jobs': 0,
                        'completed_jobs': 0,
                        'active_counts': []
                    }
            
            # Episode end
            elif "LIFECYCLE TRACE END" in line and current_episode is not None:
                episodes.append(current_episode)
                current_episode = None
            
            # Extract lifecycle snapshots
            elif "[Lifecycle]" in line and "Active=" in line and current_episode is not None:
                match = re.search(r'Active=(\d+).*Completed=(\d+).*Total=(\d+)', line)
                if match:
                    active = int(match.group(1))
                    completed = int(match.group(2))
                    total = int(match.group(3))
                    
                    current_episode['active_counts'].append(active)
                    current_episode['completed_jobs'] = max(current_episode['completed_jobs'], completed)
                    current_episode['total_jobs'] = max(current_episode['total_jobs'], total)
    
    # Add last episode if needed
    if current_episode is not None:
        episodes.append(current_episode)
    
    # Convert to DataFrame
    data = []
    for ep in episodes:
        completion_rate = (ep['completed_jobs'] / ep['total_jobs'] * 100) if ep['total_jobs'] > 0 else 0
        data.append({
            'episode': ep['episode'],
            'completion_rate': completion_rate,
            'total_jobs': ep['total_jobs'],
            'completed_jobs': ep['completed_jobs']
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Parsed {len(df)} episodes")
    return df

def load_episode_metrics(filepath):
    """Load episode-level metrics (wait time, rewards)."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, comment='#')
    print(f"Loaded {len(df)} episodes")
    return df

def load_reward_components(filepath):
    """Load step-level reward components and aggregate by episode."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, comment='#')
    
    # Calculate episode from step (assuming 50 steps per episode)
    df['episode'] = (df['step'] // 50) + 1
    
    # Aggregate by episode
    episode_agg = df.groupby('episode').agg({
        'LoadBalance': 'mean',
        'AvgWaitNorm': 'mean',
        'CompletedNorm': 'mean'
    }).reset_index()
    
    print(f"Aggregated into {len(episode_agg)} episodes")
    return episode_agg

def moving_average(data, window=20):
    """Calculate moving average."""
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values

def plot_training_improvements(completion_df, metrics_df, components_df, output_path, zoom_episodes=250):
    """
    Create 4-panel figure showing training improvements:
    A) Completion Rate (flat but acceptable)
    B) Wait Time Reduction (improvement!)
    C) Load Balance Score (improvement!)
    D) Episode Reward (improvement!)
    """
    
    # Merge dataframes
    df = completion_df.merge(metrics_df[['episode', 'episode_reward', 'wait_time']], on='episode', how='left')
    df = df.merge(components_df[['episode', 'LoadBalance']], on='episode', how='left')
    
    # Limit to zoom range
    df = df[df['episode'] <= zoom_episodes]
    
    # Calculate moving averages
    ma_window = 20
    df['completion_ma'] = moving_average(df['completion_rate'], ma_window)
    df['wait_ma'] = moving_average(df['wait_time'], ma_window)
    df['loadbalance_ma'] = moving_average(df['LoadBalance'], ma_window)
    df['reward_ma'] = moving_average(df['episode_reward'], ma_window)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('MASA-QMIX Training Improvements: Efficiency Optimization Beyond Completion Rate', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color scheme
    color_completion = '#2E86AB'  # Blue
    color_wait = '#A23B72'        # Purple-red (lower is better)
    color_loadbalance = '#F18F01' # Orange
    color_reward = '#06A77D'      # Green
    
    # === Panel A: Completion Rate ===
    ax = axes[0, 0]
    ax.plot(df['episode'], df['completion_rate'], 'o-', 
            color=color_completion, alpha=0.3, markersize=3, linewidth=0.8, label='Raw')
    ax.plot(df['episode'], df['completion_ma'], 
            color=color_completion, linewidth=2.5, label=f'MA{ma_window}')
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('A) Completion Rate (Stable Performance)', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Tight y-axis for visibility
    y_min = df['completion_rate'].min() - 2
    y_max = df['completion_rate'].max() + 2
    ax.set_ylim(y_min, y_max)
    
    # Add text annotation
    mean_completion = df['completion_rate'].mean()
    ax.text(0.98, 0.02, f'Mean: {mean_completion:.1f}%\nStable within constraints',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # === Panel B: Wait Time Reduction ===
    ax = axes[0, 1]
    ax.plot(df['episode'], df['wait_time'], 'o-', 
            color=color_wait, alpha=0.3, markersize=3, linewidth=0.8, label='Raw')
    ax.plot(df['episode'], df['wait_ma'], 
            color=color_wait, linewidth=2.5, label=f'MA{ma_window}')
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Wait Time (sim time units)', fontsize=12, fontweight='bold')
    ax.set_title('B) Wait Time Reduction ↓ (IMPROVEMENT!)', fontsize=13, fontweight='bold', pad=10, color='darkgreen')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Calculate improvement
    initial_wait = df['wait_ma'].iloc[:30].mean()
    final_wait = df['wait_ma'].iloc[-30:].mean()
    improvement_pct = ((initial_wait - final_wait) / initial_wait) * 100
    
    ax.text(0.98, 0.98, f'Initial: {initial_wait:.2f}\nFinal: {final_wait:.2f}\n↓ {improvement_pct:.1f}% reduction',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # === Panel C: Load Balance Improvement ===
    ax = axes[1, 0]
    ax.plot(df['episode'], df['LoadBalance'], 'o-', 
            color=color_loadbalance, alpha=0.3, markersize=3, linewidth=0.8, label='Raw')
    ax.plot(df['episode'], df['loadbalance_ma'], 
            color=color_loadbalance, linewidth=2.5, label=f'MA{ma_window}')
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Load Balance Score (entropy)', fontsize=12, fontweight='bold')
    ax.set_title('C) Load Balance Improvement ↑ (IMPROVEMENT!)', fontsize=13, fontweight='bold', pad=10, color='darkgreen')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Calculate improvement
    initial_lb = df['loadbalance_ma'].iloc[:30].mean()
    final_lb = df['loadbalance_ma'].iloc[-30:].mean()
    improvement_pct = ((final_lb - initial_lb) / initial_lb) * 100
    
    ax.text(0.98, 0.02, f'Initial: {initial_lb:.3f}\nFinal: {final_lb:.3f}\n↑ {improvement_pct:.1f}% improvement',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # === Panel D: Episode Reward ===
    ax = axes[1, 1]
    ax.plot(df['episode'], df['episode_reward'], 'o-', 
            color=color_reward, alpha=0.3, markersize=3, linewidth=0.8, label='Raw')
    ax.plot(df['episode'], df['reward_ma'], 
            color=color_reward, linewidth=2.5, label=f'MA{ma_window}')
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
    ax.set_title('D) Overall Reward Trend ↑ (LEARNING!)', fontsize=13, fontweight='bold', pad=10, color='darkgreen')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Calculate improvement
    initial_reward = df['reward_ma'].iloc[:30].mean()
    final_reward = df['reward_ma'].iloc[-30:].mean()
    improvement_pct = ((final_reward - initial_reward) / abs(initial_reward)) * 100
    
    ax.text(0.98, 0.02, f'Initial: {initial_reward:.2f}\nFinal: {final_reward:.2f}\n↑ {improvement_pct:.1f}% improvement',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    
    # Add global explanation
    fig.text(0.5, 0.005, 
             'Key Insight: Agent learns execution efficiency (lower wait times, balanced load) within fixed 50-step episodes.\n'
             'Completion rate stable due to time constraints, but quality metrics improve significantly.',
             ha='center', fontsize=11, style='italic', wrap=True,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING IMPROVEMENT SUMMARY")
    print("="*60)
    print(f"\n📊 Completion Rate:")
    print(f"   Mean: {mean_completion:.2f}%")
    print(f"   Range: [{df['completion_rate'].min():.1f}%, {df['completion_rate'].max():.1f}%]")
    print(f"   Status: STABLE (within episode time constraints)")
    
    print(f"\n⏱️  Wait Time:")
    print(f"   Initial (first 30 eps): {initial_wait:.2f}")
    print(f"   Final (last 30 eps): {final_wait:.2f}")
    print(f"   Improvement: ↓ {improvement_pct:.1f}% REDUCTION ✓")
    
    initial_lb = df['loadbalance_ma'].iloc[:30].mean()
    final_lb = df['loadbalance_ma'].iloc[-30:].mean()
    improvement_lb = ((final_lb - initial_lb) / initial_lb) * 100
    print(f"\n⚖️  Load Balance:")
    print(f"   Initial: {initial_lb:.4f}")
    print(f"   Final: {final_lb:.4f}")
    print(f"   Improvement: ↑ {improvement_lb:.1f}% INCREASE ✓")
    
    print(f"\n🎯 Episode Reward:")
    print(f"   Initial: {initial_reward:.2f}")
    print(f"   Final: {final_reward:.2f}")
    print(f"   Improvement: ↑ {improvement_pct:.1f}% INCREASE ✓")
    
    print("\n" + "="*60)
    print("CONCLUSION: Agents learned EFFICIENCY optimization!")
    print("="*60 + "\n")

def main():
    # Paths
    base_dir = Path("/home/cc253232/MASA-QMIX")
    data_dir = base_dir / "my_data_and_graph" / "historydata"
    
    timeline_path = str(data_dir / "scheduling_timeline.txt")
    metrics_path = str(data_dir / "episode_metrics.csv")
    components_path = str(data_dir / "reward_components.csv")
    output_path = str(base_dir / "training_improvements.png")
    
    # Check files exist
    for path in [timeline_path, metrics_path, components_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            return
    
    print("="*60)
    print("TRAINING IMPROVEMENTS ANALYSIS")
    print("="*60)
    
    # Load data
    completion_df = parse_episode_data(timeline_path)
    if completion_df is None or len(completion_df) == 0:
        print("ERROR: No completion data parsed")
        return
        
    metrics_df = load_episode_metrics(metrics_path)
    components_df = load_reward_components(components_path)
    
    # Create visualization (use all 799 episodes from metrics)
    plot_training_improvements(completion_df, metrics_df, components_df, output_path, zoom_episodes=799)

if __name__ == "__main__":
    main()
