#!/usr/bin/env python3
"""
Filter episode_metrics.csv to remove evaluation episodes and create 
separate training-only plots without overwriting existing plots.

Output files:
- episode_metrics_training_only.csv
- plots_training_only/reward_trend.png (and other plots)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def filter_episode_metrics(csv_path, n_episodes_per_epoch=4, evaluate_cycle=2, evaluate_epoch=5):
    """Remove evaluation episodes and renumber."""
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"\n📊 ORIGINAL DATA:")
    print(f"  Total rows: {len(df)}")
    print(f"  Episode range: {df['episode'].min()} - {df['episode'].max()}")
    
    # Calculate which episodes should be training
    training_episodes = []
    ep_idx = 0
    epoch = 0
    max_episodes = int(df['episode'].max()) + 1
    
    while ep_idx < max_episodes:
        for _ in range(n_episodes_per_epoch):
            training_episodes.append(ep_idx)
            ep_idx += 1
            if ep_idx >= max_episodes:
                break
        epoch += 1
        if epoch % evaluate_cycle == 0 and epoch != 0:
            ep_idx += evaluate_epoch
    
    # Filter and renumber
    df_training = df[df['episode'].isin(training_episodes)].copy()
    df_training['original_episode'] = df_training['episode']
    df_training['episode'] = range(1, len(df_training) + 1)
    
    # Reorder columns
    cols = [c for c in df_training.columns if c != 'original_episode']
    cols.append('original_episode')
    df_training = df_training[cols]
    
    print(f"\n✅ FILTERED DATA:")
    print(f"  Training episodes: {len(df_training)}")
    print(f"  Removed: {len(df) - len(df_training)} evaluation episodes")
    print(f"  New episode range: 1-{len(df_training)}")
    
    return df_training

def create_training_plots(df, output_dir):
    """Create training-only plots in separate directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    episodes = df['episode'].values
    rewards = df['episode_reward'].values
    
    # 1. Reward trend plot - EXACT SAME STYLE as plot_reward_trend()
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, color='blue', linewidth=1, alpha=0.5, label='Episode Reward')
    
    # Add epsilon min marker if epsilon data exists and compute statistics
    exploitation_start_ep = None
    if 'epsilon' in df.columns:
        # Find where epsilon reached minimum (0.1)
        epsilon_min = df['epsilon'].min()
        eps_min_idx = df[df['epsilon'] <= epsilon_min * 1.01].index.min()  # First time epsilon ~= min
        if pd.notna(eps_min_idx):
            exploitation_start_ep = df.loc[eps_min_idx, 'episode']
            plt.axvline(x=exploitation_start_ep, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Exploitation start (ep {exploitation_start_ep})')
    
    # Moving average - same window calculation as original
    if len(rewards) > 10:
        window = min(20, len(rewards) // 5)
        reward_series = pd.Series(rewards)
        reward_rolling = reward_series.rolling(window=window, center=True).mean()
        plt.plot(episodes, reward_rolling, color='red', linewidth=2.5, label=f'{window}-episode MA')
    
    # Mean line - dark blue instead of green
    mean_reward = np.mean(rewards)
    plt.axhline(y=mean_reward, color='darkblue', linestyle='--', alpha=0.5, label=f'Mean: {mean_reward:.3f}')
    
    # Compute before/after exploitation statistics
    if exploitation_start_ep is not None:
        before_mask = episodes < exploitation_start_ep
        after_mask = episodes >= exploitation_start_ep
        
        before_rewards = rewards[before_mask]
        after_rewards = rewards[after_mask]
        
        if len(before_rewards) > 0 and len(after_rewards) > 0:
            before_mean = np.mean(before_rewards)
            after_mean = np.mean(after_rewards)
            before_std = np.std(before_rewards)
            after_std = np.std(after_rewards)
            
            improvement = ((after_mean - before_mean) / before_mean) * 100
            variance_reduction = ((before_std - after_std) / before_std) * 100
            
            # Add text box with statistics
            stats_text = (
                f'Before Exploitation (ep 1-{exploitation_start_ep-1}):\n'
                f'  Mean: {before_mean:.2f}, Std: {before_std:.2f}\n'
                f'After Exploitation (ep {exploitation_start_ep}+):\n'
                f'  Mean: {after_mean:.2f}, Std: {after_std:.2f}\n'
                f'Improvement: {improvement:+.1f}%\n'
                f'Variance reduction: {variance_reduction:+.1f}%'
            )
            
            plt.text(0.98, 0.02, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                    fontsize=9,
                    family='monospace')
    
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Episode Reward', fontsize=12, fontweight='bold')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'reward_trend_training_only.png'), dpi=100)
    plt.close()
    print(f"  ✓ Created reward_trend_training_only.png")
    
    # 2. Boxplot: Reward distribution by quarter
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Divide episodes into 4 quarters
    n_episodes = len(episodes)
    quarter_size = n_episodes // 4
    
    quarters_data = []
    quarter_labels = []
    # Colors matching the reference image
    quarter_colors = ['#ADD8E6', '#90EE90', '#FFFACD', '#F08080']  # lightblue, lightgreen, lemonchiffon, lightcoral
    
    for i in range(4):
        start_idx = i * quarter_size
        if i == 3:  # Last quarter takes remaining episodes
            end_idx = n_episodes
        else:
            end_idx = (i + 1) * quarter_size
        
        quarter_rewards = rewards[start_idx:end_idx]
        quarters_data.append(quarter_rewards)
        
        start_ep = episodes[start_idx]
        end_ep = episodes[end_idx - 1]
        quarter_labels.append(f'Q{i+1}\n({start_ep}-{end_ep})')
    
    # Create boxplot
    bp = ax.boxplot(quarters_data, labels=quarter_labels, patch_artist=True,
                    showmeans=False, showfliers=True,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=3, color='darkorange'))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], quarter_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Quarter (Episode Range)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_distribution_by_quarter.png'), dpi=100)
    plt.close()
    print(f"  ✓ Created reward_distribution_by_quarter.png")
    
    # 2. Epoch-based view
    if 'epoch' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Average reward per epoch
        epoch_rewards = df.groupby('epoch')['episode_reward'].mean()
        epochs = epoch_rewards.index.values
        avg_rewards = epoch_rewards.values
        
        ax.plot(epochs, avg_rewards, marker='o', markersize=3, linewidth=1.5, 
                color='darkblue', label='Average Reward per Epoch')
        
        # Moving average
        if len(avg_rewards) > 10:
            window = min(10, len(avg_rewards) // 5)
            ma = pd.Series(avg_rewards).rolling(window=window, center=True).mean()
            ax.plot(epochs, ma, color='red', linewidth=2.5, linestyle='--', 
                   label=f'{window}-epoch MA')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Episode Reward', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'reward_by_epoch_training_only.png'), dpi=150)
        plt.close()
        print(f"  ✓ Created reward_by_epoch_training_only.png")

def main():
    base_dir = "./my_data_and_graph/historydata"
    csv_file = os.path.join(base_dir, "episode_metrics.csv")
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    # Filter training episodes
    df_training = filter_episode_metrics(
        csv_file,
        n_episodes_per_epoch=4,
        evaluate_cycle=2,
        evaluate_epoch=5
    )
    
    if df_training is None:
        return
    
    # Save filtered CSV (separate file, don't overwrite original)
    output_csv = os.path.join(base_dir, "episode_metrics_training_only.csv")
    df_training.to_csv(output_csv, index=False)
    print(f"\n💾 SAVED:")
    print(f"  Training-only CSV: {output_csv}")
    
    # Create plots in separate directory
    plots_dir = os.path.join(base_dir, "plots_training_only")
    print(f"\n🎨 CREATING PLOTS:")
    create_training_plots(df_training, plots_dir)
    print(f"\n✅ All training-only plots saved to: {plots_dir}/")
    print(f"   Original plots remain unchanged in: {base_dir}/plots/")
    
    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Original episode_metrics.csv: UNCHANGED")
    print(f"New training-only CSV: episode_metrics_training_only.csv")
    print(f"Original plots: plots/ (UNCHANGED)")
    print(f"New training-only plots: plots_training_only/")
    print("="*60)

if __name__ == "__main__":
    main()
