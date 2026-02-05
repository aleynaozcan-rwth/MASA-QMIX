#!/usr/bin/env python3
"""
Fix episode_metrics.csv by removing evaluation episodes and renumbering.
This will filter out evaluation episodes based on the known pattern.
"""

import pandas as pd
import os

def fix_episode_metrics(csv_path, n_episodes_per_epoch=4, evaluate_cycle=2, evaluate_epoch=5):
    """
    Remove evaluation episodes from episode_metrics.csv and renumber.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"\n=== ORIGINAL DATA ===")
    print(f"Total rows: {len(df)}")
    print(f"Episode range: {df['episode'].min()} - {df['episode'].max()}")
    print(f"Epoch range: {df['epoch'].min()} - {df['epoch'].max()}")
    
    # Calculate which episodes should be training episodes
    training_episodes = []
    ep_idx = 0
    epoch = 0
    
    max_episodes = int(df['episode'].max()) + 1
    
    while ep_idx < max_episodes:
        # Training episodes for this epoch
        for _ in range(n_episodes_per_epoch):
            training_episodes.append(ep_idx)
            ep_idx += 1
            if ep_idx >= max_episodes:
                break
        
        epoch += 1
        
        # Check if we need evaluation
        if epoch % evaluate_cycle == 0 and epoch != 0:
            # Skip evaluation episodes
            ep_idx += evaluate_epoch
    
    # Filter dataframe
    df_training = df[df['episode'].isin(training_episodes)].copy()
    
    print(f"\n=== FILTERED DATA ===")
    print(f"Training episodes: {len(df_training)}")
    print(f"Evaluation episodes removed: {len(df) - len(df_training)}")
    
    # Renumber episodes to be sequential (1, 2, 3, ...)
    # Keep episode 1-based numbering like original
    df_training['original_episode'] = df_training['episode']
    df_training['episode'] = range(1, len(df_training) + 1)
    
    # Reorder columns to put original_episode at the end for reference
    cols = [c for c in df_training.columns if c != 'original_episode']
    cols.append('original_episode')
    df_training = df_training[cols]
    
    print(f"\n=== STATISTICS ===")
    print(f"Episodes: 1-{len(df_training)}")
    print(f"Epochs: {df_training['epoch'].min()}-{df_training['epoch'].max()}")
    print(f"Average reward: {df_training['episode_reward'].mean():.2f}")
    print(f"Std reward: {df_training['episode_reward'].std():.2f}")
    
    return df_training

def main():
    # Fix episode_metrics.csv
    csv_file = "./my_data_and_graph/historydata/episode_metrics.csv"
    
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return
    
    # Filter training episodes
    df_fixed = fix_episode_metrics(
        csv_file,
        n_episodes_per_epoch=4,
        evaluate_cycle=2,
        evaluate_epoch=5
    )
    
    if df_fixed is not None:
        # Save backup
        backup_file = csv_file.replace('.csv', '_with_eval.csv')
        os.rename(csv_file, backup_file)
        print(f"\n=== BACKUP ===")
        print(f"Original saved to: {backup_file}")
        
        # Save fixed version
        df_fixed.to_csv(csv_file, index=False)
        print(f"\n=== SAVED ===")
        print(f"Fixed version saved to: {csv_file}")
        
        # Show sample
        print(f"\n=== SAMPLE (first 10 rows) ===")
        print(df_fixed.head(10)[['episode', 'epoch', 'episode_reward', 'original_episode']])
        
        print(f"\n=== SAMPLE (last 10 rows) ===")
        print(df_fixed.tail(10)[['episode', 'epoch', 'episode_reward', 'original_episode']])
        
        # Show removed evaluation episodes
        df_orig = pd.read_csv(backup_file)
        eval_episodes = df_orig[~df_orig['episode'].isin(df_fixed['original_episode'].values)]
        if len(eval_episodes) > 0:
            print(f"\n=== REMOVED EVALUATION EPISODES (sample) ===")
            print(eval_episodes.head(10)[['episode', 'epoch', 'episode_reward']])

if __name__ == "__main__":
    main()
