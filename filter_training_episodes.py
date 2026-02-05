#!/usr/bin/env python3
"""
Filter training episodes from learning metrics by removing evaluation episodes.
Expected pattern:
- n_epoch = 200 (or actual number from run)
- n_episodes = 4 per epoch
- evaluate_cycle = 2 (every 2 epochs)
- evaluate_epoch = 5 (5 episodes per evaluation)

Training episodes: 0-3, 4-7, 8-11, ... (sequential in groups of 4)
Evaluation episodes: inserted after every 2 epochs
"""

import os
import pandas as pd
import numpy as np

def filter_training_episodes(csv_path, n_episodes_per_epoch=4, evaluate_cycle=2, evaluate_epoch=5):
    """
    Filter learning_metrics.csv to keep only training episodes.
    
    Args:
        csv_path: Path to learning_metrics.csv
        n_episodes_per_epoch: Training episodes per epoch (default 4)
        evaluate_cycle: Evaluation frequency in epochs (default 2)
        evaluate_epoch: Evaluation episodes per eval cycle (default 5)
    
    Returns:
        DataFrame with only training episodes
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"\nOriginal data: {len(df)} episodes")
    print(f"Episode range: {df['episode'].min()} - {df['episode'].max()}")
    
    # Calculate which episodes should be training episodes
    training_episodes = []
    ep_idx = 0
    epoch = 0
    
    # Simulate the episode generation pattern
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
    
    # Renumber episodes to be sequential (0, 1, 2, ...)
    df_training['original_episode'] = df_training['episode']
    df_training['episode'] = range(len(df_training))
    
    print(f"\nFiltered data: {len(df_training)} training episodes")
    print(f"Removed: {len(df) - len(df_training)} evaluation episodes")
    
    # Show some statistics
    if len(df_training) > 0:
        print(f"\nTraining episode stats:")
        print(f"  Original episode numbers: {df_training['original_episode'].min()}-{df_training['original_episode'].max()}")
        print(f"  Renumbered: 0-{len(df_training)-1}")
        print(f"  Average reward: {df_training['episode_reward'].mean():.2f}")
    
    return df_training

def filter_from_rewards_txt(rewards_file, n_episodes_per_epoch=4, evaluate_cycle=2, evaluate_epoch=5):
    """
    Filter episode_rewards.txt to keep only training episodes.
    """
    if not os.path.exists(rewards_file):
        print(f"Error: {rewards_file} not found")
        return None
    
    # Read the rewards file
    episodes = []
    rewards = []
    with open(rewards_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            try:
                ep = int(parts[0])
                reward = float(parts[1])
                episodes.append(ep)
                rewards.append(reward)
            except:
                continue
    
    print(f"\nOriginal data: {len(episodes)} episodes")
    print(f"Episode range: {min(episodes)} - {max(episodes)}")
    
    # Calculate which episodes should be training episodes
    training_episodes = []
    ep_idx = 0
    epoch = 0
    
    max_episodes = max(episodes) + 1
    
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
    
    # Filter episodes
    training_data = [(ep, rew) for ep, rew in zip(episodes, rewards) if ep in training_episodes]
    
    print(f"\nFiltered data: {len(training_data)} training episodes")
    print(f"Removed: {len(episodes) - len(training_data)} evaluation episodes")
    
    if len(training_data) > 0:
        rewards_only = [r for _, r in training_data]
        print(f"\nTraining episode stats:")
        print(f"  Original episode numbers: {training_data[0][0]}-{training_data[-1][0]}")
        print(f"  Count: {len(training_data)}")
        print(f"  Average reward: {np.mean(rewards_only):.2f}")
    
    return training_data

def main():
    # Check for episode_rewards.txt in standard location
    base_dir = "./my_data_and_graph/historydata"
    rewards_file = os.path.join(base_dir, "episode_rewards.txt")
    
    if not os.path.exists(rewards_file):
        print(f"Looking for episode_rewards.txt in {base_dir}...")
        print("File not found. Please specify the correct path.")
        return
    
    # Filter training episodes
    training_data = filter_from_rewards_txt(
        rewards_file,
        n_episodes_per_epoch=4,
        evaluate_cycle=2,
        evaluate_epoch=5
    )
    
    if training_data is not None:
        # Save filtered data
        output_file = os.path.join(base_dir, "episode_rewards_training_only.txt")
        with open(output_file, 'w') as f:
            for i, (orig_ep, reward) in enumerate(training_data):
                f.write(f"{i},{reward}\n")  # Renumber to 0, 1, 2, ...
        print(f"\nSaved training-only rewards to: {output_file}")
        
        # Also save with original episode numbers
        output_file_orig = os.path.join(base_dir, "episode_rewards_training_original_idx.txt")
        with open(output_file_orig, 'w') as f:
            for orig_ep, reward in training_data:
                f.write(f"{orig_ep},{reward}\n")
        print(f"Saved with original indices to: {output_file_orig}")
        
        # Show which episodes were removed (evaluation)
        all_episodes = set(range(max([ep for ep, _ in training_data]) + 1))
        training_ep_set = set([ep for ep, _ in training_data])
        eval_episodes = sorted(all_episodes - training_ep_set)
        
        if eval_episodes:
            print(f"\nEvaluation episodes removed: {len(eval_episodes)}")
            print(f"Sample: {eval_episodes[:20]}{'...' if len(eval_episodes) > 20 else ''}")


if __name__ == "__main__":
    main()
