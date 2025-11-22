#!/usr/bin/env python3
"""
Generate loss/TD/reward PNG plots from current training data
without waiting for training to finish.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

history_dir = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata'

print(f"[PlotGenerator] Reading data from {history_dir}")

# Loss plot
try:
    loss_vals = np.loadtxt(os.path.join(history_dir, 'loss.txt'))
    plt.figure()
    plt.plot(loss_vals, label='Loss', color='tab:blue')
    plt.xlabel('Train Step')
    plt.ylabel('Loss')
    plt.title('Training Loss (Current)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(history_dir, 'loss.png'), dpi=150)
    plt.close()
    print(f"✅ loss.png saved ({len(loss_vals)} points)")
except Exception as e:
    print(f"❌ Could not create loss.png: {e}")

# TD error plot
try:
    td_vals = np.loadtxt(os.path.join(history_dir, 'td_error.txt'))
    plt.figure()
    plt.plot(td_vals, label='TD Error', color='tab:orange')
    plt.xlabel('Train Step')
    plt.ylabel('TD Error')
    plt.title('TD Error (Current)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(history_dir, 'td_error.png'), dpi=150)
    plt.close()
    print(f"✅ td_error.png saved ({len(td_vals)} points)")
except Exception as e:
    print(f"❌ Could not create td_error.png: {e}")

# Episode reward plot from learning_metrics.csv
try:
    lm_path = os.path.join(history_dir, 'learning_metrics.csv')
    episodes = []
    rewards = []
    if os.path.exists(lm_path):
        with open(lm_path, 'r') as lf:
            reader = csv.DictReader(lf)
            for row in reader:
                try:
                    episodes.append(int(row.get('episode', len(episodes))))
                    rewards.append(float(row.get('episode_reward', 0.0)))
                except:
                    continue
    if rewards:
        plt.figure()
        plt.plot(rewards, label='Episode Reward', color='tab:green')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Reward over Time (Current)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(history_dir, 'episode_rewards.png'), dpi=150)
        plt.close()
        print(f"✅ episode_rewards.png saved ({len(rewards)} episodes)")
    else:
        print("⚠️  No reward data found in learning_metrics.csv")
except Exception as e:
    print(f"❌ Could not create episode_rewards.png: {e}")

print(f"\n[PlotGenerator] Done! Check {history_dir} for PNG files.")
