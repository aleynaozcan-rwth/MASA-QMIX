"""Standalone plot for Reward Progression with moving average."""

import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def get_epsilon_min_step(history_dir: str = 'my_data_and_graph/historydata'):
    """Detect when epsilon reaches minimum (exploitation phase starts).
    Returns exploitation_start_epoch from episode_metrics.csv."""
    try:
        csv_path = os.path.join(history_dir, 'episode_metrics.csv')
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        if 'epsilon' not in df.columns or 'episode' not in df.columns or 'epoch' not in df.columns:
            return None
        
        epsilon_min = df['epsilon'].min()
        eps_min_idx = df[df['epsilon'] <= epsilon_min * 1.01].index.min()
        if pd.isna(eps_min_idx):
            return None
        
        exploitation_start_episode = df.loc[eps_min_idx, 'episode']
        exploitation_start_epoch = df.loc[eps_min_idx, 'epoch']
        return {
            'epsilon_min': epsilon_min,
            'epsilon_min_step': exploitation_start_episode,
            'exploitation_start_epoch': exploitation_start_epoch
        }
    except Exception as e:
        warnings.warn(f'get_epsilon_min_step error: {e}')
        return None


def plot_reward_progression(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot reward progression with moving average.
    
    Shows:
    - Raw reward per epoch (light blue, transparent)
    - 10-epoch moving average (dark blue, solid)
    - Mean reward line (red dashed)
    - ±1 std deviation band (gray)
    - Exploitation start marker (orange dashed)
    """
    try:
        csv_path = os.path.join(history_dir, 'episode_metrics.csv')
        if not os.path.exists(csv_path):
            warnings.warn(f'[plot_reward_progression] Missing {csv_path}')
            return
        
        df = pd.read_csv(csv_path)
        if 'epoch' not in df.columns or 'episode_reward' not in df.columns:
            warnings.warn('[plot_reward_progression] Missing required columns')
            return
        
        epoch_rewards = df.groupby('epoch')['episode_reward'].mean()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw rewards (transparent)
        ax.plot(epoch_rewards.index, epoch_rewards.values, alpha=0.3, color='blue', label='Reward')
        
        # Plot moving average
        window = 10
        rolling = epoch_rewards.rolling(window=window).mean()
        ax.plot(rolling.index, rolling.values, color='darkblue', linewidth=2, label=f'{window}-epoch MA')
        
        # Mean line
        mean_reward = epoch_rewards.mean()
        ax.axhline(y=mean_reward, color='red', linestyle='--', alpha=0.5, 
                   label=f'Mean: {mean_reward:.2f}')
        
        # ±1 std deviation band
        std_reward = epoch_rewards.std()
        ax.fill_between(epoch_rewards.index, 
                        mean_reward - std_reward,
                        mean_reward + std_reward,
                        alpha=0.2, color='gray', label=f'±1 Std ({std_reward:.2f})')
        
        # Add exploitation start marker
        try:
            eps_info = get_epsilon_min_step(history_dir)
            if eps_info:
                exploitation_start_epoch = eps_info['exploitation_start_epoch']
                ax.axvline(x=exploitation_start_epoch, color='green', linestyle='--', 
                          linewidth=2.5, alpha=1.0, 
                          label=f'Exploitation start (epoch {exploitation_start_epoch})')
        except:
            pass
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Episode Reward', fontweight='bold')
        ax.set_xlim(epoch_rewards.index.min(), epoch_rewards.index.max())  # Tight x-axis
        ax.set_title('Reward Progression', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save
        plots_dir = os.path.join(history_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, 'reward_progression.png')
        plt.tight_layout(pad=0.3)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f'[plot_reward_progression] Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_reward_progression] Error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    plot_reward_progression()
