"""Standalone plot for Reward Stability: Rolling Std with Mean-colored scatter."""

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


def plot_stability_std_mean(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot Rolling Std with Mean-colored scatter points.
    
    Shows:
    - Rolling Std trend line (purple)
    - Scatter points colored by Rolling Mean (colorbar)
    - CV-based stability thresholds
    - Exploitation phase marker
    """
    try:
        csv_path = os.path.join(history_dir, 'episode_metrics.csv')
        if not os.path.exists(csv_path):
            warnings.warn(f'[plot_stability_std_mean] Missing {csv_path}')
            return
        
        df = pd.read_csv(csv_path)
        if 'epoch' not in df.columns or 'episode_reward' not in df.columns:
            warnings.warn('[plot_stability_std_mean] Missing required columns')
            return
        
        epoch_rewards = df.groupby('epoch')['episode_reward'].mean()
        
        # Calculate rolling statistics
        rolling_std = epoch_rewards.rolling(window=20).std()
        rolling_mean = epoch_rewards.rolling(window=20).mean()
        rolling_cv = (rolling_std / rolling_mean) * 100  # As percentage
        
        # Find min and max CV points
        valid_cv = rolling_cv.dropna()
        min_cv = valid_cv.min()
        max_cv = valid_cv.max()
        cv_range = max_cv - min_cv
        min_cv_epoch = valid_cv.idxmin()
        max_cv_epoch = valid_cv.idxmax()
        min_cv_std = rolling_std.loc[min_cv_epoch]
        max_cv_std = rolling_std.loc[max_cv_epoch]
        
        # Calculate range-based thresholds
        excellent_threshold_cv = min_cv + cv_range * 0.333  # First 33% of range
        acceptable_threshold_cv = min_cv + cv_range * 0.666  # First 66% of range
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 5))  # Reduced height from 6 to 5
        
        # First, draw background zones based on CV at each epoch
        for i in range(len(rolling_cv)):
            if pd.notna(rolling_cv.iloc[i]):
                cv_val = rolling_cv.iloc[i]
                epoch = rolling_cv.index[i]
                if cv_val < excellent_threshold_cv:
                    color = 'green'
                elif cv_val < acceptable_threshold_cv:
                    color = 'yellow'
                else:
                    color = 'red'
                ax.axvspan(epoch - 0.5, epoch + 0.5, alpha=0.15, color=color, zorder=0)
        
        # Convert CV thresholds to Std values
        mean_reward_overall = rolling_mean.mean()
        excellent_threshold_std = excellent_threshold_cv * mean_reward_overall / 100
        acceptable_threshold_std = acceptable_threshold_cv * mean_reward_overall / 100
        
        # Plot Rolling Std line first
        ax.plot(rolling_std.index, rolling_std.values, color='purple', linewidth=2, alpha=0.6, 
                label='Rolling Std', zorder=1)
        
        # Plot scatter with color representing Rolling Mean
        valid_indices = ~rolling_std.isna()
        scatter = ax.scatter(rolling_std.index[valid_indices], 
                            rolling_std.values[valid_indices],
                            c=rolling_mean.values[valid_indices], 
                            cmap='viridis_r',  # Reversed: dark=high, light=low
                            s=30, 
                            alpha=0.8,
                            edgecolors='black',
                            linewidth=0.3,
                            zorder=2)
        
        # Add colorbar for Rolling Mean (tighter to figure)
        cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label('Rolling Mean (20 epochs)', fontsize=9, fontweight='bold')
        
        # Add legend for background zones
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='none', label='CV = (Rolling Std / Rolling Mean) × 100'),
            Patch(facecolor='green', alpha=0.15, label=f'Good Variation: {min_cv:.2f}%<CV<{excellent_threshold_cv:.2f}%'),
            Patch(facecolor='yellow', alpha=0.15, label=f'Acceptable Variation: {excellent_threshold_cv:.2f}%<CV<{acceptable_threshold_cv:.2f}%'),
            Patch(facecolor='red', alpha=0.15, label=f'Poor Variation: {acceptable_threshold_cv:.2f}%<CV<{max_cv:.2f}%')
        ]
        
        # Add annotations for min and max CV points (no arrows, aligned with points)
        # Best CV (green box) - positioned beside the point
        ax.annotate(f'CV={min_cv:.2f}%', 
                   xy=(min_cv_epoch, min_cv_std), 
                   xytext=(min_cv_epoch + 5, min_cv_std),
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='green', alpha=0.8, edgecolor='darkgreen', linewidth=1.5),
                   fontsize=7, fontweight='bold', color='white', ha='left', va='center')
        
        # Worst CV (red box) - positioned to the left of the point
        ax.annotate(f'CV={max_cv:.2f}%', 
                   xy=(max_cv_epoch, max_cv_std), 
                   xytext=(max_cv_epoch - 5, max_cv_std),
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='red', alpha=0.8, edgecolor='darkred', linewidth=1.5),
                   fontsize=7, fontweight='bold', color='white', ha='right', va='center')
        
        # Add exploitation start marker
        try:
            eps_info = get_epsilon_min_step(history_dir)
            if eps_info:
                exploitation_start_epoch = eps_info['exploitation_start_epoch']
                ax.axvline(x=exploitation_start_epoch, color='darkorange', linestyle='--', 
                          linewidth=2.5, alpha=1.0, label=f'Exploitation start')
        except:
            pass
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Rolling Std (20 epochs)', fontweight='bold')
        # Start x-axis from where rolling data actually begins (after window=20)
        first_valid_epoch = rolling_std.dropna().index.min()
        last_valid_epoch = rolling_std.dropna().index.max()
        ax.set_xlim(first_valid_epoch, last_valid_epoch)
        
        # Combine all legend elements
        handles, labels = ax.get_legend_handles_labels()
        handles.extend(legend_elements)
        ax.legend(handles=handles, fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Save
        plots_dir = os.path.join(history_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, 'stability_std_mean_analysis.png')
        plt.tight_layout(pad=0.3)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f'[plot_stability_std_mean] Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_stability_std_mean] Error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    plot_stability_std_mean()
