"""Plot Job and Operation Completion Ratios per Episode for Training Data (800 episodes)."""

import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_job_completion_ratio(history_dir: str = 'my_data_and_graph/historydata', max_episodes: int = 800) -> None:
    """Plot job completion ratio per episode (ratio of departed jobs to arrived jobs).
    
    Shows:
    - Raw completion ratio per episode (green, thin, transparent)
    - 20-episode moving average (dark green, solid, thick)
    - Completion zones (low <30%, medium 30-70%, high ≥70%)
    - 100% completion line (dashed cyan)
    - Average completion line (dotted gray)
    """
    try:
        # Read completion_ratios_training.csv (training episodes only)
        csv_path = os.path.join(history_dir, 'completion_ratios_training.csv')
        if not os.path.exists(csv_path):
            warnings.warn(f'[plot_job_completion_ratio] Missing {csv_path}. Run extract_completion_ratios.py first.')
            return
        
        df = pd.read_csv(csv_path)
        
        # Check if completion ratio columns exist
        if 'job_completion_ratio' not in df.columns:
            warnings.warn('[plot_job_completion_ratio] job_completion_ratio column not found')
            return
        
        # Use all training data (should already be ~800 episodes)
        if len(df) > max_episodes:
            df = df.head(max_episodes)
        
        episodes = df['episode'].values
        ratios = df['job_completion_ratio'].values
        
        if len(episodes) == 0:
            warnings.warn('[plot_job_completion_ratio] No valid episode data')
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 4.5))
        
        # Plot raw ratios (transparent green)
        ax.plot(episodes, ratios, color='green', linewidth=1, alpha=0.3, 
                label='Completion Ratio (Avg: {:.2f}%, λ=0.4)'.format(np.mean(ratios)))
        
        # Plot moving average
        window = 20
        if len(ratios) >= window:
            ratios_series = pd.Series(ratios)
            rolling = ratios_series.rolling(window=window, center=False).mean()
            ax.plot(episodes, rolling.values, color='darkgreen', linewidth=3, 
                    label=f'Moving Average (window={window})')
        
        # Add completion zones (background colors)
        ax.axhspan(0, 30, alpha=0.15, color='red', label='Low Completion (<30%)')
        ax.axhspan(30, 70, alpha=0.15, color='yellow', label='Medium Completion (30-70%)')
        ax.axhspan(70, 100, alpha=0.15, color='lightblue', label='High Completion (≥70%)')
        
        # Add 100% completion line
        ax.axhline(y=100, color='darkblue', linestyle='--', linewidth=2, alpha=0.8,
                   label='100% Completion (All jobs finished)')
        
        # Add average line
        avg_ratio = np.mean(ratios)
        ax.axhline(y=avg_ratio, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'Average: {avg_ratio:.2f}%')
        
        # Add exploration end line
        exploration_end = 479
        ax.axvline(x=exploration_end, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Exploitation Start (ep={exploration_end})')
        
        # Add statistics box (bottom left)
        avg_jobs_arrived = np.mean(df['jobs_arrived'].values)
        avg_ops_arrived = np.mean(df['ops_arrived'].values)
        avg_jobs_completed = np.mean(df['jobs_completed'].values)
        avg_ops_completed = np.mean(df['ops_completed'].values)
        
        # Calculate exploitation phase stats (after episode 479)
        df_explore = df[df['episode'] <= exploration_end]
        df_exploit = df[df['episode'] > exploration_end]
        explore_job_ratio = df_explore['job_completion_ratio'].mean() if len(df_explore) > 0 else 0
        exploit_job_ratio = df_exploit['job_completion_ratio'].mean() if len(df_exploit) > 0 else 0
        
        stats_text = (
            f"Avg jobs arrived/episode: {int(round(avg_jobs_arrived))}\n"
            f"Avg ops arrived/episode: {int(round(avg_ops_arrived))}\n"
            f"Avg jobs finished/episode: {int(round(avg_jobs_completed))}\n"
            f"Avg ops finished/episode: {int(round(avg_ops_completed))}\n"
            f"Before exploitation (ep≤{exploration_end}): {explore_job_ratio:.1f}%\n"
            f"After exploitation (ep>{exploration_end}): {exploit_job_ratio:.1f}%"
        )
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        
        # Formatting
        ax.set_xlim(0, max(episodes))
        ax.set_ylim(0, 105)
        ax.set_xlabel('Episode Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Completion Ratio (Departures/ Arrivals)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Save
        plots_dir = os.path.join(history_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, 'job_completion_ratio.png')
        plt.tight_layout(pad=0.3)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f'[plot_job_completion_ratio] Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_job_completion_ratio] Error: {e}')
        import traceback
        traceback.print_exc()


def plot_operation_completion_ratio(history_dir: str = 'my_data_and_graph/historydata', max_episodes: int = 800) -> None:
    """Plot operation completion ratio per episode (ratio of completed operations to arrived operations).
    
    Shows:
    - Raw completion ratio per episode (blue, thin, transparent)
    - 20-episode moving average (orange, solid, thick)
    - Completion zones (low <30%, medium 30-70%, high ≥70%)
    - 100% completion line (dashed cyan)
    - Average completion line (dotted gray)
    """
    try:
        # Read completion_ratios_training.csv (training episodes only)
        csv_path = os.path.join(history_dir, 'completion_ratios_training.csv')
        if not os.path.exists(csv_path):
            warnings.warn(f'[plot_operation_completion_ratio] Missing {csv_path}. Run extract_completion_ratios.py first.')
            return
        
        df = pd.read_csv(csv_path)
        
        # Check if completion ratio columns exist
        if 'op_completion_ratio' not in df.columns:
            warnings.warn('[plot_operation_completion_ratio] op_completion_ratio column not found')
            return
        
        # Use all training data (should already be ~800 episodes)
        if len(df) > max_episodes:
            df = df.head(max_episodes)
        
        episodes = df['episode'].values
        ratios = df['op_completion_ratio'].values
        
        if len(episodes) == 0:
            warnings.warn('[plot_operation_completion_ratio] No valid episode data')
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 4.5))
        
        # Plot raw ratios (transparent blue)
        ax.plot(episodes, ratios, color='blue', linewidth=1, alpha=0.3, 
                label='Operation Completion Ratio (Avg: {:.2f}%, λ=0.4)'.format(np.mean(ratios)))
        
        # Plot moving average
        window = 20
        if len(ratios) >= window:
            ratios_series = pd.Series(ratios)
            rolling = ratios_series.rolling(window=window, center=False).mean()
            ax.plot(episodes, rolling.values, color='orange', linewidth=3, 
                    label=f'Moving Average (window={window})')
        
        # Add completion zones (background colors)
        ax.axhspan(0, 30, alpha=0.15, color='red', label='Low Completion (<30%)')
        ax.axhspan(30, 70, alpha=0.15, color='yellow', label='Medium Completion (30-70%)')
        ax.axhspan(70, 100, alpha=0.15, color='lightblue', label='High Completion (≥70%)')
        
        # Add 100% completion line
        ax.axhline(y=100, color='darkblue', linestyle='--', linewidth=2, alpha=0.8,
                   label='100% Completion (All operations finished)')
        
        # Add average line
        avg_ratio = np.mean(ratios)
        ax.axhline(y=avg_ratio, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'Average: {avg_ratio:.2f}%')
        
        # Add exploration end line
        exploration_end = 479
        ax.axvline(x=exploration_end, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Exploitation Start (ep={exploration_end})')
        
        # Add statistics box (bottom left)
        avg_jobs_arrived = np.mean(df['jobs_arrived'].values)
        avg_ops_arrived = np.mean(df['ops_arrived'].values)
        avg_jobs_completed = np.mean(df['jobs_completed'].values)
        avg_ops_completed = np.mean(df['ops_completed'].values)
        
        # Calculate exploitation phase stats (after episode 479)
        df_explore = df[df['episode'] <= exploration_end]
        df_exploit = df[df['episode'] > exploration_end]
        explore_op_ratio = df_explore['op_completion_ratio'].mean() if len(df_explore) > 0 else 0
        exploit_op_ratio = df_exploit['op_completion_ratio'].mean() if len(df_exploit) > 0 else 0
        
        stats_text = (
            f"Avg jobs arrived/episode: {int(round(avg_jobs_arrived))}\n"
            f"Avg ops arrived/episode: {int(round(avg_ops_arrived))}\n"
            f"Avg jobs finished/episode: {int(round(avg_jobs_completed))}\n"
            f"Avg ops finished/episode: {int(round(avg_ops_completed))}\n"
            f"Before exploitation (ep≤{exploration_end}): {explore_op_ratio:.1f}%\n"
            f"After exploitation (ep>{exploration_end}): {exploit_op_ratio:.1f}%"
        )
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        
        # Formatting
        ax.set_xlim(0, max(episodes))
        ax.set_ylim(0, 105)
        ax.set_xlabel('Episode Number', fontsize=14, fontweight='bold')
        ax.set_ylabel('Operation Completion Ratio', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Save
        plots_dir = os.path.join(history_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, 'operation_completion_ratio.png')
        plt.tight_layout(pad=0.3)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(f'[plot_operation_completion_ratio] Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_operation_completion_ratio] Error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Generate both completion ratio plots
    print('[plot_completion_ratios] Starting plot generation...')
    plot_job_completion_ratio(max_episodes=800)
    plot_operation_completion_ratio(max_episodes=800)
    print('[plot_completion_ratios] Done!')
