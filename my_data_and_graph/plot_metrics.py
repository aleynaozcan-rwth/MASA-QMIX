"""Plotting helpers for MASA-QMIX historydata artifacts.

Provides small, resilient routines that read simple log files and produce
PNG visualizations under <history_dir>/plots/. Each function handles missing
inputs gracefully and is idempotent.
"""

import os
import json
import warnings
from typing import Optional

import matplotlib
# Use non-interactive backend to allow headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_plots_dir(history_dir: str) -> str:
    plots_dir = os.path.join(history_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def get_epsilon_min_step(history_dir: str = 'my_data_and_graph/historydata'):
    """Detect when epsilon reaches minimum (exploitation phase starts).
    
    Returns:
        dict with keys: 'epsilon_min_step', 'epsilon_min_value', 'epsilon_min_episode'
        or None if not found
    """
    csv_path = os.path.join(history_dir, 'training_metrics.csv')
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if 'train_step' not in df.columns or 'epsilon' not in df.columns:
            return None
        
        steps = df['train_step'].values
        epsilons = df['epsilon'].values
        
        # Find when epsilon reaches minimum (plateaus at epsilon_end)
        # Use threshold 0.11 to catch epsilon_end=0.1 with some tolerance
        mask = epsilons <= 0.11
        if not mask.any():
            return None
        
        min_step = int(steps[mask].min())
        min_eps = float(epsilons[steps == min_step][0])
        
        # Get episode number if available
        min_episode = None
        if 'episode' in df.columns:
            min_episode = int(df[df['train_step'] == min_step]['episode'].iloc[0])
        
        return {
            'epsilon_min_step': min_step,
            'epsilon_min_value': min_eps,
            'epsilon_min_episode': min_episode
        }
    
    except Exception as e:
        warnings.warn(f'[get_epsilon_min_step] Error: {e}')
        return None


def _read_series_file(path: str) -> Optional[pd.Series]:
    """Try to load a simple one- or two-column numeric log into a Series.

    The function accepts files with a single numeric column (one value per
    line) or two columns (index, value). It returns a pandas.Series or None
    when the file cannot be parsed.
    """
    if not os.path.exists(path):
        return None
    # Try common separators and formats
    for sep in [None, '\t', ',', ' ']:
        try:
            if sep is None:
                # try whitespace-delimited autodetect
                df = pd.read_csv(path, header=None, comment='#', sep=r'\s+')
            else:
                df = pd.read_csv(path, header=None, comment='#', sep=sep)
            if df.empty:
                continue
            if df.shape[1] == 1:
                ser = pd.Series(df.iloc[:, 0].astype(float).values)
                return ser
            else:
                # assume second column is the value
                ser = pd.Series(df.iloc[:, 1].astype(float).values)
                return ser
        except Exception:
            continue
    # Fallback: try simple line parse
    vals = []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                parts = ln.split()
                # take last token as numeric value
                try:
                    v = float(parts[-1])
                    vals.append(v)
                except Exception:
                    continue
        if not vals:
            return None
        return pd.Series(vals)
    except Exception:
        return None


def plot_reward_trend(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot episode reward trend aligned with training steps.
    
    Shows episode rewards over training steps (like Q-value plot) for better
    correlation analysis. Includes moving average for trend visualization.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    out_path = os.path.join(plots_dir, 'reward_trend.png')
    
    # Read episode metrics CSV (episode-based data)
    metrics_path = os.path.join(history_dir, 'episode_metrics.csv')
    if not os.path.exists(metrics_path):
        # Fallback to old learning_metrics.csv for backward compatibility
        metrics_path = os.path.join(history_dir, 'learning_metrics.csv')
        if not os.path.exists(metrics_path):
            warnings.warn(f'[plot_reward_trend] Missing episode_metrics.csv; skipping')
            return
    
    try:
        # Load episode rewards
        df = pd.read_csv(metrics_path)
        if 'episode_reward' not in df.columns or 'episode' not in df.columns:
            warnings.warn(f'[plot_reward_trend] Missing required columns in {metrics_path}')
            return
        
        episodes = df['episode'].values
        rewards = df['episode_reward'].values
        
        if len(rewards) == 0:
            warnings.warn(f'[plot_reward_trend] No reward data in {metrics_path}')
            return
        
        # Create plot (X-axis: episode number, not train_step)
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rewards, color='blue', linewidth=1, alpha=0.5, label='Episode Reward')
        
        # Add epsilon min marker (convert step to episode)
        eps_info = get_epsilon_min_step(history_dir)
        if eps_info and eps_info['epsilon_min_episode']:
            eps_ep = eps_info['epsilon_min_episode']
            plt.axvline(x=eps_ep, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Exploitation start (ep {eps_ep})')
        
        # Add moving average
        if len(rewards) > 10:
            window = min(20, len(rewards) // 5)
            reward_series = pd.Series(rewards)
            reward_rolling = reward_series.rolling(window=window, center=True).mean()
            plt.plot(episodes, reward_rolling, color='red', linewidth=2.5, label=f'{window}-episode MA')
        
        # Add mean line
        mean_reward = np.mean(rewards)
        plt.axhline(y=mean_reward, color='green', linestyle='--', alpha=0.5, label=f'Mean: {mean_reward:.3f}')
        
        # Annotations
        if len(rewards) > 0:
            first_reward = rewards[0]
            last_reward = rewards[-1]
            plt.axhline(y=first_reward, color='purple', linestyle=':', alpha=0.3, label=f'Initial: {first_reward:.3f}')
            plt.axhline(y=last_reward, color='orange', linestyle=':', alpha=0.3, label=f'Final: {last_reward:.3f}')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Episode Reward', fontsize=12)
        plt.title('Episode Reward Evolution (Episode-Based View)', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f'[plot_reward_trend] Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_reward_trend] Error: {e}')
        import traceback
        traceback.print_exc()


def plot_loss_trend(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot loss trend from loss.txt using real training steps.

    Expects loss.txt with columns: episode train_step last_loss
    Falls back to episode index if train_step not available.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    src = os.path.join(history_dir, 'loss.txt')
    out_path = os.path.join(plots_dir, 'loss_trend.png')
    """Plot loss trend from loss.txt using real training steps.

    Expects loss.txt with columns: episode train_step last_loss
    Falls back to episode index if train_step not available.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    src = os.path.join(history_dir, 'loss.txt')
    out_path = os.path.join(plots_dir, 'loss_trend.png')
    """Plot loss trend from loss.txt using real training steps.

    Expects loss.txt with columns: episode train_step last_loss
    Falls back to episode index if train_step not available.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    src = os.path.join(history_dir, 'loss.txt')
    out_path = os.path.join(plots_dir, 'loss_trend.png')

    if not os.path.exists(src):
        warnings.warn(f'[plot_loss_trend] Missing {src}; skipping plot')
        return

    try:
        df = pd.read_csv(src, sep=r'\s+', comment='#', header=0)
        
        # train_step and last_loss columns must exist - no fallback
        x_vals = df['train_step'].values
        y_vals = df['last_loss'].values
        
        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, y_vals, color='C1', linewidth=1, alpha=0.5, label='Loss')
        
        # Add epsilon min marker
        eps_info = get_epsilon_min_step(history_dir)
        if eps_info:
            eps_step = eps_info['epsilon_min_step']
            plt.axvline(x=eps_step, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Exploitation start (step {eps_step})')
        
        # Moving average ekle
        if len(y_vals) > 10:
            # pandas is already imported globally at the top
            window = min(20, len(y_vals) // 5)
            loss_series = pd.Series(y_vals)
            loss_rolling = loss_series.rolling(window=window, center=True).mean()
            plt.plot(x_vals, loss_rolling, color='red', linewidth=2.5, label=f'{window}-step MA')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Training Step')
        plt.ylabel('Loss (log scale)')
        plt.title('Loss Trend')
        plt.legend(loc='best')
        plt.tight_layout()
        try:
            plt.savefig(out_path)
        except Exception as e:
            warnings.warn(f'[plot_loss_trend] Could not write {out_path}: {e}')
        finally:
            plt.close()

    except Exception as e:
        warnings.warn(f'[plot_loss_trend] Error: {e}')
        import traceback
        traceback.print_exc()


def plot_td_error_trend(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot TD error trend from td_error.txt using real training steps.

    Expects td_error.txt with columns: episode train_step last_td
    Falls back to episode index if train_step not available.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    src = os.path.join(history_dir, 'td_error.txt')
    out_path = os.path.join(plots_dir, 'td_error_trend.png')

    if not os.path.exists(src):
        warnings.warn(f'[plot_td_error_trend] Missing {src}; skipping plot')
        return

    try:
        df = pd.read_csv(src, sep=r'\s+', comment='#', header=0)
        
        # train_step and last_td columns must exist - no fallback
        x_vals = df['train_step'].values
        y_vals = df['last_td'].values
        
        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, y_vals, color='C2', linewidth=1, label='TD Error')
        
        # Add epsilon min marker
        eps_info = get_epsilon_min_step(history_dir)
        if eps_info:
            eps_step = eps_info['epsilon_min_step']
            plt.axvline(x=eps_step, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Exploitation start (step {eps_step})')
        
        # Moving average ekle
        if len(y_vals) > 10:
            window = min(20, len(y_vals) // 5)
            td_series = pd.Series(y_vals)
            td_rolling = td_series.rolling(window=window, center=True).mean()
            plt.plot(x_vals, td_rolling, color='red', linewidth=1.2, label=f'{window}-step MA')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Training Step')
        plt.ylabel('TD Error')
        plt.title('TD Error Trend')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f'[plot_td_error_trend] Saved {out_path}', flush=True)

    except Exception as e:
        warnings.warn(f'[plot_td_error_trend] Error: {e}')
        import traceback
        traceback.print_exc()


def plot_kpi_summary(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot KPI trends from CSV kpi_log.txt and save kpi_summary.png.

    Expects a CSV file at <history_dir>/kpi_log.txt with columns:
    epoch, avg_wait_time, machine_util, operator_util, makespan
    Also includes loss and TD error trends from loss.txt and td_error.txt.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    src = os.path.join(history_dir, 'kpi_log.txt')
    out_path = os.path.join(plots_dir, 'kpi_summary.png')

    if not os.path.exists(src):
        warnings.warn(f'[plot_kpi_summary] Missing {src}; skipping plot')
        return

    try:
        # read CSV with comment lines (header starting with #)
        df = pd.read_csv(src, sep=',', comment='#', header=None)
        if df.shape[1] < 5:
            warnings.warn(f'[plot_kpi_summary] Unexpected column count in {src}; skipping plot')
            return
        df.columns = ['epoch', 'avg_wait', 'util_m', 'util_o', 'makespan']
    except Exception as e:
        warnings.warn(f'[plot_kpi_summary] Could not read {src} as CSV: {e}')
        return

    # Read loss and TD error data
    loss_series = _read_series_file(os.path.join(history_dir, 'loss.txt'))
    td_series = _read_series_file(os.path.join(history_dir, 'td_error.txt'))

    # Prepare 2x3 subplot grid for six KPIs
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes.flatten()

    try:
        # Row 1: Wait Time, Machine Util, Operator Util
        ax[0].plot(df['epoch'], df['avg_wait'], linewidth=1, color='C0', label='Avg Wait Time')
        # Moving average ekle
        if len(df['avg_wait']) > 10:
            window = min(20, len(df['avg_wait']) // 5)
            avg_wait_series = pd.Series(df['avg_wait'])
            avg_wait_rolling = avg_wait_series.rolling(window=window, center=True).mean()
            ax[0].plot(df['epoch'], avg_wait_rolling, color='magenta', linewidth=1.5, label=f'{window}-step MA')
        ax[0].legend(loc='best')
        ax[0].set_title('Avg Wait Time Trend')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Avg Wait Time')
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(df['epoch'], df['util_m'], color='C1', linewidth=1, label='Machine Utilization')
        # Moving average ekle
        if len(df['util_m']) > 10:
            window = min(20, len(df['util_m']) // 5)
            util_m_series = pd.Series(df['util_m'])
            util_m_rolling = util_m_series.rolling(window=window, center=True).mean()
            ax[1].plot(df['epoch'], util_m_rolling, color='blue', linewidth=1.5, label=f'{window}-step MA')
        ax[1].legend(loc='best')
        ax[1].set_title('Machine Utilization Trend')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Machine Util')
        ax[1].grid(True, alpha=0.3)

        ax[2].plot(df['epoch'], df['util_o'], color='C2', linewidth=1, label='Operator Utilization')
        # Moving average ekle
        if len(df['util_o']) > 10:
            window = min(20, len(df['util_o']) // 5)
            util_o_series = pd.Series(df['util_o'])
            util_o_rolling = util_o_series.rolling(window=window, center=True).mean()
            ax[2].plot(df['epoch'], util_o_rolling, color='orange', linewidth=1.5, label=f'{window}-step MA')
        ax[2].legend(loc='best')
        ax[2].set_title('Operator Utilization Trend')
        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('Operator Util')
        ax[2].grid(True, alpha=0.3)

        # Row 2: Makespan, Loss, TD Error
        ax[3].plot(df['epoch'], df['makespan'], color='C3', linewidth=1, label='Makespan')
        # Moving average ekle
        if len(df['makespan']) > 10:
            window = min(20, len(df['makespan']) // 5)
            makespan_series = pd.Series(df['makespan'])
            makespan_rolling = makespan_series.rolling(window=window, center=True).mean()
            ax[3].plot(df['epoch'], makespan_rolling, color='green', linewidth=1.5, label=f'{window}-step MA')
        ax[3].legend(loc='best')
        ax[3].set_title('Makespan Trend')
        ax[3].set_xlabel('Epoch')
        ax[3].set_ylabel('Makespan')
        ax[3].grid(True, alpha=0.3)

        # Loss trend - same style as loss_trend.png
        if loss_series is not None and len(loss_series) > 0:
            # Read CSV data for combined plot
            metrics_csv = os.path.join(history_dir, 'training_metrics.csv')
            if os.path.exists(metrics_csv):
                df_csv = pd.read_csv(metrics_csv)
                if 'train_step' in df_csv.columns and 'avg_loss' in df_csv.columns:
                    # Plot txt data (gray, transparent)
                    ax[4].plot(loss_series.index, loss_series.values, color='gray', linewidth=1, alpha=0.4, label='Last Loss (txt)')
                    # Plot csv data (orange, dashed)
                    ax[4].plot(df_csv['train_step'], df_csv['avg_loss'], color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Avg Loss (csv)')
                    # Moving average for CSV
                    if len(df_csv['avg_loss']) > 10:
                        window_csv = min(20, len(df_csv['avg_loss']) // 5)
                        rolling_csv = pd.Series(df_csv['avg_loss']).rolling(window=window_csv, center=True).mean()
                        ax[4].plot(df_csv['train_step'], rolling_csv, color='blue', linewidth=1.0, label=f'CSV MA ({window_csv})')
                    
                    # Add exploitation start marker
                    eps_info = get_epsilon_min_step(history_dir)
                    if eps_info:
                        eps_step = eps_info['epsilon_min_step']
                        ax[4].axvline(x=eps_step, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Exploitation start (step {eps_step})')
                else:
                    ax[4].plot(loss_series.index, loss_series.values, color='gray', linewidth=1, alpha=0.5, label='Loss')
            else:
                ax[4].plot(loss_series.index, loss_series.values, color='gray', linewidth=1, alpha=0.5, label='Loss')
            
            ax[4].set_yscale('log')
            ax[4].set_title('Loss Trend')
            ax[4].set_xlabel('Training Step')
            ax[4].set_ylabel('Loss (log scale)')
            ax[4].legend(loc='best', fontsize=7)
            ax[4].grid(True, alpha=0.3)
        else:
            ax[4].text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=ax[4].transAxes)
            ax[4].set_title('Loss Trend')

        # TD Error trend - same style as td_error_trend.png
        if td_series is not None and len(td_series) > 0:
            # Read CSV data for combined plot
            metrics_csv = os.path.join(history_dir, 'training_metrics.csv')
            if os.path.exists(metrics_csv):
                df_csv = pd.read_csv(metrics_csv)
                if 'train_step' in df_csv.columns and 'avg_td_error' in df_csv.columns:
                    # Plot txt data (lightblue, transparent)
                    ax[5].plot(td_series.index, td_series.values, color='lightblue', linewidth=1, alpha=0.4, label='Last TD (txt)')
                    # Plot csv data (teal, dashed)
                    ax[5].plot(df_csv['train_step'], df_csv['avg_td_error'], color='teal', linestyle='--', linewidth=1, alpha=0.7, label='Avg TD (csv)')
                    # Moving average for CSV (magenta)
                    if len(df_csv['avg_td_error']) > 10:
                        window_csv = min(20, len(df_csv['avg_td_error']) // 5)
                        rolling_csv = pd.Series(df_csv['avg_td_error']).rolling(window=window_csv, center=True).mean()
                        ax[5].plot(df_csv['train_step'], rolling_csv, color='magenta', linewidth=1.0, label=f'CSV MA ({window_csv})')
                    
                    # Add exploitation start marker
                    eps_info = get_epsilon_min_step(history_dir)
                    if eps_info:
                        eps_step = eps_info['epsilon_min_step']
                        ax[5].axvline(x=eps_step, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Exploitation start (step {eps_step})')
                else:
                    ax[5].plot(td_series.index, td_series.values, color='lightblue', linewidth=1, alpha=0.5, label='TD Error')
            else:
                ax[5].plot(td_series.index, td_series.values, color='lightblue', linewidth=1, alpha=0.5, label='TD Error')
            
            ax[5].set_title('TD Error Trend')
            ax[5].set_xlabel('Training Step')
            ax[5].set_ylabel('TD Error')
            ax[5].legend(loc='best', fontsize=7)
            ax[5].grid(True, alpha=0.3)
        else:
            ax[5].text(0.5, 0.5, 'TD Error data not available', ha='center', va='center', transform=ax[5].transAxes)
            ax[5].set_title('TD Error Trend')

        plt.tight_layout()
        try:
            plt.savefig(out_path)
            try:
                print(f'[plot_kpi_summary] Saved {out_path}', flush=True)
            except Exception:
                pass
        except Exception as e:
            warnings.warn(f'[plot_kpi_summary] Could not write {out_path}: {e}')
    finally:
        plt.close(fig)


def plot_convergence_summary(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot comprehensive convergence summary with 6 key metrics in 2x3 grid.
    
    Includes: Loss, TD Error, Q-Value, Reward, Batch Reward, Avg Wait Time
    """
    plots_dir = _ensure_plots_dir(history_dir)
    out_path = os.path.join(plots_dir, 'convergence_summary.png')
    
    # Read data files
    loss_txt = os.path.join(history_dir, 'loss.txt')
    td_txt = os.path.join(history_dir, 'td_error.txt')
    metrics_csv = os.path.join(history_dir, 'training_metrics.csv')
    kpi_log = os.path.join(history_dir, 'kpi_log.txt')
    episode_csv = os.path.join(history_dir, 'episode_metrics.csv')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Convergence Summary', fontsize=16, fontweight='bold')
    
    try:
        # Get epsilon info for markers
        eps_info = get_epsilon_min_step(history_dir)
        
        # 1. Loss Trend (top-left)
        if os.path.exists(loss_txt) and os.path.exists(metrics_csv):
            df_loss_txt = pd.read_csv(loss_txt, sep=r'\s+', header=0)
            df_csv = pd.read_csv(metrics_csv)
            if 'train_step' in df_loss_txt.columns and 'last_loss' in df_loss_txt.columns:
                axes[0,0].plot(df_loss_txt['train_step'], df_loss_txt['last_loss'], color='gray', linewidth=1, alpha=0.4, label='Last Loss (txt)')
                if 'train_step' in df_csv.columns and 'avg_loss' in df_csv.columns:
                    axes[0,0].plot(df_csv['train_step'], df_csv['avg_loss'], color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Avg Loss (csv)')
                    if len(df_csv['avg_loss']) > 10:
                        window = min(20, len(df_csv['avg_loss']) // 5)
                        rolling = pd.Series(df_csv['avg_loss']).rolling(window=window, center=True).mean()
                        axes[0,0].plot(df_csv['train_step'], rolling, color='blue', linewidth=1.0, label=f'MA ({window})')
                if eps_info:
                    axes[0,0].axvline(x=eps_info['epsilon_min_step'], color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                axes[0,0].set_yscale('log')
                axes[0,0].set_title('Loss Trend', fontsize=11, fontweight='bold')
                axes[0,0].set_xlabel('Training Step')
                axes[0,0].set_ylabel('Loss (log)')
                axes[0,0].legend(fontsize=7)
                axes[0,0].grid(True, alpha=0.3)
        
        # 2. TD Error Trend (top-middle)
        if os.path.exists(td_txt) and os.path.exists(metrics_csv):
            df_td_txt = pd.read_csv(td_txt, sep=r'\s+', header=0)
            if 'train_step' in df_td_txt.columns and 'last_td' in df_td_txt.columns:
                axes[0,1].plot(df_td_txt['train_step'], df_td_txt['last_td'], color='lightblue', linewidth=1, alpha=0.4, label='Last TD (txt)')
                if 'train_step' in df_csv.columns and 'avg_td_error' in df_csv.columns:
                    axes[0,1].plot(df_csv['train_step'], df_csv['avg_td_error'], color='teal', linestyle='--', linewidth=1, alpha=0.7, label='Avg TD (csv)')
                    if len(df_csv['avg_td_error']) > 10:
                        window = min(20, len(df_csv['avg_td_error']) // 5)
                        rolling = pd.Series(df_csv['avg_td_error']).rolling(window=window, center=True).mean()
                        axes[0,1].plot(df_csv['train_step'], rolling, color='magenta', linewidth=1.0, label=f'MA ({window})')
                if eps_info:
                    axes[0,1].axvline(x=eps_info['epsilon_min_step'], color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                axes[0,1].set_title('TD Error Trend', fontsize=11, fontweight='bold')
                axes[0,1].set_xlabel('Training Step')
                axes[0,1].set_ylabel('TD Error')
                axes[0,1].legend(fontsize=7)
                axes[0,1].grid(True, alpha=0.3)
        
        # 3. Q-Value Trend (top-right)
        diagnostics_path = os.path.join(history_dir, 'diagnostics_log.txt')
        if os.path.exists(diagnostics_path):
            train_steps, q_values = [], []
            with open(diagnostics_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('[') or 'epoch_start' in line or 'epoch_end' in line:
                        continue
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            train_steps.append(int(parts[1]))
                            q_values.append(float(parts[2]))
                        except (ValueError, IndexError):
                            continue
            if train_steps:
                import numpy as np
                q_min, q_max = min(q_values), max(q_values)
                margin = 0.05 * (q_max - q_min) if q_max > q_min else 1.0
                axes[0,2].plot(train_steps, q_values, color='darkgreen', linewidth=1.5, alpha=0.7)
                if len(q_values) > 10:
                    q_series = pd.Series(q_values)
                    q_rolling = q_series.rolling(window=min(20, len(q_values)//5), center=True).mean()
                    axes[0,2].plot(train_steps, q_rolling, color='red', linewidth=2.5, label='MA')
                if eps_info:
                    axes[0,2].axvline(x=eps_info['epsilon_min_step'], color='purple', linestyle='--', linewidth=1.5, alpha=0.7)
                axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
                axes[0,2].set_ylim(q_min - margin, q_max + margin)
                axes[0,2].set_title('Q-Value Trend', fontsize=11, fontweight='bold')
                axes[0,2].set_xlabel('Training Step')
                axes[0,2].set_ylabel('Avg Q-Value')
                axes[0,2].legend(fontsize=7)
                axes[0,2].grid(True, alpha=0.3)
        
        # 4. Reward Trend (bottom-left)
        if os.path.exists(episode_csv):
            df_ep = pd.read_csv(episode_csv)
            if 'episode' in df_ep.columns and 'episode_reward' in df_ep.columns:
                axes[1,0].plot(df_ep['episode'], df_ep['episode_reward'], color='C0', linewidth=1, alpha=0.6, label='Episode Reward')
                if len(df_ep['episode_reward']) > 10:
                    window = min(50, len(df_ep['episode_reward']) // 10)
                    rolling = df_ep['episode_reward'].rolling(window=window, center=True).mean()
                    axes[1,0].plot(df_ep['episode'], rolling, color='red', linewidth=2, label=f'MA ({window})')
                if eps_info and 'epsilon_min_episode' in eps_info:
                    axes[1,0].axvline(x=eps_info['epsilon_min_episode'], color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                axes[1,0].set_title('Reward Trend', fontsize=11, fontweight='bold')
                axes[1,0].set_xlabel('Episode')
                axes[1,0].set_ylabel('Reward')
                axes[1,0].legend(fontsize=7)
                axes[1,0].grid(True, alpha=0.3)
        
        # 5. Batch Reward Trend (bottom-middle)
        if os.path.exists(metrics_csv):
            if 'train_step' in df_csv.columns and 'avg_batch_reward' in df_csv.columns:
                axes[1,1].plot(df_csv['train_step'], df_csv['avg_batch_reward'], color='C4', linewidth=1, alpha=0.5, label='Avg Batch Reward')
                if len(df_csv['avg_batch_reward']) > 10:
                    window = 50
                    rolling = df_csv['avg_batch_reward'].rolling(window).mean()
                    axes[1,1].plot(df_csv['train_step'], rolling, color='red', linewidth=2, label=f'MA ({window})')
                if eps_info:
                    axes[1,1].axvline(x=eps_info['epsilon_min_step'], color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                axes[1,1].set_title('Batch Reward Trend', fontsize=11, fontweight='bold')
                axes[1,1].set_xlabel('Training Step')
                axes[1,1].set_ylabel('Avg Batch Reward')
                axes[1,1].legend(fontsize=7)
                axes[1,1].grid(True, alpha=0.3)
        
        # 6. Avg Wait Time Trend (bottom-right)
        if os.path.exists(kpi_log):
            df_kpi = pd.read_csv(kpi_log, sep=',', comment='#', header=None)
            if df_kpi.shape[1] >= 5:
                df_kpi.columns = ['epoch', 'avg_wait', 'util_m', 'util_o', 'makespan']
                axes[1,2].plot(df_kpi['epoch'], df_kpi['avg_wait'], color='C0', linewidth=1, label='Avg Wait Time')
                if len(df_kpi['avg_wait']) > 10:
                    window = min(20, len(df_kpi['avg_wait']) // 5)
                    rolling = pd.Series(df_kpi['avg_wait']).rolling(window=window, center=True).mean()
                    axes[1,2].plot(df_kpi['epoch'], rolling, color='magenta', linewidth=1.5, label=f'MA ({window})')
                axes[1,2].set_title('Avg Wait Time Trend', fontsize=11, fontweight='bold')
                axes[1,2].set_xlabel('Epoch')
                axes[1,2].set_ylabel('Avg Wait Time')
                axes[1,2].legend(fontsize=7)
                axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        print(f'[plot_convergence_summary] Saved {out_path}', flush=True)
    except Exception as e:
        warnings.warn(f'[plot_convergence_summary] Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)


def utilization_summary_grid(history_dir: str = 'my_data_and_graph/historydata', combine_plots: bool = True):
    """Generate 2x2 grid: per-machine util, per-operator util, makespan, reward."""
    path = os.path.join(history_dir, "run_summary.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    evols = data.get("evolutions", []) if isinstance(data, dict) else []
    if not evols:
        return

    x = list(range(len(evols)))
    makespans = []
    avg_rewards = []
    for e in evols:
        # reward
        try:
            avg_r = e.get('avg_epoch_reward') if e.get('avg_epoch_reward') is not None else None
        except Exception:
            avg_r = None
        avg_rewards.append(avg_r)
        # makespan
        m = e.get('avg_makespan', None) if isinstance(e, dict) else None
        if m is None:
            m = e.get('average_makespan', None) if isinstance(e, dict) else None
        makespans.append(float(m) if m is not None else 0.0)

    plots_dir = _ensure_plots_dir(history_dir)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax0 = axes[0,0]
        # machine trends
        machine_ids = []
        for e in evols:
            pm = e.get('per_machine_utilization', {}) or {}
            for k in pm.keys():
                ks = str(k)
                if ks not in machine_ids:
                    machine_ids.append(ks)
        cmap = plt.get_cmap('tab10')
        for i, mid in enumerate(machine_ids):
            vals = []
            for e in evols:
                pm = e.get('per_machine_utilization', {}) or {}
                v = pm.get(int(mid), None) if mid.isdigit() else pm.get(mid, None)
                if v is None:
                    v = pm.get(str(mid), 0.0)
                vals.append(float(v))
            # Her 50 epochta bir veri noktası al
            x_sparse = x[::25]
            vals_sparse = vals[::25]
            ax0.plot(x_sparse, vals_sparse, label=f"M{mid}", color=cmap(i % 10))
        ax0.set_title('Per-Machine Utilization')
        ax0.set_xlabel('Epoch')
        ax0.set_ylabel('Utilization')
        ax0.set_ylim(0,1)
        ax0.grid(True)
        ax0.legend(loc='best', fontsize='x-small')

        ax1 = axes[0,1]
        # operator trends
        op_ids = []
        for e in evols:
            po = e.get('per_operator_utilization', {}) or {}
            for k in po.keys():
                ks = str(k)
                if ks not in op_ids:
                    op_ids.append(ks)
        for i, oid in enumerate(op_ids):
            vals = []
            for e in evols:
                po = e.get('per_operator_utilization', {}) or {}
                v = po.get(oid, None)
                if v is None and oid.isdigit():
                    v = po.get(int(oid), 0.0)
                if v is None:
                    v = 0.0
                vals.append(float(v))
            # Her 50 epochta bir veri noktası al
            x_sparse = x[::25]
            vals_sparse = vals[::25]
            ax1.plot(x_sparse, vals_sparse, label=f"O{oid}", color=cmap(i % 10))
        ax1.set_title('Per-Operator Utilization')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Utilization')
        ax1.set_ylim(0,1)
        ax1.grid(True)
        ax1.legend(loc='best', fontsize='x-small')

        ax2 = axes[1,0]
        ax2.plot(x, makespans, color='tab:green', label='Average Makespan')
        # Moving average ekle
        if len(makespans) > 10:
            window = min(20, len(makespans) // 5)
            import pandas as pd
            makespan_series = pd.Series(makespans)
            makespan_rolling = makespan_series.rolling(window=window, center=True).mean()
            ax2.plot(x, makespan_rolling, color='magenta', linewidth=2, linestyle='--', label=f'{window}-epoch MA')
        ax2.legend(loc='best')
        ax2.set_title('Average Makespan')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Makespan')
        ax2.grid(True)

        ax3 = axes[1,1]
        # plot avg rewards if present
        if any(v is not None for v in avg_rewards):
            rr = [v if v is not None else float('nan') for v in avg_rewards]
            ax3.plot(x, rr, color='tab:purple', label='Average Reward')
            # Moving average ekle
            if len(rr) > 10:
                window = min(20, len(rr) // 5)
                import pandas as pd
                reward_series = pd.Series(rr)
                reward_rolling = reward_series.rolling(window=window, center=True).mean()
                ax3.plot(x, reward_rolling, color='red', linewidth=2, linestyle='--', label=f'{window}-epoch MA')
            ax3.legend(loc='best')
            ax3.set_title('Average Reward')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Reward')
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'No reward data', ha='center', va='center')
            ax3.set_title('Average Reward')

        plt.tight_layout()
        out_path = os.path.join(plots_dir, 'utilization_summary_grid.png')
        plt.savefig(out_path)
        plt.close()
        print(f"[utilization_summary_grid] Saved {out_path}", flush=True)
    except Exception as e:
        print(f"[WARN] Could not save utilization_summary_grid.png: {e}")


def plot_q_value_trend(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot Q-value evolution from diagnostics log with MA and exploitation start marker.
    
    Reads diagnostics_log.txt and extracts avg_q values over training steps.
    Shows how the network's Q-value estimates evolve during learning.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    out_path = os.path.join(plots_dir, 'q_value_trend.png')
    
    diagnostics_path = os.path.join(history_dir, 'diagnostics_log.txt')
    if not os.path.exists(diagnostics_path):
        warnings.warn(f'[plot_q_value_trend] Missing {diagnostics_path}; skipping')
        return
    
    try:
        # Parse diagnostics file
        # Format: timestamp,train_step,avg_q,grad_before,grad_after,target_updates,loss,td_error
        train_steps = []
        q_values = []
        
        with open(diagnostics_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('[') or 'epoch_start' in line or 'epoch_end' in line:
                    continue
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        step = int(parts[1])
                        q_val = float(parts[2])
                        train_steps.append(step)
                        q_values.append(q_val)
                    except (ValueError, IndexError):
                        continue
        
        if not train_steps:
            warnings.warn(f'[plot_q_value_trend] No Q-value data found in {diagnostics_path}')
            return
        
        # --- Single Q-value trend plot with tight zoom ---
        import numpy as np
        # Min ve max değerlerini bul
        q_min = min(q_values)
        q_max = max(q_values)
        q_range = q_max - q_min
        
        # Automatically calculate margin (5% of the range for tighter fit)
        margin = 0.05 * q_range if q_range > 0 else 1.0
        
        ylim_low = q_min - margin
        ylim_high = q_max + margin

        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, q_values, color='darkgreen', linewidth=1.5, alpha=0.7)
        if len(q_values) > 10:
            q_series = pd.Series(q_values)
            q_rolling = q_series.rolling(window=min(20, len(q_values)//5), center=True).mean()
            plt.plot(train_steps, q_rolling, color='red', linewidth=2.5, label='MA')
        
        # Add epsilon min marker
        eps_info = get_epsilon_min_step(history_dir)
        if eps_info:
            eps_step = eps_info['epsilon_min_step']
            plt.axvline(x=eps_step, color='purple', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Exploitation start (step {eps_step})')
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        plt.axhline(y=q_values[0], color='blue', linestyle=':', alpha=0.3, label=f'Initial: {q_values[0]:.0f}')
        plt.axhline(y=q_values[-1], color='orange', linestyle=':', alpha=0.3, label=f'Final: {q_values[-1]:.0f}')
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Average Q-Value', fontsize=12)
        plt.title('Q-Value Trend', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(ylim_low, ylim_high)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f'[plot_q_value_trend] Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_q_value_trend] Error: {e}')
        import traceback
        traceback.print_exc()


def plot_learning_analysis_comprehensive(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Generate comprehensive 6-panel learning analysis visualization.
    
    Creates a detailed analysis plot with:
    1. Reward progression with moving average
    2. Reward distribution by training quarters
    3. Makespan evolution
    4. Resource utilization trends
    5. Reward stability (rolling std)
    6. Training summary statistics
    """
    plots_dir = _ensure_plots_dir(history_dir)
    out_path = os.path.join(plots_dir, 'learning_analysis_comprehensive.png')
    
    # Load data
    metrics_path = os.path.join(history_dir, 'episode_metrics.csv')
    if not os.path.exists(metrics_path):
        # Fallback to old format
        metrics_path = os.path.join(history_dir, 'learning_metrics.csv')
        if not os.path.exists(metrics_path):
            warnings.warn(f'[plot_learning_analysis_comprehensive] Missing episode_metrics.csv; skipping')
            return
    
    kpi_path = os.path.join(history_dir, 'kpi_log.txt')
    
    try:
        metrics = pd.read_csv(metrics_path)
        epoch_rewards = metrics.groupby('epoch')['episode_reward'].mean()
        
        # Load KPI data
        if os.path.exists(kpi_path):
            kpi = pd.read_csv(kpi_path, skiprows=1, 
                             names=['epoch', 'avg_wait_time', 'machine_util', 'operator_util', 'makespan'])
        else:
            kpi = None
        
        # Create 2x3 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('MASA-QMIX Learning Analysis - Comprehensive View', fontsize=16, fontweight='bold')
        
        # 1. Reward trend with moving average
        ax = axes[0, 0]
        ax.plot(epoch_rewards.index, epoch_rewards.values, alpha=0.3, color='blue', label='Reward')
        window = 10
        rolling = epoch_rewards.rolling(window=window).mean()
        ax.plot(rolling.index, rolling.values, color='darkblue', linewidth=2, label=f'{window}-epoch MA')
        ax.axhline(y=epoch_rewards.mean(), color='red', linestyle='--', alpha=0.5, 
                   label=f'Mean: {epoch_rewards.mean():.3f}')
        ax.fill_between(epoch_rewards.index, 
                        epoch_rewards.mean() - epoch_rewards.std(),
                        epoch_rewards.mean() + epoch_rewards.std(),
                        alpha=0.2, color='gray')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Reward distribution by quarters
        ax = axes[0, 1]
        n_epochs = len(epoch_rewards)
        q_size = n_epochs // 4
        quarters = [
            epoch_rewards.iloc[:q_size],
            epoch_rewards.iloc[q_size:2*q_size],
            epoch_rewards.iloc[2*q_size:3*q_size],
            epoch_rewards.iloc[3*q_size:]
        ]
        bp = ax.boxplot(quarters, labels=[f'Q1\n(0-{q_size})', f'Q2\n({q_size}-{2*q_size})', 
                                           f'Q3\n({2*q_size}-{3*q_size})', f'Q4\n({3*q_size}-{n_epochs})'],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']):
            patch.set_facecolor(color)
        ax.set_ylabel('Reward')
        ax.set_title('Reward Distribution by Quarter')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Makespan trend
        ax = axes[0, 2]
        if kpi is not None and 'makespan' in kpi.columns:
            ax.plot(kpi['epoch'], kpi['makespan'], color='green', linewidth=1.5)
            rolling_makespan = kpi['makespan'].rolling(window=10).mean()
            ax.plot(kpi['epoch'], rolling_makespan, color='magenta', linewidth=2, label='10-epoch MA')
            ax.axhline(y=kpi['makespan'].min(), color='red', linestyle='--', alpha=0.5, 
                      label=f'Best: {kpi["makespan"].min():.1f}')
            ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Makespan')
        ax.set_title('Makespan Evolution')
        ax.grid(True, alpha=0.3)
        
        # 4. Utilization trends
        ax = axes[1, 0]
        if kpi is not None:
            ax.plot(kpi['epoch'], kpi['machine_util']*100, label='Machine', color='blue', linewidth=1.5)
            ax.plot(kpi['epoch'], kpi['operator_util']*100, label='Operator', color='orange', linewidth=1.5)
            # Machine MA
            if len(kpi['machine_util']) > 10:
                window = min(20, len(kpi['machine_util']) // 5)
                machine_ma = kpi['machine_util'].rolling(window=window, center=True).mean()*100
                ax.plot(kpi['epoch'], machine_ma, color='red', linewidth=2, linestyle='--', label=f'Machine MA ({window})')
            # Operator MA
            if len(kpi['operator_util']) > 10:
                window = min(20, len(kpi['operator_util']) // 5)
                operator_ma = kpi['operator_util'].rolling(window=window, center=True).mean()*100
                ax.plot(kpi['epoch'], operator_ma, color='green', linewidth=2, linestyle='--', label=f'Operator MA ({window})')
            ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('Resource Utilization')
        ax.grid(True, alpha=0.3)
        
        # 5. Reward variability (rolling std)
        ax = axes[1, 1]
        rolling_std = epoch_rewards.rolling(window=20).std()
        ax.plot(rolling_std.index, rolling_std.values, color='purple', linewidth=2)
        ax.axhline(y=0.015, color='green', linestyle='--', alpha=0.5, label='Good stability')
        ax.axhline(y=0.025, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
        ax.fill_between(rolling_std.index, 0, 0.015, alpha=0.2, color='green')
        ax.fill_between(rolling_std.index, 0.015, 0.025, alpha=0.2, color='yellow')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rolling Std (20 epochs)')
        ax.set_title('Reward Stability (Lower = Better)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 6. Performance metrics summary
        ax = axes[1, 2]
        ax.axis('off')
        
        # Build summary text
        best_epoch = epoch_rewards.idxmax()
        final_20_avg = epoch_rewards.iloc[-20:].mean() if len(epoch_rewards) >= 20 else epoch_rewards.mean()
        final_20_std = epoch_rewards.iloc[-20:].std() if len(epoch_rewards) >= 20 else epoch_rewards.std()
        
        summary_text = f"""
TRAINING SUMMARY
{'='*40}

Total Episodes: {len(metrics)}
Total Epochs: {len(epoch_rewards)}

REWARD METRICS
  Average: {epoch_rewards.mean():.4f}
  Best Epoch: {best_epoch} ({epoch_rewards.max():.4f})
  Final 20 Avg: {final_20_avg:.4f}
  Stability (σ): {final_20_std:.4f}
"""
        
        if kpi is not None:
            summary_text += f"""
KPI METRICS
  Avg Makespan: {kpi['makespan'].mean():.2f}
  Best Makespan: {kpi['makespan'].min():.2f}
  Machine Util: {kpi['machine_util'].mean()*100:.2f}%
  Operator Util: {kpi['operator_util'].mean()*100:.2f}%
"""
        
        # Check for convergence
        if final_20_std < 0.015:
            status = "Strong Convergence ✓"
        elif final_20_std < 0.020:
            status = "Good Convergence ✓"
        else:
            status = "Moderate Stability"
        
        summary_text += f"""
LEARNING STATUS
  Convergence: {status}
  
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[plot_learning_analysis_comprehensive] Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_learning_analysis_comprehensive] Error: {e}')
        import traceback
        traceback.print_exc()


def plot_reward_components(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot reward components over simulation time.

    Expects CSV file at <history_dir>/reward_components.csv (v4) or
    <history_dir>/reward_components_log.txt (legacy) with columns:
    step,sim_time,CompletedNorm,AvgWaitNorm,WIPNorm,ThroughputDelta,LoadBalance,R_global,R_total
    """
    plots_dir = _ensure_plots_dir(history_dir)
    
    # [v4-FIX] Try new CSV format first, then fall back to legacy
    src = os.path.join(history_dir, 'reward_components.csv')
    if not os.path.exists(src):
        src = os.path.join(history_dir, 'reward_components_log.txt')
    
    out_path = os.path.join(plots_dir, 'reward_components.png')

    if not os.path.exists(src):
        warnings.warn(f'[plot_reward_components] Missing reward components file; skipping plot')
        return

    try:
        # [v4-FIX] Read CSV with comment lines (v4 format has # header comments)
        df = pd.read_csv(src, comment='#')
        
        # v4 format: step,sim_time,CompletedNorm,AvgWaitNorm,WIPNorm,ThroughputDelta,LoadBalance,R_global,R_total
        # Legacy format: env_time,CompletedNorm,AvgWait,WIP,ThroughputDelta,LoadVariance,R_global
        
        # Normalize column names
        if 'sim_time' in df.columns:
            df.rename(columns={'sim_time': 'env_time'}, inplace=True)
        if 'AvgWaitNorm' in df.columns:
            df.rename(columns={'AvgWaitNorm': 'AvgWait'}, inplace=True)
        if 'WIPNorm' in df.columns:
            df.rename(columns={'WIPNorm': 'WIP'}, inplace=True)
        if 'LoadBalance' in df.columns:
            df.rename(columns={'LoadBalance': 'LoadVariance'}, inplace=True)
            
    except Exception as e:
        warnings.warn(f'[plot_reward_components] Could not read {src}: {e}')
        return

    if df.empty:
        warnings.warn(f'[plot_reward_components] {src} is empty; skipping plot')
        return

    # Ensure env_time is numeric and sorted
    try:
        df['env_time'] = pd.to_numeric(df['env_time'], errors='coerce')
        df = df.dropna(subset=['env_time'])
        df = df.sort_values(by='env_time')
    except Exception:
        pass

    # Read reward formula from CSV header
    formula_text = ""
    try:
        with open(src, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.startswith('#')]
            if len(lines) >= 3:
                # Extract formula lines
                formula_line = lines[1].replace('# ', '')
                coeff_line = lines[2].replace('# ', '')
                formula_text = f"{formula_line}\n{coeff_line}"
    except Exception:
        pass
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    if formula_text:
        fig.suptitle(f'Reward Components over Time (Smoothed)\n{formula_text}', 
                     fontsize=11, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Reward Components over Time (Smoothed)', fontsize=14, fontweight='bold')
    
    try:
        components = ['CompletedNorm', 'AvgWait', 'WIP', 'ThroughputDelta', 'LoadVariance', 'R_global']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, (comp, color) in enumerate(zip(components, colors)):
            if comp in df.columns:
                ax = axes[idx // 3, idx % 3]
                values = pd.to_numeric(df[comp], errors='coerce').fillna(0.0)
                time_values = df['env_time'].values
                
                # Apply EWMA smoothing - only plot the smooth line, no scatter
                if len(values) > 10:
                    smoothed = values.ewm(span=100, adjust=False).mean()
                    ax.plot(time_values, smoothed.values, color=color, linewidth=3.0, alpha=0.95)
                else:
                    ax.plot(time_values, values.values, color=color, linewidth=3.0, alpha=0.95)
                
                # Add subtle fill under the curve
                if len(values) > 10:
                    ax.fill_between(time_values, 0, smoothed.values, color=color, alpha=0.15)
                
                ax.set_title(comp, fontsize=11, fontweight='bold', color=color)
                ax.grid(True, alpha=0.2, linestyle='--')
                ax.set_xlabel('Env time', fontsize=10)
                ax.set_ylabel('Value', fontsize=10)
        
        plt.tight_layout()
        try:
            plt.savefig(out_path)
            try:
                print(f'[plot_reward_components] Saved {out_path}', flush=True)
            except Exception:
                pass
        except Exception as e:
            warnings.warn(f'[plot_reward_components] Could not write {out_path}: {e}')
    finally:
        plt.close()


def plot_epsilon_decay(history_dir: str = 'my_data_and_graph/historydata', overlay_debug: bool = False) -> None:
    """Plot epsilon decay over training steps with exploration/exploitation phases.
    
    Reads epsilon values from debug_output.txt ([EPSILON_DECAY] and [Diagnostics] lines)
    and creates a comprehensive plot showing:
    - Epsilon decay curve
    - Phase transitions (exploration → exploitation)
    - Key milestones (epsilon thresholds)
    """
    plots_dir = _ensure_plots_dir(history_dir)
    out_path = os.path.join(plots_dir, 'epsilon_decay.png')
    
    # Prefer authoritative training CSV (train_step, epsilon) when present
    csv_path = os.path.join(history_dir, 'training_metrics.csv')
    debug_file = os.path.join(history_dir, 'debug_output.txt')

    try:
        import re

        # Fail-fast behavior: training_metrics.csv must exist and have required columns
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Required file not found: {csv_path}")

        df_csv = pd.read_csv(csv_path)
        if 'train_step' not in df_csv.columns or 'epsilon' not in df_csv.columns:
            raise ValueError(f"training_metrics.csv missing required columns: 'train_step' and 'epsilon'")

        # Use CSV as authoritative source
        steps = df_csv['train_step'].astype(int).values
        epsilons = df_csv['epsilon'].astype(float).values


        # Create plot (step-style to show plateaus clearly)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.step(steps, epsilons, where='post', linewidth=1.8, color='#2E86AB', label='Epsilon (training_metrics.csv)')
        ax.scatter(steps, epsilons, s=8, color='#2E86AB', alpha=0.7)

        # Phase annotations (always draw the bands so the thresholds are visible regardless of data)
        ax.axhspan(0.5, 1.0, alpha=0.08, color='red', label='High Exploration (ε>0.5)')
        ax.axhspan(0.2, 0.5, alpha=0.08, color='orange', label='Medium Exploration (0.2<ε≤0.5)')
        ax.axhspan(0.05, 0.2, alpha=0.08, color='yellow', label='Low Exploration (0.05<ε≤0.2)')

        ax.axhline(y=0.05, color='green', linestyle='--', linewidth=2, label='Min Epsilon (0.05) - Exploitation')

        # Find when epsilon reaches ~0.05 (may raise if arrays are empty — intentional fail-fast)
        mask = epsilons <= 0.051
        exploitation_start = int(steps[mask].min()) if mask.any() else None
        if exploitation_start is not None:
            ax.axvline(x=exploitation_start, color='green', linestyle=':', alpha=0.5)
            ax.text(exploitation_start, 0.5, f'  Exploitation\n  starts ~step {exploitation_start}',
                    rotation=0, va='center', ha='left', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

        # Mark key milestones
        milestones = [0.9, 0.5, 0.2, 0.1]
        for milestone in milestones:
            ms_mask = (epsilons >= milestone - 0.01) & (epsilons <= milestone + 0.01)
            milestone_steps = steps[ms_mask]
            if milestone_steps.size > 0:
                step_at_milestone = int(milestone_steps[0])
                ax.plot(step_at_milestone, milestone, 'ro', markersize=6, alpha=0.6)
                ax.annotate(f'ε≈{milestone:.1f}\nstep {step_at_milestone}',
                            xy=(step_at_milestone, milestone), xytext=(10, 10), textcoords='offset points',
                            fontsize=8, alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Epsilon (ε)', fontsize=12)
        ax.set_title('Epsilon Decay: Exploration → Exploitation Transition', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(-0.05, 1.05)

        # Add statistics box
        initial_eps = float(epsilons[0])
        final_eps = float(epsilons[-1])
        total_steps = int(steps[-1])
        decay_rate = (initial_eps - final_eps) / total_steps * 1000.0
        stats_text = f"""Training Stats:
Initial ε: {initial_eps:.4f}
Final ε: {final_eps:.4f}
Total steps: {total_steps}
Decay rate: {decay_rate:.6f} per 1k steps"""
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        # Try to read epsilon_anneal_fraction directly from arguments.py
        anneal_fraction = None
        try:
            import sys
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MARL/common')))
            import arguments
            args = arguments.get_mutable_args()
            anneal_fraction = getattr(args, 'epsilon_anneal_fraction', None)
        except Exception:
            anneal_fraction = None
        if anneal_fraction is not None:
            stats_text += f"\nEpsilon Anneal Fraction: {anneal_fraction:.2f}"

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[plot_epsilon_decay] Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_epsilon_decay] Error: {e}')
        import traceback
        traceback.print_exc()

def plot_loss_trend_combined(history_dir='my_data_and_graph/historydata'):
    """Plot both last_loss (txt) and avg_loss (csv) on same graph."""
    plots_dir = _ensure_plots_dir(history_dir)
    out = os.path.join(plots_dir, 'loss_trend.png')

    loss_txt = os.path.join(history_dir, 'loss.txt')
    metrics_csv = os.path.join(history_dir, 'training_metrics.csv')

    if not (os.path.exists(loss_txt) and os.path.exists(metrics_csv)):
        warnings.warn("[plot_loss_trend_combined] Missing files")
        return

    try:
        df_txt = pd.read_csv(loss_txt, sep=r'\s+', header=0)
        df_csv = pd.read_csv(metrics_csv)

        if 'train_step' not in df_txt.columns or 'last_loss' not in df_txt.columns:
            warnings.warn("loss.txt invalid format")
            return
        if 'train_step' not in df_csv.columns or 'avg_loss' not in df_csv.columns:
            warnings.warn("CSV missing avg_loss")
            return

        plt.figure(figsize=(8, 4))
        # Ham değerler
        plt.plot(df_txt['train_step'], df_txt['last_loss'], label='Last Loss (txt)', color='gray', alpha=0.4)
        plt.plot(df_csv['train_step'], df_csv['avg_loss'], label='Avg Loss (csv)', color='orange', linestyle='--', alpha=0.7)
        
        # Add epsilon min marker
        eps_info = get_epsilon_min_step(history_dir)
        if eps_info:
            eps_step = eps_info['epsilon_min_step']
            plt.axvline(x=eps_step, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Exploitation start (step {eps_step})')
        
        # Sadece CSV moving average (mavi)
        if len(df_csv['avg_loss']) > 10:
            window_csv = min(20, len(df_csv['avg_loss']) // 5)
            rolling_csv = pd.Series(df_csv['avg_loss']).rolling(window=window_csv, center=True).mean()
            plt.plot(df_csv['train_step'], rolling_csv, color='blue', linewidth=1.0, label=f'CSV MA ({window_csv})')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Loss Trend (txt vs csv)')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
    except Exception as e:
        warnings.warn(f"[plot_loss_trend_combined] Error: {e}")
def plot_td_error_trend_combined(history_dir='my_data_and_graph/historydata'):
    """Plot last_td (txt) + avg_td_error (csv) together."""
    plots_dir = _ensure_plots_dir(history_dir)
    out = os.path.join(plots_dir, 'td_error_trend.png')

    td_txt = os.path.join(history_dir, 'td_error.txt')
    metrics_csv = os.path.join(history_dir, 'training_metrics.csv')

    if not (os.path.exists(td_txt) and os.path.exists(metrics_csv)):
        warnings.warn("[plot_td_error_trend_combined] Missing files")
        return

    try:
        df_txt = pd.read_csv(td_txt, sep=r'\s+', header=0)
        df_csv = pd.read_csv(metrics_csv)

        if 'train_step' not in df_txt.columns or 'last_td' not in df_txt.columns:
            warnings.warn("td_error.txt invalid format")
            return
        if 'train_step' not in df_csv.columns or 'avg_td_error' not in df_csv.columns:
            warnings.warn("CSV missing avg_td_error")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(df_txt['train_step'], df_txt['last_td'], label='Last TD (txt)', color='lightblue', alpha=0.4)
        plt.plot(df_csv['train_step'], df_csv['avg_td_error'], label='Avg TD (csv)', color='teal', linestyle='--', alpha=0.7)
        
        # Add epsilon min marker
        eps_info = get_epsilon_min_step(history_dir)
        if eps_info:
            eps_step = eps_info['epsilon_min_step']
            plt.axvline(x=eps_step, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Exploitation start (step {eps_step})')
        
        # Add moving average for CSV data
        if len(df_csv['avg_td_error']) > 10:
            window_csv = min(20, len(df_csv['avg_td_error']) // 5)
            rolling_csv = pd.Series(df_csv['avg_td_error']).rolling(window=window_csv, center=True).mean()
            plt.plot(df_csv['train_step'], rolling_csv, color='magenta', linewidth=1.0, label=f'CSV MA ({window_csv})')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Training Step')
        plt.ylabel('TD Error')
        plt.title('TD Error Trend (txt vs csv)')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
    except Exception as e:
        warnings.warn(f"[plot_td_error_trend_combined] Error: {e}")
def plot_q_value_csv(history_dir='my_data_and_graph/historydata'):
    """Uses training_metrics.csv to plot avg_q_value over training steps."""
    plots_dir = _ensure_plots_dir(history_dir)
    out = os.path.join(plots_dir, 'q_value_csv.png')

    csv_path = os.path.join(history_dir, 'training_metrics.csv')
    if not os.path.exists(csv_path):
        warnings.warn("[plot_q_value_csv] CSV missing")
        return

    try:
        df = pd.read_csv(csv_path)
        if 'train_step' not in df.columns or 'avg_q_value' not in df.columns:
            warnings.warn("CSV missing avg_q_value")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(df['train_step'], df['avg_q_value'], color='darkgreen', linewidth=1.5, label='Avg Q-value')
        
        # Add epsilon min marker
        eps_info = get_epsilon_min_step(history_dir)
        if eps_info:
            eps_step = eps_info['epsilon_min_step']
            plt.axvline(x=eps_step, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Exploitation start (step {eps_step})')
        
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Training Step")
        plt.ylabel("Avg Q-Value")
        plt.title("Q-Value Trend (from training_metrics.csv)")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
    except Exception as e:
        warnings.warn(f"[plot_q_value_csv] Error: {e}")
def plot_batch_reward_csv(history_dir='my_data_and_graph/historydata'):
    """Plot avg_batch_reward from training_metrics.csv."""
    plots_dir = _ensure_plots_dir(history_dir)
    out = os.path.join(plots_dir, 'batch_reward_trend.png')

    csv_path = os.path.join(history_dir, 'training_metrics.csv')
    if not os.path.exists(csv_path):
        warnings.warn("[plot_batch_reward_csv] CSV missing")
        return

    try:
        df = pd.read_csv(csv_path)
        if 'train_step' not in df.columns or 'avg_batch_reward' not in df.columns:
            warnings.warn("CSV missing avg_batch_reward")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(df['train_step'], df['avg_batch_reward'], color='C4', alpha=0.5, label='Avg Batch Reward')
        # Moving average (smoothing) line
        window = 50  # You can adjust window size
        ma = df['avg_batch_reward'].rolling(window).mean()
        plt.plot(df['train_step'], ma, color='red', linewidth=2, label=f'Moving Average ({window})')
        
        # Add epsilon min marker
        eps_info = get_epsilon_min_step(history_dir)
        if eps_info:
            eps_step = eps_info['epsilon_min_step']
            plt.axvline(x=eps_step, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Exploitation start (step {eps_step})')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel("Training Step")
        plt.ylabel("Avg Batch Reward")
        plt.title("Batch Reward Trend (training_metrics.csv)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
    except Exception as e:
        warnings.warn(f"[plot_batch_reward_csv] Error: {e}")

# Convenience function to run all plots
def generate_all_plots(history_dir: str = 'my_data_and_graph/historydata', overlay_debug: bool = True) -> None:
    plot_reward_trend(history_dir)
    plot_q_value_trend(history_dir)  # Q-value plot with MA and exploitation start
    plot_epsilon_decay(history_dir, overlay_debug=overlay_debug)  # New epsilon decay plot!
    plot_kpi_summary(history_dir)
    plot_convergence_summary(history_dir)  # Comprehensive convergence summary (6 metrics)
    plot_loss_trend_combined(history_dir)  # Combined loss plot (txt + csv)
    plot_td_error_trend_combined(history_dir)  # Combined TD error plot (txt + csv)
    plot_batch_reward_csv(history_dir)
    plot_gradient_norms(history_dir)  # NEW: Gradient norm monitoring (before/after clipping)

    # reward components plot (optional)
    try:
        plot_reward_components(history_dir)
    except Exception:
        warnings.warn('[generate_all_plots] plot_reward_components failed; continuing')
    # utilization summary grid (optional)
    try:
        utilization_summary_grid(history_dir)
    except Exception:
        warnings.warn('[generate_all_plots] utilization_summary_grid failed; continuing')
    # comprehensive learning analysis (new!)
    try:
        plot_learning_analysis_comprehensive(history_dir)
    except Exception:
        warnings.warn('[generate_all_plots] plot_learning_analysis_comprehensive failed; continuing')
    try:
        plot_loss_trend_combined(history_dir)
    except Exception:
        warnings.warn('[generate_all_plots] plot_loss_trend_combined failed; continuing')

    try:
        plot_td_error_trend_combined(history_dir)
    except Exception:
        warnings.warn('[generate_all_plots] plot_td_error_trend_combined failed; continuing')

    try:
        plot_batch_reward_csv(history_dir)
    except Exception:
        warnings.warn('[generate_all_plots] plot_batch_reward_csv failed; continuing')
    try:
        print('[plot_metrics] All plots generated successfully.', flush=True)
    except Exception:
        pass


# Stub functions for backwards compatibility with runner.py
def plot_utilization_and_makespan(history_dir: str = 'my_data_and_graph/historydata'):
    """Deprecated: Use utilization_summary_grid instead."""
    pass

def plot_per_machine_utilization(history_dir: str = 'my_data_and_graph/historydata'):
    """Deprecated: Use utilization_summary_grid instead."""
    pass

def plot_per_operator_utilization(history_dir: str = 'my_data_and_graph/historydata'):
    """Deprecated: Use utilization_summary_grid instead."""
    pass

def check_timeline_consistency(history_dir: str = 'my_data_and_graph/historydata', util_threshold: float = 0.6):
    """Deprecated: No-op stub."""
    pass

def plot_gradient_norms(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot gradient norms before and after clipping from diagnostics_log.txt.
    
    Shows both grad_before and grad_after on the same plot with different colors.
    Helps diagnose gradient explosion issues.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    out_path = os.path.join(plots_dir, 'gradient_norms.png')
    diagnostics_path = os.path.join(history_dir, 'diagnostics_log.txt')
    
    if not os.path.exists(diagnostics_path):
        warnings.warn(f'[plot_gradient_norms] Missing {diagnostics_path}; skipping plot')
        return
    
    try:
        # Parse diagnostics file
        # Format: timestamp,train_step,avg_q,grad_before,grad_after,target_updates,loss,td_error
        train_steps = []
        grad_before_vals = []
        grad_after_vals = []
        
        with open(diagnostics_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('[') or 'epoch_start' in line or 'epoch_end' in line:
                    continue
                parts = line.split(',')
                if len(parts) >= 5:
                    try:
                        step = int(parts[1])
                        grad_before = float(parts[3])
                        grad_after = float(parts[4])
                        train_steps.append(step)
                        grad_before_vals.append(grad_before)
                        grad_after_vals.append(grad_after)
                    except (ValueError, IndexError):
                        continue
        
        if not train_steps:
            warnings.warn(f'[plot_gradient_norms] No gradient norm data found in {diagnostics_path}')
            return
        
        # Create figure with 2 subplots stacked vertically
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # --- Top plot: Both grad_before and grad_after ---
        ax1.plot(train_steps, grad_before_vals, color='darkorange', linewidth=1.0, alpha=0.6, label='Before clipping')
        ax1.plot(train_steps, grad_after_vals, color='darkblue', linewidth=1.5, alpha=0.8, label='After clipping')
        
        # Add moving average
        if len(grad_before_vals) > 10:
            window = min(20, len(grad_before_vals)//5)
            grad_before_ma = pd.Series(grad_before_vals).rolling(window=window, center=True).mean()
            grad_after_ma = pd.Series(grad_after_vals).rolling(window=window, center=True).mean()
            ax1.plot(train_steps, grad_before_ma, color='red', linewidth=2.0, linestyle='--', alpha=0.7, label='Before MA')
            ax1.plot(train_steps, grad_after_ma, color='navy', linewidth=2.0, linestyle='--', alpha=0.7, label='After MA')
        
        # Add exploitation start marker
        eps_info = get_epsilon_min_step(history_dir)
        if eps_info:
            eps_step = eps_info['epsilon_min_step']
            ax1.axvline(x=eps_step, color='purple', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'Exploitation start (step {eps_step})')
        
        ax1.set_ylabel('Gradient Norm', fontsize=11)
        ax1.set_title('Gradient Norms: Before vs After Clipping', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visibility
        
        # --- Bottom plot: Clipping effect (before - after) ---
        clipping_effect = [b - a for b, a in zip(grad_before_vals, grad_after_vals)]
        ax2.fill_between(train_steps, 0, clipping_effect, color='crimson', alpha=0.4, label='Clipping effect')
        ax2.plot(train_steps, clipping_effect, color='darkred', linewidth=1.0, alpha=0.8)
        
        # Add moving average
        if len(clipping_effect) > 10:
            window = min(20, len(clipping_effect)//5)
            clipping_ma = pd.Series(clipping_effect).rolling(window=window, center=True).mean()
            ax2.plot(train_steps, clipping_ma, color='black', linewidth=2.0, linestyle='--', alpha=0.7, label='MA')
        
        # Add exploitation start marker
        if eps_info:
            ax2.axvline(x=eps_step, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Training Step', fontsize=11)
        ax2.set_ylabel('Clipping Effect (before - after)', fontsize=11)
        ax2.set_title('Gradient Clipping Effect', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f'[plot_gradient_norms] Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_gradient_norms] Error: {e}')
        import traceback
        traceback.print_exc()


def plot_lr_decay(history_dir: str = 'my_data_and_graph/historydata', base_lr: float = 0.0005) -> None:
    """Plot learning rate decay (adaptive) from training_metrics.csv and save as lr_decay.png."""
    plots_dir = _ensure_plots_dir(history_dir)
    out_path = os.path.join(plots_dir, 'lr_decay.png')
    metrics_path = os.path.join(history_dir, 'training_metrics.csv')
    if not os.path.exists(metrics_path):
        warnings.warn(f'[plot_lr_decay] Missing training_metrics.csv; skipping plot')
        return
    try:
        df = pd.read_csv(metrics_path)
        if 'train_step' not in df.columns or 'epsilon' not in df.columns:
            warnings.warn(f'[plot_lr_decay] Missing required columns in {metrics_path}')
            return
        steps = df['train_step'].values
        epsilons = df['epsilon'].values
        lrs = [base_lr * max(0.5, eps) for eps in epsilons]
        plt.figure(figsize=(10, 5))
        plt.plot(steps, lrs, color='blue', linewidth=1.5, label='Learning Rate')
        plt.plot(steps, [base_lr * eps for eps in epsilons], color='orange', linestyle='--', alpha=0.5, label='base_lr * epsilon')
        plt.axhline(y=base_lr * 0.5, color='red', linestyle=':', alpha=0.7, label='Min LR (base_lr * 0.5)')
        plt.xlabel('Train Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Decay (Adaptive)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f'[plot_lr_decay] Saved {out_path}', flush=True)
    except Exception as e:
        warnings.warn(f'[plot_lr_decay] Error: {e}')
        import traceback
        traceback.print_exc()
