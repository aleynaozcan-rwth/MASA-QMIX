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
    
    # Read learning metrics CSV
    metrics_path = os.path.join(history_dir, 'learning_metrics.csv')
    if not os.path.exists(metrics_path):
        warnings.warn(f'[plot_reward_trend] Missing {metrics_path}; skipping')
        return
    
    try:
        # Load episode rewards
        df = pd.read_csv(metrics_path)
        if 'episode_reward' not in df.columns or 'episode' not in df.columns:
            warnings.warn(f'[plot_reward_trend] Missing required columns in {metrics_path}')
            return
        
        episodes = df['episode'].values
        rewards = df['episode_reward'].values
        
        # Estimate training steps: assume ~50 training steps per episode (4 episodes per epoch)
        # This is approximate - real correlation would need exact mapping
        # For better visualization, we use episode count * 50 as proxy for training steps
        training_steps = episodes * 50
        
        if len(rewards) == 0:
            warnings.warn(f'[plot_reward_trend] No reward data in {metrics_path}')
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(training_steps, rewards, color='blue', linewidth=1, alpha=0.5, label='Episode Reward')
        
        # Add moving average
        if len(rewards) > 10:
            window = min(20, len(rewards) // 5)
            reward_series = pd.Series(rewards)
            reward_rolling = reward_series.rolling(window=window, center=True).mean()
            plt.plot(training_steps, reward_rolling, color='red', linewidth=2.5, label=f'{window}-episode MA')
        
        # Add mean line
        mean_reward = np.mean(rewards)
        plt.axhline(y=mean_reward, color='green', linestyle='--', alpha=0.5, label=f'Mean: {mean_reward:.3f}')
        
        # Annotations
        if len(rewards) > 0:
            first_reward = rewards[0]
            last_reward = rewards[-1]
            plt.axhline(y=first_reward, color='purple', linestyle=':', alpha=0.3, label=f'Initial: {first_reward:.3f}')
            plt.axhline(y=last_reward, color='orange', linestyle=':', alpha=0.3, label=f'Final: {last_reward:.3f}')
        
        plt.xlabel('Approximate Training Step', fontsize=12)
        plt.ylabel('Episode Reward', fontsize=12)
        plt.title('Episode Reward Evolution During Training', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f'[plot_reward_trend] Saved {out_path}')
        
    except Exception as e:
        warnings.warn(f'[plot_reward_trend] Error: {e}')
        import traceback
        traceback.print_exc()


def plot_loss_trend(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot loss trend from loss_log.txt and save loss_trend.png.

    Expects a simple log file ./history_dir/loss_log.txt containing numeric
    loss values (one per line) or two-column index/value pairs.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    src = os.path.join(history_dir, 'loss.txt')
    ser = _read_series_file(src)
    out_path = os.path.join(plots_dir, 'loss_trend.png')

    if ser is None:
        warnings.warn(f'[plot_loss_trend] Missing or unreadable {src}; skipping plot')
        return

    plt.figure(figsize=(8, 4))
    plt.plot(ser.index, ser.values, color='C1', linewidth=1)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Training Step')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Trend')
    plt.tight_layout()
    try:
        plt.savefig(out_path)
    except Exception as e:
        warnings.warn(f'[plot_loss_trend] Could not write {out_path}: {e}')
    finally:
        plt.close()


def plot_td_error_trend(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot TD-error trend from td_error_log.txt and save td_error_trend.png.

    Expects simple numeric log similar to loss_log.txt.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    src = os.path.join(history_dir, 'td_error.txt')
    ser = _read_series_file(src)
    out_path = os.path.join(plots_dir, 'td_error_trend.png')

    if ser is None:
        warnings.warn(f'[plot_td_error_trend] Missing or unreadable {src}; skipping plot')
        return

    plt.figure(figsize=(8, 4))
    plt.plot(ser.index, ser.values, color='C2', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Training Step')
    plt.ylabel('TD Error')
    plt.title('TD Error Trend')
    plt.tight_layout()
    try:
        plt.savefig(out_path)
    except Exception as e:
        warnings.warn(f'[plot_td_error_trend] Could not write {out_path}: {e}')
    finally:
        plt.close()


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
        ax[0].plot(df['epoch'], df['avg_wait'], marker='o', linewidth=1, color='C0')
        ax[0].set_title('Avg Wait Time Trend')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Avg Wait Time')
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(df['epoch'], df['util_m'], color='C1', linewidth=1)
        ax[1].set_title('Machine Utilization Trend')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Machine Util')
        ax[1].grid(True, alpha=0.3)

        ax[2].plot(df['epoch'], df['util_o'], color='C2', linewidth=1)
        ax[2].set_title('Operator Utilization Trend')
        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('Operator Util')
        ax[2].grid(True, alpha=0.3)

        # Row 2: Makespan, Loss, TD Error
        ax[3].plot(df['epoch'], df['makespan'], color='C3', linewidth=1)
        ax[3].set_title('Makespan Trend')
        ax[3].set_xlabel('Epoch')
        ax[3].set_ylabel('Makespan')
        ax[3].grid(True, alpha=0.3)

        if loss_series is not None and len(loss_series) > 0:
            ax[4].plot(loss_series.index, loss_series.values, color='C4', linewidth=1)
            ax[4].set_yscale('log')
            ax[4].set_title('Loss Trend (Training Steps)')
            ax[4].set_xlabel('Gradient Update Step')
            ax[4].set_ylabel('Loss (log scale)')
            ax[4].grid(True, alpha=0.3)
        else:
            ax[4].text(0.5, 0.5, 'Loss data not available', ha='center', va='center', transform=ax[4].transAxes)
            ax[4].set_title('Loss Trend')

        if td_series is not None and len(td_series) > 0:
            ax[5].plot(td_series.index, td_series.values, color='C5', linewidth=1)
            ax[5].set_title('TD Error Trend (Training Steps)')
            ax[5].set_xlabel('Gradient Update Step')
            ax[5].set_ylabel('TD Error')
            ax[5].grid(True, alpha=0.3)
        else:
            ax[5].text(0.5, 0.5, 'TD Error data not available', ha='center', va='center', transform=ax[5].transAxes)
            ax[5].set_title('TD Error Trend')

        plt.tight_layout()
        try:
            plt.savefig(out_path)
            try:
                print(f'[plot_kpi_summary] Saved {out_path}')
            except Exception:
                pass
        except Exception as e:
            warnings.warn(f'[plot_kpi_summary] Could not write {out_path}: {e}')
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
            ax0.plot(x, vals, marker='o', label=f"M{mid}", color=cmap(i % 10))
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
            ax1.plot(x, vals, marker='o', label=f"O{oid}", color=cmap(i % 10))
        ax1.set_title('Per-Operator Utilization')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Utilization')
        ax1.set_ylim(0,1)
        ax1.grid(True)
        ax1.legend(loc='best', fontsize='x-small')

        ax2 = axes[1,0]
        ax2.plot(x, makespans, marker='o', color='tab:green')
        ax2.set_title('Average Makespan')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Makespan')
        ax2.grid(True)

        ax3 = axes[1,1]
        # plot avg rewards if present
        if any(v is not None for v in avg_rewards):
            rr = [v if v is not None else float('nan') for v in avg_rewards]
            ax3.plot(x, rr, marker='o', color='tab:purple')
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
        print(f"[utilization_summary_grid] Saved {out_path}")
    except Exception as e:
        print(f"[WARN] Could not save utilization_summary_grid.png: {e}")


def plot_q_value_trend(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot Q-value evolution from diagnostics log.
    
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
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_steps, q_values, color='darkgreen', linewidth=1.5, alpha=0.7)
        
        # Add moving average
        if len(q_values) > 10:
            window = min(20, len(q_values) // 5)
            q_series = pd.Series(q_values)
            q_rolling = q_series.rolling(window=window, center=True).mean()
            plt.plot(train_steps, q_rolling, color='red', linewidth=2.5, label=f'{window}-step MA')
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        # Annotations
        if q_values:
            first_q = q_values[0]
            last_q = q_values[-1]
            plt.axhline(y=first_q, color='blue', linestyle=':', alpha=0.3, label=f'Initial: {first_q:.0f}')
            plt.axhline(y=last_q, color='orange', linestyle=':', alpha=0.3, label=f'Final: {last_q:.0f}')
        
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Average Q-Value', fontsize=12)
        plt.title('Q-Value Evolution During Training', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f'[plot_q_value_trend] Saved {out_path}')
        
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
    metrics_path = os.path.join(history_dir, 'learning_metrics.csv')
    kpi_path = os.path.join(history_dir, 'kpi_log.txt')
    
    if not os.path.exists(metrics_path):
        warnings.warn(f'[plot_learning_analysis_comprehensive] Missing {metrics_path}; skipping')
        return
    
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
            ax.plot(kpi['epoch'], rolling_makespan, color='darkgreen', linewidth=2, label='10-epoch MA')
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
        print(f'[plot_learning_analysis_comprehensive] Saved {out_path}')
        
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

    plt.figure(figsize=(10, 5))
    try:
        components = ['CompletedNorm', 'AvgWait', 'WIP', 'ThroughputDelta', 'LoadVariance', 'R_global']
        for comp in components:
            if comp in df.columns:
                plt.plot(df['env_time'].values, pd.to_numeric(df[comp], errors='coerce').fillna(0.0).values, label=comp, linewidth=1)

        plt.grid(True, alpha=0.3)
        plt.xlabel('Env time')
        plt.ylabel('Value')
        plt.title('Reward Components over Time')
        plt.legend(loc='best')
        plt.tight_layout()
        try:
            plt.savefig(out_path)
            try:
                print(f'[plot_reward_components] Saved {out_path}')
            except Exception:
                pass
        except Exception as e:
            warnings.warn(f'[plot_reward_components] Could not write {out_path}: {e}')
    finally:
        plt.close()


def plot_epsilon_decay(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot epsilon decay over training steps with exploration/exploitation phases.
    
    Reads epsilon values from debug_output.txt ([EPSILON_DECAY] and [Diagnostics] lines)
    and creates a comprehensive plot showing:
    - Epsilon decay curve
    - Phase transitions (exploration → exploitation)
    - Key milestones (epsilon thresholds)
    """
    plots_dir = _ensure_plots_dir(history_dir)
    out_path = os.path.join(plots_dir, 'epsilon_decay.png')
    
    debug_file = os.path.join(history_dir, 'debug_output.txt')
    if not os.path.exists(debug_file):
        warnings.warn(f'[plot_epsilon_decay] Missing {debug_file}; skipping plot')
        return
    
    try:
        import re
        
        # Parse epsilon values from debug_output.txt
        epsilon_data = []
        
        with open(debug_file, 'r') as f:
            for line in f:
                # Match [EPSILON_DECAY] step=X, epsilon=Y
                match = re.search(r'\[EPSILON_DECAY\] step=(\d+), epsilon=([\d.]+)', line)
                if match:
                    step = int(match.group(1))
                    eps = float(match.group(2))
                    epsilon_data.append((step, eps))
                    continue
                
                # Match [Diagnostics] Epoch X start | epsilon=Y
                match = re.search(r'Epoch (\d+) start \| epsilon=([\d.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    eps = float(match.group(2))
                    # Approximate step from epoch (assuming ~50 steps/epoch)
                    step = epoch * 50
                    epsilon_data.append((step, eps))
        
        if not epsilon_data:
            warnings.warn('[plot_epsilon_decay] No epsilon data found')
            return
        
        # Convert to arrays
        epsilon_data = sorted(set(epsilon_data))  # Remove duplicates
        steps = np.array([x[0] for x in epsilon_data])
        epsilons = np.array([x[1] for x in epsilon_data])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Main epsilon curve
        ax.plot(steps, epsilons, linewidth=2, color='#2E86AB', label='Epsilon')
        ax.fill_between(steps, epsilons, alpha=0.3, color='#2E86AB')
        
        # Phase annotations
        # High exploration: epsilon > 0.5
        # Medium exploration: 0.2 < epsilon <= 0.5
        # Low exploration: 0.05 < epsilon <= 0.2
        # Exploitation: epsilon = 0.05
        
        high_exp = epsilons > 0.5
        if high_exp.any():
            ax.axhspan(0.5, 1.0, alpha=0.1, color='red', label='High Exploration (ε>0.5)')
        
        med_exp = (epsilons > 0.2) & (epsilons <= 0.5)
        if med_exp.any():
            ax.axhspan(0.2, 0.5, alpha=0.1, color='orange', label='Medium Exploration (0.2<ε≤0.5)')
        
        low_exp = (epsilons > 0.05) & (epsilons <= 0.2)
        if low_exp.any():
            ax.axhspan(0.05, 0.2, alpha=0.1, color='yellow', label='Low Exploration (0.05<ε≤0.2)')
        
        # Mark epsilon=0.05 (exploitation phase)
        ax.axhline(y=0.05, color='green', linestyle='--', linewidth=2, 
                   label='Min Epsilon (0.05) - Exploitation')
        
        # Find when epsilon reaches 0.05
        exploitation_start = steps[epsilons <= 0.051].min() if (epsilons <= 0.051).any() else None
        if exploitation_start is not None:
            ax.axvline(x=exploitation_start, color='green', linestyle=':', alpha=0.5)
            ax.text(exploitation_start, 0.5, f'  Exploitation\n  starts ~step {exploitation_start}',
                   rotation=0, va='center', ha='left', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        
        # Mark key milestones
        milestones = [0.9, 0.5, 0.2, 0.1]
        for milestone in milestones:
            milestone_steps = steps[(epsilons >= milestone - 0.01) & (epsilons <= milestone + 0.01)]
            if len(milestone_steps) > 0:
                step_at_milestone = milestone_steps[0]
                ax.plot(step_at_milestone, milestone, 'ro', markersize=8, alpha=0.6)
                ax.annotate(f'ε≈{milestone:.1f}\nstep {step_at_milestone}',
                           xy=(step_at_milestone, milestone),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Epsilon (ε)', fontsize=12)
        ax.set_title('Epsilon Decay: Exploration → Exploitation Transition', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        
        # Add statistics box
        final_eps = epsilons[-1]
        initial_eps = epsilons[0]
        stats_text = f"""Training Stats:
Initial ε: {initial_eps:.4f}
Final ε: {final_eps:.4f}
Total steps: {steps[-1]}
Decay rate: {(initial_eps - final_eps) / steps[-1] * 1000:.6f} per 1k steps"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[plot_epsilon_decay] Saved {out_path}')
        
    except Exception as e:
        warnings.warn(f'[plot_epsilon_decay] Error: {e}')
        import traceback
        traceback.print_exc()


# Convenience function to run all plots
def generate_all_plots(history_dir: str = 'my_data_and_graph/historydata') -> None:
    plot_reward_trend(history_dir)
    plot_loss_trend(history_dir)
    plot_td_error_trend(history_dir)
    plot_q_value_trend(history_dir)  # New Q-value plot!
    plot_epsilon_decay(history_dir)  # New epsilon decay plot!
    plot_kpi_summary(history_dir)
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
        print('[plot_metrics] All plots generated successfully.')
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
