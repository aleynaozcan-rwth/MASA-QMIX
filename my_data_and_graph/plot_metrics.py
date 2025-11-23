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
    """Plot reward trend from reward_log.txt and save reward_trend.png.

    Expects a simple log file ./history_dir/reward_log.txt containing numeric
    reward values (one per line) or two-column timestamp/value pairs.
    """
    plots_dir = _ensure_plots_dir(history_dir)
    src = os.path.join(history_dir, 'episode_rewards.txt')
    ser = _read_series_file(src)
    out_path = os.path.join(plots_dir, 'reward_trend.png')

    if ser is None:
        warnings.warn(f'[plot_reward_trend] Missing or unreadable {src}; skipping plot')
        return

    plt.figure(figsize=(10, 5))
    
    # Raw data with transparency
    plt.plot(ser.index, ser.values, marker='o', linewidth=0.5, markersize=2, 
             alpha=0.3, label='Raw reward', color='lightblue')
    
    # Moving average (window=50)
    if len(ser) > 50:
        ma_50 = ser.rolling(window=50, min_periods=1).mean()
        plt.plot(ser.index, ma_50, linewidth=2, label='MA(50)', color='blue')
    
    # Moving average (window=100) for longer runs
    if len(ser) > 100:
        ma_100 = ser.rolling(window=100, min_periods=1).mean()
        plt.plot(ser.index, ma_100, linewidth=2.5, label='MA(100)', color='darkblue')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Step / Epoch')
    plt.ylabel('Reward')
    plt.title('Reward Trend (with Moving Averages)')
    plt.legend(loc='best')
    plt.tight_layout()
    try:
        plt.savefig(out_path, dpi=150)
    except Exception as e:
        warnings.warn(f'[plot_reward_trend] Could not write {out_path}: {e}')
    finally:
        plt.close()


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


def plot_reward_components(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot reward component breakdown over simulation time.

    Expects CSV file at <history_dir>/reward_components_log.txt with rows
    env_time,CompletedNorm,AvgWait,WIP,ThroughputDelta,LoadVariance,R_global
    (no header).
    """
    plots_dir = _ensure_plots_dir(history_dir)
    src = os.path.join(history_dir, 'reward_components_log.txt')
    out_path = os.path.join(plots_dir, 'reward_components.png')

    if not os.path.exists(src):
        warnings.warn(f'[plot_reward_components] Missing {src}; skipping plot')
        return

    try:
        # Try reading with or without header
        try:
            df = pd.read_csv(src, header=None)
            if df.shape[1] >= 7:
                df = df.iloc[:, :7]
                df.columns = ['env_time', 'CompletedNorm', 'AvgWait', 'WIP', 'ThroughputDelta', 'LoadVariance', 'R_global']
            else:
                # maybe file has header row
                df = pd.read_csv(src)
        except Exception:
            df = pd.read_csv(src)
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


# Convenience function to run all plots
def generate_all_plots(history_dir: str = 'my_data_and_graph/historydata') -> None:
    plot_reward_trend(history_dir)
    plot_loss_trend(history_dir)
    plot_td_error_trend(history_dir)
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
