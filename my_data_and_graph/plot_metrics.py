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


# Backwards-compatible re-exports: some callers (Runner) expect several
# utilization/makespan plotting helpers to be available from
# `my_data_and_graph.plot_metrics`. The implementation was moved to
# `reports.metrics.plot_metrics` during a refactor; import from there and
# expose the symbols. If that import fails, provide safe no-op fallbacks
# so existing code paths (Runner) don't crash.
try:
    from reports.metrics.plot_metrics import (
        plot_utilization_and_makespan,
        plot_per_machine_utilization,
        plot_per_operator_utilization,
        utilization_summary_grid,
        check_timeline_consistency,
    )
except Exception:
    # Define no-op fallbacks with the expected signatures.
    def plot_utilization_and_makespan(history_dir: str = 'my_data_and_graph/historydata'):
        return None

    def plot_per_machine_utilization(history_dir: str = 'my_data_and_graph/historydata'):
        return None

    def plot_per_operator_utilization(history_dir: str = 'my_data_and_graph/historydata'):
        return None

    def utilization_summary_grid(history_dir: str = 'my_data_and_graph/historydata', combine_plots: bool = True):
        return None

    def check_timeline_consistency(history_dir: str = 'my_data_and_graph/historydata', util_threshold: float = 0.6):
        return None


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
                df = pd.read_csv(path, header=None, comment='#', delim_whitespace=True)
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

    plt.figure(figsize=(8, 4))
    plt.plot(ser.index, ser.values, marker='o', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Step / Epoch')
    plt.ylabel('Reward')
    plt.title('Reward Trend')
    plt.tight_layout()
    try:
        plt.savefig(out_path)
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

    Expects a CSV file at <history_dir>/kpi_log.txt with columns
    (no header): epoch, reward, util_m, util_o, makespan
    """
    plots_dir = _ensure_plots_dir(history_dir)
    src = os.path.join(history_dir, 'kpi_log.txt')
    out_path = os.path.join(plots_dir, 'kpi_summary.png')

    if not os.path.exists(src):
        warnings.warn(f'[plot_kpi_summary] Missing {src}; skipping plot')
        return

    try:
        # read headerless CSV with exact column order
        df = pd.read_csv(src, sep=',', header=None)
        if df.shape[1] < 5:
            warnings.warn(f'[plot_kpi_summary] Unexpected column count in {src}; skipping plot')
            return
        df.columns = ['epoch', 'reward', 'util_m', 'util_o', 'makespan']
    except Exception as e:
        warnings.warn(f'[plot_kpi_summary] Could not read {src} as CSV: {e}')
        return

    # Prepare 2x2 subplot grid for four KPIs
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes.flatten()

    try:
        ax[0].plot(df['epoch'], df['reward'], marker='o', linewidth=1)
        ax[0].set_title('Reward Trend')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Reward')
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

        ax[3].plot(df['epoch'], df['makespan'], color='C3', linewidth=1)
        ax[3].set_title('Makespan Trend')
        ax[3].set_xlabel('Epoch')
        ax[3].set_ylabel('Makespan')
        ax[3].grid(True, alpha=0.3)

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
    try:
        print('[plot_metrics] All plots generated successfully.')
    except Exception:
        pass
