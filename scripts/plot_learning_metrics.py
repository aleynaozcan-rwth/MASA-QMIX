#!/usr/bin/env python3
"""
Plot learning metrics from my_data_and_graph/historydata/learning_metrics.csv*

Usage:
  PYTHONPATH=. python3 scripts/plot_learning_metrics.py

The script will automatically pick the most-recent file whose name starts with
`learning_metrics.csv` in `my_data_and_graph/historydata/`, read numeric columns
if present (episode, episode_reward, last_loss, last_td, epsilon, q_value_mean)
and produce `plots/learning_trends.png`.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import sys
from typing import List, Dict, Tuple

import numpy as np
import matplotlib

# Use a non-interactive backend so the script can run in headless CI/machines
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HIST_DIR = os.path.join("my_data_and_graph", "historydata")
PLOT_DIR = "plots"
OUT_PNG = os.path.join(PLOT_DIR, "learning_trends.png")


def find_latest_metrics_file(hist_dir: str) -> str:
    pattern = os.path.join(hist_dir, "learning_metrics.csv*")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No metrics files matching 'learning_metrics.csv*' in {hist_dir}")
    latest = max(candidates, key=os.path.getmtime)
    return latest


def read_csv_floats(path: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Read CSV at path and return (fieldnames, dict of column->np.array floats).

    Non-numeric values are converted to np.nan. Missing columns will not be present.
    """
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        if not rows:
            return reader.fieldnames or [], {}
        fieldnames = reader.fieldnames or []
        cols: Dict[str, List[float]] = {fn: [] for fn in fieldnames}
        for r in rows:
            for fn in fieldnames:
                v = r.get(fn, "")
                if v is None or v == "":
                    cols[fn].append(np.nan)
                    continue
                # try to parse numeric-like values (strip commas)
                try:
                    # handle strings like '1.23e-4' or '123'
                    fv = float(v.replace(",", ""))
                except Exception:
                    fv = np.nan
                cols[fn].append(fv)

    # convert to numpy arrays
    arrs: Dict[str, np.ndarray] = {k: np.array(v, dtype=float) for k, v in cols.items()}
    return fieldnames, arrs


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    # use nan-aware moving average
    ret = np.full_like(x, np.nan, dtype=float)
    cumsum = np.nancumsum(np.nan_to_num(x, nan=0.0))
    counts = np.cumsum(~np.isnan(x)).astype(float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        s = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        n = counts[i] - (counts[start - 1] if start > 0 else 0.0)
        ret[i] = s / n if n > 0 else np.nan
    return ret


def summarize_reward(rewards: np.ndarray, window: int = 10) -> str:
    if rewards.size == 0:
        return "No reward data"
    last = rewards[-1]
    window = max(1, min(window, rewards.size))
    ma = moving_average(rewards, window)
    last_ma = ma[-1]
    # compute a simple trend: compare mean of last window vs first window
    first_ma = np.nanmean(rewards[:window]) if rewards.size >= window else np.nanmean(rewards)
    trend = "stable"
    if not math.isnan(first_ma) and not math.isnan(last_ma):
        if last_ma > first_ma * 1.05:
            trend = "increasing"
        elif last_ma < first_ma * 0.95:
            trend = "decreasing"
    return f"Final reward={last:.3f}, {window}-step MA={last_ma:.3f}, trend={trend}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_metrics(fieldnames: List[str], data: Dict[str, np.ndarray], out_png: str) -> None:
    # Try to find common column names
    # episode: try 'episode' or 'ep'
    ep_keys = [k for k in fieldnames if k.lower() in ("episode", "ep", "episodes")]
    x = None
    if ep_keys:
        x = data[ep_keys[0]]
    else:
        # fallback to index
        # create x as 1..N
        any_col = next(iter(data.values())) if data else np.array([])
        x = np.arange(1, len(any_col) + 1)

    # reward keys
    reward_keys = [k for k in fieldnames if k.lower() in ("episode_reward", "reward", "ep_reward")]
    reward = data[reward_keys[0]] if reward_keys else (data.get("episode_reward") or None)
    if reward is None or len(reward) == 0:
        # try common alternate
        reward = data.get("episode_reward") if "episode_reward" in data else None

    # loss and td
    loss_keys = [k for k in fieldnames if k.lower() in ("last_loss", "loss")]
    td_keys = [k for k in fieldnames if k.lower() in ("last_td", "td", "td_error", "td_loss")]
    epsilon_keys = [k for k in fieldnames if k.lower() in ("epsilon", "eps")]
    q_keys = [k for k in fieldnames if k.lower() in ("q_value_mean", "q_mean", "q_value")]

    loss = data[loss_keys[0]] if loss_keys else None
    td = data[td_keys[0]] if td_keys else None
    eps = data[epsilon_keys[0]] if epsilon_keys else None
    qv = data[q_keys[0]] if q_keys else None

    # Prepare plotting layout: 2 rows, maybe 3rd if optional
    n_plots = 2 + (1 if (eps is not None or qv is not None) else 0)
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), squeeze=False)
    axs = axs.flatten()

    # Plot reward
    ax = axs[0]
    if reward is not None:
        ax.plot(x, reward, label="episode_reward", color="tab:blue", alpha=0.6)
        win = max(1, min(10, max(1, int(len(reward) / 20))))
        ma = moving_average(reward, window=win)
        ax.plot(x, ma, label=f"MA({win})", color="tab:orange")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Episode Reward vs Episode")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No episode_reward column found", ha="center", va="center")
        ax.set_title("Episode Reward")

    # Plot TD or loss
    ax2 = axs[1]
    if td is not None:
        ax2.plot(x, td, label="TD (last_td)", color="tab:red")
        ax2.set_ylabel("TD")
        ax2.set_title("TD Error / TD Loss vs Episode")
        ax2.legend(loc="best")
    elif loss is not None:
        ax2.plot(x, loss, label="Loss (last_loss)", color="tab:green")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss vs Episode")
        ax2.legend(loc="best")
    else:
        ax2.text(0.5, 0.5, "No TD or Loss column found", ha="center", va="center")
        ax2.set_title("TD / Loss")

    # optional third plot for epsilon or q-values
    if n_plots > 2:
        ax3 = axs[2]
        if eps is not None:
            ax3.plot(x, eps, label="epsilon", color="tab:purple")
            ax3.set_ylabel("Epsilon")
            ax3.set_title("Epsilon vs Episode")
            ax3.legend(loc="best")
        elif qv is not None:
            ax3.plot(x, qv, label="q_value_mean", color="tab:brown")
            ax3.set_ylabel("Q value mean")
            ax3.set_title("Q Value Mean vs Episode")
            ax3.legend(loc="best")

    for a in axs:
        a.set_xlabel("Episode")
        a.grid(True, alpha=0.3)

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_png) or ".")
    fig.savefig(out_png)
    print(f"Saved plot to {out_png}")


def main() -> None:
    try:
        metrics = find_latest_metrics_file(HIST_DIR)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(2)

    print(f"Reading metrics from: {metrics}")
    fieldnames, data = read_csv_floats(metrics)

    if not data:
        print("No numeric data parsed from metrics file.")
        sys.exit(3)

    # Print a short summary
    # try to find reward column
    reward_keys = [k for k in fieldnames if k.lower() in ("episode_reward", "reward", "ep_reward")]
    if reward_keys:
        rewards = data[reward_keys[0]]
        summary = summarize_reward(rewards, window=10)
        print(summary)
    else:
        print("No episode_reward column found for summary.")

    # plot
    ensure_dir(PLOT_DIR)
    plot_metrics(fieldnames, data, OUT_PNG)


if __name__ == "__main__":
    main()
