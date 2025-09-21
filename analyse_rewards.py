import matplotlib.pyplot as plt
import numpy as np
import os

# === File paths ===
reward_file = "./my_data_and_graph/historydata/episode_rewards.txt"
loss_file = "./my_data_and_graph/historydata/loss.txt"
time_file = "./my_data_and_graph/historydata/times.txt"

save_dir = "./my_data_and_graph/historydata/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# === Helper: smooth values (with edge fix) ===
def smooth(y, box_pts=50):
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="valid")
    pad = (len(y) - len(y_smooth)) // 2
    y_smooth = np.concatenate([
        [np.nan] * pad,
        y_smooth,
        [np.nan] * (len(y) - len(y_smooth) - pad)
    ])
    return y_smooth


# === Plot 1: Reward Curve ===
def plot_rewards():
    if not os.path.exists(reward_file):
        print("[WARN] No reward file found.")
        return
    data = np.loadtxt(reward_file, delimiter=",")
    if data.ndim == 1:
        episodes = [int(data[0])]
        rewards = [float(data[1])]
    else:
        episodes = data[:, 0]
        rewards = data[:, 1]

    # --- Normal Reward Curve ---
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, rewards, label="Raw (global per-episode reward)", alpha=0.5)
    if len(rewards) >= 50:
        plt.plot(episodes, smooth(rewards, 50),
                 label="Smoothed", color="red")

    plt.title("Reward Curve (Global reward per episode)")
    plt.xlabel("Episode Index")
    plt.ylabel("Global Reward (average per agent)")
    plt.legend()
    plt.grid(True)
    # PlotFixText: normal reward curve, shows all values including outliers
    plt.figtext(
        0.5, -0.08,
        "Note: Rewards are averaged per agent.\n"
        "Smoothed curve uses moving average with window=50 episodes.",
        wrap=True, ha="center", fontsize=9
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "rewards_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Reward curve saved to {save_path}")

    # --- Zoomed Reward Curve ---
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, rewards, label="Raw (global per-episode reward)", alpha=0.5)
    if len(rewards) >= 50:
        plt.plot(episodes, smooth(rewards, 50),
                 label="Smoothed", color="red")

    # Focus on 5th–95th percentile range
    low, high = np.percentile(rewards, 5), np.percentile(rewards, 95)
    plt.ylim(low, high)

    plt.title("Reward Curve (Zoomed)")
    plt.xlabel("Episode Index")
    plt.ylabel("Global Reward (average per agent)")
    plt.legend()
    plt.grid(True)
    # PlotFixText: zoomed reward curve, focuses on stable region by removing outliers
    plt.figtext(
        0.5, -0.08,
        "Zoomed view: rewards between 5th and 95th percentile.\n"
        "Highlights main convergence trend without extreme outliers.",
        wrap=True, ha="center", fontsize=9
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "rewards_curve_zoomed.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Reward curve (zoomed) saved to {save_path}")

    # Print analysis
    avg_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    print("=== Training Reward Analysis ===")
    print(f"Total episodes logged: {len(rewards)}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")


# === Plot 2: Loss Curve ===
def plot_loss():
    if not os.path.exists(loss_file):
        print("[WARN] No loss file found.")
        return

    def clean_loss_line(line):
        line = line.strip()
        if line.startswith("tensor("):
            line = line.replace("tensor(", "").split(",")[0]
        return line

    with open(loss_file, "r") as f:
        cleaned_lines = [clean_loss_line(l) for l in f if l.strip() != ""]

    try:
        data = np.array([float(x) for x in cleaned_lines])
    except ValueError as e:
        print("[ERROR] Could not parse loss file:", e)
        return

    steps = np.arange(len(data))

    # --- Normal loss curve ---
    plt.figure(figsize=(12, 7))
    plt.plot(steps, data, label="Raw loss (per mini-batch update)", alpha=0.5)
    if len(data) >= 50:
        plt.plot(steps, smooth(data, 50),
                 label="Smoothed", color="red")
    plt.title("Loss Curve")
    plt.xlabel("Training Step (mini-batch updates)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    # PlotFixText: loss curve over all steps, raw + smoothed
    plt.figtext(
        0.5, -0.08,
        "One training step = one mini-batch update "
        "(not to be confused with environment episode steps).",
        wrap=True, ha="center", fontsize=9
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Loss curve saved to {save_path}")

    # --- Zoomed-in loss curve ---
    plt.figure(figsize=(12, 7))
    plt.plot(steps, data, label="Raw loss (per mini-batch update)", alpha=0.5)
    if len(data) >= 50:
        plt.plot(steps, smooth(data, 50),
                 label="Smoothed (50-update avg)", color="red")
    plt.ylim(0, np.percentile(data, 95))
    plt.title("Loss Curve (Zoomed In)")
    plt.xlabel("Training Step (mini-batch updates)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    # PlotFixText: zoomed loss curve, hides extreme outliers
    plt.figtext(
        0.5, -0.08,
        "Zoomed view: y-limit set to 95th percentile.\n"
        "Removes extreme outliers to highlight main trend.",
        wrap=True, ha="center", fontsize=9
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "loss_curve_zoomed.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Loss curve (zoomed) saved to {save_path}")


# === Plot 3: Training Duration per Episode ===
def plot_training_time():
    if not os.path.exists(time_file):
        print("[WARN] No times file found.")
        return
    times = np.loadtxt(time_file, delimiter=",")[:, 1]  # second column = duration
    episodes = np.arange(len(times))

    plt.figure(figsize=(12, 7))
    plt.plot(episodes, times, label="Raw episode duration", alpha=0.5)
    if len(times) >= 50:
        plt.plot(episodes, smooth(times, 50),
                 label="Smoothed", color="red")

    plt.title("Training Duration per Episode")
    plt.xlabel("Episode Index")
    plt.ylabel("Episode Duration (steps)")
    plt.legend()
    plt.grid(True)
    # PlotFixText: training duration = makespan estimate per episode
    plt.figtext(
        0.5, -0.08,
        "Logged value = sum(episode_time_slice) + max(state_left_time)\n"
        "= elapsed time + longest remaining job duration.\n"
        "Interpretation: estimated makespan if jobs continue unchanged.",
        wrap=True, ha="center", fontsize=9
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_time.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Training time curve saved to {save_path}")


# === Main ===
if __name__ == "__main__":
    plot_rewards()
    plot_loss()
    plot_training_time()
