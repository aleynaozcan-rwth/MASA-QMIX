import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

HIST_DIR = "./my_data_and_graph/historydata"

# SLURM Job ID (for logging context)
JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def analyse_rewards(file_path, label="Training"):
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    try:
        rewards_df = pd.read_csv(file_path, header=None)
        if rewards_df.shape[1] == 2:
            episodes = rewards_df.iloc[:, 0].astype(int).tolist()
            rewards = rewards_df.iloc[:, 1].astype(float).tolist()
        else:
            rewards = pd.to_numeric(
                rewards_df.squeeze("columns"), errors="coerce"
            ).dropna().tolist()
            episodes = np.arange(len(rewards))
    except Exception as e:
        print(f"[!] Error reading {file_path}: {e}")
        return

    if len(rewards) == 0:
        print(f"[!] No rewards data in {file_path}")
        return

    smoothed = pd.Series(rewards).rolling(50, min_periods=1).mean()

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, color="lightblue", alpha=0.4, label="Raw (global per-episode reward)")
    plt.plot(episodes, smoothed, color="red", linewidth=2, label="Smoothed (50-episode avg)")
    plt.title("Reward Curve (Global reward averaged per episode)")
    plt.xlabel("Episode Index")
    plt.ylabel("Reward (global, already averaged over agents)")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(HIST_DIR, "rewards_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Reward curve saved to {out_path}")

    print(f"=== {label} Reward Analysis (JobID: {JOB_ID}) ===")
    print(f"Total episodes logged: {len(rewards)}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")


def analyse_times(file_path):
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    try:
        times = pd.read_csv(file_path, header=None).squeeze("columns")
        times = pd.to_numeric(times, errors="coerce").dropna().tolist()
    except Exception as e:
        print(f"[!] Error reading {file_path}: {e}")
        return

    if len(times) == 0:
        print(f"[!] No time data in {file_path}")
        return

    episodes = np.arange(len(times))
    smoothed = pd.Series(times).rolling(50, min_periods=1).mean()

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, times, color="lightblue", alpha=0.4, label="Raw episode duration")
    plt.plot(episodes, smoothed, color="red", linewidth=2, label="Smoothed (50-episode avg)")
    plt.title("Training Duration per Episode")
    plt.xlabel("Episode Index")
    plt.ylabel("Episode Duration (steps)")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(HIST_DIR, "training_time.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Training time curve saved to {out_path} (JobID: {JOB_ID})")


def analyse_loss(file_path, zoom=False):
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    try:
        losses = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("tensor("):
                    try:
                        val = float(line.split("(")[1].split(",")[0])
                        losses.append(val)
                    except:
                        continue
                elif line != "":
                    try:
                        losses.append(float(line))
                    except:
                        continue
    except Exception as e:
        print(f"[!] Error reading {file_path}: {e}")
        return

    if len(losses) == 0:
        print(f"[!] No loss data in {file_path}")
        return

    steps = np.arange(len(losses))
    smoothed = pd.Series(losses).rolling(50, min_periods=1).mean()

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, color="lightblue", alpha=0.4, label="Raw loss (per mini-batch update)")
    plt.plot(steps, smoothed, color="red", linewidth=2, label="Smoothed (50-update avg)")
    plt.title("Loss Curve")
    plt.xlabel("Training Step (mini-batch updates)")
    plt.ylabel("Loss")

    if zoom:
        plt.ylim(0, np.percentile(losses, 95))  # zoom in to 95th percentile

    plt.legend()
    plt.grid(True)

    out_path = os.path.join(HIST_DIR, "loss_curve.png")
    if zoom:
        out_path = out_path.replace(".png", "_zoomed.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Loss curve saved to {out_path} (JobID: {JOB_ID})")


if __name__ == "__main__":
    rewards_file = os.path.join(HIST_DIR, "episode_rewards.txt")
    times_file = os.path.join(HIST_DIR, "times.txt")
    loss_file = os.path.join(HIST_DIR, "loss.txt")

    analyse_rewards(rewards_file, "Training")
    analyse_times(times_file)
    analyse_loss(loss_file, zoom=True)  # zoom=True enables zoomed-in version
