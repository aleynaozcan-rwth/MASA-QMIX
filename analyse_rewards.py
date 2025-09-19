import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

HIST_DIR = "./my_data_and_graph/historydata"

def analyse_rewards(file_path, label="Training"):
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    try:
        # Try reading as CSV: (episode_idx, reward)
        rewards_df = pd.read_csv(file_path, header=None)
        if rewards_df.shape[1] == 2:
            episodes = rewards_df.iloc[:, 0].astype(int).tolist()
            rewards = rewards_df.iloc[:, 1].astype(float).tolist()
        else:
            rewards = pd.to_numeric(rewards_df.squeeze("columns"), errors="coerce").dropna().tolist()
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
    plt.plot(episodes, rewards, color="lightblue", alpha=0.4, label="Raw Reward")
    plt.plot(episodes, smoothed, color="red", linewidth=2, label="Smoothed (50 ep)")
    plt.title("QMIX Reward Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(HIST_DIR, "rewards_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Reward curve saved to {out_path}")

    print(f"=== {label} Reward Analysis ===")
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

    runs = np.arange(len(times))
    smoothed = pd.Series(times).rolling(50, min_periods=1).mean()

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(runs, times, color="lightblue", alpha=0.4, label="Raw Time")
    plt.plot(runs, smoothed, color="red", linewidth=2, label="Smoothed (50)")
    plt.title("Training Duration per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(HIST_DIR, "training_time.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Training time curve saved to {out_path}")


def analyse_loss(file_path):
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    try:
        losses = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("tensor("):
                    # "tensor(3.6622, grad_fn=<...>)" → 3.6622
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
    plt.plot(steps, losses, color="lightblue", alpha=0.4, label="Raw Loss")
    plt.plot(steps, smoothed, color="red", linewidth=2, label="Smoothed (50)")
    plt.title("Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(HIST_DIR, "loss_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Loss curve saved to {out_path}")


if __name__ == "__main__":
    rewards_file = os.path.join(HIST_DIR, "episode_rewards.txt")  # 2-column (episode,reward)
    times_file = os.path.join(HIST_DIR, "times.txt")
    loss_file = os.path.join(HIST_DIR, "loss.txt")

    analyse_rewards(rewards_file, "Training")
    analyse_times(times_file)
    analyse_loss(loss_file)
