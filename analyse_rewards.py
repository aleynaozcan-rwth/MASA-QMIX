import matplotlib.pyplot as plt
import numpy as np
import os
from MARL.common.terms import t  # Step 8A terminology layer

# === File paths ===
reward_file = "./my_data_and_graph/historydata/episode_rewards.txt"
loss_file   = "./my_data_and_graph/historydata/loss.txt"
time_file   = "./my_data_and_graph/historydata/times.txt"
wait_file   = "./my_data_and_graph/historydata/waittimes.txt"

save_dir = "./my_data_and_graph/historydata/"
os.makedirs(save_dir, exist_ok=True)

# === Helper: smooth values with adaptive moving average ===
def smooth(y, box_pts=None):
    if len(y) < 3:
        return y
    if box_pts is None:
        box_pts = max(3, len(y)//10)
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="valid")
    pad = (len(y) - len(y_smooth)) // 2
    y_smooth = np.concatenate([
        [np.nan]*pad, y_smooth, [np.nan]*(len(y) - len(y_smooth) - pad)
    ])
    return y_smooth

# === Plot 1: Reward Curve ===
def plot_rewards():
    if not os.path.exists(reward_file):
        print("[WARN] No reward file found.")
        return
    data = np.loadtxt(reward_file, delimiter=",")
    if data.ndim == 1:
        episodes, rewards = [int(data[0])], [float(data[1])]
    else:
        episodes, rewards = data[:, 0], data[:, 1]

    plt.figure(figsize=(10,6))
    plt.plot(episodes, rewards, label="Raw", alpha=0.5)
    if len(rewards) >= 5:
        plt.plot(episodes, smooth(rewards), label="Smoothed", color="red")
    plt.title(f"Reward Curve ({t('JobAgent')}-level Global Reward)")
    plt.xlabel("Episode Index")
    plt.ylabel(f"Average {t('Reward')}")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rewards_curve.png"))
    plt.close()

# === Plot 2: Loss Curve ===
def plot_loss():
    if not os.path.exists(loss_file):
        print("[WARN] No loss file found.")
        return

    cleaned_lines = []
    with open(loss_file, "r") as f:
        for line in f:
            l = line.strip()
            if not l or l.startswith(("version https://", "oid sha256", "size ")):
                continue
            if l.startswith("tensor("):
                l = l.replace("tensor(", "").split(",")[0]
            try:
                float(l)
                cleaned_lines.append(l)
            except:
                continue
    if not cleaned_lines:
        print("[WARN] Loss file empty or unreadable.")
        return

    data = np.array([float(x) for x in cleaned_lines])
    steps = np.arange(len(data))
    plt.figure(figsize=(10,6))
    plt.plot(steps, data, alpha=0.5)
    if len(data) >= 5:
        plt.plot(steps, smooth(data), color="red", label="Smoothed")
    plt.title("Training Loss Curve")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

# === Plot 3: Episode Duration Curve (SimPy time diff) ===
def plot_training_time():
    if not os.path.exists(time_file):
        print("[WARN] No times file found.")
        return
    data = np.loadtxt(time_file, delimiter=",")
    if data.ndim == 1:
        times = [float(data[1])]
    else:
        times = data[:, 1]
    durations = np.diff(times, prepend=0)
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(len(durations)), durations, alpha=0.6)
    if len(durations) >= 5:
        plt.plot(smooth(durations), color="red", label="Smoothed")
    plt.title(f"Episode Duration per {t('JobAgent')} (ΔSimPy Time)")
    plt.xlabel("Episode Index")
    plt.ylabel("Duration (SimPy time units)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_time.png"))
    plt.close()

# === Plot 4: Global Wait Curve ===
def plot_wait_curve():
    if not os.path.exists(wait_file):
        print("[WARN] No waittimes file found.")
        return
    episodes, waits = [], []
    with open(wait_file, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 4: continue
            ep_str, _, _, wait_str = parts
            try:
                ep = int(ep_str.replace("Episode", "").strip())
                wt = float(wait_str.replace("Wait", "").strip())
                episodes.append(ep)
                waits.append(wt)
            except: continue

    if not episodes:
        print("[INFO] No valid wait entries found.")
        return

    episodes, waits = np.array(episodes), np.array(waits)
    unique_eps = np.unique(episodes)
    ep_sum = [np.sum(waits[episodes == e]) for e in unique_eps]

    plt.figure(figsize=(10,6))
    plt.plot(unique_eps, ep_sum, alpha=0.6, label="Raw")
    if len(ep_sum) >= 5:
        plt.plot(unique_eps, smooth(ep_sum), color="red", label="Smoothed")
    plt.title(f"Global {t('Wait')} Time Curve (Sum per Episode)")
    plt.xlabel("Episode Index")
    plt.ylabel(f"Total {t('Wait')} Time")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "wait_curve.png"))
    plt.close()

# === Main ===
if __name__ == "__main__":
    print("Starting post-training analysis...")
    plot_rewards()
    plot_loss()
    plot_training_time()
    plot_wait_curve()
    print("✅ Analysis complete.")
