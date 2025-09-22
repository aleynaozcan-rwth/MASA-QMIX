import matplotlib.pyplot as plt
import numpy as np
import os

# === File paths ===
reward_file = "./my_data_and_graph/historydata/episode_rewards.txt"
loss_file = "./my_data_and_graph/historydata/loss.txt"
time_file = "./my_data_and_graph/historydata/times.txt"
wait_file = "./my_data_and_graph/historydata/waittimes.txt"

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

    # Focus on 10th–90th percentile range
    low, high = np.percentile(rewards, 10), np.percentile(rewards, 90)
    plt.ylim(low, high)
    plt.title("Reward Curve (Zoomed)")
    plt.xlabel("Episode Index")
    plt.ylabel("Global Reward (average per agent)")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, "rewards_curve_zoomed.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Reward curve (zoomed) saved to {save_path}")


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
    save_path = os.path.join(save_dir, "training_time.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Training time curve saved to {save_path}")


# === Plot 4: Wait Times (per plane, per episode) ===
def plot_wait_times():
    if not os.path.exists(wait_file):
        print("[WARN] No waittimes file found.")
        return

    # --- Parse custom text format ---
    episodes, planes, jobs, waits = [], [], [], []
    with open(wait_file, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 4:
                continue
            ep_str, plane_str, job_str, wait_str = parts
            try:
                ep = int(ep_str.replace("Episode", "").strip())
                plane = plane_str.strip()
                job = job_str.strip()
                wt = float(wait_str.replace("Wait", "").strip())
            except ValueError:
                continue
            episodes.append(ep)
            planes.append(plane)
            jobs.append(job)
            waits.append(wt)

    episodes = np.array(episodes)
    planes = np.array(planes)
    jobs = np.array(jobs)
    waits = np.array(waits)

    unique_planes = np.unique(planes)

    plt.figure(figsize=(14, 8))

    for plane in unique_planes:
        mask = planes == plane
        plt.barh([plane] * np.sum(mask),
                 waits[mask],
                 alpha=0.6)

        # Annotate each bar
        for ep, job, wt in zip(episodes[mask], jobs[mask], waits[mask]):
            plt.text(wt + 0.5, plane,
                     f"Ep{ep}, {job}\n{wt:.1f}",
                     va='center', fontsize=7)

    plt.title("Wait Times per Plane (per Episode & Job)")
    plt.xlabel("Wait Time")
    plt.ylabel("Planes")
    plt.grid(axis='x', linestyle="--", alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "wait_times.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Wait times plot saved to {save_path}")


# === Main ===
if __name__ == "__main__":
    plot_rewards()
    plot_loss()
    plot_training_time()
    plot_wait_times()
