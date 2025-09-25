import matplotlib.pyplot as plt
import numpy as np
import os

# === File paths ===
reward_file = "./my_data_and_graph/historydata/episode_rewards.txt"
loss_file   = "./my_data_and_graph/historydata/loss.txt"
time_file   = "./my_data_and_graph/historydata/times.txt"
wait_file   = "./my_data_and_graph/historydata/waittimes.txt"

save_dir = "./my_data_and_graph/historydata/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# === Helper: smooth values with moving average (with edge fix) ===
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
        rewards  = [float(data[1])]
    else:
        episodes = data[:, 0]
        rewards  = data[:, 1]

    # Normal Reward Curve
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, rewards, label="Raw (global per-episode reward)", alpha=0.5)
    if len(rewards) >= 50:
        plt.plot(episodes, smooth(rewards, 50), label="Smoothed", color="red")
    plt.title("Reward Curve (Global reward per episode)")
    plt.xlabel("Episode Index")
    plt.ylabel("Global Reward (average per agent)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "rewards_curve.png"))
    plt.close()

    # Zoomed Reward Curve
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, rewards, label="Raw (global per-episode reward)", alpha=0.5)
    if len(rewards) >= 50:
        plt.plot(episodes, smooth(rewards, 50), label="Smoothed", color="red")
    low, high = np.percentile(rewards, 10), np.percentile(rewards, 90)
    plt.ylim(low, high)
    plt.title("Reward Curve (Zoomed)")
    plt.xlabel("Episode Index")
    plt.ylabel("Global Reward (average per agent)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "rewards_curve_zoomed.png"))
    plt.close()


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

    # Normal Loss Curve
    plt.figure(figsize=(12, 7))
    plt.plot(steps, data, label="Raw loss (per mini-batch update)", alpha=0.5)
    if len(data) >= 50:
        plt.plot(steps, smooth(data, 50), label="Smoothed", color="red")
    plt.title("Loss Curve")
    plt.xlabel("Training Step (mini-batch updates)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # Zoomed Loss Curve
    plt.figure(figsize=(12, 7))
    plt.plot(steps, data, label="Raw loss (per mini-batch update)", alpha=0.5)
    if len(data) >= 50:
        plt.plot(steps, smooth(data, 50), label="Smoothed", color="red")
    plt.ylim(0, np.percentile(data, 95))
    plt.title("Loss Curve (Zoomed In)")
    plt.xlabel("Training Step (mini-batch updates)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve_zoomed.png"))
    plt.close()


# === Plot 3: Training Duration per Episode ===
def plot_training_time():
    if not os.path.exists(time_file):
        print("[WARN] No times file found.")
        return
    times = np.loadtxt(time_file, delimiter=",")[:, 1]
    episodes = np.arange(len(times))

    # Normal
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, times, label="Raw episode duration", alpha=0.5)
    if len(times) >= 50:
        plt.plot(episodes, smooth(times, 50), label="Smoothed", color="red")
    plt.title("Training Duration per Episode")
    plt.xlabel("Episode Index")
    plt.ylabel("Episode Duration (simulation time units)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_time.png"))
    plt.close()

    # Zoomed
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, times, label="Raw episode duration", alpha=0.5)
    if len(times) >= 50:
        plt.plot(episodes, smooth(times, 50), label="Smoothed", color="red")
    low, high = np.percentile(times, 10), np.percentile(times, 90)
    plt.ylim(low, high)
    plt.title("Training Duration per Episode (Zoomed)")
    plt.xlabel("Episode Index")
    plt.ylabel("Episode Duration (simulation time units)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_time_zoomed.png"))
    plt.close()


# === Plot 4: Wait Times per Plane (snapshot every N episodes) ===
def plot_wait_times_snapshot(interval=250):
    if not os.path.exists(wait_file):
        print("[WARN] No waittimes file found.")
        return

    episodes, planes, jobs, waits = [], [], [], []
    with open(wait_file, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 4:
                continue
            ep_str, plane_str, job_str, wait_str = parts
            try:
                ep   = int(ep_str.replace("Episode", "").strip())
                plane= plane_str.strip()
                job  = int(job_str.replace("Job", "").strip())
                wt   = float(wait_str.replace("Wait", "").strip())
            except ValueError:
                continue
            episodes.append(ep)
            planes.append(plane)
            jobs.append(job)
            waits.append(wt)

    episodes, planes, jobs, waits = map(np.array, [episodes, planes, jobs, waits])
    unique_jobs = np.unique(jobs)
    cmap = plt.colormaps["tab20"]
    job_to_color = {job: cmap(i % cmap.N) for i, job in enumerate(unique_jobs)}

    for target_ep in range(interval, max(episodes)+1, interval):
        mask = (episodes == target_ep) & (waits > 0)
        if not np.any(mask):
            continue

        plt.figure(figsize=(14, 8))
        for plane in np.unique(planes[mask]):
            plane_mask = (planes == plane) & (episodes == target_ep) & (waits > 0)
            plt.barh([plane] * np.sum(plane_mask),
                     waits[plane_mask],
                     color=[job_to_color[j] for j in jobs[plane_mask]],
                     alpha=0.7)
            for wt, job in zip(waits[plane_mask], jobs[plane_mask]):
                plt.text(wt + 0.5, plane, f"{wt:.1f}", va="center", fontsize=7)

        handles = [plt.Rectangle((0,0),1,1, color=job_to_color[j]) for j in unique_jobs]
        labels = [f"Job {j}" for j in unique_jobs]
        plt.legend(handles, labels, title="Jobs", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(f"Wait Times per Plane (Episode {target_ep})")
        plt.xlabel("Wait Time")
        plt.ylabel("Planes")
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"wait_times_ep{target_ep}.png"), dpi=200)
        plt.close()


# === Plot 5: Global Wait Curve ===
def plot_wait_curve():
    if not os.path.exists(wait_file):
        print("[WARN] No waittimes file found.")
        return

    episodes, waits = [], []
    with open(wait_file, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 4:
                continue
            ep_str, _, _, wait_str = parts
            try:
                ep   = int(ep_str.replace("Episode", "").strip())
                wt   = float(wait_str.replace("Wait", "").strip())
            except ValueError:
                continue
            episodes.append(ep)
            waits.append(wt)

    episodes, waits = np.array(episodes), np.array(waits)
    unique_eps = np.unique(episodes)
    ep_sums = [np.sum(waits[episodes == e]) for e in unique_eps]

    # Normal curve
    plt.figure(figsize=(12, 7))
    plt.plot(unique_eps, ep_sums, label="Total wait per episode", alpha=0.5)
    if len(ep_sums) >= 50:
        plt.plot(unique_eps, smooth(ep_sums, 50), label="Smoothed", color="red")
    plt.title("Global Wait Time Curve")
    plt.xlabel("Episode Index")
    plt.ylabel("Total Wait Time (sum per episode)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "wait_curve.png"))
    plt.close()

    # Zoomed curve
    non_zero = [v for v in ep_sums if v > 0]
    if len(non_zero) > 0:
        low, high = np.percentile(non_zero, 10), np.percentile(non_zero, 90)
    else:
        low, high = 0, 1

    plt.figure(figsize=(12, 7))
    plt.plot(unique_eps, ep_sums, label="Total wait per episode", alpha=0.5)
    if len(ep_sums) >= 50:
        plt.plot(unique_eps, smooth(ep_sums, 50), label="Smoothed", color="red")
    plt.ylim(low, high)
    plt.title("Global Wait Time Curve (Zoomed)")
    plt.xlabel("Episode Index")
    plt.ylabel("Total Wait Time (sum per episode)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "wait_curve_zoomed.png"))
    plt.close()


# === Main ===
if __name__ == "__main__":
    print("Starting reward/loss/time/wait analysis...")
    plot_rewards()
    plot_loss()
    plot_training_time()
    plot_wait_times_snapshot(interval=250)
    plot_wait_curve()
    print("Analysis complete.")
