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

    # --- Normal Reward Curve ---
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, rewards, label="Raw (global per-episode reward)", alpha=0.5)
    if len(rewards) >= 50:
        plt.plot(episodes, smooth(rewards, 50), label="Smoothed", color="red")
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
        plt.plot(episodes, smooth(rewards, 50), label="Smoothed", color="red")

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
        plt.plot(steps, smooth(data, 50), label="Smoothed", color="red")
    plt.title("Loss Curve")
    plt.xlabel("Training Step (mini-batch updates)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Loss curve saved to {save_path}")

    # --- Zoomed loss curve ---
    plt.figure(figsize=(12, 7))
    plt.plot(steps, data, label="Raw loss (per mini-batch update)", alpha=0.5)
    if len(data) >= 50:
        plt.plot(steps, smooth(data, 50), label="Smoothed (50-update avg)", color="red")
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

    # --- Normal curve ---
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, times, label="Raw episode duration", alpha=0.5)
    if len(times) >= 50:
        plt.plot(episodes, smooth(times, 50), label="Smoothed", color="red")
    plt.title("Training Duration per Episode")
    plt.xlabel("Episode Index")
    plt.ylabel("Episode Duration (simulation time units)")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, "training_time.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Training time curve saved to {save_path}")

    # --- Zoomed curve ---
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
    save_path = os.path.join(save_dir, "training_time_zoomed.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Training time (zoomed) saved to {save_path}")


# === Plot 4: Wait Times (subset: every 250th episode) ===
def plot_wait_times_subset(interval=250):
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
                job  = job_str.replace("Job", "").strip()
                wt   = float(wait_str.replace("Wait", "").strip())
            except ValueError:
                continue
            episodes.append(ep)
            planes.append(plane)
            jobs.append(int(job))
            waits.append(wt)

    episodes = np.array(episodes)
    planes   = np.array(planes)
    jobs     = np.array(jobs)
    waits    = np.array(waits)

    unique_jobs = np.unique(jobs)
    cmap = plt.colormaps["tab20"]
    job_to_color = {job: cmap(i % cmap.N) for i, job in enumerate(unique_jobs)}

    for ep in np.unique(episodes):
        if ep % interval != 0:
            continue
        plt.figure(figsize=(14, 8))
        mask_ep = (episodes == ep) & (waits > 0)
        unique_planes = np.unique(planes[mask_ep])

        for plane in unique_planes:
            mask = mask_ep & (planes == plane)
            plt.barh([plane] * np.sum(mask),
                     waits[mask],
                     color=[job_to_color[j] for j in jobs[mask]],
                     alpha=0.7)
            for wt in waits[mask]:
                plt.text(wt + 0.5, plane, f"{wt:.1f}",
                         va='center', ha='left', fontsize=7, color="black")

        handles = [plt.Rectangle((0,0),1,1, color=job_to_color[j]) for j in unique_jobs]
        labels = [f"Job {j}" for j in unique_jobs]
        plt.legend(handles, labels, title="Jobs", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title(f"Wait Times per Plane (Episode {ep})")
        plt.xlabel("Wait Time")
        plt.ylabel("Planes")
        plt.grid(axis='x', linestyle="--", alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"wait_times_ep{ep}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[+] Wait times plot saved to {save_path}")


# === Plot 5: Global Wait Curve (episode-wise total/avg) ===
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

    episodes = np.array(episodes)
    waits    = np.array(waits)

    unique_eps = np.unique(episodes)
    ep_sums = [np.sum(waits[episodes == e]) for e in unique_eps]

    # --- Normal curve ---
    plt.figure(figsize=(12, 7))
    plt.plot(unique_eps, ep_sums, label="Total wait per episode", alpha=0.5)
    if len(ep_sums) >= 50:
        plt.plot(unique_eps, smooth(ep_sums, 50), label="Smoothed", color="red")
    plt.title("Global Wait Time Curve")
    plt.xlabel("Episode Index")
    plt.ylabel("Total Wait Time (sum per episode)")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, "wait_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Global wait curve saved to {save_path}")

    # --- Zoomed curve ---
    plt.figure(figsize=(12, 7))
    plt.plot(unique_eps, ep_sums, label="Total wait per episode", alpha=0.5)
    if len(ep_sums) >= 50:
        plt.plot(unique_eps, smooth(ep_sums, 50), label="Smoothed", color="red")

    low, high = np.percentile(ep_sums, 5), np.percentile(ep_sums, 95)
    if low == high:
        low, high = min(ep_sums), max(ep_sums)
        if low == high:  # tümü aynıysa margin ekle
            low -= 1
            high += 1
    plt.ylim(low, high)
    plt.title("Global Wait Time Curve (Zoomed)")
    plt.xlabel("Episode Index")
    plt.ylabel("Total Wait Time (sum per episode)")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, "wait_curve_zoomed.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Global wait curve (zoomed) saved to {save_path}")


# === Main entry ===
if __name__ == "__main__":
    plot_rewards()
    plot_loss()
    plot_training_time()
    plot_wait_times_subset(interval=250)   # <-- her 250 episode’da bir
    plot_wait_curve()
