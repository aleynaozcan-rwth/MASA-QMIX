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
    times = np.loadtxt(time_file, delimiter=",")[:, 1]  # second column = duration
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

    # Zoomed curve (non-zero values only)
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


# === NEW: Convergence Assessment Function ===
def assess_convergence():
    """Automated convergence detection based on reward stability and trend."""
    if not os.path.exists(reward_file):
        print("[WARN] Cannot assess convergence - no reward file.")
        return "NO_DATA"
        
    data = np.loadtxt(reward_file, delimiter=",")
    if data.ndim == 1 or len(data) < 100:
        print("[INFO] Insufficient data for convergence assessment.")
        return "INSUFFICIENT_DATA"
        
    rewards = data[:, 1]
    episodes = data[:, 0]
    
    # Take last 100 episodes for assessment
    recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
    
    # Calculate key metrics
    mean_reward = np.mean(recent_rewards)
    std_reward = np.std(recent_rewards)
    cv = std_reward / abs(mean_reward) if mean_reward != 0 else float('inf')
    
    # Linear trend (improvement rate)
    if len(recent_rewards) > 1:
        trend_slope = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
    else:
        trend_slope = 0
    
    # Convergence criteria (MASA-QMIX specific)
    stable = cv < 0.15  # Coefficient of variation < 15%
    minimal_improvement = abs(trend_slope) < 0.02  # < 2% change per episode
    acceptable_performance = mean_reward > 3.0  # Above minimum threshold
    
    # Assessment
    print(f"\n=== CONVERGENCE ASSESSMENT ===")
    print(f"Episodes analyzed: {len(recent_rewards)}")
    print(f"Mean reward (last 100): {mean_reward:.3f}")
    print(f"Std deviation: {std_reward:.3f}")
    print(f"Coefficient of Variation: {cv:.3f} (target: <0.15)")
    print(f"Trend slope: {trend_slope:.4f} (target: <0.02)")
    print(f"Stable: {'YES' if stable else 'NO'}")
    print(f"Minimal improvement: {'YES' if minimal_improvement else 'NO'}")
    print(f"Performance acceptable: {'YES' if acceptable_performance else 'NO'}")
    
    if stable and minimal_improvement and acceptable_performance:
        status = "CONVERGED"
        print(f"STATUS: {status} ✓")
    elif stable and minimal_improvement:
        status = "STAGNATED_AT_SUBOPTIMAL"  
        print(f"STATUS: {status} ⚠")
    else:
        status = "STILL_LEARNING"
        print(f"STATUS: {status} →")
    
    print("=" * 35)
    
    # Save assessment to file
    with open(os.path.join(save_dir, "convergence_assessment.txt"), "w") as f:
        f.write(f"Mean Reward: {mean_reward:.3f}\n")
        f.write(f"CV: {cv:.3f}\n") 
        f.write(f"Trend: {trend_slope:.4f}\n")
        f.write(f"Status: {status}\n")
    
    return status


# === NEW: Performance Comparison Against Baselines ===
def plot_performance_comparison():
    """Compare current performance against expected baselines."""
    if not os.path.exists(reward_file):
        print("[WARN] No reward file for performance comparison.")
        return
        
    data = np.loadtxt(reward_file, delimiter=",")
    if data.ndim == 1:
        rewards = [data[1]]
        episodes = [data[0]]
    else:
        rewards = data[:, 1]
        episodes = data[:, 0]
    
    # MASA-QMIX specific baselines
    random_baseline = -20  # Expected random performance
    heuristic_baseline = 2  # Simple heuristic performance  
    target_performance = 8  # Target performance level
    
    plt.figure(figsize=(12, 8))
    
    # Plot performance
    plt.plot(episodes, rewards, alpha=0.6, label="Actual Performance", color='blue')
    if len(rewards) >= 50:
        plt.plot(episodes, smooth(rewards, 50), label="Smoothed", color='red', linewidth=2)
    
    # Plot baselines
    plt.axhline(y=random_baseline, color='gray', linestyle='--', 
               label=f'Random Baseline ({random_baseline})')
    plt.axhline(y=heuristic_baseline, color='orange', linestyle='--', 
               label=f'Heuristic Baseline ({heuristic_baseline})')
    plt.axhline(y=target_performance, color='green', linestyle='--', 
               label=f'Target Performance ({target_performance})')
    
    # Performance zones
    plt.fill_between(episodes, random_baseline, heuristic_baseline, 
                    alpha=0.1, color='red', label='Poor Performance Zone')
    plt.fill_between(episodes, heuristic_baseline, target_performance, 
                    alpha=0.1, color='yellow', label='Acceptable Performance Zone') 
    plt.fill_between(episodes, target_performance, max(max(rewards), target_performance+2), 
                    alpha=0.1, color='green', label='Target Performance Zone')
    
    plt.title("Performance vs Baselines")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_comparison.png"))
    plt.close()
    print(f"[+] Performance comparison saved")


# === NEW: Training Health Check ===
def training_health_check():
    """Check if training is healthy or has issues."""
    print(f"\n=== TRAINING HEALTH CHECK ===")
    
    health_status = "HEALTHY"
    issues = []
    
    # Check reward file
    if not os.path.exists(reward_file):
        print("[ERROR] No reward file found!")
        return "ERROR"
    
    # Check loss file  
    if not os.path.exists(loss_file):
        print("[WARN] No loss file found!")
        issues.append("No loss tracking")
    else:
        try:
            def clean_loss_line(line):
                line = line.strip()
                if line.startswith("tensor("):
                    line = line.replace("tensor(", "").split(",")[0]
                return line
            
            with open(loss_file, "r") as f:
                cleaned_lines = [clean_loss_line(l) for l in f if l.strip()]
            
            if len(cleaned_lines) < 10:
                print("[WARN] Very few loss values recorded")
                issues.append("Insufficient loss data")
            else:
                recent_losses = [float(line) for line in cleaned_lines[-10:]]
                avg_recent_loss = np.mean(recent_losses)
                print(f"Recent average loss: {avg_recent_loss:.3f}")
                
                if avg_recent_loss > 20:
                    print("[WARN] Loss is very high - possible training instability")
                    health_status = "UNSTABLE"
                    issues.append("High loss values")
                elif avg_recent_loss < 0.1:
                    print("[INFO] Loss is very low - possible convergence")
                else:
                    print("[INFO] Loss appears normal")
        except Exception as e:
            print(f"[WARN] Could not parse loss file: {e}")
            issues.append("Loss file parsing error")
    
    # Check episode completion
    if os.path.exists(time_file):
        times = np.loadtxt(time_file, delimiter=",")[:, 1]
        avg_time = np.mean(times[-50:]) if len(times) >= 50 else np.mean(times)
        print(f"Average episode duration: {avg_time:.2f}")
        
        if avg_time > 200:
            print("[WARN] Episodes taking very long - possible inefficiency")
            issues.append("Long episode duration")
        elif avg_time < 20:
            print("[WARN] Episodes very short - possible premature termination")
            issues.append("Short episode duration")  
        else:
            print("[INFO] Episode duration appears normal")
    else:
        print("[WARN] No episode timing data found")
        issues.append("No timing data")
    
    # Overall health assessment
    if len(issues) == 0:
        print(f"OVERALL STATUS: {health_status} ✓")
    else:
        print(f"OVERALL STATUS: {health_status} with {len(issues)} issues ⚠")
        for issue in issues:
            print(f"  - {issue}")
    
    print("=" * 35)
    return health_status


# === NEW: Robustness Test ===
def test_convergence_robustness():
    """Test if convergence is robust or fragile."""
    if not os.path.exists(reward_file):
        print("[WARN] No reward file for robustness test.")
        return "NO_DATA"
        
    data = np.loadtxt(reward_file, delimiter=",")
    rewards = data[:, 1]
    
    if len(rewards) < 200:
        print("[INFO] Insufficient data for robustness test (need 200+ episodes)")
        return "INSUFFICIENT_DATA"
    
    # Test stability of last 100 vs previous 100 episodes
    last_100 = rewards[-100:]
    prev_100 = rewards[-200:-100] 
    
    mean_last = np.mean(last_100)
    mean_prev = np.mean(prev_100)
    performance_retention = mean_last / mean_prev if mean_prev != 0 else 0
    
    print(f"\n=== CONVERGENCE ROBUSTNESS TEST ===")
    print(f"Previous 100 episodes mean: {mean_prev:.3f}")
    print(f"Last 100 episodes mean: {mean_last:.3f}")  
    print(f"Performance retention: {performance_retention:.3f}")
    
    if performance_retention >= 0.9:
        status = "ROBUST"
        print("STATUS: Robust convergence ✓")
    elif performance_retention >= 0.8:
        status = "ACCEPTABLE"
        print("STATUS: Acceptable stability →")
    else:
        status = "UNSTABLE"
        print("STATUS: Unstable convergence ⚠")
    print("=" * 35)
    
    return status


# === Main ===
if __name__ == "__main__":
    print("Starting comprehensive analysis...")
    plot_rewards()
    plot_loss()
    plot_training_time()
    plot_wait_times_snapshot(interval=250)
    plot_wait_curve()
    
    # NEW: Add convergence assessment
    convergence_status = assess_convergence()
    plot_performance_comparison()
    health_status = training_health_check()
    robustness_status = test_convergence_robustness()
    
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Convergence Status: {convergence_status}")
    print(f"Training Health: {health_status}")
    print(f"Robustness: {robustness_status}")
    print("=" * 35)
    print("Analysis complete!")