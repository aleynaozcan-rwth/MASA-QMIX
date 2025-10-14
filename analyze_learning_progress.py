#!/usr/bin/env python3
"""
analyze_learning_progress.py
--------------------------------
Step 7B.3 automatic post-training analysis.

Reads loss.txt and episode_rewards.txt from ./my_data_and_graph/historydata/
and determines whether learning occurred (loss ↓, reward ↑).
Generates both textual and graphical summaries.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

DATA_DIR = "./my_data_and_graph/historydata/"
LOSS_FILE = os.path.join(DATA_DIR, "loss.txt")
REWARD_FILE = os.path.join(DATA_DIR, "episode_rewards.txt")

print("[Learning Summary] Starting analysis...")

def load_numeric(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() != ""]
    vals = []
    for l in lines:
        try:
            vals.append(float(l.split(",")[-1]))
        except:
            try:
                vals.append(float(l))
            except:
                pass
    return np.array(vals) if len(vals) > 0 else None


loss_vals = load_numeric(LOSS_FILE)
reward_vals = load_numeric(REWARD_FILE)

summary_text = []

if loss_vals is not None and len(loss_vals) > 3:
    loss_start, loss_end = np.mean(loss_vals[:10]), np.mean(loss_vals[-10:])
    loss_drop = 100 * (1 - loss_end / max(loss_start, 1e-6))
    summary_text.append(f"Loss: {loss_start:.3f} → {loss_end:.3f}  (↓ {loss_drop:.1f}%)")
else:
    summary_text.append("Loss: insufficient data")

if reward_vals is not None and len(reward_vals) > 3:
    rew_start, rew_end = np.mean(reward_vals[:3]), np.mean(reward_vals[-3:])
    reward_gain = 100 * ((rew_end - rew_start) / max(abs(rew_start), 1e-6))
    summary_text.append(f"Reward: {rew_start:.1f} → {rew_end:.1f}  (Δ {reward_gain:+.1f}%)")
else:
    summary_text.append("Reward: insufficient data")

if loss_vals is not None and reward_vals is not None:
    if (len(loss_vals) > 3 and len(reward_vals) > 3 
        and loss_drop > 0 and reward_gain > 0):
        summary_text.append("✅ Learning detected (loss decreasing, reward increasing)")
    else:
        summary_text.append("⚠️ No clear learning trend detected")
else:
    summary_text.append("⚠️ Missing data; unable to assess learning")

print("\n".join(summary_text))

# Save summary text
with open(os.path.join(DATA_DIR, "learning_summary.txt"), "w") as f:
    f.write("\n".join(summary_text))

# Plot summary
plt.figure(figsize=(10, 5))
if reward_vals is not None:
    plt.plot(reward_vals, label="Reward", color="tab:blue")
if loss_vals is not None and np.max(loss_vals) > 0:
    scale = np.max(reward_vals) / np.max(loss_vals) if reward_vals is not None else 1.0
    plt.plot(loss_vals * scale, label="Loss (scaled)", color="tab:red", linestyle="--")

plt.title("Learning Progress Overview")
plt.xlabel("Training step / Episode")
plt.ylabel("Value (scaled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "learning_summary.png"))
plt.close()

print("\n[Learning Summary] Analysis complete.")
