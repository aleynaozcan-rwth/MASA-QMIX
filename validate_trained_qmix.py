# validate_trained_qmix.py – MASA-QMIX Validation (Auto Checkpoint Loader)
# ------------------------------------------------------------
# Automatically detects and loads latest saved QMIX checkpoint
# and evaluates on unseen job scenarios with makespan/wait/reward.
# ------------------------------------------------------------

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from MARL.policy.qmix import QMIX
from MARL.common.arguments import get_common_args, get_mixer_args
from environment import MASAEnv


def find_latest_checkpoint(base_dir="./MARL/model/qmix"):
    """Finds latest saved QMIX checkpoint automatically (supports both naming styles)."""
    candidates = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith("qmix_net_params.pkl"):
                path = os.path.join(root, f)
                timestamp = os.path.getmtime(path)
                candidates.append((timestamp, path))
    if not candidates:
        print("[Warning] No trained QMIX checkpoint found anywhere under ./MARL/model/qmix/")
        return None, None

    # Sort by timestamp → newest last
    candidates.sort(key=lambda x: x[0])
    latest_qmix = candidates[-1][1]
    latest_rnn = latest_qmix.replace("qmix_net_params.pkl", "rnn_net_params.pkl")

    if not os.path.exists(latest_rnn):
        print(f"[Warning] Found mixer {latest_qmix} but missing RNN counterpart.")
        return latest_qmix, None

    print(f"[AutoLoad] Latest QMIX checkpoint detected:\n  Mixer → {latest_qmix}\n  RNN   → {latest_rnn}")
    return latest_qmix, latest_rnn


def evaluate_policy(policy, env, n_episodes=10):
    """Runs multiple unseen validation episodes using greedy (argmax) policy."""
    rewards, waits, durations = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward, total_wait = 0.0, 0.0
        done = False
        t = 0

        while not done:
            num_agents = len(obs)
            obs_t = torch.tensor(np.array(obs), dtype=torch.float32)

            # --- Build full RNN input like training: [obs, last_action, agent_id] ---
            last_action = torch.zeros((num_agents, policy.n_actions))  # no exploration
            agent_ids = torch.eye(policy.n_agents)[:num_agents]
            inputs = torch.cat([obs_t, last_action, agent_ids], dim=1)
            hidden = torch.zeros((num_agents, policy.args.rnn_hidden_dim))

            q_t, _ = policy.eval_rnn(inputs, hidden)
            actions = q_t.argmax(dim=1).cpu().numpy()

            obs, reward, done, info = env.step(actions)
            total_reward += reward
            total_wait += env.total_wait_time
            t += 1
            if t >= env.episode_limit:
                break

        rewards.append(total_reward)
        waits.append(total_wait)
        durations.append(t)
        print(f"[Eval] Episode {ep+1:02d} | Reward={total_reward:.2f} | Wait={total_wait:.1f} | Duration={t}")

    return np.array(rewards), np.array(waits), np.array(durations)


if __name__ == "__main__":
    print("[Validation] Starting MASA-QMIX policy evaluation...")
    args = get_common_args()
    args = get_mixer_args(args)
    env = MASAEnv(num_jobs=args.n_agents, num_operators=args.num_operators, num_wcs=args.n_actions)

    # --- Load latest checkpoint automatically ---
    qmix_path, rnn_path = find_latest_checkpoint()
    policy = QMIX(args)

    if qmix_path and rnn_path:
        policy.eval_mixer.load_state_dict(torch.load(qmix_path, map_location="cpu"))
        policy.eval_rnn.load_state_dict(torch.load(rnn_path, map_location="cpu"))
        print("[QMIX] ✅ Loaded pretrained weights successfully.")
    else:
        print("[Warning] No trained QMIX model found. Run training first.")
        exit(0)

    # --- Evaluate on unseen scenarios ---
    rewards, waits, durations = evaluate_policy(policy, env, n_episodes=10)

    print("\n[Summary] Validation complete:")
    print(f"  → Avg Reward:   {np.mean(rewards):.3f}")
    print(f"  → Avg WaitTime: {np.mean(waits):.2f}")
    print(f"  → Avg Makespan: {np.mean(durations):.1f}")

    # --- Save comparison plots ---
    os.makedirs("./validation_results", exist_ok=True)

    plt.figure()
    plt.plot(rewards, label="Episode Reward")
    plt.title("QMIX Validation – Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("./validation_results/validation_rewards.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(waits, label="Total Wait Time", color="orange")
    plt.title("QMIX Validation – Total Wait Times")
    plt.xlabel("Episode")
    plt.ylabel("Wait Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("./validation_results/validation_waits.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(durations, label="Episode Duration", color="green")
    plt.title("QMIX Validation – Episode Durations (Makespan)")
    plt.xlabel("Episode")
    plt.ylabel("Duration (steps)")
    plt.legend()
    plt.grid(True)
    plt.savefig("./validation_results/validation_durations.png", dpi=200)
    plt.close()

    print("[Validation] Results saved to ./validation_results/")
