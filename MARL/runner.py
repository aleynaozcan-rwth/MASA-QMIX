import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Rollout
from MARL.common.rollout import RolloutWorker
try:
    from MARL.common.rollout import CommRolloutWorker  # type: ignore
except Exception:
    CommRolloutWorker = RolloutWorker

# Agents / Buffer
from MARL.agent.agent import Agents, CommAgents
from MARL.common.replay_buffer import ReplayBuffer
from MARL.common.terms import t  # unified terminology helper


def plot_gantt(for_gantt_data, filename="gantt.png"):
    """Step 8A.6.6 Gantt – Machine-level layout with WC + Operator labels."""
    if not for_gantt_data or not isinstance(for_gantt_data, (list, tuple)):
        return False
    if len(for_gantt_data) == 0 or len(for_gantt_data[0]) not in [5, 6]:
        return False

    fig, ax = plt.subplots(figsize=(13, 6))
    operation_types = sorted(set([rec[2] for rec in for_gantt_data]))
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_map = {op: colors[i % len(colors)] for i, op in enumerate(operation_types)}

    max_end = 0
    for rec in for_gantt_data:
        if len(rec) == 6:
            start, end, operation, workcenter, jobagent, operator = rec
        else:
            start, end, operation, workcenter, jobagent = rec
            operator = "?"
        ax.barh(jobagent, end - start, left=start, color=color_map.get(operation, "gray"), edgecolor="black")
        ax.text((start + end) / 2, jobagent, f"M{operation} | WC{workcenter} | O{operator}",
                va="center", ha="center", fontsize=7, color="black")
        max_end = max(max_end, end)

    jobagent_ids = sorted(set([rec[4] for rec in for_gantt_data]))
    ax.set_yticks(jobagent_ids)
    ax.set_yticklabels([str(j) for j in jobagent_ids])
    ax.set_xlabel("Simulation Time (SimPy clock)")
    ax.set_ylabel(f"{t('JobAgent')} (ID)")
    ax.set_title("Step 8A.6.6 – Machine-level Schedule (WC + Operator + Dynamic Arrivals)")
    ax.set_xlim(0, max_end + 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[op]) for op in operation_types]
    labels = [f"Operation {op}" for op in operation_types]
    ax.legend(handles, labels, title="Operation Types", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return True


class Runner:
    """
    Step 8A.6.6 Runner – Replay-based QMIX + KPI Logging + Learning Stability
    -------------------------------------------------------------------------
    - 11D observations (progress_ratio)
    - Extended KPI metrics (avg wait, utilization, makespan)
    - Moving-average loss/TD tracking
    """

    def __init__(self, env, args):
        self.env = env
        self.args = args

        # Agents & rollout setup
        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:
            self.agents = CommAgents(args)
            self.buffer = ReplayBuffer(episode_capacity=args.buffer_size, seed=args.seed) if getattr(args, "learn", True) else None
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args, buffer=self.buffer)
        else:
            self.agents = Agents(args)
            self.buffer = ReplayBuffer(episode_capacity=args.buffer_size, seed=args.seed) if getattr(args, "learn", True) else None
            self.rolloutWorker = RolloutWorker(env, self.agents, args, buffer=self.buffer)

        self.win_rates = []
        self.episode_rewards = []
        self.episode_durations = []
        self.wait_time_records = []

        self.save_path = os.path.join(self.args.result_dir, args.alg, args.map)
        os.makedirs(self.save_path, exist_ok=True)
        self.history_dir = "./my_data_and_graph/historydata/"
        os.makedirs(self.history_dir, exist_ok=True)

        print(f"[Runner 8A.6.6] Initialized | alg={args.alg} | buffer={getattr(args,'buffer_size','-')} | batch={getattr(args,'batch_size','-')}")

    def run(self, num):
        train_steps = 0
        all_gantt_data = []
        avg_rewards = [0]
        global_ep_idx = 0

        print("[Runner] === Training loop started ===")

        for epoch in range(self.args.n_epoch):
            sys.stdout.write(f"\rRun {num}, epoch {epoch}, avg rewards {np.mean(avg_rewards):.2f}")
            sys.stdout.flush()

            # === Periodic evaluation ===
            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                win_rate, ep_reward, global_ep_idx, gantt_eval = self.evaluate(all_gantt_data, global_ep_idx)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(ep_reward)
                print(f"\n[Eval] Epoch {epoch} | Reward={ep_reward:.2f}")
                try:
                    plot_gantt(gantt_eval, filename=f"{self.history_dir}/gantt_epoch{epoch}.png")
                except Exception as e:
                    print("[WARN] Gantt plot failed:", e)

            episodes, avg_rewards = [], []

            for _ in range(self.args.n_episodes):
                episode, _, _, gantt_data = self.rolloutWorker.generate_episode(global_ep_idx)
                all_gantt_data.extend(gantt_data)
                ep_r = float(np.sum(episode.get('r', 0)))
                avg_rewards.append(ep_r)
                episodes.append(episode)
                global_ep_idx += 1

                # Episode durations
                if hasattr(self.rolloutWorker, "episode_duration"):
                    self.episode_durations.append(self.rolloutWorker.episode_duration)
                else:
                    self.episode_durations.append(len(episode.get('r', [])))

                # Wait times
                if hasattr(self.env, "wait_time_dict"):
                    self.wait_time_records.append(dict(self.env.wait_time_dict))
                else:
                    self.wait_time_records.append({0: ep_r / 20.0})

            # === Training updates (Replay-based) ===
            if self.args.alg not in ['coma', 'central_v', 'reinforce'] and self.buffer is not None:
                if len(self.buffer) < self.args.batch_size:
                    print(f"\n[DEBUG] Buffer warm-up ({len(self.buffer)}/{self.args.batch_size}) – skipping training")
                else:
                    print(f"\n[DEBUG] Training active (buffer={len(self.buffer)}) – gradient updates...")
                    for _ in range(self.args.train_steps):
                        mini_batch = self.buffer.sample(self.args.batch_size, n_actions=self.args.n_actions)
                        if mini_batch is None:
                            break
                        result = self.agents.train(mini_batch, train_steps)
                        train_steps += 1
                        if isinstance(result, dict):
                            loss, td = result.get("loss"), result.get("td_error")
                            if (train_steps % 20 == 0) and (loss is not None):
                                msg = f"[TRAIN] step={train_steps}, loss={loss:.4f}"
                                if td is not None:
                                    msg += f", td={td:.4f}"
                                msg += f", buffer={len(self.buffer)}"
                                print(msg)

            # === KPI LOGGING (Step 8A.6.6) ===
            try:
                avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                util_m = self.env._util_machines()
                util_o = self.env._util_ops()
                makespan = self.env.t
                with open(os.path.join(self.history_dir, "kpi_log.txt"), "a") as f:
                    f.write(f"{epoch},{avg_wait:.4f},{util_m:.4f},{util_o:.4f},{makespan:.2f}\n")
            except Exception as e:
                print(f"[WARN] KPI logging failed: {e}")

        # === Save episode statistics ===
        try:
            with open(os.path.join(self.history_dir, "episode_rewards.txt"), "w") as f_r:
                for ep_idx, ep_r in enumerate(self.episode_rewards):
                    f_r.write(f"{ep_idx},{ep_r:.2f}\n")

            with open(os.path.join(self.history_dir, "times.txt"), "w") as f_t:
                for ep_idx, dur in enumerate(self.episode_durations):
                    f_t.write(f"{ep_idx},{dur:.2f}\n")

            with open(os.path.join(self.history_dir, "waittimes.txt"), "w") as f_w:
                for ep_idx, waits in enumerate(self.wait_time_records):
                    for job_id, wait_val in waits.items():
                        f_w.write(f"Episode {ep_idx} | JobAgent J{job_id} | Wait {wait_val:.2f}\n")
            print("[Runner] Reward/time/wait logs saved for analysis.")
        except Exception as e:
            print("[WARN] Could not save episode stats:", e)

        print("\n✅ [Runner] Training completed successfully.")

        # === Post-training moving average plot ===
        try:
            loss = np.loadtxt(f"{self.history_dir}/loss.txt")
            td = np.loadtxt(f"{self.history_dir}/td_error.txt")
            window = 50
            if len(loss) > window:
                loss_smooth = np.convolve(loss, np.ones(window)/window, mode='valid')
                td_smooth = np.convolve(td, np.ones(window)/window, mode='valid')
                plt.figure()
                plt.plot(loss_smooth, label="Loss (avg)")
                plt.plot(td_smooth, label="TD Error (avg)", alpha=0.7)
                plt.legend(); plt.xlabel("Training Step"); plt.ylabel("Value")
                plt.title("Step 8A.6.6 – Learning Stability Trends")
                plt.tight_layout()
                plt.savefig(f"{self.history_dir}/learning_stability.png", dpi=300)
                plt.close()
                print("[Runner] Learning stability plot saved.")
        except Exception as e:
            print("[WARN] Could not plot learning stability:", e)

    def evaluate(self, all_gantt_data, global_ep_idx):
        win_number, episode_rewards, gantt_eval = 0, 0, []
        for _ in range(self.args.evaluate_epoch):
            _, ep_reward, win_tag, gantt_eval = self.rolloutWorker.generate_episode(global_ep_idx, evaluate=True)
            all_gantt_data.extend(gantt_eval)
            episode_rewards += ep_reward
            if win_tag:
                win_number += 1
            global_ep_idx += 1
        avg_reward = episode_rewards / self.args.evaluate_epoch
        win_rate = win_number / self.args.evaluate_epoch
        return win_rate, avg_reward, global_ep_idx, gantt_eval

    def plt(self, num):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel(f'epoch × {self.args.evaluate_cycle}')
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel(f'epoch × {self.args.evaluate_cycle}')
        plt.ylabel('episode_rewards')

        plt.tight_layout()
        outdir = self.save_path
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f'plt_{num}.png'), format='png')
        np.save(os.path.join(outdir, f'win_rates_{num}.npy'), self.win_rates)
        np.save(os.path.join(outdir, f'episode_rewards_{num}.npy'), self.episode_rewards)
        print(f"[Runner] Plots saved to {outdir}")
