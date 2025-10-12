import numpy as np
import os
from MARL.common.rollout import RolloutWorker, CommRolloutWorker
from MARL.agent.agent import Agents, CommAgents
from MARL.common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys


def plot_gantt(for_gantt_data, filename="gantt.png"):
    """
    Step 6B – Plot one episode's Gantt chart.
    Supports both 5-element (old) and 6-element (with operator) tuples.
    """
    if not for_gantt_data or not isinstance(for_gantt_data, (list, tuple)):
        return False
    if len(for_gantt_data) == 0 or len(for_gantt_data[0]) not in [5, 6]:
        print("[WARN] Gantt plot skipped: wrong or empty tuple format")
        return False

    fig, ax = plt.subplots(figsize=(12, 6))
    job_types = sorted(set([rec[2] for rec in for_gantt_data]))  # job_id
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_map = {jt: colors[i % len(colors)] for i, jt in enumerate(job_types)}

    max_end = 0
    for rec in for_gantt_data:
        # Support old (5-tuple) or new (6-tuple) format
        if len(rec) == 6:
            start, end, job, site, plane, operator = rec
        else:
            start, end, job, site, plane = rec
            operator = "?"
        ax.barh(plane, end - start, left=start,
                color=color_map[job], edgecolor="black")
        ax.text((start + end) / 2, plane,
                f"J{job}|S{site}|O{operator}",
                va="center", ha="center", fontsize=6, color="black")
        max_end = max(max_end, end)

    ax.set_xlabel("Simulation Time (SimPy clock)")
    ax.set_ylabel("Plane (Agent)")
    ax.set_title("Step 6B – SimPy Schedule with Operators")
    ax.set_xlim(0, max_end + 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[jt]) for jt in job_types]
    labels = [f"Job {jt}" for jt in job_types]
    ax.legend(handles, labels, title="Job Types",
              bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return True


class Runner:
    def __init__(self, env, args):
        self.env = env

        # === Agent initialization ===
        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)

        if args.learn and args.alg not in ['coma', 'central_v', 'reinforce']:
            self.buffer = ReplayBuffer(args)

        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        self.save_path = os.path.join(self.args.result_dir, args.alg, args.map)
        os.makedirs(self.save_path, exist_ok=True)

    # ============================================================
    # === Main Training Loop =====================================
    # ============================================================

    def run(self, num):
        train_steps = 0
        all_gantt_data = []
        r_s = [0]
        global_ep_idx = 0

        for epoch in range(self.args.n_epoch):
            text = '\rRun {}, epoch {}, avg rewards {:.2f}'
            sys.stdout.write(text.format(num, epoch, np.mean(r_s)))
            sys.stdout.flush()

            # --- Evaluation phase ---
            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                win_rate, ep_reward, global_ep_idx, gantt_eval = \
                    self.evaluate(all_gantt_data, global_ep_idx)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(ep_reward)
                print(f"\n[Eval] Epoch {epoch} | Reward={ep_reward:.2f}")

                try:
                    plot_gantt(gantt_eval,
                               filename=f"./my_data_and_graph/historydata/"
                                        f"gantt_epoch{epoch}.png")
                except Exception as e:
                    print("[WARN] Gantt plot failed:", e)

            # --- Training episodes ---
            episodes = []
            r_s = []

            for episode_idx in range(self.args.n_episodes):
                episode, _, _, gantt_data = \
                    self.rolloutWorker.generate_episode(global_ep_idx)
                all_gantt_data.extend(gantt_data)
                ep_r = np.sum(episode['r'])
                r_s.append(ep_r)
                episodes.append(episode)
                global_ep_idx += 1

            # --- Merge batch ---
            episode_batch = episodes[0]
            episodes.pop(0)
            for ep in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate(
                        (episode_batch[key], ep[key]), axis=0
                    )

            # --- Training step ---
            if self.args.alg in ['coma', 'central_v', 'reinforce']:
                self.agents.train(
                    episode_batch, train_steps, self.rolloutWorker.epsilon
                )
            else:
                self.buffer.store_episode(episode_batch)
                for _ in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(
                        min(self.buffer.current_size, self.args.batch_size)
                    )
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1

    # ============================================================
    # === Evaluation =============================================
    # ============================================================

    def evaluate(self, all_gantt_data, global_ep_idx):
        win_number = 0
        episode_rewards = 0
        gantt_eval = []
        for _ in range(self.args.evaluate_epoch):
            _, ep_reward, win_tag, gantt_eval = \
                self.rolloutWorker.generate_episode(global_ep_idx, evaluate=True)
            all_gantt_data.extend(gantt_eval)
            episode_rewards += ep_reward
            if win_tag:
                win_number += 1
            global_ep_idx += 1
        return (win_number / self.args.evaluate_epoch,
                episode_rewards / self.args.evaluate_epoch,
                global_ep_idx, gantt_eval)

    # ============================================================
    # === Plot Results ===========================================
    # ============================================================

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
        plt.savefig(os.path.join(self.save_path, f'plt_{num}.png'),
                    format='png')
        np.save(os.path.join(self.save_path, f'win_rates_{num}.npy'),
                self.win_rates)
        np.save(os.path.join(self.save_path, f'episode_rewards_{num}.npy'),
                self.episode_rewards)
