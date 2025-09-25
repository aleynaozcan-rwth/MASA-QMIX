import numpy as np
import os
from MARL.common.rollout import RolloutWorker, CommRolloutWorker
from MARL.agent.agent import Agents, CommAgents
from MARL.common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys


# --- Gantt chart plotting function ---
def plot_gantt(for_gantt_data, filename="gantt.png"):
    """
    Expected format: (start_time, end_time, job_id, site_id, plane_id)
    Creates and saves a Gantt chart of scheduled jobs.

    NOTE (English):
    - Each call plots only ONE episode's scheduling (for clarity).
    - For global comparison across episodes, keep logs in scheduleresults.txt.
    """
    if not for_gantt_data or not isinstance(for_gantt_data, (list, tuple)):
        return False
    if len(for_gantt_data) == 0 or len(for_gantt_data[0]) != 5:
        print("[WARN] Gantt plot skipped: wrong or empty tuple format")
        return False

    fig, ax = plt.subplots(figsize=(12, 6))

    # Fixed color mapping for JobTypes
    job_types = sorted(set([rec[2] for rec in for_gantt_data]))  # 2 = job_id
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_map = {jt: colors[i % len(colors)] for i, jt in enumerate(job_types)}

    max_end = 0
    for (start, end, job, site, plane) in for_gantt_data:
        # Draw bar for each scheduled job
        ax.barh(plane, end - start, left=start,
                color=color_map[job], edgecolor="black")

        # Add JobID|SiteID label on the bar
        ax.text((start + end) / 2, plane,
                f"J{job}|S{site}",
                va="center", ha="center",
                fontsize=6, color="black")

        max_end = max(max_end, end)

    # Axis labels and title
    ax.set_xlabel("Simulation Time (environment clock)")
    ax.set_ylabel("Plane (Agent ID)")
    ax.set_title("Gantt Chart (JobID|SiteID)")
    ax.set_xlim(0, max_end + 1)

    # Legend showing JobType ↔ Color
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[jt]) for jt in job_types]
    labels = [f"Job {jt}" for jt in job_types]
    ax.legend(handles, labels, title="Job Types", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return True
# --- End of function ---


class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-policy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # Used to save plt and pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        train_steps = 0
        all_gantt_data = []   # store all gantt data across episodes
        r_s = [0]

        # === NEW: global episode counter (ensures unique IDs across all epochs) ===
        global_ep_idx = 0

        for epoch in range(self.args.n_epoch):
            text = '\rRun {}, train epoch {}, ave_rewards {}'
            sys.stdout.write(text.format(num, epoch, sum(r_s)/len(r_s)))
            sys.stdout.flush()

            # === Evaluation phase ===
            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                # NOTE (English): evaluation also increments global_ep_idx internally
                win_rate, episode_reward, global_ep_idx, gantt_eval = self.evaluate(all_gantt_data, global_ep_idx)
                self.win_rates.append(win_rate)
                print("\nepisode_reward:", episode_reward, "epoch:", epoch)
                self.episode_rewards.append(episode_reward)

                # Save global gantt data (optional long-term log)
                with open("./my_data_and_graph/historydata/scheduleresults.txt", "a") as f:
                    print(all_gantt_data, file=f)

                # Plot only the last evaluation episode's Gantt chart
                try:
                    plot_gantt(gantt_eval,
                               filename=f"./my_data_and_graph/historydata/gantt_epoch{epoch}_ep{global_ep_idx}.png")
                except Exception as e:
                    print("[WARN] Gantt plot skipped:", e)

            episodes = []
            r_s = []

            # === Training episodes inside this epoch ===
            for episode_idx in range(self.args.n_episodes):
                # FIX: pass global_ep_idx instead of episode_idx
                episode, _, _, gantt_data = self.rolloutWorker.generate_episode(global_ep_idx)
                all_gantt_data.extend(gantt_data)   # accumulate gantt data (global log)

                # calculate episode reward correctly
                ep_r = np.sum(episode['r']) / self.args.n_agents
                r_s.append(ep_r)

                episodes.append(episode)

                # log with global episode counter
                with open("./my_data_and_graph/historydata/episode_rewards.txt", "a") as f:
                    f.write(f"{global_ep_idx},{ep_r}\n")

                #  Per-episode Gantt PNG removed (only logs kept)

                # === Added for Cluster Convergence Test 25.09.2025 Version 4.8 ===
                # Epsilon log per episode (helps track exploration vs exploitation)
                with open("./my_data_and_graph/historydata/epsilon_log.txt", "a") as f:
                    f.write(f"Episode {global_ep_idx}, epsilon={self.rolloutWorker.epsilon}\n")
                # === End of addition ===

                global_ep_idx += 1

            # Merge batch episodes
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1

    def evaluate(self, all_gantt_data, global_ep_idx):
        win_number = 0
        episode_rewards = 0
        gantt_eval = []  # store last eval episode gantt
        for _ in range(self.args.evaluate_epoch):
            # FIX: pass global_ep_idx to rollout
            _, episode_reward, win_tag, gantt_eval = self.rolloutWorker.generate_episode(global_ep_idx, evaluate=True)
            all_gantt_data.extend(gantt_eval)   # add eval gantt data too (global log)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
            global_ep_idx += 1   # increment global counter also during evaluation
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch, global_ep_idx, gantt_eval

    def plt(self, num):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
