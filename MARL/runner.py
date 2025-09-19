import numpy as np
import os
from MARL.common.rollout import RolloutWorker, CommRolloutWorker
from MARL.agent.agent import Agents, CommAgents
from MARL.common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import sys


# --- Added: Gantt chart plotting function ---
def plot_gantt(for_gantt_data, filename="gantt.png"):
    """
    Expected format: (start_time, end_time, job_id, site_id, plane_id)
    Creates and saves a simple Gantt chart of scheduled jobs.
    """
    if not for_gantt_data or not isinstance(for_gantt_data, (list, tuple)):
        return False
    if len(for_gantt_data[0]) != 5:
        print("[WARN] Gantt plot skipped: wrong tuple format")
        return False

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.get_cmap("tab20")
    color_map = {}
    cidx = 0
    max_end = 0

    for (start, end, job, site, plane) in for_gantt_data:
        if job not in color_map:
            color_map[job] = cmap(cidx % 20)
            cidx += 1
        ax.barh(plane, end - start, left=start, color=color_map[job])
        ax.text((start + end) / 2, plane, f"J{job}@S{site}",
                va="center", ha="center", fontsize=7, color="white")
        max_end = max(max_end, end)

    ax.set_xlabel("Time")
    ax.set_ylabel("Plane")
    ax.set_title("Schedule Gantt Chart")
    ax.set_xlim(0, max_end + 1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return True
# --- End of added function ---


class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # # Used to save plt and pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        train_steps = 0
        for_gantt_data =[]
        # print('Run {} start'.format(num))
        r_s = [0]
        for epoch in range(self.args.n_epoch):
            # # Display output

            text = '\rRun {}, train epoch {}, ave_rewards {}'
            sys.stdout.write(text.format(num, epoch, sum(r_s)/len(r_s)))
            sys.stdout.flush()

            # print('Run {}, train epoch {}'.format(num, epoch), flush=False)
            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                print("\nepisode_reward:", episode_reward, "epoch:", epoch)
                self.episode_rewards.append(episode_reward)
                # self.plt(num)
                with open("./my_data_and_graph/historydata/scheduleresults.txt", "a") as f:
                    print(for_gantt_data, file=f)

                # --- Added: call Gantt plot after saving results ---
                try:
                    plot_gantt(for_gantt_data,
                               filename=f"./my_data_and_graph/historydata/gantt_epoch{epoch}.png")
                except Exception as e:
                    print("[WARN] Gantt plot skipped:", e)
                # --- End of added part ---

            episodes = []
            r_s = []
            #  # Collect self.args.n_episodes episodes


            #for episode_idx in range(self.args.n_episodes):
                #episode, _, _, for_gantt_data = self.rolloutWorker.generate_episode(episode_idx)
                #episodes.append(episode)
                #r_s.append(sum(episode['r'][0])[0])
                # print(_)
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, for_gantt_data = self.rolloutWorker.generate_episode(episode_idx)

                # --- FIX: episode reward correct calculation ---
                # episode['r'] shape: (1, episode_len, n_agents, 1)
                ep_r = np.sum(episode['r']) / self.args.n_agents
                r_s.append(ep_r)
                # ------------------------------------------

                episodes.append(episode)

                # Logla (opsiyonel, CSV’ye)
                with open("./my_data_and_graph/historydata/episode_rewards.txt", "a") as f:
                    f.write(f"{episode_idx},{ep_r}\n")

            # Each field of an episode is a 4-D array with shape (1, episode_len, n_agents, <dim>);
            # concatenate all episodes along the first dimension
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                # These algorithms need to store into the replay buffer
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1

        # self.plt(num)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, for_gant = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        # # Returns average win count and average reward
        print(for_gant)
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

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
