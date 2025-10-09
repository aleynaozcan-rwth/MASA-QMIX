import numpy as np
import torch
from torch.distributions import one_hot_categorical


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, global_ep_idx=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and global_ep_idx == 0:
            self.env.close()

        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if global_ep_idx == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state0 = self.env.get_state()
            state0 = torch.tensor(state0, dtype=torch.float32)
            if self.args.cuda:
                state0 = state0.cuda()
            z_prob = self.agents.policy.z_policy(state0)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        for_gantt = []
        while not terminated and step < self.episode_limit:
            obs_list = self.env.get_obs()                             # list of (1,) arrays
            state_vec = self.env.get_state()                          # (state_shape,)
            # enforce consistent shapes
            obs = np.asarray(obs_list, dtype=np.float32).reshape(self.n_agents, self.obs_shape)
            state = np.asarray(state_vec, dtype=np.float32).reshape(self.state_shape,)

            actions, avail_actions, actions_onehot = [], [], []

            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(
                    obs[agent_id], last_action[agent_id], agent_id,
                    avail_action, epsilon, evaluate
                )

                if action not in [18, 19, 20]:
                    self.env.has_chosen_action(action, agent_id)

                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False

            o.append(obs)                                             # (n_agents, obs_shape)
            s.append(state)                                           # (state_shape,)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

            if terminated:
                for_gantt = info.get("episodes_situation", [])

        # --- episode-level logs ---
        with open("./my_data_and_graph/historydata/scheduleresults.txt", "a") as f:
            if for_gantt:
                for rec in for_gantt:
                    start, end, job_id, site_id, plane_id = rec
                    f.write(f"Episode {global_ep_idx} | Plane {plane_id} | Job {job_id} "
                            f"| Site {site_id} | Start {start} | End {end}\n")
                f.write("---- End of episode ----\n")
            else:
                f.write(f"[DEBUG] Episode {global_ep_idx} ended without gantt records.\n")

        if for_gantt:
            wait_times = {}
            plane_last_end = {}
            for (start, end, job, site, plane) in sorted(for_gantt, key=lambda x: (x[4], x[0])):
                last_end = plane_last_end.get(plane, 0)
                wait = max(0, start - last_end)
                wait_times[(plane, job)] = wait
                plane_last_end[plane] = end
            with open("./my_data_and_graph/historydata/waittimes.txt", "a") as f:
                for (plane, job), w in wait_times.items():
                    f.write(f"Episode {global_ep_idx} | Plane {plane} | Job {job} | Wait {w}\n")
                f.write("---- End of episode ----\n")

        with open("./my_data_and_graph/historydata/times.txt", "a") as f:
            f.write(f"{global_ep_idx},{info.get('time',0)}\n")

        # last obs/state for *_next
        obs_list = self.env.get_obs()
        state_vec = self.env.get_state()
        obs = np.asarray(obs_list, dtype=np.float32).reshape(self.n_agents, self.obs_shape)
        state = np.asarray(state_vec, dtype=np.float32).reshape(self.state_shape,)

        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        # get avail_action for last obs
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # padding if episode ended early
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape), dtype=np.float32))
            u.append(np.zeros([self.n_agents, 1], dtype=np.float32))
            s.append(np.zeros(self.state_shape, dtype=np.float32))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape), dtype=np.float32))
            s_next.append(np.zeros(self.state_shape, dtype=np.float32))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            avail_u.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(
            o=np.array([o], dtype=np.float32),                  # (1, T, n_agents, obs_shape)
            s=np.array([s], dtype=np.float32),                  # (1, T, state_shape)
            u=np.array([u], dtype=np.float32),                  # (1, T, n_agents, 1)
            r=np.array([r], dtype=np.float32),                  # (1, T, 1)
            avail_u=np.array([avail_u], dtype=np.float32),      # (1, T, n_agents, n_actions)
            o_next=np.array([o_next], dtype=np.float32),
            s_next=np.array([s_next], dtype=np.float32),
            avail_u_next=np.array([avail_u_next], dtype=np.float32),
            u_onehot=np.array([u_onehot], dtype=np.float32),
            padded=np.array([padded], dtype=np.float32),
            terminated=np.array([terminate], dtype=np.float32)
        )

        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and global_ep_idx == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()

        return episode, episode_reward, win_tag, for_gantt


# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode(self, global_ep_idx=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and global_ep_idx == 0:
            self.env.close()

        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if global_ep_idx == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        for_gantt = []
        while not terminated and step < self.episode_limit:
            obs_list = self.env.get_obs()
            state_vec = self.env.get_state()
            obs = np.asarray(obs_list, dtype=np.float32).reshape(self.n_agents, self.obs_shape)
            state = np.asarray(state_vec, dtype=np.float32).reshape(self.state_shape,)

            weights = self.agents.get_action_weights(np.array(obs), last_action)

            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(weights[agent_id], avail_action, epsilon, evaluate)

                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False

            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1

            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

            if terminated:
                for_gantt = info.get("episodes_situation", [])

        with open("./my_data_and_graph/historydata/times.txt", "a") as f:
            f.write(f"{global_ep_idx},{info.get('time',0)}\n")

        if for_gantt:
            wait_times = {}
            plane_last_end = {}
            for (start, end, job, site, plane) in sorted(for_gantt, key=lambda x: (x[4], x[0])):
                last_end = plane_last_end.get(plane, 0)
                wait = max(0, start - last_end)
                wait_times[(plane, job)] = wait
                plane_last_end[plane] = end
            with open("./my_data_and_graph/historydata/waittimes.txt", "a") as f:
                for (plane, job), w in wait_times.items():
                    f.write(f"Episode {global_ep_idx} | Plane {plane} | Job {job} | Wait {w}\n")
                f.write("---- End of episode ----\n")

        # last obs/state
        obs_list = self.env.get_obs()
        state_vec = self.env.get_state()
        obs = np.asarray(obs_list, dtype=np.float32).reshape(self.n_agents, self.obs_shape)
        state = np.asarray(state_vec, dtype=np.float32).reshape(self.state_shape,)

        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape), dtype=np.float32))
            u.append(np.zeros([self.n_agents, 1], dtype=np.float32))
            s.append(np.zeros(self.state_shape, dtype=np.float32))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape), dtype=np.float32))
            s_next.append(np.zeros(self.state_shape, dtype=np.float32))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            avail_u.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(
            o=np.array([o], dtype=np.float32),
            s=np.array([s], dtype=np.float32),
            u=np.array([u], dtype=np.float32),
            r=np.array([r], dtype=np.float32),
            avail_u=np.array([avail_u], dtype=np.float32),
            o_next=np.array([o_next], dtype=np.float32),
            s_next=np.array([s_next], dtype=np.float32),
            avail_u_next=np.array([avail_u_next], dtype=np.float32),
            u_onehot=np.array([u_onehot], dtype=np.float32),
            padded=np.array([padded], dtype=np.float32),
            terminated=np.array([terminate], dtype=np.float32)
        )

        if not evaluate:
            self.epsilon = epsilon
        if evaluate and global_ep_idx == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()

        return episode, episode_reward, win_tag, for_gantt
