import numpy as np
import torch


class RolloutWorker:
    """
    Step 6B – SimPy-synchronized RolloutWorker.
    Works with (site, operator) action pairs and collects availability masks
    from the environment at every step.
    """
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
        print("Init RolloutWorker (Step 6B)")

    # ------------------------------------------------------------

    def generate_episode(self, global_ep_idx=None, evaluate=False):
        """Run one complete SimPy-driven episode and return the trajectory."""
        self.env.reset()
        terminated = False
        step = 0
        episode_reward = 0
        self.agents.policy.init_hidden(1)

        # containers
        o, o_next, s, s_next, u, r, avail_u, avail_u_next, u_onehot, terminate, padded = \
            [], [], [], [], [], [], [], [], [], [], []

        epsilon = 0 if evaluate else self.epsilon
        for_gantt = []

        while not terminated and step < self.episode_limit:
            obs = np.zeros((self.n_agents, self.obs_shape))
            state = np.zeros(self.state_shape)

            # --- step environment ---
            reward, terminated, info = self.env.step(None)
            episode_reward += reward

            # --- availability mask from environment (Step 6B) ---
            avail_mask = info.get("avail_actions", np.ones(self.n_actions))

            # --- store transition ---
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            u.append(np.zeros((self.n_agents, 1)))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(avail_mask)
            o.append(obs)
            s.append(state)

            step += 1
            if terminated:
                for_gantt = info.get("episodes_situation", [])
                break

        # pad if episode ended early
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            s.append(np.zeros(self.state_shape))
            u.append(np.zeros((self.n_agents, 1)))
            r.append([0.])
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros(self.n_actions))
            terminate.append([1.])
            padded.append([1.])

        # next-state placeholders
        o_next = o[1:] + [np.zeros_like(o[0])]
        s_next = s[1:] + [np.zeros_like(s[0])]
        avail_u_next = avail_u[1:] + [np.zeros_like(avail_u[0])]

        episode = dict(
            o=np.array([o]),
            s=np.array([s]),
            u=np.array([u]),
            r=np.array([r]),
            avail_u=np.array([avail_u]),
            o_next=np.array([o_next]),
            s_next=np.array([s_next]),
            avail_u_next=np.array([avail_u_next]),
            u_onehot=np.array([u_onehot]),
            padded=np.array([padded]),
            terminated=np.array([terminate]),
        )
        return episode, episode_reward, True, for_gantt


# ============================================================
# === CommRolloutWorker (CommNet / G2ANet) ===================
# ============================================================

class CommRolloutWorker:
    """Placeholder identical to RolloutWorker for Step 6B."""
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.args = args
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print("[INFO] CommRolloutWorker (Step 6B placeholder) initialized")

    def generate_episode(self, global_ep_idx=None, evaluate=False):
        self.env.reset()
        terminated = False
        step = 0
        episode_reward = 0
        self.agents.policy.init_hidden(1)

        for_gantt = []
        o, o_next, s, s_next, u, r, avail_u, avail_u_next, u_onehot, terminate, padded = \
            [], [], [], [], [], [], [], [], [], [], []

        while not terminated and step < self.episode_limit:
            obs = np.zeros((self.n_agents, self.obs_shape))
            state = np.zeros(self.state_shape)

            reward, terminated, info = self.env.step(None)
            episode_reward += reward

            avail_mask = info.get("avail_actions", np.ones(self.n_actions))

            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            u.append(np.zeros((self.n_agents, 1)))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(avail_mask)
            o.append(obs)
            s.append(state)

            step += 1
            if terminated:
                for_gantt = info.get("episodes_situation", [])
                break

        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            s.append(np.zeros(self.state_shape))
            u.append(np.zeros((self.n_agents, 1)))
            r.append([0.])
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros(self.n_actions))
            terminate.append([1.])
            padded.append([1.])

        o_next = o[1:] + [np.zeros_like(o[0])]
        s_next = s[1:] + [np.zeros_like(s[0])]
        avail_u_next = avail_u[1:] + [np.zeros_like(avail_u[0])]

        episode = dict(
            o=np.array([o]),
            s=np.array([s]),
            u=np.array([u]),
            r=np.array([r]),
            avail_u=np.array([avail_u]),
            o_next=np.array([o_next]),
            s_next=np.array([s_next]),
            avail_u_next=np.array([avail_u_next]),
            u_onehot=np.array([u_onehot]),
            padded=np.array([padded]),
            terminated=np.array([terminate]),
        )
        return episode, episode_reward, True, for_gantt
