import numpy as np
import torch

class RolloutWorker:
    """
    rollout.py
    Step 8A.6.3 – Full Compatible Learning Rollout
    ----------------------------------------------
    - Compatible with env.step() → (obs_next, reward, done, info)
    - Handles inconsistent obs_next shapes robustly
    - Records transitions into replay buffer (with next state)
    - Prepares environment for QMix-style centralized training
    """

    def __init__(self, env, agents, args, buffer=None):
        self.env = env
        self.agents = agents
        self.args = args
        self.buffer = buffer

        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        print("[INFO] RolloutWorker 8A.6.3 initialized — full learning + shape-safe observation handling")

    # ============================================================
    # === Helper: normalize available actions ====================
    # ============================================================
    def _normalize_avail_mask(self, mask):
        if mask is None:
            return np.ones((self.n_agents, self.n_actions), dtype=np.float32)
        mask = np.asarray(mask)
        if mask.ndim == 1 and mask.shape[0] == self.n_actions:
            return np.tile(mask[None, :], (self.n_agents, 1)).astype(np.float32)
        if mask.ndim == 2 and mask.shape[1] == self.n_actions:
            rows = min(mask.shape[0], self.n_agents)
            out = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
            out[:rows] = mask[:rows]
            return out
        return np.ones((self.n_agents, self.n_actions), dtype=np.float32)

    # ============================================================
    # === Action selection (epsilon-greedy) =======================
    # ============================================================
    def _select_actions(self, obs, avail_actions, epsilon, evaluate=False):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        h_in = self.agents.policy.eval_hidden

        q_values, self.agents.policy.eval_hidden = self.agents.policy.eval_rnn(obs_t, h_in)
        q_values = q_values.detach().cpu().numpy()

        actions = np.zeros(self.n_agents, dtype=np.int64)
        for i in range(self.n_agents):
            if np.random.uniform() < epsilon and not evaluate:
                avail = np.where(avail_actions[i] > 0)[0]
                actions[i] = np.random.choice(avail)
            else:
                avail = np.where(avail_actions[i] > 0)[0]
                masked_q = q_values[i][avail]
                actions[i] = avail[np.argmax(masked_q)]
        return actions

    # ============================================================
    # === Core episode generation ================================
    # ============================================================
    def generate_episode(self, global_ep_idx=None, evaluate=False):
        self.env.reset()
        self.agents.policy.init_hidden(1)

        # initialize hidden state for RNN-based policy
        self.agents.policy.eval_hidden = torch.zeros(self.args.n_agents, 64)

        terminated = False
        step = 0
        episode_reward = 0.0
        gantt_records = []

        epsilon = 0.0 if evaluate else self.epsilon
        o, s, u, r, avail_u, terminate, padded = [], [], [], [], [], [], []

        print(f"[Rollout 8A.6.3] === New episode (ep={global_ep_idx}) ===")

        # Initial dummy obs/state
        obs = np.zeros((self.n_agents, self.obs_shape), dtype=np.float32)
        state = np.zeros((self.state_shape,), dtype=np.float32)

        while not terminated and step < self.episode_limit:
            avail = np.ones((self.n_agents, self.n_actions), dtype=np.float32)
            actions = self._select_actions(obs, avail, epsilon, evaluate)

            # Environment step: expects 4 returns
            obs_next, reward, done_flag, info = self.env.step(actions)

            # --- Observation shape normalization ---
            if obs_next is None:
                obs_next = np.zeros((self.n_agents, self.obs_shape), dtype=np.float32)
            else:
                try:
                    obs_next = np.asarray(obs_next, dtype=np.float32)
                    if obs_next.ndim == 1:
                        obs_next = np.expand_dims(obs_next, axis=0)
                except Exception as e:
                    print(f"[WARN] obs_next shape inconsistent → {type(obs_next)} | {e}")
                    obs_next = np.zeros((self.n_agents, self.obs_shape), dtype=np.float32)

            # --- Compute next state (mean pooling of obs) ---
            try:
                state_next = np.mean(obs_next, axis=0)
            except Exception as e:
                print(f"[WARN] state_next mean failed → {e}")
                state_next = np.zeros((self.state_shape,), dtype=np.float32)

            # --- Store transition ---
            o.append(obs)
            s.append(state)
            u.append(actions)
            r.append([reward])
            avail_u.append(avail)
            terminate.append([done_flag])
            padded.append([0.0])

            episode_reward += float(reward)
            step += 1
            terminated = bool(done_flag)

            # --- Replay buffer logging ---
            if self.buffer is not None:
                self.buffer.add_transition(
                    jobagent_id=0,
                    time=float(info.get("time", 0.0)),
                    state=state,
                    action=actions,
                    reward=reward,
                    next_state=state_next,
                    done=terminated,
                )

            obs = obs_next
            state = state_next

            if terminated:
                gantt_records = info.get("episodes_situation", [])
                break

        print(f"[Rollout] Episode finished in {step} steps | total reward={episode_reward:.2f}")
        return {"r": np.array(r)}, episode_reward, True, gantt_records


# =============================================================
# === Communication-enabled variant ============================
# =============================================================
class CommRolloutWorker(RolloutWorker):
    def __init__(self, env, agents, args, buffer=None):
        super().__init__(env, agents, args, buffer)
        print("[INFO] CommRolloutWorker 8A.6.3 initialized — shape-safe full compatibility")
