import numpy as np

class RolloutWorker:
    """
    rollout.py
    Step 8A.5.3 — Learning-Activated Rollout (ReplayBuffer-integrated)
    -------------------------------------------------------------------
    - Consumes env-provided avail_actions mask
    - Epsilon-greedy action selection (policy-aware, masked)
    - Uses obs/state from env info (safe fallbacks)
    - Writes per-agent transitions into ReplayBuffer (if provided):
        add_transition(jobagent_id, time, state, action, reward, next_state, done)
      Also optional credits: add_time_penalty / add_completion_bonus
    """

    def __init__(self, env, agents, args, buffer=None):
        self.env = env
        self.agents = agents
        self.args = args
        self.buffer = buffer

        # Shapes (prefer env.get_env_info if provided)
        try:
            env_info = self.env.get_env_info()
            self.episode_limit = env_info.get("episode_limit", args.episode_limit)
            self.n_actions     = env_info.get("n_actions",     args.n_actions)
            self.n_agents      = env_info.get("n_agents",      args.n_agents)
            self.state_shape   = env_info.get("state_shape",   args.state_shape)
            self.obs_shape     = env_info.get("obs_shape",     args.obs_shape)
        except Exception:
            self.episode_limit = args.episode_limit
            self.n_actions     = args.n_actions
            self.n_agents      = args.n_agents
            self.state_shape   = args.state_shape
            self.obs_shape     = args.obs_shape

        # Exploration
        self.epsilon = getattr(args, "epsilon", 0.05)
        self.anneal_epsilon = getattr(args, "anneal_epsilon", 0.0)
        self.min_epsilon    = getattr(args, "min_epsilon", 0.0)

        print("[INFO] RolloutWorker initialized (Learning-Activated, Replay-integrated)")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _normalize_avail_mask(self, mask):
        """Return (n_agents, n_actions) float32 mask."""
        if mask is None:
            return np.ones((self.n_agents, self.n_actions), dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim == 1:
            mask = np.tile(mask[None, :], (self.n_agents, 1))
        if mask.shape != (self.n_agents, self.n_actions):
            out = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
            rows = min(mask.shape[0], self.n_agents)
            cols = min(mask.shape[1], self.n_actions)
            out[:rows, :cols] = mask[:rows, :cols]
            mask = out
        return mask

    def _masked_random(self, mask_row):
        """Pick a random valid action given a 1D mask row; if none, return 0."""
        valid = np.where(mask_row > 0.5)[0]
        if valid.size == 0:
            return 0
        return int(np.random.choice(valid))

    def _select_actions(self, obs_batch, avail_mask, epsilon, evaluate=False):
        """
        obs_batch: (n_agents, obs_shape)
        avail_mask: (n_agents, n_actions)
        Return: actions (n_agents,), onehot (n_agents, n_actions)
        """
        actions = np.zeros((self.n_agents,), dtype=np.int64)
        onehot  = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)

        for aid in range(self.n_agents):
            mask_row = avail_mask[aid]
            # Try policy if available
            picked = None
            if hasattr(self.agents, "policy") and hasattr(self.agents.policy, "choose_action"):
                try:
                    picked = int(self.agents.policy.choose_action(
                        obs=obs_batch[aid],
                        last_action=None,
                        agent_num=aid,
                        avail_actions_mask=mask_row,
                        epsilon=epsilon,
                        evaluate=evaluate
                    ))
                except Exception:
                    picked = None

            if picked is None or mask_row[picked] < 0.5:
                # epsilon-greedy fallback on masked random
                if (not evaluate) and (np.random.rand() < epsilon):
                    picked = self._masked_random(mask_row)
                else:
                    # no logits yet → use masked random as "greedy" fallback
                    picked = self._masked_random(mask_row)

            actions[aid] = picked
            onehot[aid, picked] = 1.0

        return actions, onehot

    def _query_env_avail_mask(self):
        # Prefer live call if available; else 1s
        if hasattr(self.env, "get_avail_actions"):
            try:
                m = self.env.get_avail_actions()
                return self._normalize_avail_mask(m)
            except Exception:
                pass
        return np.ones((self.n_agents, self.n_actions), dtype=np.float32)

    # ------------------------------------------------------------
    # Main
    # ------------------------------------------------------------
    def generate_episode(self, global_ep_idx=None, evaluate=False):
        # Reset env & agent hidden states
        try:
            self.env.reset()
        except Exception:
            pass

        if hasattr(self.agents, "policy") and hasattr(self.agents.policy, "init_hidden"):
            try:
                self.agents.policy.init_hidden(1)
            except Exception:
                pass

        terminated = False
        step = 0
        episode_reward = 0.0
        gantt_records = []
        episode_end_time = 0.0

        o, s, u, u_onehot, r, terminate, padded, avail_u = [], [], [], [], [], [], [], []

        # Initial obs/state (fallback if env doesn’t expose pre-step info)
        try:
            obs0 = self.env._build_obs()
            state0 = self.env._build_state()
        except Exception:
            obs0 = np.zeros((self.obs_shape,), dtype=np.float32)
            state0 = np.zeros((self.state_shape,), dtype=np.float32)

        obs_t = np.tile(obs0, (self.n_agents, 1))
        state_t = state0

        while not terminated and step < self.episode_limit:
            # --- Query availability & select actions ---
            avail_mask = self._query_env_avail_mask()
            eps = 0.0 if evaluate else self.epsilon
            actions, onehot_t = self._select_actions(obs_t, avail_mask, eps, evaluate=evaluate)

            # --- Environment step ---
            reward, done_flag, info = self.env.step(actions)
            episode_reward += float(reward)
            episode_end_time = info.get("time", episode_end_time)

            # --- Next obs/state from env info (fallback safe) ---
            obs_raw = info.get("obs", None)
            state_raw = info.get("state", None)
            if obs_raw is None:
                obs_t_next = np.zeros((self.n_agents, self.obs_shape), dtype=np.float32)
            else:
                obs_t_next = np.tile(np.asarray(obs_raw, dtype=np.float32), (self.n_agents, 1))
            state_t_next = np.asarray(state_raw, dtype=np.float32) if state_raw is not None \
                           else np.zeros((self.state_shape,), dtype=np.float32)

            # --- Store transition (episode storage for mixers) ---
            r.append([reward])
            terminate.append([done_flag])
            padded.append([0.0])
            u.append(actions.reshape(self.n_agents, 1))
            u_onehot.append(onehot_t.astype(np.float32))
            avail_u.append(avail_mask.astype(np.float32))
            o.append(obs_t.astype(np.float32))
            s.append(state_t.astype(np.float32))

            # --- Write per-agent transitions to ReplayBuffer (flat) ---
            if self.buffer is not None:
                t_now = float(info.get("time", 0.0))

                # Global “tick” penalty (optional shaping)
                try:
                    if hasattr(self.buffer, "add_time_penalty"):
                        self.buffer.add_time_penalty(time=t_now, penalty=-1.0)
                except Exception:
                    pass

                # Credit completions (optional shaping)
                try:
                    for jid in info.get("newly_completed_by", []):
                        if hasattr(self.buffer, "add_completion_bonus"):
                            self.buffer.add_completion_bonus(jobagent_id=int(jid), time=t_now, bonus=10.0)
                except Exception:
                    pass

                # Main per-agent transitions
                try:
                    for aid in range(self.n_agents):
                        self.buffer.add_transition(
                            jobagent_id=aid,
                            time=t_now,
                            state=state_t,                          # global state (10-dim)
                            action=np.array([actions[aid]], dtype=np.int64),
                            reward=float(reward),                   # shared reward signal
                            next_state=state_t_next,
                            done=bool(done_flag),
                        )
                except Exception:
                    # Fail-safe: do not crash training loop if buffer write fails
                    pass

            # --- Advance ---
            step += 1
            terminated = bool(done_flag)
            if hasattr(self.env, "all_jobs_completed") and self.env.all_jobs_completed():
                terminated = True
            obs_t, state_t = obs_t_next, state_t_next

            if terminated:
                gantt_records = info.get("episodes_situation", [])
                break

        # --- Padding for mixer compatibility ---
        for t in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape), dtype=np.float32))
            s.append(np.zeros((self.state_shape,), dtype=np.float32))
            u.append(np.zeros((self.n_agents, 1), dtype=np.int64))
            r.append([0.0])
            u_onehot.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            avail_u.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            terminate.append([1.0])
            padded.append([1.0])

        # Shifted next-views
        o_next = o[1:] + [np.zeros_like(o[0])]
        s_next = s[1:] + [np.zeros_like(s[0])]
        avail_u_next = avail_u[1:] + [np.zeros_like(avail_u[0])]

        # Pack episode (for mixer-based learners)
        episode = dict(
            o=np.array([o], dtype=np.float32),
            s=np.array([s], dtype=np.float32),
            u=np.array([u], dtype=np.int64),
            r=np.array([r], dtype=np.float32),
            avail_u=np.array([avail_u], dtype=np.float32),
            o_next=np.array([o_next], dtype=np.float32),
            s_next=np.array([s_next], dtype=np.float32),
            avail_u_next=np.array([avail_u_next], dtype=np.float32),
            u_onehot=np.array([u_onehot], dtype=np.float32),
            padded=np.array([padded], dtype=np.float32),
            terminated=np.array([terminate], dtype=np.float32),
        )

        # Epsilon anneal (train only)
        if (not evaluate) and (self.anneal_epsilon > 0.0):
            self.epsilon = max(self.min_epsilon, self.epsilon - self.anneal_epsilon)

        print(f"[Rollout] Episode finished in {step} steps | total reward={episode_reward:.2f} | t_end={episode_end_time:.1f}")
        return episode, episode_reward, True, gantt_records


class CommRolloutWorker(RolloutWorker):
    def __init__(self, env, agents, args, buffer=None):
        super().__init__(env, agents, args, buffer=buffer)
        print("[INFO] CommRolloutWorker initialized (Learning-Activated, Replay-integrated)")
