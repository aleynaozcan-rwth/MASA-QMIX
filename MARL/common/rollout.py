import numpy as np

class RolloutWorker:
    """
    rollout.py
    Step 7B.3 – Replay Transition Training
    --------------------------------------
    - Keeps fixed shapes for episodic buffers (compat).
    - Additionally writes *transition-level* tuples into ReplayBuffer each env step:
        * time penalty (global)
        * per-plane completion bonus (for planes that finished a job this step)
        * optional per-plane "heartbeat" transition (0-reward) to densify buffer
    """

    def __init__(self, env, agents, args, buffer=None):
        self.env = env
        self.agents = agents
        self.args = args
        self.buffer = buffer  # replay buffer or None

        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        print("[INFO] RolloutWorker (Step 7B.3) initialized (replay transitions enabled)")

    # ------ helper ------
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

    # ------ main ------
    def generate_episode(self, global_ep_idx=None, evaluate=False):
        self.env.reset()
        self.agents.policy.init_hidden(1)

        terminated = False
        step = 0
        episode_reward = 0.0
        gantt_records = []

        # episodic containers (kept for compatibility with mixers)
        o, s, u, u_onehot, r, terminate, padded, avail_u = [], [], [], [], [], [], [], []

        epsilon = 0.0 if evaluate else self.epsilon

        print("\n[Rollout 7B.3] === New episode started ===")
        last_state = np.zeros((self.n_agents, self.obs_shape), dtype=np.float32)

        while not terminated and step < self.episode_limit:
            # placeholder obs/state (your obs encoder can replace these later)
            obs_t = np.zeros((self.n_agents, self.obs_shape), dtype=np.float32)
            state_t = np.zeros((self.state_shape,), dtype=np.float32)

            # env ignores actions; put zeros for shapes
            actions = None

            reward, done_flag, info = self.env.step(actions)
            episode_reward += float(reward)

            # episodic buffers (for mixer compat)
            avail_mask = self._normalize_avail_mask(info.get("avail_actions", None))
            r.append([reward])
            terminate.append([done_flag])
            padded.append([0.0])
            u.append(np.zeros((self.n_agents, 1), dtype=np.int64))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            avail_u.append(avail_mask)
            o.append(obs_t)
            s.append(state_t)

            # ---------- Step 7B.3: push transitions to ReplayBuffer ----------
            if self.buffer is not None:
                t_now = float(info.get("time", 0.0))

                # Global time penalty every step (helps fill buffer quickly)
                self.buffer.add_time_penalty(time=t_now, penalty=-1.0)

                # Per-plane completion bonuses this step
                for pid in info.get("newly_completed_by", []):
                    self.buffer.add_completion_bonus(plane_id=int(pid), time=t_now, bonus=10.0)

                # Optional: per-active-plane heartbeat (0-reward)
                # Gives the learner some (s,a,r,s') density even without completions.
                for pid in info.get("active_agents", []):
                    self.buffer.add_transition(
                        plane_id=int(pid),
                        time=t_now,
                        state=last_state[0] if last_state.ndim == 2 else np.zeros(10, np.float32),
                        action=np.array([0], dtype=np.int64),
                        reward=0.0,
                        next_state=np.zeros(10, dtype=np.float32),
                        done=False,
                    )

            step += 1
            terminated = bool(done_flag)

            if hasattr(self.env, "all_jobs_completed") and self.env.all_jobs_completed():
                terminated = True

            if terminated:
                gantt_records = info.get("episodes_situation", [])
                break

        # pad episode for mixer compatibility
        for t in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape), dtype=np.float32))
            s.append(np.zeros((self.state_shape,), dtype=np.float32))
            u.append(np.zeros((self.n_agents, 1), dtype=np.int64))
            r.append([0.0])
            u_onehot.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            avail_u.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            terminate.append([1.0])
            padded.append([1.0])

        o_next = o[1:] + [np.zeros_like(o[0])]
        s_next = s[1:] + [np.zeros_like(s[0])]
        avail_u_next = avail_u[1:] + [np.zeros_like(avail_u[0])]

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

        print(f"[Rollout 7B.3] Episode finished in {step} steps | total reward={episode_reward:.1f}")
        print("-----------------------------------------------------------------")
        return episode, episode_reward, True, gantt_records


class CommRolloutWorker(RolloutWorker):
    def __init__(self, env, agents, args, buffer=None):
        super().__init__(env, agents, args, buffer=buffer)
        print("[INFO] CommRolloutWorker (Step 7B.3) initialized")
