import numpy as np
import torch


class RolloutWorker:
    """
    rollout.py
    Step 7A – Dynamic Job Arrivals + Operator Constraints (SimPy)
    -------------------------------------------------------------
    Collects complete episodes from ScheduleEnv.

    Key points:
    - Environment may spawn new planes (agents) during the episode.
    - We keep buffer shapes constant using args.n_agents (max agents); if env
      temporarily has fewer/more active planes, we still log with fixed shapes.
    - Availability masks are stored per-agent: (n_agents, n_actions).
      If env provides a 1-D mask, we tile it across agents.
    - Episode auto-terminates when env.all_jobs_completed() is True.
    - Actions are placeholders for now (env ignores them in Step 7A).
    """

    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.args = args

        # Fixed shapes for the buffer
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        # ε-greedy params (kept for compatibility; not used here yet)
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        print("[INFO] RolloutWorker (Step 7A) initialized")

    # ----------------------------------------------------------------------

    def _normalize_avail_mask(self, mask):
        """
        Ensure availability mask has shape (n_agents, n_actions).
        Accepts:
          - None (returns ones)
          - 1D (n_actions,) → tiles to (n_agents, n_actions)
          - 2D (n_agents, n_actions) → returned as-is
        Anything else → defaults to ones.
        """
        if mask is None:
            return np.ones((self.n_agents, self.n_actions), dtype=np.float32)

        mask = np.asarray(mask)
        if mask.ndim == 1 and mask.shape[0] == self.n_actions:
            return np.tile(mask[None, :], (self.n_agents, 1)).astype(np.float32)
        if mask.ndim == 2 and mask.shape[1] == self.n_actions:
            # If env reported a different live agent count this step,
            # we fit/crop/pad to self.n_agents for buffer consistency.
            live_agents = mask.shape[0]
            out = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
            rows = min(live_agents, self.n_agents)
            out[:rows] = mask[:rows]
            # Any extra slots remain zeros (interpretable as "no available actions")
            return out
        # Fallback
        return np.ones((self.n_agents, self.n_actions), dtype=np.float32)

    # ----------------------------------------------------------------------

    def generate_episode(self, global_ep_idx=None, evaluate=False):
        """
        Run one complete SimPy-driven episode and return the trajectory bundle.

        Returns
        -------
        episode : dict[str, np.ndarray]
            Keys: o, s, u, r, avail_u, o_next, s_next, avail_u_next, u_onehot, padded, terminated
            Each is shaped with a leading batch dimension = 1, e.g. (1, T, n_agents, obs_dim).
        episode_reward : float
        need_save : bool
        gantt_records : list[tuple]
        """
        # Reset environment and policy hidden states
        self.env.reset()
        self.agents.policy.init_hidden(1)

        terminated = False
        step = 0
        episode_reward = 0.0
        gantt_records = []

        # Containers (lists of time steps)
        o, s = [], []
        u, u_onehot = [], []
        r, terminate, padded = [], [], []
        avail_u = []

        # (Optional) ε=0 during evaluation
        epsilon = 0.0 if evaluate else self.epsilon

        print("\n[Rollout] === New episode started ===")

        while not terminated and step < self.episode_limit:
            # Observations/State placeholders (plug real obs/state later)
            obs_t = np.zeros((self.n_agents, self.obs_shape), dtype=np.float32)
            state_t = np.zeros((self.state_shape,), dtype=np.float32)

            # Placeholder actions: env ignores them in Step 7A
            actions = None

            # Step environment
            reward, terminated_flag, info = self.env.step(actions)
            episode_reward += float(reward)

            # Availability mask (may be missing or 1D or 2D)
            avail_mask = self._normalize_avail_mask(info.get("avail_actions", None))

            # Store time-step transition
            r.append([reward])  # shape (1,)
            terminate.append([terminated_flag])  # shape (1,)
            padded.append([0.0])  # not padded at real steps

            # Action placeholders: shapes must match buffer contract
            u.append(np.zeros((self.n_agents, 1), dtype=np.int64))  # discrete action indices
            u_onehot.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))

            avail_u.append(avail_mask)
            o.append(obs_t)
            s.append(state_t)

            step += 1
            terminated = bool(terminated_flag)

            # Dynamic agent logging (optional)
            if "n_planes" in info:
                print(f"[Rollout] t={info.get('time', 0):.1f} | "
                      f"active planes={info['n_planes']} | "
                      f"completed_jobs={info.get('completed_jobs', 0)}")

            # Early termination when all planes have no remaining jobs
            if hasattr(self.env, "all_jobs_completed") and self.env.all_jobs_completed():
                terminated = True
                terminated_flag = True

            if terminated:
                gantt_records = info.get("episodes_situation", [])
                break

        # Pad to episode_limit for uniform batch shapes
        for t in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape), dtype=np.float32))
            s.append(np.zeros((self.state_shape,), dtype=np.float32))
            u.append(np.zeros((self.n_agents, 1), dtype=np.int64))
            r.append([0.0])
            u_onehot.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            avail_u.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            terminate.append([1.0])  # mark as terminated
            padded.append([1.0])     # padded step

        # Next observations/states/masks (shifted by one)
        o_next = o[1:] + [np.zeros_like(o[0])]
        s_next = s[1:] + [np.zeros_like(s[0])]
        avail_u_next = avail_u[1:] + [np.zeros_like(avail_u[0])]

        # Convert lists to arrays with leading batch dimension = 1
        episode = dict(
            o=np.array([o], dtype=np.float32),                           # (1, T, n_agents, obs_dim)
            s=np.array([s], dtype=np.float32),                           # (1, T, state_dim)
            u=np.array([u], dtype=np.int64),                             # (1, T, n_agents, 1)
            r=np.array([r], dtype=np.float32),                           # (1, T, 1)
            avail_u=np.array([avail_u], dtype=np.float32),               # (1, T, n_agents, n_actions)
            o_next=np.array([o_next], dtype=np.float32),                 # (1, T, n_agents, obs_dim)
            s_next=np.array([s_next], dtype=np.float32),                 # (1, T, state_dim)
            avail_u_next=np.array([avail_u_next], dtype=np.float32),     # (1, T, n_agents, n_actions)
            u_onehot=np.array([u_onehot], dtype=np.float32),             # (1, T, n_agents, n_actions)
            padded=np.array([padded], dtype=np.float32),                 # (1, T, 1)
            terminated=np.array([terminate], dtype=np.float32),          # (1, T, 1)
        )

        print(f"[Rollout] Episode finished in {step} steps | total reward={episode_reward:.1f}")
        print("------------------------------------------------------------")
        return episode, episode_reward, True, gantt_records


# ============================================================
# === CommRolloutWorker (CommNet / G2ANet) ===================
# ============================================================

class CommRolloutWorker(RolloutWorker):
    """
    Communication-enabled variant (kept for interface compatibility).
    In Step 7A it behaves the same as RolloutWorker.
    """
    def __init__(self, env, agents, args):
        super().__init__(env, agents, args)
        print("[INFO] CommRolloutWorker (Step 7A) initialized")
