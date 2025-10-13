import numpy as np
import torch
from MARL.common.replay_buffer import ReplayBuffer



class RolloutWorker:
    """
    rollout.py
    Step 7B – Dynamic arrivals + Replay integration (SimPy)
    --------------------------------------------------------
    Collects full episodes from ScheduleEnv and streams
    per-plane transitions into ReplayBuffer.

    Key features:
    - Uses env.info["active_agents"], ["new_records"], ["newly_completed_by"].
    - Credits each plane with +10 per completed job (completion bonus).
    - Applies −1 time penalty shared among currently active planes.
    - Keeps fixed-shape episode arrays for MARLlib-compatibility.
    """

    def __init__(self, env, agents, args, buffer: ReplayBuffer):
        self.env = env
        self.agents = agents
        self.args = args
        self.buffer = buffer

        # Fixed buffer shapes (for MARLlib episode storage)
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        print("[INFO] RolloutWorker (Step 7B) initialized with replay buffer")

    # ----------------------------------------------------------------------
    def _normalize_avail_mask(self, mask):
        """Ensure (n_agents, n_actions) availability mask."""
        if mask is None:
            return np.ones((self.n_agents, self.n_actions), dtype=np.float32)
        mask = np.asarray(mask)
        if mask.ndim == 1 and mask.shape[0] == self.n_actions:
            return np.tile(mask[None, :], (self.n_agents, 1)).astype(np.float32)
        if mask.ndim == 2 and mask.shape[1] == self.n_actions:
            out = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
            rows = min(mask.shape[0], self.n_agents)
            out[:rows] = mask[:rows]
            return out
        return np.ones((self.n_agents, self.n_actions), dtype=np.float32)

    # ----------------------------------------------------------------------
    def generate_episode(self, global_ep_idx=None, evaluate=False):
        """
        Run one SimPy-driven episode.
        During rollout, transitions are also streamed to ReplayBuffer.
        """
        self.env.reset()
        self.agents.policy.init_hidden(1)

        terminated = False
        step = 0
        episode_reward = 0.0
        gantt_records = []

        # fixed-shape episode containers
        o, s, u, u_onehot, r, term, pad, avail_u = ([] for _ in range(8))

        print("\n[Rollout 7B] === New episode started ===")

        while not terminated and step < self.episode_limit:
            obs_t = np.zeros((self.n_agents, self.obs_shape), dtype=np.float32)
            state_t = np.zeros((self.state_shape,), dtype=np.float32)
            actions = None

            reward, done_flag, info = self.env.step(actions)
            episode_reward += float(reward)

            active_agents = info.get("active_agents", [])
            new_records = info.get("new_records", [])
            newly_completed = info.get("newly_completed_by", [])
            t_now = info.get("time", 0.0)

            # ---------- plane-aware reward credits ----------
            if len(active_agents) > 0:
                penalty_share = -1.0 / len(active_agents)
                for pid in active_agents:
                    self.buffer.add_time_penalty(plane_id=pid, time=t_now, penalty=penalty_share)

            for pid in newly_completed:
                self.buffer.add_completion_bonus(plane_id=pid, time=t_now, bonus=10.0,
                                                 aux={"kind": "completion", "records": len(new_records)})

            # ---------- episode-array bookkeeping ----------
            avail_mask = self._normalize_avail_mask(info.get("avail_actions", None))
            r.append([reward])
            term.append([done_flag])
            pad.append([0.0])
            u.append(np.zeros((self.n_agents, 1), dtype=np.int64))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            avail_u.append(avail_mask)
            o.append(obs_t)
            s.append(state_t)

            step += 1
            terminated = bool(done_flag)
            if terminated or self.env.all_jobs_completed():
                gantt_records = info.get("episodes_situation", [])
                terminated = True
                break

            if step % 5 == 0:
                print(f"[Rollout 7B] t={t_now:.1f} | active={len(active_agents)} "
                      f"| new_records={len(new_records)} | total_jobs={info.get('completed_jobs', 0)}")

        # pad to episode_limit
        for t in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape), dtype=np.float32))
            s.append(np.zeros((self.state_shape,), dtype=np.float32))
            u.append(np.zeros((self.n_agents, 1), dtype=np.int64))
            r.append([0.0])
            u_onehot.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            avail_u.append(np.zeros((self.n_agents, self.n_actions), dtype=np.float32))
            term.append([1.0])
            pad.append([1.0])

        # shift next-states/masks
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
            padded=np.array([pad], dtype=np.float32),
            terminated=np.array([term], dtype=np.float32),
        )

        print(f"[Rollout 7B] Episode finished in {step} steps | total reward={episode_reward:.1f}")
        print(f"[Rollout 7B] Buffer size = {len(self.buffer)} transitions across "
              f"{len(self.buffer.active_plane_ids())} planes")
        print("------------------------------------------------------------")
        return episode, episode_reward, True, gantt_records


# ----------------------------------------------------------------------
# Communication-enabled wrapper (same behavior for Step 7B)
# ----------------------------------------------------------------------

class CommRolloutWorker(RolloutWorker):
    def __init__(self, env, agents, args, buffer: ReplayBuffer):
        super().__init__(env, agents, args, buffer)
        print("[INFO] CommRolloutWorker (Step 7B) initialized")
