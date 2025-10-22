import numpy as np
import torch

class RolloutWorker:
    """
    Step 8A.6.6 – Full learning rollout worker (RNN-safe, replay-aware, KPI-compatible)

    • Resets env and builds replay-ready transitions
    • Epsilon-greedy action selection (with availability masks)
    • Maintains RNN hidden state per agent for eval_rnn()
    • Inserts full episode into ReplayBuffer if provided
    • Reports episode duration for KPI tracking
    """

    def __init__(
        self,
        env,
        agents,
        buffer=None,
        args=None,
        episode_limit=200,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_anneal_steps=50000,
        device="cpu",
        log_prefix="8A.6.6",
    ):
        self.env = env
        self.agents = agents
        self.buffer = buffer
        self.args = args
        self.episode_limit = getattr(args, "episode_limit", episode_limit)
        self.device = getattr(args, "device", device)
        self.log_prefix = log_prefix

        # --- epsilon schedule ---
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_anneal_steps = int(epsilon_anneal_steps)
        self._eps_decay = (
            (self.epsilon_start - self.epsilon_end)
            / max(1, self.epsilon_anneal_steps)
        )

        # --- hidden state placeholder ---
        self._reset_hidden_states()
        self.episode_duration = 0  # new KPI-compatible field

        print(
            f"[INFO] RolloutWorker {self.log_prefix} initialized — RNN-compatible + replay-aware"
        )
        print(
            f"  → Episode limit: {self.episode_limit}\n"
            f"  → Epsilon decay: {self.epsilon_start} → {self.epsilon_end} over {self.epsilon_anneal_steps} steps\n"
            f"  → Using device: {self.device}"
        )

    # ============================================================
    #                    MAIN ROLLOUT
    # ============================================================
    def generate_episode(self, global_ep_idx=0, evaluate=False):
        """Generate one full episode rollout."""
        self._maybe_decay_epsilon(evaluate)
        self._reset_hidden_states()

        obs_list, info = self._reset_env()
        s = self._get_state_from_info(info)
        avail = self._get_avail_from_info(info, self.agents.n_actions)

        ep_transitions, ep_reward, t = [], 0.0, 0
        gantt_data = []  # placeholder for future visualization support

        while t < self.episode_limit:
            actions, q_vals = self._select_actions(obs_list, avail, evaluate)
            obs_next, reward, done, info_next = self.env.step(actions)

            s_next = self._get_state_from_info(info_next)
            avail_next = self._get_avail_from_info(info_next, self.agents.n_actions)

            trans = dict(
                o=self._obs_list_to_array(obs_list),
                o_next=self._obs_list_to_array(obs_next),
                s=s.astype(np.float32),
                s_next=s_next.astype(np.float32),
                u=np.asarray(actions, np.int64),
                avail_a=self._normalize_avail(avail),
                avail_a_next=self._normalize_avail(avail_next),
                r=float(reward),
                terminated=bool(done),
                padded=False,
            )
            ep_transitions.append(trans)
            ep_reward += float(reward)

            obs_list, s, avail = obs_next, s_next, avail_next
            t += 1
            if done:
                break

        # episode duration tracking for KPI reporting
        self.episode_duration = len(ep_transitions)

        episode = self._pack_episode(ep_transitions)
        if self.buffer is not None:
            self._insert_episode(episode)

        print(
            f"[Rollout] Episode finished in {self.episode_duration} steps | total reward={ep_reward:.2f}"
        )
        if self.buffer is not None:
            print(f"Episode reward: {ep_reward:.2f}, buffer length: {len(self.buffer)}")

        # Return with gantt_data placeholder (empty list for now)
        return episode, ep_reward, bool(ep_transitions[-1]["terminated"]), gantt_data

    # ============================================================
    #                    ACTION SELECTION
    # ============================================================
    def _select_actions(self, obs_list, avail, evaluate=False):
        n_agents, n_actions = self.agents.n_agents, self.agents.n_actions
        obs_t = torch.as_tensor(
            self._obs_list_to_array(obs_list), dtype=torch.float32, device=self.device
        )

        if hasattr(self.agents.policy, "eval_rnn"):
            q_t, h_out = self.agents.policy.eval_rnn(obs_t, self.eval_hidden)
            self.eval_hidden = h_out
        elif hasattr(self.agents.policy, "get_q_values"):
            batch = {"o": obs_t.unsqueeze(0)}
            q_t, _ = self.agents.policy.get_q_values(batch, t=0)
            q_t = q_t.squeeze(0)
        else:
            raise RuntimeError("Policy must implement eval_rnn() or get_q_values().")

        q_np = q_t.detach().cpu().numpy()

        mask = (
            np.ones((n_agents, n_actions), bool)
            if avail is None
            else np.asarray(avail, bool)
        )
        if mask.ndim == 1:
            mask = np.tile(mask[None, :], (n_agents, 1))
        masked_q = np.where(mask, q_np, -1e9)

        actions = []
        for i in range(n_agents):
            if (not evaluate) and (np.random.rand() < self.epsilon):
                valid = np.where(mask[i])[0]
                a = np.random.choice(valid) if valid.size else np.random.randint(0, n_actions)
            else:
                a = int(np.argmax(masked_q[i]))
            actions.append(a)
        return actions, q_np

    # ============================================================
    #                    HELPERS / UTILS
    # ============================================================
    def _reset_env(self):
        out = self.env.reset()
        return out if isinstance(out, tuple) and len(out) == 2 else (out, {})

    def _get_state_from_info(self, info):
        s = info.get("state_vec", None)
        if s is None:
            dim = getattr(self.env, "state_dim", 64)
            return np.zeros((dim,), np.float32)
        return np.asarray(s, np.float32).reshape(-1)

    def _get_avail_from_info(self, info, n_actions):
        avail = info.get("avail_actions", None)
        if avail is None:
            return None
        avail = np.asarray(avail, bool)
        if avail.ndim == 1 and avail.size != n_actions:
            return None
        if avail.ndim == 2 and avail.shape[1] != n_actions:
            return None
        return avail

    def _obs_list_to_array(self, obs_list):
        arr = (
            obs_list
            if isinstance(obs_list, np.ndarray)
            else np.asarray([np.asarray(o, np.float32).ravel() for o in obs_list], np.float32)
        )
        target = getattr(self.env, "obs_dim_agent", arr.shape[1])
        if arr.shape[1] < target:
            arr = np.pad(arr, ((0, 0), (0, target - arr.shape[1])))
        elif arr.shape[1] > target:
            arr = arr[:, :target]
        return arr.astype(np.float32)

    def _normalize_avail(self, avail):
        if avail is None:
            return None
        mask = np.asarray(avail, bool)
        if mask.ndim == 1:
            mask = np.tile(mask[None, :], (self.agents.n_agents, 1))
        return mask

    def _pack_episode(self, trans):
        T = len(trans)
        def stack_or_none(k):
            vals = [t[k] for t in trans]
            return None if vals[0] is None else np.stack(vals)
        ep = {
            "o": np.stack([t["o"] for t in trans]).astype(np.float32),
            "o_next": np.stack([t["o_next"] for t in trans]).astype(np.float32),
            "s": np.stack([t["s"] for t in trans]).astype(np.float32),
            "s_next": np.stack([t["s_next"] for t in trans]).astype(np.float32),
            "u": np.stack([t["u"] for t in trans]).astype(np.int64),
            "avail_a": stack_or_none("avail_a"),
            "avail_a_next": stack_or_none("avail_a_next"),
            "r": np.array([t["r"] for t in trans], np.float32),
            "terminated": np.array([t["terminated"] for t in trans], np.float32),
            "padded": np.array([t["padded"] for t in trans], np.float32),
            "episode_len": T,
        }
        return ep

    def _insert_episode(self, ep):
        for name in ("store_episode", "insert_episode", "push_episode", "add_episode", "push"):
            fn = getattr(self.buffer, name, None)
            if callable(fn):
                fn(ep)
                return
        add_fn = getattr(self.buffer, "add", None)
        if callable(add_fn):
            for t in range(ep["episode_len"]):
                add_fn({k: v[t] for k, v in ep.items() if k not in ("episode_len",)})

    def _maybe_decay_epsilon(self, evaluate):
        if not evaluate and self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon - self._eps_decay)

    def _reset_hidden_states(self):
        n_agents = getattr(self.agents, "n_agents", 1)
        hdim = 64
        if hasattr(self.agents.policy, "rnn") and hasattr(self.agents.policy.rnn, "hidden_size"):
            hdim = self.agents.policy.rnn.hidden_size
        self.eval_hidden = torch.zeros((n_agents, hdim), dtype=torch.float32, device=self.device)
