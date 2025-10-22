import numpy as np
import torch

class RolloutWorker:
    """
    Step 8A.7.3 – Final RolloutWorker for MASA-QMIX
    ------------------------------------------------
    • Builds full RNN input (obs + last_action + agent_id)
    • RNN-safe, replay-ready, epsilon-decay compatible
    • Always includes s / s_next for ReplayBuffer
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
        log_prefix="8A.7.3",
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
        self._eps_decay = (self.epsilon_start - self.epsilon_end) / max(1, self.epsilon_anneal_steps)

        # --- hidden state placeholder ---
        self.eval_hidden = None
        self.episode_duration = 0

        print(f"[INFO] RolloutWorker {self.log_prefix} initialized — unified RNN inputs")
        print(f"  → Episode limit: {self.episode_limit}")
        print(f"  → Epsilon decay: {self.epsilon_start} → {self.epsilon_end} over {self.epsilon_anneal_steps} steps")
        print(f"  → Using device: {self.device}")

    # ============================================================
    #                    MAIN ROLLOUT
    # ============================================================
    def generate_episode(self, global_ep_idx=0, evaluate=False):
        """Generate one full episode rollout."""
        self._maybe_decay_epsilon(evaluate)

        obs_list, info = self._reset_env()
        self._ensure_hidden(len(obs_list))

        s = self._get_state_from_info(info)
        n_actions = getattr(self.agents, "n_actions", getattr(self.env, "num_wcs", None))
        avail = self._get_avail_from_info(info, n_actions)

        ep_transitions, ep_reward, t = [], 0.0, 0
        gantt_data = []

        while t < self.episode_limit:
            actions, q_vals = self._select_actions(obs_list, avail, evaluate)
            obs_next, reward, done, info_next = self.env.step(actions)

            s_next = self._get_state_from_info(info_next)
            avail_next = self._get_avail_from_info(info_next, n_actions)

            # --- Safe conversion for replay buffer ---
            s_arr = np.asarray(s, np.float32).reshape(-1)
            s_next_arr = np.asarray(s_next, np.float32).reshape(-1)
            if s_arr.size == 0:
                s_arr = np.zeros((64,), np.float32)
            if s_next_arr.size == 0:
                s_next_arr = np.zeros((64,), np.float32)

            trans = dict(
                o=self._obs_list_to_array(obs_list),
                o_next=self._obs_list_to_array(obs_next),
                s=s_arr,
                s_next=s_next_arr,
                u=np.asarray(actions, np.int64),
                avail_a=self._normalize_avail(avail, len(obs_list), n_actions),
                avail_a_next=self._normalize_avail(avail_next, len(obs_list), n_actions),
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

        self.episode_duration = len(ep_transitions)
        episode = self._pack_episode(ep_transitions)

        if self.buffer is not None:
            self._insert_episode(episode)

        print(f"[Rollout] Episode finished in {self.episode_duration} steps | total reward={ep_reward:.2f}")
        if self.buffer is not None:
            print(f"Episode reward: {ep_reward:.2f}, buffer length: {len(self.buffer)}")

        return episode, ep_reward, bool(ep_transitions[-1]["terminated"]), gantt_data

    # ============================================================
    #                    ACTION SELECTION
    # ============================================================
    def _select_actions(self, obs_list, avail, evaluate=False):
        n_agents = len(obs_list)
        obs_t = torch.as_tensor(self._obs_list_to_array(obs_list), dtype=torch.float32, device=self.device)

        # --- Build full RNN input (obs + last_action + agent_id) ---
        obs_dim = obs_t.shape[1]
        n_actions = getattr(self.agents, "n_actions", 0)
        last_action = torch.zeros((n_agents, n_actions), dtype=torch.float32, device=self.device)
        agent_ids = torch.eye(n_agents, dtype=torch.float32, device=self.device)

        rnn_input = torch.cat([obs_t, last_action, agent_ids], dim=1)

        # --- RNN Forward Pass ---
        if hasattr(self.agents.policy, "eval_rnn"):
            if self.eval_hidden is None or self.eval_hidden.shape[0] != n_agents:
                self._ensure_hidden(n_agents)
            q_t, h_out = self.agents.policy.eval_rnn(rnn_input, self.eval_hidden)
            self.eval_hidden = h_out
        else:
            raise RuntimeError("Policy must implement eval_rnn().")

        q_np = q_t.detach().cpu().numpy()
        n_actions = q_np.shape[1]

        # --- Mask handling ---
        if avail is None:
            mask = np.ones_like(q_np, dtype=bool)
        else:
            mask = np.asarray(avail, bool)
            if mask.ndim == 1:
                mask = np.tile(mask[None, :], (n_agents, 1))
            if mask.shape != q_np.shape:
                mask = np.ones_like(q_np, dtype=bool)

        masked_q = np.where(mask, q_np, -1e9)
        actions = []
        for i in range(n_agents):
            if not evaluate and np.random.rand() < self.epsilon:
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
        if avail.ndim == 2 and (n_actions is None or avail.shape[1] == int(n_actions)):
            return avail
        if avail.ndim == 1 and n_actions is not None and avail.size == int(n_actions):
            return np.tile(avail[None, :], (len(self.env.jobs), 1))
        return None

    def _obs_list_to_array(self, obs_list):
        arr = np.asarray([np.asarray(o, np.float32).ravel() for o in obs_list], np.float32)
        target = getattr(self.env, "obs_dim_agent", arr.shape[1])
        if arr.shape[1] < target:
            arr = np.pad(arr, ((0, 0), (0, target - arr.shape[1])))
        elif arr.shape[1] > target:
            arr = arr[:, :target]
        return arr.astype(np.float32)

    def _normalize_avail(self, avail, n_agents, n_actions):
        if avail is None:
            return None
        mask = np.asarray(avail, bool)
        if mask.ndim == 1 and n_actions is not None:
            mask = np.tile(mask[None, :], (n_agents, 1))
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

    def _ensure_hidden(self, n_agents: int):
        """Ensure RNN hidden state matches current agent count."""
        hdim = 64
        if hasattr(self.agents.policy, "rnn") and hasattr(self.agents.policy.rnn, "hidden_size"):
            hdim = self.agents.policy.rnn.hidden_size
        self.eval_hidden = torch.zeros((n_agents, hdim), dtype=torch.float32, device=self.device)
