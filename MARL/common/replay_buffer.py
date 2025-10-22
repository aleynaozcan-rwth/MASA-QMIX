# MARL/common/replay_buffer.py
# Step 8A.6.5 – Episodic Replay Buffer (heterogeneous job-aware)
# --------------------------------------------------------------
# Updates from 8A.6.4:
#   • Fallback obs_dim updated from 10 → 11 (progress_ratio feature)
#   • Job-agent IDs are tracked in batches for analysis/debug
#   • Sampling remains fully job-agnostic (pattern-based generalization)
#   • Optional batch summary print for monitoring (disabled by default)

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
import numpy as np
import random


Transition = Dict[str, Any]
Episode = List[Transition]


class ReplayBuffer:
    def __init__(
        self,
        episode_capacity: int = 1000,
        seed: int = 123,
    ):
        self._episodes: deque[Episode] = deque(maxlen=int(episode_capacity))
        self._current: Episode = []   # used by streaming API (add_transition)
        self._rng = random.Random(seed)
        self._num_transitions: int = 0

    # ---------------------------------------------------------------------
    # Streaming API
    # ---------------------------------------------------------------------
    def add_transition(
        self,
        jobagent_id: int,
        time: float,
        state: Optional[np.ndarray],
        action: Union[int, np.ndarray],
        reward: float,
        next_state: Optional[np.ndarray],
        done: bool,
        obs: Optional[Sequence[np.ndarray]] = None,
        obs_next: Optional[Sequence[np.ndarray]] = None,
        avail_u: Optional[np.ndarray] = None,
        avail_u_next: Optional[np.ndarray] = None,
        agent_actions: Optional[Sequence[int]] = None,
        n_actions: Optional[int] = None,
    ):
        """Insert a single transition. When done=True, closes current episode automatically."""
        tr: Transition = {
            "jobagent_id": int(jobagent_id),
            "time": float(time),
            "reward": float(reward),
            "done": bool(done),
        }

        # Global state (optional)
        if state is not None:
            tr["state"] = np.asarray(state, dtype=np.float32)
        if next_state is not None:
            tr["state_next"] = np.asarray(next_state, dtype=np.float32)

        # Per-agent observations
        if obs is not None:
            tr["o"] = _as_agents_obs(obs)
        if obs_next is not None:
            tr["o_next"] = _as_agents_obs(obs_next)

        # Avail actions
        if avail_u is not None:
            tr["avail_u"] = np.asarray(avail_u, dtype=np.float32)
        if avail_u_next is not None:
            tr["avail_u_next"] = np.asarray(avail_u_next, dtype=np.float32)

        # Actions
        if agent_actions is not None:
            u = np.asarray(agent_actions, dtype=np.int64).reshape(-1, 1)
            tr["u"] = u
            if n_actions is not None:
                tr["u_onehot"] = _to_onehot(u.squeeze(-1), n_actions)
                tr["n_actions"] = int(n_actions)
        else:
            a = np.asarray(action, dtype=np.int64).reshape(1, 1)
            tr["u"] = a

        # Append to current episode
        self._current.append(tr)
        self._num_transitions += 1

        # Close episode when done=True
        if done:
            self._close_current_episode()

    def _close_current_episode(self):
        if len(self._current) > 0:
            self._episodes.append(self._current)
            self._current = []

    # ---------------------------------------------------------------------
    # Episodic API
    # ---------------------------------------------------------------------
    def store_episode(self, episode_batch: Union[Episode, Dict[str, np.ndarray]]):
        """Store a full episode (either list-of-transitions or packed dict)."""
        if isinstance(episode_batch, list):
            self._episodes.append(episode_batch)
            self._num_transitions += len(episode_batch)
            return

        if isinstance(episode_batch, dict):
            T = _time_len_from_dict(episode_batch)
            ep: Episode = []
            for t in range(T):
                tr: Transition = {}
                for k, arr in episode_batch.items():
                    tr[k] = _slice_first_dim(arr, t)
                if "terminated" in tr:
                    tr["done"] = bool(np.asarray(tr["terminated"]).item() > 0.5)
                ep.append(tr)
            self._episodes.append(ep)
            self._num_transitions += len(ep)
            return

        raise TypeError("Unsupported episode format passed to store_episode().")

    # ---------------------------------------------------------------------
    # Sampling (job-agnostic, pattern-based)
    # ---------------------------------------------------------------------
    def sample(
        self,
        batch_size: int = 32,
        n_actions: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Sample a batch of episodes and return a padded dict for QMIX training."""
        if len(self._episodes) == 0 and len(self._current) == 0:
            return None

        pool = list(self._episodes)
        if len(self._current) > 0:
            pool.append(list(self._current))

        B = min(batch_size, len(pool))
        episodes = self._rng.sample(pool, k=B)
        ep_lengths = [len(ep) for ep in episodes]
        T = max(ep_lengths) if max_seq_len is None else int(max_seq_len)
        n_agents, obs_dim = _infer_agents_obs(episodes)

        # Pre-allocate main tensors
        o = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
        o_next = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
        u = np.zeros((B, T, n_agents, 1), dtype=np.int64)
        r = np.zeros((B, T, 1), dtype=np.float32)
        terminated = np.zeros((B, T, 1), dtype=np.float32)
        filled = np.zeros((B, T, 1), dtype=np.float32)

        have_avail = have_state = have_u_onehot = False
        avail_u = avail_u_next = state = state_next = u_onehot = None

        for b, ep in enumerate(episodes):
            t_limit = min(T, len(ep))
            for t in range(t_limit):
                tr = ep[t]
                # Observations
                if "o" in tr:
                    o[b, t] = _as_agents_obs(tr["o"], ensure_shape=(n_agents, obs_dim))
                if "o_next" in tr:
                    o_next[b, t] = _as_agents_obs(tr["o_next"], ensure_shape=(n_agents, obs_dim))
                # Actions
                if "u" in tr:
                    ut = np.asarray(tr["u"], dtype=np.int64).reshape(-1, 1)
                    if ut.shape[0] != n_agents:
                        ut = np.repeat(ut[:1, :], n_agents, axis=0)
                    u[b, t] = ut
                # Reward
                if "reward" in tr:
                    r[b, t, 0] = float(tr["reward"])
                # Done
                done_val = tr.get("done", tr.get("terminated", False))
                terminated[b, t, 0] = 1.0 if bool(done_val) else 0.0
                # Avail actions
                if "avail_u" in tr:
                    if not have_avail:
                        _na = tr["avail_u"].shape[-1]
                        avail_u = np.zeros((B, T, n_agents, _na), dtype=np.float32)
                        avail_u_next = np.zeros((B, T, n_agents, _na), dtype=np.float32)
                        have_avail = True
                    avail_u[b, t] = np.asarray(tr["avail_u"], dtype=np.float32)
                if "avail_u_next" in tr and have_avail:
                    avail_u_next[b, t] = np.asarray(tr["avail_u_next"], dtype=np.float32)
                # States
                if "state" in tr:
                    if not have_state:
                        sd = int(np.asarray(tr["state"]).size)
                        state = np.zeros((B, T, sd), dtype=np.float32)
                        state_next = np.zeros((B, T, sd), dtype=np.float32)
                        have_state = True
                    state[b, t] = np.asarray(tr["state"], dtype=np.float32).reshape(-1)
                if "state_next" in tr and have_state:
                    state_next[b, t] = np.asarray(tr["state_next"], dtype=np.float32).reshape(-1)
                # One-hot
                if "u_onehot" in tr:
                    if not have_u_onehot:
                        _na = tr["u_onehot"].shape[-1]
                        u_onehot = np.zeros((B, T, n_agents, _na), dtype=np.float32)
                        have_u_onehot = True
                    u_onehot[b, t] = np.asarray(tr["u_onehot"], dtype=np.float32)
                # Filled mask
                filled[b, t, 0] = 1.0

        # One-hot fallback
        if u_onehot is None and n_actions is not None:
            u_onehot = np.zeros((B, T, n_agents, n_actions), dtype=np.float32)
            idx = u.squeeze(-1)
            for b in range(B):
                for t in range(T):
                    for a in range(n_agents):
                        if filled[b, t, 0] > 0.5:
                            u_onehot[b, t, a, idx[b, t, a]] = 1.0

        # Job-agent IDs (for analysis)
        job_ids = np.zeros((B, 1), dtype=np.int32)
        for b, ep in enumerate(episodes):
            if len(ep) > 0 and "jobagent_id" in ep[0]:
                job_ids[b, 0] = int(ep[0]["jobagent_id"])

        batch = {
            "o": o,
            "o_next": o_next,
            "u": u,
            "r": r,
            "terminated": terminated,
            "filled": filled,
            "job_ids": job_ids,
        }
        if u_onehot is not None:
            batch["u_onehot"] = u_onehot
        if have_avail:
            batch["avail_u"] = avail_u
            batch["avail_u_next"] = avail_u_next
        if have_state:
            batch["state"] = state
            batch["state_next"] = state_next

        return batch

    def __len__(self) -> int:
        return self._num_transitions

    def clear(self):
        self._episodes.clear()
        self._current = []
        self._num_transitions = 0


# ============================== helpers ==============================

def _as_agents_obs(x: Union[np.ndarray, Sequence[np.ndarray]], ensure_shape: Optional[tuple] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if ensure_shape is not None:
        n_agents, obs_dim = ensure_shape
        if arr.shape[0] < n_agents:
            pad = np.zeros((n_agents - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > n_agents:
            arr = arr[:n_agents]
        if arr.shape[1] < obs_dim:
            pad = np.zeros((arr.shape[0], obs_dim - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > obs_dim:
            arr = arr[:, :obs_dim]
    return arr


def _to_onehot(indices: np.ndarray, n_actions: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    onehot = np.zeros((idx.shape[0], int(n_actions)), dtype=np.float32)
    for i, a in enumerate(idx):
        if 0 <= a < n_actions:
            onehot[i, a] = 1.0
    return onehot


def _time_len_from_dict(d: Dict[str, np.ndarray]) -> int:
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            return v.shape[0]
    raise ValueError("Cannot infer time length from episode dict.")


def _slice_first_dim(arr: Any, t: int) -> Any:
    if isinstance(arr, np.ndarray) and arr.ndim >= 1 and t < arr.shape[0]:
        return arr[t]
    return arr


def _infer_agents_obs(episodes: List[Episode]) -> Tuple[int, int]:
    """Infer (n_agents, obs_dim). Fallback now 11D due to progress_ratio feature."""
    for ep in episodes:
        for tr in ep:
            if "o" in tr:
                o = np.asarray(tr["o"], dtype=np.float32)
                if o.ndim == 1:
                    return 1, int(o.shape[0])
                if o.ndim == 2:
                    return int(o.shape[0]), int(o.shape[1])
                if o.ndim == 3 and o.shape[0] == 1:
                    return int(o.shape[1]), int(o.shape[2])
    return 1, 11  # updated fallback
