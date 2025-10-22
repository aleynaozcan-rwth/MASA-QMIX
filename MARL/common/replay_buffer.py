# MARL/common/replay_buffer.py
# Step 8A.7.2 – Episodic Replay Buffer (MASA-QMIX, replay-aware)
# --------------------------------------------------------------
# • Accepts whole episodes (list-of-transitions or dict-of-arrays)
# • Safe handling of 'terminated' shape (scalar/1D/2D)
# • Samples (B, T, ...) mini-batches with keys expected by QMIX.learn()

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
import numpy as np
import random

Transition = Dict[str, Any]
Episode = List[Transition]


class ReplayBuffer:
    def __init__(self, episode_capacity: int = 1000, seed: int = 123):
        self._episodes: deque[Episode] = deque(maxlen=int(episode_capacity))
        self._current: Episode = []
        self._rng = random.Random(seed)
        self._num_transitions: int = 0

    # -----------------------------------------------------------
    # Store a full episode (as list of transitions OR dict batch)
    # -----------------------------------------------------------
    def store_episode(self, episode_batch: Union[Episode, Dict[str, np.ndarray]]):
        # Case 1: classic list of transitions
        if isinstance(episode_batch, list):
            self._episodes.append(episode_batch)
            self._num_transitions += len(episode_batch)
            return

        # Case 2: dict of arrays stacked over time (T first dimension)
        if isinstance(episode_batch, dict):
            T = _infer_time_len_from_dict(episode_batch)
            ep: Episode = []
            for t in range(T):
                tr: Transition = {}
                for k, arr in episode_batch.items():
                    tr[k] = _take_step(arr, t)

                # Robust 'done' from 'terminated'
                if "terminated" in tr:
                    term = np.asarray(tr["terminated"]).astype(np.float32).flatten()
                    tr["done"] = bool(term[-1] > 0.5)

                ep.append(tr)

            self._episodes.append(ep)
            self._num_transitions += len(ep)
            return

        raise TypeError("Unsupported episode format passed to store_episode().")

    # -----------------------------------------------------------
    # Sample a mini-batch of episodes (B, T, ...)
    # -----------------------------------------------------------
    def sample(
        self,
        batch_size: int = 32,
        n_actions: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
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

        # Allocate tensors
        o      = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
        o_next = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
        u      = np.zeros((B, T, n_agents, 1), dtype=np.int64)
        r      = np.zeros((B, T, 1), dtype=np.float32)
        terminated = np.zeros((B, T, 1), dtype=np.float32)
        filled     = np.zeros((B, T, 1), dtype=np.float32)

        # Optional blocks
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

                # Actions (indices)
                if "u" in tr:
                    ut = np.asarray(tr["u"], dtype=np.int64).reshape(-1, 1)
                    if ut.shape[0] != n_agents:
                        ut = np.repeat(ut[:1, :], n_agents, axis=0)
                    u[b, t] = ut

                # Rewards
                if "r" in tr:
                    r[b, t, 0] = float(tr["r"])

                # Termination
                done_val = tr.get("done", tr.get("terminated", False))
                terminated[b, t, 0] = 1.0 if bool(done_val) else 0.0

                # Avail masks
                if "avail_a" in tr:
                    if not have_avail:
                        _na = int(np.asarray(tr["avail_a"]).shape[-1])
                        avail_u = np.zeros((B, T, n_agents, _na), dtype=np.float32)
                        avail_u_next = np.zeros((B, T, n_agents, _na), dtype=np.float32)
                        have_avail = True
                    avail_u[b, t] = np.asarray(tr["avail_a"], dtype=np.float32)
                if "avail_a_next" in tr and have_avail:
                    avail_u_next[b, t] = np.asarray(tr["avail_a_next"], dtype=np.float32)

                # State vectors (global state)
                if "s" in tr:
                    if not have_state:
                        sd = int(np.asarray(tr["s"]).size)
                        state = np.zeros((B, T, sd), dtype=np.float32)
                        state_next = np.zeros((B, T, sd), dtype=np.float32)
                        have_state = True
                    state[b, t] = np.asarray(tr["s"], dtype=np.float32).reshape(-1)
                if "s_next" in tr and have_state:
                    state_next[b, t] = np.asarray(tr["s_next"], dtype=np.float32).reshape(-1)

                # One-hot actions (optional)
                if "u_onehot" in tr:
                    if not have_u_onehot:
                        _na = tr["u_onehot"].shape[-1]
                        u_onehot = np.zeros((B, T, n_agents, _na), dtype=np.float32)
                        have_u_onehot = True
                    u_onehot[b, t] = np.asarray(tr["u_onehot"], dtype=np.float32)

                filled[b, t, 0] = 1.0

        # If u_onehot missing but n_actions known → build from 'u'
        if u_onehot is None and n_actions is not None:
            u_onehot = np.zeros((B, T, n_agents, n_actions), dtype=np.float32)
            idx = u.squeeze(-1)
            for b in range(B):
                for t in range(T):
                    if filled[b, t, 0] > 0.5:
                        for a in range(n_agents):
                            u_onehot[b, t, a, idx[b, t, a]] = 1.0

        # Build batch dict (keys that QMIX.learn expects)
        batch = {
            "o": o,
            "o_next": o_next,
            "u": u,
            "r": r,
            "terminated": terminated,
            "filled": filled,
        }
        if have_state:
            batch["state"] = state
            batch["state_next"] = state_next
        if have_avail:
            batch["avail_u"] = avail_u
            batch["avail_u_next"] = avail_u_next
        if u_onehot is not None:
            batch["u_onehot"] = u_onehot

        return batch

    def __len__(self) -> int:
        return self._num_transitions

    def clear(self):
        self._episodes.clear()
        self._current = []
        self._num_transitions = 0


# ============================== helpers ==============================

def _as_agents_obs(x: Union[np.ndarray, Sequence[np.ndarray]], ensure_shape: Optional[tuple] = None) -> np.ndarray:
    """Convert a variety of obs formats into (n_agents, obs_dim) float32."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if ensure_shape is not None:
        n_agents, obs_dim = ensure_shape
        # pad/trim agents
        if arr.shape[0] < n_agents:
            pad = np.zeros((n_agents - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > n_agents:
            arr = arr[:n_agents]
        # pad/trim obs dim
        if arr.shape[1] < obs_dim:
            pad = np.zeros((arr.shape[0], obs_dim - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > obs_dim:
            arr = arr[:, :obs_dim]
    return arr


def _infer_time_len_from_dict(d: Dict[str, np.ndarray]) -> int:
    """Infer T from first array key; rollout packs time on axis 0."""
    for _, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            return int(v.shape[0])
    raise ValueError("Cannot infer time length from episode dict.")


def _take_step(arr: Any, t: int) -> Any:
    """Take time step t from stacked arrays (T first)."""
    if isinstance(arr, np.ndarray) and arr.ndim >= 1 and t < arr.shape[0]:
        return arr[t]
    return arr


def _infer_agents_obs(episodes: List[Episode]) -> Tuple[int, int]:
    """Infer (n_agents, obs_dim) by inspecting first available 'o'."""
    for ep in episodes:
        for tr in ep:
            if "o" in tr:
                o = np.asarray(tr["o"], dtype=np.float32)
                if o.ndim == 1:
                    return 1, int(o.shape[0])
                if o.ndim == 2:
                    return int(o.shape[0]), int(o.shape[1])
                if o.ndim == 3 and o.shape[0] == 1:  # (1, n_agents, obs_dim)
                    return int(o.shape[1]), int(o.shape[2])
    return 1, 11  # safe fallback
