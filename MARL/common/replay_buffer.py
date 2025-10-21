# MARL/common/replay_buffer.py
# Step 8A.6.4 – Episodic Replay Buffer (QMIX-ready, multi-agent)
# --------------------------------------------------------------
# - Stores full episodes with variable lengths.
# - Supports both streaming step-by-step inserts (add_transition) and
#   direct episodic inserts (store_episode).
# - Sampling returns padded, shape-consistent batches for QMIX training.
#
# Batch keys returned by sample():
#   o:           (B, T, n_agents, obs_dim)
#   o_next:      (B, T, n_agents, obs_dim)
#   u:           (B, T, n_agents, 1)           action indices (int64)
#   u_onehot:    (B, T, n_agents, n_actions)   one-hot actions (float32)
#   r:           (B, T, 1)                     scalar reward per step
#   terminated:  (B, T, 1)                     episode-done flags
#   filled:      (B, T, 1)                     padding mask (1=real, 0=pad)
# Optional (if provided at insert time):
#   avail_u:       (B, T, n_agents, n_actions)
#   avail_u_next:  (B, T, n_agents, n_actions)
#   state:         (B, T, state_dim)
#   state_next:    (B, T, state_dim)

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Union
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

        # Stats
        self._num_transitions: int = 0

    # ---------------------------------------------------------------------
    # Streaming API (compatible with earlier flat usage)
    # ---------------------------------------------------------------------
    def add_transition(
        self,
        jobagent_id: int,
        time: float,
        state: Optional[np.ndarray],      # global state (optional)
        action: Union[int, np.ndarray],   # per-agent or joint; we normalize below
        reward: float,
        next_state: Optional[np.ndarray], # global state next (optional)
        done: bool,

        # NEW (recommended for MARL/QMIX):
        obs: Optional[Sequence[np.ndarray]] = None,          # list/array (n_agents, obs_dim)
        obs_next: Optional[Sequence[np.ndarray]] = None,     # list/array (n_agents, obs_dim)
        avail_u: Optional[np.ndarray] = None,                # (n_agents, n_actions)
        avail_u_next: Optional[np.ndarray] = None,           # (n_agents, n_actions)
        agent_actions: Optional[Sequence[int]] = None,       # per-agent action indices
        n_actions: Optional[int] = None,                     # to build onehots (if agent_actions given)
    ):
        """
        Streaming insert of a single transition. When done=True arrives, the
        current episode is moved into the ring buffer automatically.

        Notes:
          - If you pass per-agent observations in `obs` / `obs_next`, they will
            be used as 'o' / 'o_next'. Otherwise they are omitted.
          - If you pass `agent_actions` (length n_agents), it will be stored as 'u'
            (shape normalized to (n_agents, 1)) and onehot built if n_actions is known.
          - If you only pass scalar `action`, we still store it (as array shape (1, 1)).
        """
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

        # Per-agent obs (optional but recommended for QMIX)
        if obs is not None:
            tr["o"] = _as_agents_obs(obs)
        if obs_next is not None:
            tr["o_next"] = _as_agents_obs(obs_next)

        # Avail actions (optional)
        if avail_u is not None:
            tr["avail_u"] = np.asarray(avail_u, dtype=np.float32)
        if avail_u_next is not None:
            tr["avail_u_next"] = np.asarray(avail_u_next, dtype=np.float32)

        # Actions
        if agent_actions is not None:
            u = np.asarray(agent_actions, dtype=np.int64).reshape(-1, 1)  # (n_agents, 1)
            tr["u"] = u
            if n_actions is not None:
                tr["u_onehot"] = _to_onehot(u.squeeze(-1), n_actions)  # (n_agents, n_actions)
                tr["n_actions"] = int(n_actions)
        else:
            # Fallback: single scalar action
            a = np.asarray(action, dtype=np.int64).reshape(1, 1)        # (1, 1)
            tr["u"] = a

        # Append to current episode
        self._current.append(tr)
        self._num_transitions += 1

        # Close episode on done
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
        """
        Store a full episode.

        Accepted formats:
          1) List[Transition]: each Transition may contain keys described above.
          2) Dict[str, np.ndarray] packed with timesteps first (T, ...)
             and containing at least:
               - 'o', 'o_next' with shape (T, n_agents, obs_dim)
               - 'u' with shape (T, n_agents, 1) or (T, 1, 1)
               - 'r' with shape (T, 1)
               - 'terminated' with shape (T, 1)
             Optional: 'u_onehot', 'avail_u', 'avail_u_next', 'state', 'state_next'
        """
        if isinstance(episode_batch, list):
            # Already a list of transitions
            self._episodes.append(episode_batch)
            self._num_transitions += len(episode_batch)
            return

        if isinstance(episode_batch, dict):
            # Unpack dict-of-arrays into list of per-timestep transitions
            T = _time_len_from_dict(episode_batch)
            ep: Episode = []
            for t in range(T):
                tr: Transition = {}
                for k, arr in episode_batch.items():
                    tr[k] = _slice_first_dim(arr, t)
                # Align field names
                if "terminated" in tr:
                    tr["done"] = bool(np.asarray(tr["terminated"]).item() > 0.5)
                ep.append(tr)
            self._episodes.append(ep)
            self._num_transitions += len(ep)
            return

        raise TypeError("Unsupported episode format passed to store_episode().")

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    def sample(
        self,
        batch_size: int = 32,
        n_actions: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Sample a batch of episodes and return a padded dict suitable for QMIX.

        Args:
          batch_size: number of episodes to sample.
          n_actions: if given, ensures 'u_onehot' is present (build if missing).
          max_seq_len: if given, pad/clip to this T; otherwise use max T among sampled episodes.

        Returns:
          dict with arrays described at the top. Returns None if buffer is empty.
        """
        if len(self._episodes) == 0 and len(self._current) == 0:
            return None

        # If a streaming episode is open, do not lose it
        if len(self._current) > 0:
            # move a copy into the pool (do not clear the live current)
            # This ensures sampling sees latest context without truncating it.
            tmp_ep = list(self._current)
            pool = list(self._episodes) + [tmp_ep]
        else:
            pool = list(self._episodes)

        # Choose episodes
        B = min(batch_size, len(pool))
        episodes = self._rng.sample(pool, k=B)

        # Determine shapes by scanning episodes
        ep_lengths = [len(ep) for ep in episodes]
        T = max(ep_lengths) if max_seq_len is None else int(max_seq_len)

        # Infer n_agents and obs_dim from first episode that has 'o'
        n_agents, obs_dim = _infer_agents_obs(episodes)

        # Pre-allocate
        o        = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
        o_next   = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
        u        = np.zeros((B, T, n_agents, 1), dtype=np.int64)
        r        = np.zeros((B, T, 1), dtype=np.float32)
        terminated = np.zeros((B, T, 1), dtype=np.float32)
        filled     = np.zeros((B, T, 1), dtype=np.float32)

        # Optional blocks – allocate lazily when first seen
        have_avail = False
        have_state = False
        have_u_onehot = False
        avail_u = avail_u_next = None
        state = state_next = None
        u_onehot = None

        # Pack
        for b, ep in enumerate(episodes):
            t_limit = min(T, len(ep))
            for t in range(t_limit):
                tr = ep[t]

                # Observations (per-agent)
                if "o" in tr:
                    ot = _as_agents_obs(tr["o"], ensure_shape=(n_agents, obs_dim))
                    o[b, t] = ot
                if "o_next" in tr:
                    on = _as_agents_obs(tr["o_next"], ensure_shape=(n_agents, obs_dim))
                    o_next[b, t] = on

                # Actions
                if "u" in tr:
                    ut = np.asarray(tr["u"], dtype=np.int64).reshape(-1, 1)
                    if ut.shape[0] != n_agents:
                        # Fallback: single action for all agents? broadcast to n_agents
                        ut = np.repeat(ut[:1, :], n_agents, axis=0)
                    u[b, t] = ut

                # Reward
                if "reward" in tr:
                    r[b, t, 0] = float(tr["reward"])

                # Done flag
                done_val = tr.get("done", tr.get("terminated", False))
                terminated[b, t, 0] = 1.0 if bool(done_val) else 0.0

                # Avail actions
                if "avail_u" in tr:
                    if not have_avail:
                        # Infer n_actions from first seen avail (n_agents, n_actions)
                        _na = tr["avail_u"].shape[-1]
                        avail_u       = np.zeros((B, T, n_agents, _na), dtype=np.float32)
                        avail_u_next  = np.zeros((B, T, n_agents, _na), dtype=np.float32)
                        have_avail = True
                    avail_u[b, t] = np.asarray(tr["avail_u"], dtype=np.float32)
                if "avail_u_next" in tr and have_avail:
                    avail_u_next[b, t] = np.asarray(tr["avail_u_next"], dtype=np.float32)

                # Global state
                if "state" in tr:
                    if not have_state:
                        sd = int(np.asarray(tr["state"]).size)
                        state      = np.zeros((B, T, sd), dtype=np.float32)
                        state_next = np.zeros((B, T, sd), dtype=np.float32)
                        have_state = True
                    state[b, t] = np.asarray(tr["state"], dtype=np.float32).reshape(-1)
                if "state_next" in tr and have_state:
                    state_next[b, t] = np.asarray(tr["state_next"], dtype=np.float32).reshape(-1)

                # Onehot (if already present)
                if "u_onehot" in tr:
                    if not have_u_onehot:
                        _na = tr["u_onehot"].shape[-1]
                        u_onehot = np.zeros((B, T, n_agents, _na), dtype=np.float32)
                        have_u_onehot = True
                    u_onehot[b, t] = np.asarray(tr["u_onehot"], dtype=np.float32)

                # Filled mask
                filled[b, t, 0] = 1.0

        # If onehot missing but n_actions provided, build it
        if u_onehot is None and n_actions is not None:
            u_onehot = np.zeros((B, T, n_agents, n_actions), dtype=np.float32)
            # take indices from u
            idx = u.squeeze(-1)  # (B, T, n_agents)
            for b in range(B):
                for t in range(T):
                    for a in range(n_agents):
                        if filled[b, t, 0] > 0.5:
                            u_onehot[b, t, a, idx[b, t, a]] = 1.0

        batch = {
            "o": o,
            "o_next": o_next,
            "u": u,
            "r": r,
            "terminated": terminated,
            "filled": filled,
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

    # ---------------------------------------------------------------------
    # Python protocol
    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        """Total number of stored transitions across all closed episodes (open current excluded)."""
        return self._num_transitions

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def clear(self):
        self._episodes.clear()
        self._current = []
        self._num_transitions = 0


# ============================== helpers ==============================

def _as_agents_obs(x: Union[np.ndarray, Sequence[np.ndarray]], ensure_shape: Optional[tuple] = None) -> np.ndarray:
    """
    Normalize per-agent observation into array of shape (n_agents, obs_dim).
    Accepts list[ndarray] or ndarray; will attempt to stack/reshape.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        # (obs_dim,) -> (1, obs_dim)
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3:
        # sometimes comes as (1, n_agents, obs_dim)
        if arr.shape[0] == 1:
            arr = arr[0]
    # else expect (n_agents, obs_dim)
    if ensure_shape is not None:
        n_agents, obs_dim = ensure_shape
        # Pad or crop agents dimension if necessary (rare)
        if arr.shape[0] < n_agents:
            pad = np.zeros((n_agents - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > n_agents:
            arr = arr[:n_agents]
        # Pad/crop obs_dim if necessary
        if arr.shape[1] < obs_dim:
            pad = np.zeros((arr.shape[0], obs_dim - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > obs_dim:
            arr = arr[:, :obs_dim]
    return arr


def _to_onehot(indices: np.ndarray, n_actions: int) -> np.ndarray:
    """indices: (n_agents,) -> (n_agents, n_actions) onehot."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    onehot = np.zeros((idx.shape[0], int(n_actions)), dtype=np.float32)
    for i, a in enumerate(idx):
        if 0 <= a < n_actions:
            onehot[i, a] = 1.0
    return onehot


def _time_len_from_dict(d: Dict[str, np.ndarray]) -> int:
    """Infer T from first array whose first dim is time."""
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            return v.shape[0]
    raise ValueError("Cannot infer time length from episode dict.")


def _slice_first_dim(arr: Any, t: int) -> Any:
    """Safely slice arr[t] if possible, else return arr."""
    if isinstance(arr, np.ndarray) and arr.ndim >= 1 and t < arr.shape[0]:
        return arr[t]
    return arr


def _infer_agents_obs(episodes: List[Episode]) -> (int, int):
    """Scan episodes to infer (n_agents, obs_dim). Fallback to (1, 10)."""
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
    # Fallback defaults aligned with earlier code (obs_dim_agent=10)
    return 1, 10
