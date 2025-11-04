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
import logging

Transition = Dict[str, Any]
Episode = List[Transition]


class ReplayBuffer:
    def __init__(self, episode_capacity: int = 1000, seed: int = 123, n_agents: Optional[int] = None, obs_dim: Optional[int] = None):
        self._episodes: deque[Episode] = deque(maxlen=int(episode_capacity))
        self._current: Episode = []
        self._rng = random.Random(seed)
        self._num_transitions: int = 0
        # Optional pre-specified runtime shapes (preferred over inference)
        self._n_agents = int(n_agents) if n_agents is not None else None
        self._obs_dim = int(obs_dim) if obs_dim is not None else None

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

        # Prefer runner-provided shapes when available; otherwise infer from episodes
        if getattr(self, '_n_agents', None) is not None and getattr(self, '_obs_dim', None) is not None:
            n_agents, obs_dim = int(self._n_agents), int(self._obs_dim)
        else:
            n_agents, obs_dim = _infer_agents_obs(episodes)

        # Pre-scan episodes to find the maximum availability vector length so
        # we can allocate a consistent avail_u tensor even when some
        # transitions contain flattened operator-level masks and others
        # contain per-machine rows. This avoids broadcasting errors when
        # episode transitions have variable avail lengths.
        max_avail_len = 0
        for ep in episodes:
            for tr in ep:
                if "avail_a" in tr:
                    try:
                        _len = int(np.asarray(tr["avail_a"]).shape[-1])
                        if _len > max_avail_len:
                            max_avail_len = _len
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        continue

        # Allocate tensors
        o      = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
        o_next = np.zeros((B, T, n_agents, obs_dim), dtype=np.float32)
        u      = np.zeros((B, T, n_agents, 1), dtype=np.int64)
        u_machine = np.full((B, T, n_agents, 1), -1, dtype=np.int64)
        r      = np.zeros((B, T, 1), dtype=np.float32)
        terminated = np.zeros((B, T, 1), dtype=np.float32)
        filled     = np.zeros((B, T, 1), dtype=np.float32)

        # Optional blocks
        have_avail = have_state = have_u_onehot = False
        avail_u = avail_u_next = state = state_next = u_onehot = None

        # If we pre-detected avail vectors across episodes, pre-allocate
        # the avail arrays to the maximum observed length. This avoids
        # reallocating during the main loop and keeps shapes consistent.
        if max_avail_len > 0:
            have_avail = True
            avail_u = np.zeros((B, T, n_agents, max_avail_len), dtype=np.float32)
            avail_u_next = np.zeros((B, T, n_agents, max_avail_len), dtype=np.float32)

        for b, ep in enumerate(episodes):
            t_limit = min(T, len(ep))
            for t in range(t_limit):
                tr = ep[t]

                # Observations
                if "o" in tr:
                    try:
                        arr_o = _as_agents_obs(tr["o"], ensure_shape=(n_agents, obs_dim))
                    except Exception:
                        arr_o = _as_agents_obs(tr["o"]) if 'o' in tr else np.zeros((n_agents, obs_dim), dtype=np.float32)
                    if __debug__:
                        try:
                            print(f"[DEBUG shapes] o target={(n_agents, obs_dim)} src={arr_o.shape} (b={b},t={t})")
                        except Exception:
                            print(f"[DEBUG shapes] o src={getattr(arr_o, 'shape', None)} (b={b},t={t})")
                    # ensure safe copy to avoid broadcasting errors
                    if arr_o.shape != (n_agents, obs_dim):
                        tmp = np.zeros((n_agents, obs_dim), dtype=np.float32)
                        r = min(arr_o.shape[0], n_agents)
                        c = min(arr_o.shape[1], obs_dim if arr_o.ndim > 1 else arr_o.shape[1])
                        try:
                            tmp[:r, :c] = arr_o[:r, :c]
                        except Exception:
                            # final fallback: flatten and copy leading elements
                            flat = np.asarray(arr_o).reshape(-1)
                            flat = np.pad(flat, (0, max(0, n_agents * obs_dim - flat.size)), mode='constant')[: n_agents * obs_dim]
                            tmp = flat.reshape((n_agents, obs_dim))
                        arr_o = tmp
                    o[b, t] = arr_o
                if "o_next" in tr:
                    try:
                        arr_on = _as_agents_obs(tr["o_next"], ensure_shape=(n_agents, obs_dim))
                    except Exception:
                        arr_on = _as_agents_obs(tr.get("o_next", []))
                    if __debug__:
                        try:
                            print(f"[DEBUG shapes] o_next target={(n_agents, obs_dim)} src={arr_on.shape} (b={b},t={t})")
                        except Exception:
                            print(f"[DEBUG shapes] o_next src={getattr(arr_on, 'shape', None)} (b={b},t={t})")
                    if arr_on.shape != (n_agents, obs_dim):
                        tmp = np.zeros((n_agents, obs_dim), dtype=np.float32)
                        r = min(arr_on.shape[0], n_agents)
                        c = min(arr_on.shape[1], obs_dim if arr_on.ndim > 1 else arr_on.shape[1])
                        try:
                            tmp[:r, :c] = arr_on[:r, :c]
                        except Exception:
                            flat = np.asarray(arr_on).reshape(-1)
                            flat = np.pad(flat, (0, max(0, n_agents * obs_dim - flat.size)), mode='constant')[: n_agents * obs_dim]
                            tmp = flat.reshape((n_agents, obs_dim))
                        arr_on = tmp
                    o_next[b, t] = arr_on

                # Actions (indices)
                if "u" in tr:
                    ut = np.asarray(tr["u"], dtype=np.int64).reshape(-1, 1)
                    if ut.shape[0] != n_agents:
                        ut = np.repeat(ut[:1, :], n_agents, axis=0)
                    u[b, t] = ut
                # Machine-level action id (optional)
                if "u_machine" in tr:
                    utm = np.asarray(tr["u_machine"], dtype=np.int64).reshape(-1, 1)
                    if utm.shape[0] != n_agents:
                        utm = np.repeat(utm[:1, :], n_agents, axis=0)
                    u_machine[b, t] = utm

                # Rewards
                if "r" in tr:
                    r[b, t, 0] = float(tr["r"])

                # Termination
                done_val = tr.get("done", tr.get("terminated", False))
                terminated[b, t, 0] = 1.0 if bool(done_val) else 0.0

                # Avail masks
                if "avail_a" in tr:
                    try:
                        mask = _as_agents_mask(tr["avail_a"], n_agents)
                        # mask shape: (n_agents, current_len)
                        curr_len = mask.shape[1]
                        if __debug__:
                            try:
                                print(f"[DEBUG shapes] avail_a target=(B={B},T={T},n_agents={n_agents},max_avail={max_avail_len}) src={mask.shape} (b={b},t={t})")
                            except Exception:
                                print(f"[DEBUG shapes] avail_a src={getattr(mask,'shape',None)} (b={b},t={t})")
                        if avail_u is None:
                            # allocate conservative buffer if not pre-allocated
                            avail_u = np.zeros((B, T, n_agents, curr_len), dtype=np.float32)
                            avail_u_next = np.zeros((B, T, n_agents, curr_len), dtype=np.float32)
                            have_avail = True
                        # copy into the pre-allocated buffer (pad/truncate as needed)
                        # be careful: mask may have different second-dim than buffer
                        cpy = min(curr_len, avail_u.shape[3]) if avail_u.ndim >= 4 else curr_len
                        try:
                            avail_u[b, t, :mask.shape[0], :cpy] = mask[:, :cpy]
                        except Exception:
                            # best-effort flatten-copy
                            mm = np.zeros((n_agents, avail_u.shape[3]), dtype=np.float32)
                            mm[:, :min(mask.shape[1], mm.shape[1])] = mask[:, :mm.shape[1]]
                            avail_u[b, t] = mm
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        pass
                if "avail_a_next" in tr and have_avail:
                    try:
                        maskn = _as_agents_mask(tr["avail_a_next"], n_agents)
                        curr_len_n = maskn.shape[1]
                        cpy = min(curr_len_n, avail_u_next.shape[3]) if avail_u_next.ndim >= 4 else curr_len_n
                        try:
                            avail_u_next[b, t, :maskn.shape[0], :cpy] = maskn[:, :cpy]
                        except Exception:
                            mm = np.zeros((n_agents, avail_u_next.shape[3]), dtype=np.float32)
                            mm[:, :min(maskn.shape[1], mm.shape[1])] = maskn[:, :mm.shape[1]]
                            avail_u_next[b, t] = mm
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        pass

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
        # include machine mapping per action if present
        try:
            batch["u_machine"] = u_machine
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass
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


def _as_agents_mask(x: Any, n_agents: int) -> np.ndarray:
    """Convert availability masks into (n_agents, n_actions) float32.

    Pads with zeros or trims extra rows so the returned array has exactly
    n_agents rows. Accepts 1D vectors, 2D arrays, or lists.
    """
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim == 0:
        a = a.reshape(1, -1)
    # ensure at least 2D
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[0] < n_agents:
        pad = np.zeros((n_agents - a.shape[0], a.shape[1]), dtype=np.float32)
        a = np.concatenate([a, pad], axis=0)
    elif a.shape[0] > n_agents:
        a = a[:n_agents]
    return a


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
    """Infer (n_agents, obs_dim) by inspecting episode transitions.

    Robustly consider multiple possible keys that indicate agent-count
    (for example: 'o', 'avail_a', 'u') and return the maximum agent count
    observed across the episode pool. This avoids shape-mismatch when the
    environment emits variable-sized decision batches across timesteps.
    """
    max_agents = 1
    obs_dim = 11
    for ep in episodes:
        for tr in ep:
            # observations
            if "o" in tr:
                o = np.asarray(tr["o"], dtype=np.float32)
                if o.ndim == 1:
                    max_agents = max(max_agents, 1)
                    obs_dim = max(obs_dim, int(o.shape[0]))
                elif o.ndim == 2:
                    max_agents = max(max_agents, int(o.shape[0]))
                    obs_dim = max(obs_dim, int(o.shape[1]))
                elif o.ndim == 3 and o.shape[0] == 1:
                    max_agents = max(max_agents, int(o.shape[1]))
                    obs_dim = max(obs_dim, int(o.shape[2]))

            # availability masks (per-agent × n_actions)
            if "avail_a" in tr:
                a = np.asarray(tr["avail_a"])
                if a.ndim >= 2:
                    max_agents = max(max_agents, int(a.shape[0]))

            # actions vector (may be 1D per-agent)
            if "u" in tr:
                u = np.asarray(tr["u"])
                if u.ndim >= 1:
                    max_agents = max(max_agents, int(u.shape[0]))

    return int(max_agents), int(obs_dim)