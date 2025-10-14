"""
replay_buffer.py
Step 7B.3 – Replay Transition Training

- Global (flat) transition buffer + per-plane views.
- Each entry is a full transition dict: state, action, reward, next_state, done, plane_id, time.
- Sampling returns a packed numpy batch ready for policy.learn_from_transitions().
"""

from collections import defaultdict, deque
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, size: int = 100000, seed: int = 123):
        self.size = int(size)
        self.rng = random.Random(seed)

        # Global flat ring buffer (O(1) append/pop)
        self._flat = deque(maxlen=self.size)

        # Optional: per-plane views (not required by learner but handy for debug/analysis)
        self._by_plane = defaultdict(lambda: deque(maxlen=self.size))

    # ---------- Agent/plane management ----------
    def register_agent(self, plane_id: int):
        _ = self._by_plane[plane_id]

    # ---------- Core API ----------
    def add_transition(
        self,
        plane_id: int,
        time: float,
        state,
        action,
        reward: float,
        next_state,
        done: bool,
    ):
        """Store one full transition."""
        self.register_agent(plane_id)
        item = {
            "plane_id": int(plane_id),
            "time": float(time),
            "state": np.array(state, copy=False),
            "action": np.array(action, copy=False),
            "reward": float(reward),
            "next_state": np.array(next_state, copy=False),
            "done": bool(done),
        }
        self._flat.append(item)
        self._by_plane[plane_id].append(item)

    # ---------- Convenience credits (kept from 7B) ----------
    def add_time_penalty(self, time: float, penalty: float = -1.0):
        """Global shaping: pushes a dummy 'global' transition."""
        self.add_transition(
            plane_id=-1,
            time=time,
            state=np.zeros(10, dtype=np.float32),
            action=np.array([0], dtype=np.int64),
            reward=penalty,
            next_state=np.zeros(10, dtype=np.float32),
            done=False,
        )

    def add_completion_bonus(self, plane_id: int, time: float, bonus: float = 10.0):
        """Credit assignment when a plane completes an operation."""
        self.add_transition(
            plane_id=plane_id,
            time=time,
            state=np.zeros(10, dtype=np.float32),
            action=np.array([0], dtype=np.int64),
            reward=bonus,
            next_state=np.zeros(10, dtype=np.float32),
            done=False,
        )

    # ---------- Python protocol ----------
    def __len__(self):
        return len(self._flat)

    # ---------- Sampling ----------
    def sample(self, batch_size: int = 32):
        if len(self._flat) == 0:
            return None
        bs = min(batch_size, len(self._flat))
        batch = self.rng.sample(self._flat, k=bs)

        states = np.stack([b["state"] for b in batch])
        actions = np.stack([b["action"] for b in batch])
        rewards = np.array([b["reward"] for b in batch], dtype=np.float32)
        next_states = np.stack([b["next_state"] for b in batch])
        dones = np.array([b["done"] for b in batch], dtype=np.float32)
        plane_ids = np.array([b["plane_id"] for b in batch], dtype=np.int32)
        times = np.array([b["time"] for b in batch], dtype=np.float32)

        return {
            "state": states,
            "action": actions,
            "reward": rewards,
            "next_state": next_states,
            "done": dones,
            "plane_id": plane_ids,
            "time": times,
        }

    # ---------- Episodic API (compatibility no-op) ----------
    def store_episode(self, episode_batch):
        # Not used in 7B.3 (transition-based training). Kept for backward-compat.
        pass
