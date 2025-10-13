"""
replay_buffer.py
Step 7B – Dynamic, plane-aware replay buffer (extended version)

Key upgrades vs previous:
- Fully dynamic agent (plane) registration.
- Each transition carries optional metadata (aux dict: site, operator, kind, etc.).
- Flattened global sampling + optional per-plane sampling.
- Diagnostics: recent transitions, per-plane reward stats, etc.
"""

from collections import deque, defaultdict
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, size: int = 100_000, per_plane_limit: int = 10_000, seed: int = 123):
        self.size = int(size)
        self.per_plane_limit = int(per_plane_limit)
        self.rng = random.Random(seed)
        # plane_id -> deque of transitions
        self.buffer = defaultdict(lambda: deque(maxlen=self.per_plane_limit))
        self.num_transitions = 0
        self.plane_reward_sums = defaultdict(float)
        self.plane_counts = defaultdict(int)

    # ============================================================
    # === Agent / plane management ===============================
    # ============================================================
    def register_agent(self, plane_id: int):
        """Ensure a buffer exists for a (possibly new) plane_id."""
        _ = self.buffer[plane_id]

    def active_plane_ids(self):
        """Return a list of planes that currently have transitions."""
        return list(self.buffer.keys())

    # ============================================================
    # === Core transition API ====================================
    # ============================================================
    def add_transition(
        self,
        plane_id: int,
        time: float,
        state,
        action,
        reward: float,
        next_state,
        done: bool,
        aux: dict | None = None,
    ):
        """Append a single transition for a specific plane."""
        self.register_agent(plane_id)
        record = {
            "plane_id": plane_id,
            "time": float(time),
            "state": np.array(state, copy=False),
            "action": np.array(action, copy=False),
            "reward": float(reward),
            "next_state": np.array(next_state, copy=False),
            "done": bool(done),
            "aux": aux or {},
        }
        self.buffer[plane_id].append(record)
        self.num_transitions += 1
        self.plane_reward_sums[plane_id] += reward
        self.plane_counts[plane_id] += 1

    # ============================================================
    # === Helper transitions =====================================
    # ============================================================
    def add_completion_bonus(self, plane_id: int, time: float, bonus: float = 10.0, aux: dict | None = None):
        """
        Lightweight helper: used when a job completion is detected for plane_id.
        Stores a synthetic transition (state/action placeholders) to credit the plane.
        """
        aux = aux or {"kind": "completion_bonus"}
        self.add_transition(
            plane_id=plane_id,
            time=time,
            state=np.zeros(10, dtype=np.float32),
            action=np.array([0], dtype=np.int64),
            reward=bonus,
            next_state=np.zeros(10, dtype=np.float32),
            done=False,
            aux=aux,
        )

    def add_time_penalty(self, plane_id: int, time: float, penalty: float = -1.0, aux: dict | None = None):
        """Add a small negative reward to simulate waiting/time cost."""
        aux = aux or {"kind": "time_penalty"}
        self.add_transition(
            plane_id=plane_id,
            time=time,
            state=np.zeros(10, dtype=np.float32),
            action=np.array([0], dtype=np.int64),
            reward=penalty,
            next_state=np.zeros(10, dtype=np.float32),
            done=False,
            aux=aux,
        )

    # ============================================================
    # === Sampling ===============================================
    # ============================================================
    def __len__(self):
        return self.num_transitions

    def sample(self, batch_size: int = 64, flatten: bool = True):
        """
        Sample across all planes uniformly.
        Returns dict of np.arrays with aligned shapes.
        """
        all_records = []
        for dq in self.buffer.values():
            all_records.extend(dq)
        if len(all_records) == 0:
            return None

        batch = self.rng.sample(all_records, k=min(batch_size, len(all_records)))

        states = np.stack([b["state"] for b in batch])
        actions = np.stack([b["action"] for b in batch])
        rewards = np.array([b["reward"] for b in batch], dtype=np.float32)
        next_states = np.stack([b["next_state"] for b in batch])
        dones = np.array([b["done"] for b in batch], dtype=np.bool_)
        plane_ids = np.array([b["plane_id"] for b in batch], dtype=np.int32)
        times = np.array([b["time"] for b in batch], dtype=np.float32)
        aux = [b["aux"] for b in batch]

        return {
            "state": states,
            "action": actions,
            "reward": rewards,
            "next_state": next_states,
            "done": dones,
            "plane_id": plane_ids,
            "time": times,
            "aux": aux,
        }

    def sample_plane(self, plane_id: int, batch_size: int = 32):
        """Sample transitions from a specific plane's buffer."""
        if plane_id not in self.buffer or len(self.buffer[plane_id]) == 0:
            return None
        dq = self.buffer[plane_id]
        batch = self.rng.sample(dq, k=min(batch_size, len(dq)))

        states = np.stack([b["state"] for b in batch])
        actions = np.stack([b["action"] for b in batch])
        rewards = np.array([b["reward"] for b in batch], dtype=np.float32)
        next_states = np.stack([b["next_state"] for b in batch])
        dones = np.array([b["done"] for b in batch], dtype=np.bool_)
        times = np.array([b["time"] for b in batch], dtype=np.float32)
        aux = [b["aux"] for b in batch]

        return {
            "state": states,
            "action": actions,
            "reward": rewards,
            "next_state": next_states,
            "done": dones,
            "plane_id": np.full(len(batch), plane_id, dtype=np.int32),
            "time": times,
            "aux": aux,
        }

    # ============================================================
    # === Diagnostics / Utilities ================================
    # ============================================================
    def recent_transitions(self, n: int = 5):
        """Return the last n global transitions (flattened across all planes)."""
        all_records = []
        for dq in self.buffer.values():
            all_records.extend(dq)
        if len(all_records) == 0:
            return []
        return list(all_records)[-n:]

    def plane_stats(self):
        """Return per-plane average reward and count."""
        stats = {}
        for pid in self.buffer.keys():
            total = self.plane_reward_sums[pid]
            count = self.plane_counts[pid]
            avg = total / max(1, count)
            stats[pid] = {"count": count, "avg_reward": avg}
        return stats

    def clear(self):
        """Completely reset the buffer."""
        self.buffer.clear()
        self.num_transitions = 0
        self.plane_reward_sums.clear()
        self.plane_counts.clear()

    # ============================================================
    # === Optional compatibility (older MARLlib APIs) ============
    # ============================================================
    def store_episode(self, episode_batch):
        """Kept for backward-compatibility; not used in Step 7B."""
        pass
