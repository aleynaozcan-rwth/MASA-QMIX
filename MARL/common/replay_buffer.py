"""
replay_buffer.py
Step 8A.2.4 – Unified Terminology (JobAgent / WorkCenter)
----------------------------------------------------------
- Global (flat) transition buffer + per-JobAgent views.
- Each entry is a transition dict: state, action, reward, next_state, done, jobagent_id, time.
- Sampling returns packed numpy batches for policy.learn_from_transitions().
- No functional changes; terminology unified with environment and rollout.
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

        # Optional: per-JobAgent views (for debug/analysis)
        self._by_jobagent = defaultdict(lambda: deque(maxlen=self.size))

    # ---------- Agent management ----------
    def register_agent(self, jobagent_id: int):
        _ = self._by_jobagent[jobagent_id]

    # ---------- Core API ----------
    def add_transition(
        self,
        jobagent_id: int,
        time: float,
        state,
        action,
        reward: float,
        next_state,
        done: bool,
    ):
        """Store one full transition for a JobAgent."""
        self.register_agent(jobagent_id)
        item = {
            "jobagent_id": int(jobagent_id),
            "time": float(time),
            "state": np.array(state, copy=False),
            "action": np.array(action, copy=False),
            "reward": float(reward),
            "next_state": np.array(next_state, copy=False),
            "done": bool(done),
        }
        self._flat.append(item)
        self._by_jobagent[jobagent_id].append(item)

    # ---------- Convenience credits ----------
    def add_time_penalty(self, time: float, penalty: float = -1.0):
        """Global shaping: pushes a dummy 'global' transition."""
        self.add_transition(
            jobagent_id=-1,
            time=time,
            state=np.zeros(10, dtype=np.float32),
            action=np.array([0], dtype=np.int64),
            reward=penalty,
            next_state=np.zeros(10, dtype=np.float32),
            done=False,
        )

    def add_completion_bonus(self, jobagent_id: int, time: float, bonus: float = 10.0):
        """Credit assignment when a JobAgent completes an operation."""
        self.add_transition(
            jobagent_id=jobagent_id,
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
        """Sample a batch of transitions across all JobAgents."""
        if len(self._flat) == 0:
            return None
        bs = min(batch_size, len(self._flat))
        batch = self.rng.sample(self._flat, k=bs)

        states = np.stack([b["state"] for b in batch])
        actions = np.stack([b["action"] for b in batch])
        rewards = np.array([b["reward"] for b in batch], dtype=np.float32)
        next_states = np.stack([b["next_state"] for b in batch])
        dones = np.array([b["done"] for b in batch], dtype=np.float32)
        jobagent_ids = np.array([b["jobagent_id"] for b in batch], dtype=np.int32)
        times = np.array([b["time"] for b in batch], dtype=np.float32)

        return {
            "state": states,
            "action": actions,
            "reward": rewards,
            "next_state": next_states,
            "done": dones,
            "jobagent_id": jobagent_ids,
            "time": times,
        }

    # ---------- Episodic API (compatibility no-op) ----------
    def store_episode(self, episode_batch):
        """Kept for backward compatibility; unused in 8A.2.4."""
        pass
