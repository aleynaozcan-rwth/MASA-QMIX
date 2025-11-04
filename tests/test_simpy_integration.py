import types
from types import SimpleNamespace
import os

import numpy as np


class FakeJob:
    def __init__(self, jid):
        self.id = jid
        self.finished = False


class MinimalSimPyEnv:
    """A tiny deterministic SimPy-like environment for integration testing.

    Implements the minimal API expected by the event-driven Runner:
      - reset()
      - get_env_info()
      - wait_for_decisions()
      - pop_decision_reward()
      - gantt_records, jobs
    """

    def __init__(self):
        self.jobs = [FakeJob(0), FakeJob(1)]
        self.gantt_records = []
        self._call_count = 0
        self._pending_rewards = []
        self.t = 0.0
        # make some env metadata used by RolloutWorker
        self.num_wcs = 1
        self.num_ops = 1

    def reset(self):
        # no observations needed for this mini-env
        self._call_count = 0
        self._pending_rewards = []
        for j in self.jobs:
            j.finished = False
        self.t = 0.0
        return None

    def get_env_info(self):
        return dict(n_agents=2, n_actions=1, obs_shape=1, state_shape=1, episode_limit=50)

    def wait_for_decisions(self):
        # On first call, return two decision items (one per job). On second call return empty -> done.
        if self._call_count == 0:
            self._call_count += 1
            sim_time = float(self.t)
            batch = []
            for job in self.jobs:
                def make_resume(j):
                    def resume(choice):
                        # mark job finished and register a deterministic reward
                        j.finished = True
                        self._pending_rewards.append(1.0)
                    return resume

                item = {
                    'job_id': job.id,
                    'obs': None,
                    'allowed_wcs': [0],
                    'avail_row': [1],
                    'resume': make_resume(job),
                }
                batch.append(item)
            return batch, sim_time
        else:
            # subsequent calls: done
            return [], float(self.t)

    def pop_decision_reward(self):
        # return the sum of pending rewards and clear
        r = float(sum(self._pending_rewards)) if self._pending_rewards else 0.0
        self._pending_rewards = []
        return r

    # minimal helpers used in some mask/avail computations
    def _build_state_vector(self):
        return [0.0]

    def _build_avail_actions(self):
        return {0: [1]}


def test_simpy_integration_runner_and_worker(tmp_path, monkeypatch):
    """Integration test: run the event-driven episode loop end-to-end.

    We create a minimal SimPy-like env and run the RolloutWorker via Runner to
    ensure the event-driven path executes and terminates deterministically.
    """
    # import runner module and use a lightweight stub Agents/ReplayBuffer so Runner init succeeds
    import MARL.runner as runner_mod

    class StubAgents:
        def __init__(self, args):
            self.rng = np.random.RandomState(0)

        def select_actions(self, obs_batch, avail_batch, evaluate=False):
            return [0] * len(obs_batch)

    class StubReplayBuffer:
        def __init__(self, episode_capacity=None, seed=None):
            self._store = []

        def __len__(self):
            return 0

    monkeypatch.setattr(runner_mod, 'Agents', StubAgents)
    monkeypatch.setattr(runner_mod, 'ReplayBuffer', StubReplayBuffer)

    env = MinimalSimPyEnv()

    args = SimpleNamespace()
    args.alg = 'stub'
    args.result_dir = str(tmp_path)
    args.map = 'testmap'
    args.n_epoch = 1
    args.evaluate_cycle = 1
    args.n_episodes = 1
    args.evaluate_epoch = 1
    args.n_steps = 100
    args.batch_size = 1
    args.buffer_size = 10
    args.train_steps = 1
    args.gantt_csv = False
    args.use_granular_actions = False
    args.use_machine_actions = False
    args.snapshot_on_eval = True
    args.clean_history = False
    args.result_dir = str(tmp_path)
    args.history_dir = str(tmp_path)
    args.seed = 0
    args.learn = True

    # instantiate Runner (which will create a RolloutWorker using our stubs)
    r = runner_mod.Runner(env, args)

    # run a single event-driven episode using Runner's delegated method
    episode, ep_reward, win_tag, gantt = r._run_event_driven_episode(0)

    # assertions: deterministic behavior
    assert isinstance(episode, dict)
    assert 'r' in episode
    # two jobs each awarded reward 1.0 -> total reward 2.0
    assert ep_reward == 2.0
    assert win_tag is True
    assert isinstance(gantt, list)
