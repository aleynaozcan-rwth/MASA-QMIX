import numpy as np

from MARL.common.rollout import RolloutWorker


class DummyEnv:
    def __init__(self):
        # workcenters_meta must provide machine_index and machine_list for mapping
        class WCMeta:
            pass

        self.workcenters_meta = WCMeta()
        self.workcenters_meta.machine_index = {"m0": 0, "m1": 1}
        self.workcenters_meta.machine_list = ["m0", "m1"]
        # simple eligible groups per workcenter
        self.workcenters_meta.eligible_operator_groups_by_wc = {0: [0], 1: [0]}

        self.num_wcs = 2
        self.num_ops = 1

        # simple resources placeholders
        self.wc_resources = [object(), object()]
        self.operator_groups = [object()]

        # keep done flag
        self.done = False

    def _resource_free(self, r):
        # always free for deterministic tests
        return True


class Args:
    def __init__(self, n_agents=2, obs_shape=3, n_actions=2, use_granular_actions=False):
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.use_granular_actions = use_granular_actions


def make_batch():
    # Two decision items with simple obs vectors and per-machine avail_row
    return [
        {"job_id": 0, "obs": [0.1, 0.2, 0.3], "avail_row": [1, 0], "allowed_machine_indices": [0]},
        {"job_id": 1, "obs": [0.4, 0.5, 0.6], "avail_row": [0, 1], "allowed_machine_indices": [1]},
    ]


def test_build_transitions_basic_shapes():
    env = DummyEnv()
    args = Args(n_agents=2, obs_shape=3, n_actions=2, use_granular_actions=False)
    rw = RolloutWorker(env, agents=None, args=args)

    batch = make_batch()
    processed_actions = [0, 1]
    processed_machine_names = ["m0", "m1"]
    r = 1.0
    s_before = np.array([0.0, 1.0], dtype=np.float32)
    s_after = np.array([0.5, 0.5], dtype=np.float32)

    tr_list = rw.build_transitions_from_decision_batch(batch, processed_actions, processed_machine_names, r, s_before=s_before, s_after=s_after)

    assert isinstance(tr_list, list)
    assert len(tr_list) == len(batch)

    for tr in tr_list:
        # mandatory keys
        for k in ["o", "u", "u_machine", "u_machine_name", "r", "done"]:
            assert k in tr

        # shapes
        assert tr["o"].shape == (args.n_agents, args.obs_shape)
        assert isinstance(tr["u"], list)
        assert len(tr["u"]) == args.n_agents
        # avail_a should be present and have shape (n_agents, n_actions)
        assert "avail_a" in tr and tr["avail_a"].shape == (args.n_agents, args.n_actions)
        # state keys
        assert np.allclose(tr["s"], s_before)
        assert np.allclose(tr["s_next"], s_after)


def test_build_transitions_granular_fallback():
    # When granular actions are requested but mask building is not available,
    # the helper should fall back to expanding per-machine avail_row into a
    # flattened avail_a. This test ensures no exceptions and reasonable shapes.
    env = DummyEnv()
    args = Args(n_agents=2, obs_shape=3, n_actions=2, use_granular_actions=True)
    rw = RolloutWorker(env, agents=None, args=args)

    batch = make_batch()
    processed_actions = [0, 1]
    processed_machine_names = ["m0", "m1"]
    r = 0.5

    tr_list = rw.build_transitions_from_decision_batch(batch, processed_actions, processed_machine_names, r, s_before=None, s_after=None)

    assert isinstance(tr_list, list)
    assert len(tr_list) == 2

    for tr in tr_list:
        assert "avail_a" in tr
        # flattened actions dimension should be at least n_actions (fallback)
        assert tr["avail_a"].ndim == 2
        assert tr["avail_a"].shape[0] == args.n_agents
