import math

import pytest

# Import MASAEnv from the top-level environment module and the argument helper
from environment import MASAEnv
from MARL.common.arguments import get_mutable_args as get_args


def test_hybrid_reward_smoke():
    """Lightweight smoke test for the hybrid reward implementation.

    - Initializes the environment with default args
    - Ensures pop_decision_reward() returns a float
    - Ensures last_reward_components contains the expected keys and numeric values
    """
    args = get_args()

    # Initialize environment (should not raise)
    env = MASAEnv(args)

    assert hasattr(env, "pop_decision_reward"), "MASAEnv missing pop_decision_reward"

    # Call the reward function; it should handle empty/initial state gracefully
    r = env.pop_decision_reward()

    assert isinstance(r, float), "Reward must be a float"
    assert hasattr(env, "last_reward_components"), "Environment must expose last_reward_components"

    keys = [
        "CompletedNorm",
        "AvgWaitNorm",
        "WIPNorm",
        "ThroughputDelta",
        "LoadVariance",
        "R_global",
        "R_local_mean",
        "R_total",
    ]

    components = getattr(env, "last_reward_components", {})

    for k in keys:
        assert k in components, f"Missing key {k} in last_reward_components"
        v = components[k]
        # numeric check: ints and floats allowed
        assert isinstance(v, (float, int)), f"Invalid type for {k}: {type(v)}"
        # ensure not NaN for float values
        if isinstance(v, float):
            assert not math.isnan(v), f"NaN encountered for {k}"


if __name__ == "__main__":
    test_hybrid_reward_smoke()
    print("✅ Hybrid reward smoke test passed.")
