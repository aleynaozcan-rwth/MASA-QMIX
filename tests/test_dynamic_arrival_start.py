import pytest

from environment import MASAEnv
from MARL.common.arguments import get_mutable_args


def test_dynamic_arrival_auto_start():
    """MASAEnv should automatically create and schedule a TaskGenerator
    when constructed with auto_start_arrivals=True and arrival_lambda>0.
    """
    args = get_mutable_args()
    args.arrival_lambda = 0.05

    env = MASAEnv(args=args, auto_start_arrivals=True)
    env.reset()

    # TaskGenerator instance was created and the arrival loop was scheduled
    assert env._task_generator is not None, "TaskGenerator was not created"
    assert len(env.env._queue) > 0, "SimPy event queue should contain scheduled arrival loop"

    # Helpful output when running tests manually
    print("Dynamic arrivals active")
