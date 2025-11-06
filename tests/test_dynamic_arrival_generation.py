import pytest

from environment import MASAEnv
from MARL.common.arguments import get_mutable_args


def test_dynamic_arrival_generates_jobs():
    """Verify TaskGenerator generates new jobs over simulation time when
    MASAEnv is constructed with auto_start_arrivals=True and arrival_lambda>0.
    """
    args = get_mutable_args()
    args.arrival_lambda = 0.05
    # constrain generated job complexity so TaskGenerator selects ops that
    # map to available workcenters in the test machine registry
    args.job_min_ops = 1
    args.job_max_ops = 2

    # Construct env that auto-starts arrivals using MASAEnv.reset(). The
    # TaskGenerator will be attached and scheduled on the MASAEnv wrapper
    # (so capability resolution works) as of the recent TaskGenerator.start
    # fix.
    env = MASAEnv(args=args, auto_start_arrivals=True)
    env.reset()

    # run the simulation for a while so arrivals can occur
    env.env.run(until=100)

    assert len(env.jobs) > int(getattr(env, 'initial_jobs', 0)), (
        f"Expected more jobs than initial ({len(env.jobs)} <= {env.initial_jobs})"
    )

    print("Dynamic arrivals successfully generated")
