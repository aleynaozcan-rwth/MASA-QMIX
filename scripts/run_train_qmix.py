#!/usr/bin/env python3
"""
Lightweight training entrypoint for QMIX smoke tests.
Reads environment variables:
  EXP_SEED           -> int (default: 0)
  EXP_EPISODE_LIMIT  -> int total episodes to run (default: 100)
  EXP_LAMBDA         -> float arrival lambda override for env (optional)

This script maps EXP_EPISODE_LIMIT to runner.n_epoch (episodes per epoch = 1)
so that `EXP_EPISODE_LIMIT=500` runs 500 episodes quickly in default config.
"""
import os
import logging
from MARL.common.arguments import get_common_args
from environment import MASAEnv
from MARL.runner import Runner


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_common_args()

    # Read env overrides
    seed = int(os.environ.get('EXP_SEED', os.environ.get('SEED', 0)))
    total_eps = int(os.environ.get('EXP_EPISODE_LIMIT', 100))
    lam = os.environ.get('EXP_LAMBDA', None)

    # Note: this script should not redefine centralized hyperparameters.
    # Any runtime overrides should be provided via CLI flags or by modifying
    # `MARL/common/arguments.py`. We read environment variables here but do
    # not assign them back into the global `args` namespace to avoid ad-hoc
    # default duplication. Consumers (Runner) will read `args` as-is.
    if seed != 0:
        # if a seed was explicitly provided via environment, prefer using it
        # but do not reassign into args; log a reminder for reproducibility.
        logging.getLogger(__name__).info("[run_train_qmix] NOTE: EXP_SEED=%s provided; pass --seed %s to override centrally", seed, seed)
    if total_eps != 100:
        logging.getLogger(__name__).info("[run_train_qmix] NOTE: EXP_EPISODE_LIMIT=%s provided; pass --n_epoch %s to override centrally", total_eps, total_eps)

    # Instantiate environment. Inject centralized args so the environment
    # does not perform any ad-hoc parsing or fallback logic. If the
    # orchestrator provided config flags, forward them so MASAEnv can
    # auto-load and merge the YAML into its defaults.
    # Config flags are deprecated: MASAEnv now relies on in-module defaults.
    # Keep behavior identical by only passing the central `args` namespace.
    env = MASAEnv(args=args)
    if lam is not None:
        try:
            lamf = float(lam)
            # Prefer starting a TaskGenerator properly on the MASAEnv wrapper
            # so arrival loop receives the wrapper (with workcenters_meta) and
            # can resolve capabilities. Fall back gracefully if unavailable.
            try:
                if getattr(env, '_task_generator', None) is not None:
                    env._task_generator.start(env, lamf)
                else:
                    # Try to construct a TaskGenerator and attach it to env
                    try:
                        from utils.task_generator import TaskGenerator  # type: ignore
                        tg = TaskGenerator(py_rng=getattr(env, '_py_rng', None), np_rng=getattr(env, '_np_rng', None))
                        try:
                            setattr(tg, '_owner_env', env)
                        except Exception as e:
                            logging.getLogger(__name__).warning(f"[C1] Could not set _owner_env on TaskGenerator: {e}")
                        tg.start(env, lamf)
                        # keep reference so callers can inspect active generator
                        setattr(env, '_task_generator', tg)
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Could not construct TaskGenerator; dynamic arrivals disabled: {e}")
                logging.getLogger(__name__).info("[run_train_qmix] Started dynamic arrivals with lambda=%s", lamf)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Could not start dynamic arrivals: {e}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Dynamic arrivals setup failed: {e}")

    runner = Runner(env, args)
    runner.run(0)


if __name__ == '__main__':
    main()
