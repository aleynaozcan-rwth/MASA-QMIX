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
        print(f"[run_train_qmix] NOTE: EXP_SEED={seed} provided; pass --seed {seed} to override centrally")
    if total_eps != 100:
        print(f"[run_train_qmix] NOTE: EXP_EPISODE_LIMIT={total_eps} provided; pass --n_epoch {total_eps} to override centrally")

    # Instantiate environment. Inject centralized args so the environment
    # does not perform any ad-hoc parsing or fallback logic. If the
    # orchestrator provided config flags, forward them so MASAEnv can
    # auto-load and merge the YAML into its defaults.
    env = MASAEnv(
        args=args,
        config_path=getattr(args, 'config_path', None),
        auto_load_config=getattr(args, 'auto_load_config', False),
    )
    if lam is not None:
        try:
            lamf = float(lam)
            # start a background process that injects arrivals at the requested rate
            try:
                env.env.process(env._dynamic_arrival_loop(lamf))
                print(f"[run_train_qmix] Started dynamic arrival loop with lambda={lamf}")
            except Exception:
                # if the env hasn't created a TaskGenerator, this may be a no-op
                print("[run_train_qmix] Could not start dynamic arrival loop; TaskGenerator may be missing in config")
        except Exception:
            pass

    runner = Runner(env, args)
    runner.run(0)


if __name__ == '__main__':
    main()
