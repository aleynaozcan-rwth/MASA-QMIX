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

    args.seed = seed
    # run each episode as an epoch (simple mapping for smoke tests)
    args.n_epoch = max(1, total_eps)
    args.n_episodes = 1
    args.evaluate_cycle = max(10, args.evaluate_cycle if hasattr(args, 'evaluate_cycle') else 10)
    args.evaluate_epoch = 1
    args.episode_limit = int(os.environ.get('EPISODE_LIMIT', getattr(args, 'episode_limit', 200)))

    # conservative but visible defaults for smoke tests
    args.learn = True
    args.buffer_size = int(getattr(args, 'buffer_size', 10000))
    args.batch_size = int(getattr(args, 'batch_size', 32))
    args.train_steps = int(getattr(args, 'train_steps', 10))
    args.save_cycle = int(getattr(args, 'save_cycle', 500))
    args.cuda = False

    # Instantiate environment. If EXP_LAMBDA is provided, start the dynamic
    # arrival loop on the created env rather than passing an unsupported
    # constructor argument (MASAEnv.__init__ doesn't accept arrival overrides).
    env = MASAEnv()
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
