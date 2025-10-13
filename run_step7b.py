"""
run_step7b.py
-------------------------------------------------------------
Runs Step 7B environment with dynamic job arrivals,
SimPy scheduling, replay buffer, and QMIX training loop.

This file is for testing the Step 7B integration
(environment + rollout + replay + runner) before merging
into main.py.
"""

import sys, os

# === Project Path Setup ===
sys.path.append("/home/cc253232/MASA-QMIX")

# === Imports ===
from MARL.common.arguments import get_common_args, get_mixer_args
from MARL.runner import Runner
from environment import ScheduleEnv


if __name__ == "__main__":
    print("\n🚀 Starting Step 7B simulation test...")

    # === 1. Load arguments ===
    args = get_common_args()
    args = get_mixer_args(args)

    # === 2. Create environment ===
    env = ScheduleEnv(
        start_planes=args.start_planes,
        max_planes=args.max_planes,
        arrival_prob=args.arrival_prob,
        variable_ops=args.variable_ops,
        seed=args.seed,
    )

    # === 3. Initialize runner ===
    runner = Runner(env, args)

    # === 4. Quick rollout test ===
    print("\n[TEST] Running one rollout episode (evaluation mode)...")
    runner.rolloutWorker.generate_episode(global_ep_idx=0, evaluate=True)

    # === 5. Training ===
    print("\n[TRAIN] Starting Step 7B training loop...")
    runner.run(num=1)

    print("\n✅ Step 7B finished successfully.")
