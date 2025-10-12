# main.py — cumulative entrypoint for MASA-QMIX
# Modes:
#   4c  : Run one plane's full SimPy workflow (Step 4C)
#   4b  : Co-execution smoke test (Step 4B)
#   rl  : Full RL pipeline (Step 5+)
#   6b  : Test machine + operator pair action space (Step 6B)

import argparse
import importlib
import sys
from environment import ScheduleEnv


def run_mode_4c(plane_id: int):
    env = ScheduleEnv()
    env.test_single_job_run(plane_id=plane_id)


def run_mode_4b(steps: int):
    env = ScheduleEnv()
    env.test_coexecution(steps=steps)


def run_mode_rl():
    if importlib.util.find_spec("torch") is None:
        print(
            "[ERROR] PyTorch ('torch') is not installed.\n"
            "To enable RL mode: pip install torch torchvision torchaudio\n"
            "Or load your cluster's PyTorch module, then re-run with --mode rl."
        )
        sys.exit(1)

    try:
        from MARL.runner import Runner
        from MARL.common.arguments import (
            get_common_args, get_coma_args, get_mixer_args,
            get_centralv_args, get_reinforce_args,
            get_commnet_args, get_g2anet_args
        )
    except Exception as e:
        print("[ERROR] Could not import MARL modules:", e)
        sys.exit(1)

    args = get_common_args()
    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)

    env = ScheduleEnv()
    env.reset()

    if not hasattr(env, "get_env_info"):
        print("[ERROR] get_env_info missing in environment.py")
        sys.exit(1)

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    print("\n=== Environment Info ===")
    for k, v in env_info.items():
        print(f"{k}: {v}")
    print("====================================\n")

    runner = Runner(env, args)
    if args.learn:
        runner.run(0)
    else:
        win_rate, reward, _ = runner.evaluate([], 0)
        print(f"The ave_reward of {args.alg} is {reward}")


def run_mode_6b():
    """Standalone test for Step 6B (machine + operator action space)."""
    env = ScheduleEnv()
    env.test_coexecution()


def parse_args():
    p = argparse.ArgumentParser(description="MASA-QMIX cumulative entrypoint")
    p.add_argument("--mode",
                   choices=["4c", "4b", "rl", "6b"],
                   default="4c",
                   help=("4c: single-plane SimPy run; "
                         "4b: co-exec smoke test; "
                         "rl: full MARL pipeline; "
                         "6b: test machine+operator pairs"))
    p.add_argument("--plane-id", type=int, default=0,
                   help="Plane ID for 4c mode")
    p.add_argument("--steps", type=int, default=5,
                   help="Number of steps to tick in 4b mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "4c":
        run_mode_4c(args.plane_id)
    elif args.mode == "4b":
        run_mode_4b(args.steps)
    elif args.mode == "rl":
        run_mode_rl()
    elif args.mode == "6b":
        run_mode_6b()
# --- End of file main.py ---
