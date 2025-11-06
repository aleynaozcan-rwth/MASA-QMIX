import argparse
import numpy as np
import pickle
import os
from environment import MASAEnv, LOG
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))

from MARL.runner import Runner
from MARL.common.arguments import (
    get_common_args,
    get_mutable_args,
    get_coma_args,
    get_mixer_args,
    get_centralv_args,
    get_reinforce_args,
    get_commnet_args,
    get_g2anet_args,
)


# ============================================================
# === MARL Agent Wrapper =====================================
# ============================================================

def marl_agent_wrapper(args):
    """Standard MARL training loop (supports QMIX, COMA, etc.)."""

    # --- Environment (MASAEnv) ---
    # Inject the centralized args namespace — environment will not read CLI
    # itself and must be driven by the orchestrator.
    # If the orchestrator provided config_path/auto_load_config flags on args
    # pass them into MASAEnv so it can load/merge runtime configuration.
    # Construct environment without auto-loading YAML/config to ensure
    # the environment runs purely from its module defaults and injected args.
    auto_arrivals = getattr(args, 'arrival_lambda', 0.0) > 0.0
    try:
        if auto_arrivals:
            LOG.info("[Main] Auto-starting TaskGenerator (arrival_lambda=%.3f)", float(getattr(args, 'arrival_lambda', 0.0)))
    except Exception:
        pass
    env = MASAEnv(
        args=args,
        config_path=getattr(args, 'config_path', None),
        auto_load_config=False,
        auto_start_arrivals=auto_arrivals,
    )
    # Runner will query the environment and initialize any runtime-derived
    # shapes (n_actions, n_agents, obs/state dims, episode_limit). Keep
    # `main.py` strictly as orchestration so it does not set or mutate args.

    # --- Algorithm-specific args (env'den sonra çağrılmalı!) ---
    if args.alg.find("coma") > -1:
        args = get_coma_args(args)
    elif args.alg.find("central_v") > -1:
        args = get_centralv_args(args)
    elif args.alg.find("reinforce") > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)

    if args.alg.find("commnet") > -1:
        args = get_commnet_args(args)
    if args.alg.find("g2anet") > -1:
        args = get_g2anet_args(args)

    # Note: Runner will initialize environment-derived shapes (n_agents,
    # n_actions, obs/state dims, episode_limit). Do not mutate `args` here.

    # --- Summary printout ---
    print("\n=== Training Setup Summary (Args) ===")
    print(f"Algorithm: {args.alg}")
    print(f"Random seed: {args.seed}")
    print(f"Replay buffer size: {getattr(args, 'buffer_size', 'N/A')}")
    print(f"Batch size: {getattr(args, 'batch_size', 'N/A')}")
    print(f"Total epochs: {args.n_epoch}")
    print(f"Episodes per epoch: {args.n_episodes}")
    print(f"Evaluation every {args.evaluate_cycle} epochs")
    print(f"Observation dim: {args.obs_shape}")
    print(f"State dim: {args.state_shape}")
    print("====================================")

    runner = Runner(env, args)
    if args.learn:
        runner.run(num=1)
        # Post-run quick summary for developer visibility
        try:
            for idx, r in enumerate(getattr(runner, 'episode_rewards', [])):
                print(f"Episode {idx} | Total reward: {r:.4f}")
        except Exception:
            pass
        print("Episode finished")
    else:
        win_rate, reward, _ = runner.evaluate([], 0)
        print(f"Avg reward for {args.alg}: {reward}")


# ============================================================
# === Random Baseline ========================================
# ============================================================

def random_agent_wrapper(args):
    episodes = 10
    # For the random baseline use a pure environment (no YAML/config auto-load)
    auto_arrivals = getattr(args, 'arrival_lambda', 0.0) > 0.0
    try:
        if auto_arrivals:
            LOG.info("[Main] Auto-starting TaskGenerator (arrival_lambda=%.3f)", float(getattr(args, 'arrival_lambda', 0.0)))
    except Exception:
        pass
    env = MASAEnv(
        args=args,
        config_path=getattr(args, 'config_path', None),
        auto_load_config=False,
        auto_start_arrivals=auto_arrivals,
    )
    EATs = []
    schedule_processes = []

    # ensure output directory exists before writing pickle
    out_dir = "./my_data_and_graph/pickles"
    os.makedirs(out_dir, exist_ok=True)

    for episode in range(episodes):
        s = env.reset()
        done = False
        while not done:
            actions = []
            for i in range(len(env.jobs)):
                avail_actions = env._build_avail_actions()[i]
                valid_actions = [k for k, v in enumerate(avail_actions) if v == 1]
                if valid_actions:
                    action = np.random.choice(valid_actions)
                else:
                    action = np.random.randint(0, env.num_wcs)
                actions.append(action)
            _, _, done, info = env.step(actions)

        EATs.append(env.t)
        schedule_processes.append(info)
        print(f"[Episode {episode}] Completion time: {env.t}")

    print(f"Average completion time: {sum(EATs) / len(EATs)}")
    out_path = os.path.join(out_dir, "process.pk")
    try:
        with open(out_path, "wb") as f:
            pickle.dump(schedule_processes, f)
    except Exception as e:
        print(f"ERROR: failed to write pickle to {out_path}: {e}")


# ============================================================
# === Entry Point ============================================
# ============================================================

if __name__ == "__main__":
    # Centralized argument parsing
    # Use a mutable args object here so we can override values for the
    # developer mini-run without hitting ReadOnlyArgs protections.
    args = get_mutable_args()

    # No in-code mini-run overrides: trust CLI/config to provide runtime values
    # (This avoids accidental mutation of fundamental settings like n_agents
    # or initial_jobs during developer quick-runs. To run a short test, set
    # the desired args via the command line or a wrapper script.)
    # Ensure a sensible default history_dir if the caller didn't provide one.
    # Avoid assigning to `args.*` at module import time (hygiene rule); use a
    # local variable instead so callers/tests that import this module don't
    # observe mutated globals.
    history_dir = getattr(args, 'history_dir', None) or "./my_data_and_graph/historydata"

    # Clean previous run artifacts in the history directory to avoid mixing
    # results from earlier runs. Remove all files and subdirectories inside
    # the history dir (preserve the directory itself). This gives a fully
    # fresh history folder for each run.
    try:
        import os, shutil
        hist = history_dir
        os.makedirs(hist, exist_ok=True)
        for name in os.listdir(hist):
            path = os.path.join(hist, name)
            try:
                if os.path.islink(path) or os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception:
                # best-effort: skip items we can't remove
                pass
    except Exception:
        pass

    # Select execution mode
    if getattr(args, 'mode', 'marl') == "marl":
        marl_agent_wrapper(args)
    elif getattr(args, 'mode', 'marl') == "random":
        random_agent_wrapper(args)
    else:
        raise ValueError(f"Unknown mode: {getattr(args, 'mode', None)}")
