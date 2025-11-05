import argparse
import numpy as np
import pickle
import os
from environment import MASAEnv
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
    env = MASAEnv(
        args=args,
        config_path=getattr(args, 'config_path', None),
        auto_load_config=getattr(args, 'auto_load_config', False),
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
    env = MASAEnv(
        args=args,
        config_path=getattr(args, 'config_path', None),
        auto_load_config=getattr(args, 'auto_load_config', False),
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

    # --- Mini-run override for quick QMIX smoke test ---
    # These in-code overrides are intentional for the developer quick-run
    # requested: reproducible short training session.
    args.seed = getattr(args, 'seed', 0) if args.seed is None else 42
    args.n_epoch = 1
    args.n_episodes = 3
    args.learn = True
    args.evaluate_cycle = 1
    args.n_agents = 4
    args.initial_jobs = 2
    args.episode_limit = 64
    args.history_dir = getattr(args, 'history_dir', None) or "./my_data_and_graph/historydata"

    print(f"[Mini-run] Overriding args for quick test: seed={args.seed}, n_epoch={args.n_epoch}, n_episodes={args.n_episodes}, n_agents={args.n_agents}, initial_jobs={args.initial_jobs}")
    # Allow writing artifacts for this quick developer run so we can inspect outputs
    # (this is safe for local developer runs; CI/test harnesses rely on the default False)
    args.allow_history_writes = True

    # Select execution mode
    if getattr(args, 'mode', 'marl') == "marl":
        marl_agent_wrapper(args)
    elif getattr(args, 'mode', 'marl') == "random":
        random_agent_wrapper(args)
    else:
        raise ValueError(f"Unknown mode: {getattr(args, 'mode', None)}")
