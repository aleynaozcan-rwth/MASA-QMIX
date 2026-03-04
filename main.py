import argparse
import numpy as np
import pickle
import os
import sys
from os.path import dirname, abspath

# Ensure the project root is on sys.path before importing local modules
sys.path.append(dirname(dirname(abspath(__file__))))

from environment import MASAEnv, LOG

from MARL.runner import Runner
from MARL.common.arguments import (
    get_common_args,
    get_mutable_args,
    get_mixer_args,
)


# ============================================================
# === MARL Agent Wrapper =====================================
# ============================================================

def marl_agent_wrapper(args):
    """Standard MARL training loop (supports QMIX, COMA, etc.)."""
    
    # [PHASE9-FIX] Task 9.4: Set all random seeds for full reproducibility
    if hasattr(args, 'seed'):
        import torch
        import logging
        logger = logging.getLogger(__name__)
        
        # NumPy seed
        np.random.seed(args.seed)
        logger.info(f"[PHASE9] NumPy random seed set to {args.seed}")
        
        # PyTorch seed (CPU)
        torch.manual_seed(args.seed)
        logger.info(f"[PHASE9] PyTorch CPU seed set to {args.seed}")
        
        # PyTorch seed (GPU) if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            logger.info(f"[PHASE9] PyTorch CUDA seed set to {args.seed}")
            
            # Deterministic mode for CUDA (may reduce performance but ensures reproducibility)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(f"[PHASE9] PyTorch CUDNN deterministic mode enabled")

    # --- Environment (MASAEnv) ---
    # Construct environment without config - all parameters come from args
    auto_arrivals = getattr(args, 'arrival_lambda', 0.0) > 0.0
    try:
        if auto_arrivals:
            LOG.info("[Main] Auto-starting TaskGenerator (arrival_lambda=%.3f)", float(getattr(args, 'arrival_lambda', 0.0)))
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
    env = MASAEnv(
        args=args,
        config_path=None,
        auto_load_config=False,
        auto_start_arrivals=auto_arrivals,
    )
    # Runner will query the environment and initialize any runtime-derived
    # shapes (n_actions, n_agents, obs/state dims, episode_limit). Keep
    # `main.py` strictly as orchestration so it does not set or mutate args.

    # --- Apply QMIX-specific args ---
    args = get_mixer_args(args)

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
    print("====================================")

    runner = Runner(env, args)
    
    # Print environment-derived shapes after Runner initialization
    print(f"\n=== Environment-Derived Shapes ===")
    print(f"Observation dim: {args.obs_shape}")
    print(f"State dim: {args.state_shape}")
    print(f"Number of agents: {args.n_agents}")
    print(f"Number of actions: {args.n_actions}")
    print("==================================\n")
    
    if args.learn:
        runner.run(num=1)
        # Post-run quick summary for developer visibility
        try:
            for idx, r in enumerate(getattr(runner, 'episode_rewards', [])):
                print(f"Episode {idx} | Total reward: {r:.4f}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
        print("Episode finished")
    else:
        win_rate, reward, _ = runner.evaluate([], 0)
        print(f"Avg reward for {args.alg}: {reward}")


# ============================================================
# === Random Baseline ========================================
# ============================================================

def random_agent_wrapper(args):
    episodes = 10
    # For the random baseline use a pure environment 
    auto_arrivals = getattr(args, 'arrival_lambda', 0.0) > 0.0
    try:
        if auto_arrivals:
            LOG.info("[Main] Auto-starting TaskGenerator (arrival_lambda=%.3f)", float(getattr(args, 'arrival_lambda', 0.0)))
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
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
    # Automatically redirect stdout and stderr to terminal_output_log.txt
    import sys
    import os
    log_path = os.path.join(os.path.dirname(__file__), "terminal_output_log.txt")
    sys.stdout = open(log_path, "w")
    sys.stderr = sys.stdout

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
            except Exception as e:
                # best-effort: skip items we can't remove
                logging.getLogger(__name__).warning(f"[C1] Failed to remove {path}: {e}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

    # Select execution mode
    if getattr(args, 'mode', 'marl') == "marl":
        marl_agent_wrapper(args)
    elif getattr(args, 'mode', 'marl') == "random":
        random_agent_wrapper(args)
    else:
        raise ValueError(f"Unknown mode: {getattr(args, 'mode', None)}")
