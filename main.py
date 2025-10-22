import argparse
import numpy as np
import pickle
from environment import MASAEnv
import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))

from MARL.runner import Runner
from MARL.common.arguments import (
    get_common_args,
    get_coma_args,
    get_mixer_args,
    get_centralv_args,
    get_reinforce_args,
    get_commnet_args,
    get_g2anet_args,
)

from utils.PDRs.shortestDistence import SDrules


# ============================================================
# === MARL Agent Wrapper =====================================
# ============================================================

def marl_agent_wrapper(args):
    """Standard MARL training loop (supports QMIX, COMA, etc.)."""

    # --- Environment (MASAEnv) ---
    env = MASAEnv()
    env.reset()
    env_info = env.get_env_info()

    # --- Shape bilgilerini environment'tan al ---
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

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

    # ✅ get_mixer_args env boyutlarını ezdiği için tekrar sabitle
    args.obs_shape = env_info["obs_shape"]      # 11D observation
    args.state_shape = env_info["state_shape"]  # 64D global state

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
    else:
        win_rate, reward, _ = runner.evaluate([], 0)
        print(f"Avg reward for {args.alg}: {reward}")


# ============================================================
# === Random Baseline ========================================
# ============================================================

def random_agent_wrapper():
    episodes = 10
    env = MASAEnv()
    EATs = []
    schedule_processes = []

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
    with open("./my_data_and_graph/pickles/process.pk", "wb") as f:
        pickle.dump(schedule_processes, f)


# ============================================================
# === Rule-Based Baseline (SDrules) ==========================
# ============================================================

def SDrules_agent_wrapper():
    EPISODES = 10
    sd_rules = SDrules()
    env = MASAEnv()

    for episode in range(EPISODES):
        done = False
        env.reset()
        while not done:
            actions = []
            agents_id_sequence = sd_rules.FIFO_generate_agents_sequence(len(env.jobs))

            for agent_id in agents_id_sequence:
                avail_actions = env._build_avail_actions()[agent_id]
                valid_actions = [k for k, v in enumerate(avail_actions) if v == 1]
                if valid_actions:
                    action = sd_rules.choose_action(agent_id, avail_actions, None, None)
                    actions.append(action)
                else:
                    actions.append(0)

            _, _, done, info = env.step(actions)
        print(f"[Episode {episode}] Completion time: {env.t}")


# ============================================================
# === Step 7B Integration Mode ===============================
# ============================================================

def step7b_agent_wrapper(args):
    """Replay-aware QMIX training using dynamic arrivals (Step 7B mode)."""
    print("\n🚀 Starting Step 7B replay-aware QMIX training...")

    env = MASAEnv(
        num_jobs=args.start_planes,
        job_spawn_baseline=args.arrival_prob,
        seed=args.seed,
    )

    args = get_mixer_args(args)
    args.obs_shape = 11
    args.state_shape = 64

    runner = Runner(env, args)

    print("\n[TEST] Running one rollout episode (evaluation mode)...")
    runner.rolloutWorker.generate_episode(global_ep_idx=0, evaluate=True)

    print("\n[TRAIN] Starting replay-buffer aware training loop...")
    runner.run(num=1)

    print("\n✅ Step 7B training finished successfully.")


# ============================================================
# === Entry Point ============================================
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["marl", "random", "rules", "7b"], default="marl",
                        help="Select which mode to run: marl / random / rules / 7b.")
    parser.add_argument("--alg", type=str, default="qmix",
                        help="Algorithm type (qmix, coma, reinforce, etc.)")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for reproducibility.")
    args_main = parser.parse_args()

    # Integrate with common args
    args = get_common_args()
    args.alg = args_main.alg
    args.seed = args_main.seed

    # Select execution mode
    if args_main.mode == "marl":
        marl_agent_wrapper(args)
    elif args_main.mode == "random":
        random_agent_wrapper()
    elif args_main.mode == "rules":
        SDrules_agent_wrapper()
    elif args_main.mode == "7b":
        step7b_agent_wrapper(args)
    else:
        raise ValueError(f"Unknown mode: {args_main.mode}")
