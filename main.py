import argparse
import numpy as np
import pickle
from environment import ScheduleEnv
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

    # --- Environment ---
    env = ScheduleEnv()
    env.reset()
    env_info = env.get_env_info()

    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

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
    env = ScheduleEnv()
    EATs = []
    schedule_processes = []

    for episode in range(episodes):
        s = env.reset()
        done = False
        while not done:
            actions = []
            for i in range(len(env.planes)):
                avail_actions = env.get_avail_agent_actions(i)
                if isinstance(avail_actions, str):
                    actions.append(18)
                else:
                    valid_actions = [k for k, v in enumerate(avail_actions) if v == 1]
                    if valid_actions:
                        action = np.random.choice(valid_actions)
                        env.has_chosen_action(action, i)
                    else:
                        action = 18
                    actions.append(action)
            _, _, done, info = env.step(actions)

        EATs.append(info["time"])
        schedule_processes.append(env.job_record_for_gant)
        print(f"[Episode {episode}] Completion time: {info['time']}")

    print(f"Average completion time: {sum(EATs) / len(EATs)}")
    with open("./my_data_and_graph/pickles/process.pk", "wb") as f:
        pickle.dump(schedule_processes, f)


# ============================================================
# === Rule-Based Baseline (SDrules) ==========================
# ============================================================

def SDrules_agent_wrapper():
    EPISODES = 10
    sd_rules = SDrules()
    env = ScheduleEnv()
    sites_locations = env.sites_obj.sites_position

    for episode in range(EPISODES):
        done = False
        env.reset()
        while not done:
            actions = []
            agents_id_sequence = sd_rules.FIFO_generate_agents_sequence(8)

            for agent_id in agents_id_sequence:
                avail_actions = env.get_avail_agent_actions(agent_id)
                current_plane_location = env.planes[agent_id].position
                action = sd_rules.choose_action(agent_id, avail_actions, current_plane_location, sites_locations)
                actions.append(action)
                if action < 18:
                    env.has_chosen_action(action, agent_id)

            reorder_actions = [-1 for _ in range(8)]
            for i in range(8):
                reorder_actions[agents_id_sequence[i]] = actions[i]

            _, done, info = env.step(reorder_actions)
        print(f"[Episode {episode}] Completion time: {info['time']}")

    print(f"Final schedule: {info['episodes_situation']}")


# ============================================================
# === Step 7B Integration Mode ===============================
# ============================================================

def step7b_agent_wrapper(args):
    """Replay-aware QMIX training using dynamic arrivals (Step 7B mode)."""
    print("\n🚀 Starting Step 7B replay-aware QMIX training...")

    args = get_mixer_args(args)
    env = ScheduleEnv(
        start_planes=args.start_planes,
        max_planes=args.max_planes,
        arrival_prob=args.arrival_prob,
        variable_ops=args.variable_ops,
        seed=args.seed,
    )

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
