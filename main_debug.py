# main_debug.py
import os, sys

# --- Add root and MARL paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MARL_DIR = os.path.join(BASE_DIR, "MARL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MARL_DIR not in sys.path:
    sys.path.insert(0, MARL_DIR)

print(">>> Debug PATH added:", BASE_DIR)
print(">>> Debug PATH added:", MARL_DIR)

# --- Imports ---
from MARL.common.arguments import get_common_args, get_mixer_args
from MARL.agent.agent import Agents
from MARL.runner import Runner                  # ✅ düzeltildi
from environment import MASAEnv


from MARL.common.replay_buffer import ReplayBuffer
from MARL.common.rollout import RolloutWorker


if __name__ == "__main__":
    args = get_common_args()
    args = get_mixer_args(args)

    env = MASAEnv(
        num_operators=4,
        obs_dim_agent=10,
        state_dim=64,
        episode_limit=200,
        seed=args.seed
    )

    # >>> BURASI KRİTİK: agents'i kurmadan önce n_agents = env.num_ops
    args.n_agents = env.num_ops

    env.reset()

    agents = Agents(args)
    buffer = ReplayBuffer(episode_capacity=args.buffer_size, seed=args.seed)

    rollout = RolloutWorker(env, agents, buffer=buffer, args=args)



    episode, reward, _, _ = rollout.generate_episode(global_ep_idx=0)
    print(f"\nEpisode reward: {reward:.2f}, buffer length: {len(buffer)}")


