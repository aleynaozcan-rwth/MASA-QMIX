import argparse

"""
Arguments for MASA-QMIX / ScheduleEnv
Step 7B update:
----------------
- Added replay and rollout configuration (episode_limit, buffer sizes, etc.)
- Added extra parameters for explainable logs and training stability.
- Fully backward-compatible with Step 7A.
"""

def get_common_args():
    parser = argparse.ArgumentParser()

    # ============================================================
    # === Environment settings ==================================
    # ============================================================
    parser.add_argument('--difficulty', type=str, default='7')
    parser.add_argument('--game_version', type=str, default='latest')
    parser.add_argument('--map', type=str, default='boatschedule')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--step_mul', type=int, default=8)
    parser.add_argument('--replay_dir', type=str, default='')
    parser.add_argument('--alg', type=str, default='qmix')
    parser.add_argument('--last_action', type=bool, default=True)
    parser.add_argument('--reuse_network', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--optimizer', type=str, default='RMS')
    parser.add_argument('--evaluate_epoch', type=int, default=5)
    parser.add_argument('--model_dir', type=str, default='./MARL/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--learn', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--havelook', type=bool, default=False)

    # ============================================================
    # === Step 7A — Dynamic Job Arrival Parameters ===============
    # ============================================================
    parser.add_argument('--start_planes', type=int, default=4)
    parser.add_argument('--max_planes', type=int, default=12)
    parser.add_argument('--arrival_prob', type=float, default=0.20)
    parser.add_argument('--variable_ops', type=bool, default=True)
    parser.add_argument('--num_operators', type=int, default=4)
    parser.add_argument('--machine_speed_range', type=float, nargs=2, default=[0.7, 1.4])

    # ============================================================
    # === Step 7B — Replay + Rollout parameters ==================
    # ============================================================
    parser.add_argument('--episode_limit', type=int, default=200,
                        help='max timesteps per episode (for rollout buffer shape)')
    parser.add_argument('--n_agents', type=int, default=12,
                        help='max number of active agents (planes) during episode')
    parser.add_argument('--n_actions', type=int, default=21,
                        help='discrete action space size (site selection, etc.)')
    parser.add_argument('--state_shape', type=int, default=10)
    parser.add_argument('--obs_shape', type=int, default=10)

    # Replay buffer parameters
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help='max total transitions in replay buffer')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size for training updates')
    parser.add_argument('--train_steps', type=int, default=4,
                        help='number of gradient updates per epoch')

    # Evaluation / checkpoint cadence
    parser.add_argument('--n_epoch', type=int, default=400)
    parser.add_argument('--n_episodes', type=int, default=4)
    parser.add_argument('--evaluate_cycle', type=int, default=20)
    parser.add_argument('--save_cycle', type=int, default=100)
    parser.add_argument('--target_update_cycle', type=int, default=50)

    # Exploration / annealing
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--min_epsilon', type=float, default=0.05)
    parser.add_argument('--anneal_epsilon', type=float, default=0.0001)
    parser.add_argument('--epsilon_anneal_scale', type=str, default='step')

    # Gradient / optimization
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--grad_norm_clip', type=float, default=10)

    # Misc logging / explainability
    parser.add_argument('--enable_logs', type=bool, default=True,
                        help='enable step-by-step explainable console logs')
    parser.add_argument('--save_gantt', type=bool, default=True,
                        help='save gantt chart per evaluation cycle')

    args = parser.parse_args()
    return args


# ===============================================================
# === Algorithm-specific argument groups ========================
# ===============================================================

def get_mixer_args(args):
    args.rnn_hidden_dim   = 64
    args.qmix_hidden_dim  = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr               = 5e-4

    args.epsilon          = 1.0
    args.min_epsilon      = 0.05
    anneal_steps          = 10000
    args.anneal_epsilon   = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    args.n_epoch     = 400
    args.n_episodes  = 4
    args.train_steps = 2
    args.evaluate_cycle = 20
    args.batch_size  = 24
    args.buffer_size = 3000
    args.save_cycle  = 100
    args.target_update_cycle = 50

    args.lambda_opt  = 1
    args.lambda_nopt = 1
    args.grad_norm_clip = 10

    args.noise_dim            = 16
    args.lambda_mi            = 0.001
    args.lambda_ql            = 1
    args.entropy_coefficient  = 0.001
    return args


def get_coma_args(args):
    args.rnn_hidden_dim = 64
    args.critic_dim     = 128
    args.lr_actor       = 1e-4
    args.lr_critic      = 1e-3
    args.epsilon         = 0.5
    args.anneal_epsilon  = 0.00064
    args.min_epsilon     = 0.02
    args.epsilon_anneal_scale = 'epoch'
    args.td_lambda = 0.8
    args.n_epoch         = 20000
    args.n_episodes      = 1
    args.evaluate_cycle  = 100
    args.save_cycle      = 5000
    args.target_update_cycle = 200
    args.grad_norm_clip  = 10
    return args


def get_centralv_args(args):
    args.rnn_hidden_dim = 64
    args.critic_dim     = 128
    args.lr_actor       = 1e-4
    args.lr_critic      = 1e-3
    args.epsilon         = 0.5
    args.anneal_epsilon  = 0.00064
    args.min_epsilon     = 0.02
    args.epsilon_anneal_scale = 'epoch'
    args.n_epoch         = 20000
    args.n_episodes      = 1
    args.evaluate_cycle  = 100
    args.save_cycle      = 5000
    args.target_update_cycle = 200
    args.grad_norm_clip  = 10
    return args


def get_reinforce_args(args):
    args.rnn_hidden_dim = 64
    args.critic_dim     = 128
    args.lr_actor       = 1e-4
    args.lr_critic      = 1e-3
    args.epsilon         = 0.5
    args.anneal_epsilon  = 0.00064
    args.min_epsilon     = 0.02
    args.epsilon_anneal_scale = 'epoch'
    args.n_epoch         = 20000
    args.n_episodes      = 1
    args.evaluate_cycle  = 100
    args.save_cycle      = 5000
    args.grad_norm_clip  = 10
    return args


def get_commnet_args(args):
    args.k = 2 if args.map == '3m' else 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args
