import argparse

"""
Here are the params for training
Step 7A Update:
- Added parameters for dynamic job arrivals (start_planes, max_planes, arrival_prob, variable_ops).
- Compatible with ScheduleEnv dynamic simulation.
"""

def get_common_args():
    parser = argparse.ArgumentParser()
    # environment settings (SMAC-flavored; mostly unused by ScheduleEnv)
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='boatschedule', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default=r'', help='absolute path to save the replay')

    # In total, 13 algorithm variants can be tested, but there are effectively 8 agent types.
    # The alternative algorithms are:
    #   vdn, coma, central_v, qmix, qtran_base, qtran_alt, reinforce,
    #   coma+commnet, central_v+commnet, reinforce+commnet,
    #   coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')

    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')

    # NOTE on CLI (Command-Line Interface):
    # In terminals (e.g., WSL Ubuntu / VS Code Terminal), passing booleans like "--learn False"
    # can be tricky because bool("False") becomes True in Python. We keep defaults here.
    parser.add_argument('--evaluate_epoch', type=int, default=5, help='number of episodes in each evaluation')

    parser.add_argument('--model_dir', type=str, default='./MARL/model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')

    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--havelook', type=bool, default=False, help='whether to print intermediate info')

    # ============================================================
    # Step 7A — Dynamic Job Arrival Parameters
    # ============================================================
    parser.add_argument('--start_planes', type=int, default=4,
                        help='initial number of planes (jobs) at t=0')
    parser.add_argument('--max_planes', type=int, default=12,
                        help='maximum number of planes (jobs) allowed per episode')
    parser.add_argument('--arrival_prob', type=float, default=0.2,
                        help='probability of spawning a new job per environment step')
    parser.add_argument('--variable_ops', type=bool, default=True,
                        help='whether newly generated jobs have variable operation counts')
    parser.add_argument('--num_operators', type=int, default=4,
                        help='number of operators in the environment')
    parser.add_argument('--machine_speed_range', type=float, nargs=2, default=[0.7, 1.4],
                        help='min and max relative machine speed factors across sites')
    # ============================================================

    args = parser.parse_args()
    return args


# ===============================================================
# Arguments for VDN / QMIX / QTRAN
# ===============================================================
def get_mixer_args(args):
    # ---------- QUICK-RUN CHANGES (original → new) ----------
    # network
    args.rnn_hidden_dim   = 64
    args.qmix_hidden_dim  = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr               = 5e-4

    # epsilon-greedy
    args.epsilon          = 1.0
    args.min_epsilon      = 0.05
    anneal_steps          = 10000
    args.anneal_epsilon   = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # training schedule
    args.n_epoch     = 400
    args.n_episodes  = 4
    args.train_steps = 2

    # evaluation / saving cadence
    args.evaluate_cycle = 20
    args.batch_size     = 24
    args.buffer_size    = 3000
    args.save_cycle         = 100
    args.target_update_cycle= 50

    # QTRAN lambda (unused for plain QMIX)
    args.lambda_opt  = 1
    args.lambda_nopt = 1
    args.grad_norm_clip = 10

    # MAVEN (kept for compatibility)
    args.noise_dim            = 16
    args.lambda_mi            = 0.001
    args.lambda_ql            = 1
    args.entropy_coefficient  = 0.001
    return args


# ===============================================================
# Arguments of COMA
# ===============================================================
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


# ===============================================================
# Arguments of Central-V
# ===============================================================
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


# ===============================================================
# Arguments of Reinforce
# ===============================================================
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


# ===============================================================
# Arguments for CommNet / G2ANet
# ===============================================================
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args
