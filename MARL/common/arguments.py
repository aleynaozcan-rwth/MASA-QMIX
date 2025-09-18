import argparse

"""
Here are the params for training
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
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of epochs between evaluations')

    # NOTE on CLI (Command-Line Interface):
    # In terminals (e.g., WSL Ubuntu / VS Code Terminal), passing booleans like "--learn False"
    # can be tricky because bool("False") becomes True in Python. For now we keep defaults here.
    # (We can switch to store_true/store_false later if you want to flip them from CLI.)
    parser.add_argument('--model_dir', type=str, default='./MARL/model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')

    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--havelook', type=bool, default=False, help='whether to print intermediate info')

    args = parser.parse_args()
    return args


# arguments of vdn / qmix / qtran
def get_mixer_args(args):
    # ---------- QUICK-RUN CHANGES (original → new) ----------
    # network
    args.rnn_hidden_dim   = 64     # CHANGED (64 → 32)
    args.qmix_hidden_dim  = 32    # CHANGED (32 → 16)
    args.two_hyper_layers = False  # (unchanged)
    args.hyper_hidden_dim = 64    # CHANGED (64 → 32)
    args.qtran_hidden_dim = 64     # (unchanged)
    args.lr               = 5e-4   # CHANGED (5e-4 → 1e-3)

    # epsilon-greedy
    args.epsilon          = 1.0    # (unchanged)
    args.min_epsilon      = 0.05   # (unchanged)
    anneal_steps          = 50000   # CHANGED (50000 → 5000->2000)
    args.anneal_epsilon   = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'  # (unchanged)

    # training schedule
    args.n_epoch     = 15000   # CHANGED (15000 → 200-> 300->150)
    args.n_episodes  = 5    # CHANGED (5 → 2->3)
    args.train_steps = 2     # CHANGED (2 → 1)

    # evaluation / saving cadence
    args.evaluate_cycle = 50     # CHANGED (50 → 25)
    args.batch_size     = 32    # CHANGED (32 → 16->8)
    args.buffer_size    = 5000   # CHANGED (5000 → 2000->1000)

    args.save_cycle         = 50   # CHANGED (50 → 200)  # save less often in quick runs
    args.target_update_cycle= 200   # CHANGED (200 → 100)

    # QTRAN lambda (unused for plain QMIX, kept for compatibility)
    args.lambda_opt  = 1     # (unchanged)
    args.lambda_nopt = 1     # (unchanged)

    # prevent gradient explosion
    args.grad_norm_clip = 10 # (unchanged)

    # MAVEN (left as-is; only used if alg == 'maven')
    args.noise_dim            = 16    # (unchanged)
    args.lambda_mi            = 0.001 # (unchanged)
    args.lambda_ql            = 1     # (unchanged)
    args.entropy_coefficient  = 0.001 # (unchanged)
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim     = 128
    args.lr_actor       = 1e-4
    args.lr_critic      = 1e-3

    # epsilon-greedy
    args.epsilon         = 0.5
    args.anneal_epsilon  = 0.00064
    args.min_epsilon     = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # td-lambda
    args.td_lambda = 0.8

    # training/eval/saving
    args.n_epoch         = 20000
    args.n_episodes      = 1
    args.evaluate_cycle  = 100
    args.save_cycle      = 5000
    args.target_update_cycle = 200

    args.grad_norm_clip  = 10
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim     = 128
    args.lr_actor       = 1e-4
    args.lr_critic      = 1e-3

    # epsilon-greedy
    args.epsilon         = 0.5
    args.anneal_epsilon  = 0.00064
    args.min_epsilon     = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # training/eval/saving
    args.n_epoch         = 20000
    args.n_episodes      = 1
    args.evaluate_cycle  = 100
    args.save_cycle      = 5000
    args.target_update_cycle = 200

    args.grad_norm_clip  = 10
    return args


# arguments of reinforce
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim     = 128
    args.lr_actor       = 1e-4
    args.lr_critic      = 1e-3

    # epsilon-greedy
    args.epsilon         = 0.5
    args.anneal_epsilon  = 0.00064
    args.min_epsilon     = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # training/eval/saving
    args.n_epoch         = 20000
    args.n_episodes      = 1
    args.evaluate_cycle  = 100
    args.save_cycle      = 5000

    args.grad_norm_clip  = 10
    return args


# arguments of coma+commnet
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
