#!/usr/bin/env python3
"""
run_long.py
Small wrapper that runs the MARL training but preserves CLI overrides for
n_epoch and n_episodes (get_mixer_args previously overwrote them).

Usage example:
  python3 run_long.py --alg qmix --seed 123 --n_epoch 200 --n_episodes 8 --quiet_env
"""
import argparse
import sys
from os.path import dirname, abspath
sys.path.append(dirname(abspath(__file__)))
sys.path.append(dirname(dirname(abspath(__file__))))

from environment import MASAEnv
from MARL.common.arguments import get_common_args, get_mixer_args, get_coma_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
from MARL.runner import Runner


def parse_args():
    # Use parse_known_args so extra hyperparameter flags (e.g. --lr, --batch_size)
    # passed by the user are not rejected here and can be parsed later by
    # `get_common_args()` which reads the full CLI. We only consume the small
    # set of flags this wrapper explicitly cares about.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--alg', type=str, default='qmix')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--n_epoch', type=int, default=None)
    parser.add_argument('--n_episodes', type=int, default=None)
    parser.add_argument('--quiet_env', action='store_true', default=False)
    parser.add_argument('--snapshot_on_eval', action='store_true', default=True)
    parser.add_argument('--clean_history', action='store_true', default=False)
    args, _unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    cli = parse_args()

    # load common args (reads CLI too, so user can pass many flags)
    args = get_common_args()
    args.alg = cli.alg
    args.seed = cli.seed

    # build env and populate env-dependent shapes
    env = MASAEnv()
    env.reset()
    env_info = env.get_env_info()
    args.n_actions = env_info['n_actions']
    args.n_agents = env_info['n_agents']
    args.state_shape = env_info['state_shape']
    args.obs_shape = env_info['obs_shape']
    args.episode_limit = env_info['episode_limit']

    # apply algorithm-specific defaults (may override some fields)
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

    # now re-apply CLI overrides for run length if provided
    if cli.n_epoch is not None:
        args.n_epoch = cli.n_epoch
    if cli.n_episodes is not None:
        args.n_episodes = cli.n_episodes
    args.quiet_env = cli.quiet_env
    args.snapshot_on_eval = cli.snapshot_on_eval
    args.clean_history = cli.clean_history

    # print summary
    print('\n=== Long Training Setup ===')
    print(f'Algorithm: {args.alg}')
    print(f'Seed: {args.seed}')
    print(f'Total epochs: {args.n_epoch}')
    print(f'Episodes per epoch: {args.n_episodes}')
    print('===========================\n')

    runner = Runner(env, args)
    runner.run(num=1)
