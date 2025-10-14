import numpy as np
import torch
from MARL.policy.vdn import VDN
from MARL.policy.qmix import QMIX
from MARL.policy.coma import COMA
from MARL.policy.reinforce import Reinforce
from MARL.policy.central_v import CentralV
from MARL.policy.qtran_alt import QtranAlt
from MARL.policy.qtran_base import QtranBase
from MARL.policy.maven import MAVEN
from torch.distributions import Categorical


class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        if args.alg == 'vdn':
            self.policy = VDN(args)
        elif args.alg == 'qmix':
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            self.policy = Reinforce(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('[Agents] Initialized')

    # (unchanged choose_action helpers omitted for brevity)

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        """
        Dual-path training:
        - If batch looks like replay transitions (has 'state'), call policy.learn_from_transitions()
        - Else, assume episodic mixer training and call policy.learn()
        """
        if isinstance(batch, dict) and 'state' in batch:
            # Transition-based update (Step 7B.3)
            return self.policy.learn_from_transitions(batch, train_step)
        else:
            # Episodic training path (original mixers)
            max_episode_len = self._get_max_episode_len(batch)
            for key in batch.keys():
                if key != 'z':
                    batch[key] = batch[key][:, :max_episode_len]
            self.policy.learn(batch, max_episode_len, train_step, epsilon)
            if train_step > 0 and train_step % self.args.save_cycle == 0:
                print("\nStart saving model", train_step, self.args.save_cycle)
                self.policy.save_model(train_step)
            return None


class CommAgents:
    # unchanged aside from init message
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('[CommAgents] Initialized')

    # rest stays as before
