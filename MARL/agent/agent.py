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


# Agent with no communication
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
        print('Init Agents')

    # Prevent choosing WAIT (18) if other actions are available
    def random_choice_with_mask(self, avail_actions):
        valid = [i for i, a in enumerate(avail_actions) if a == 1]
        if not valid:
            return 18  # default WAIT if nothing else is available
        if len(valid) > 1 and 18 in valid:
            valid.remove(18)
        return np.random.choice(valid)

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()

        # agent id one-hot
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # prepare tensors
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q values
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        else:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # action selection
        if self.args.alg in ['coma', 'central_v', 'reinforce']:
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.rand() < epsilon:  # exploration
                avail_numpy = avail_actions.cpu().numpy()[0]
                action = self.random_choice_with_mask(avail_numpy)
                print(f"[DEBUG] Random action selected → {action}")
            else:  # exploitation
                action = torch.argmax(q_value).item()
                print(f"[DEBUG] Greedy action selected → {action}")

        return action


    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: q_value/logits of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # First convert the Actor network output to a probability distribution via softmax
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add epsilon noise
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # probability of invalid actions is 0

        """
        After setting the probability of invalid actions to 0, the sum of prob may not be 1.
        We do NOT need to renormalize here, because torch.distributions.Categorical
        will renormalize internally. Note: during training we don't use Categorical,
        so when taking the probability of the executed action, renormalization is needed.
        """

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    # This function has an issue
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        # Because within episode_limit there may be no terminal==1, this can lead to max_episode_len == 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episodes have different lengths, so we need to get the max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)

        if train_step > 0 and train_step % self.args.save_cycle == 0:
            print("\nStart saving model", train_step, self.args.save_cycle)
            self.policy.save_model(train_step)


# Agent with communication
class CommAgents:
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
        print('Init CommAgents')

    # Obtain probabilities from weights, then select an action with epsilon
    def choose_action(self, weights, avail_actions, epsilon, evaluate=False):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # number of available actions
        # First convert the Actor network output to a probability distribution via softmax
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # During training, add noise to the probability distribution
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # probability of invalid actions is 0

        """
        After setting the probability of invalid actions to 0, the sum of prob may not be 1.
        We do NOT need to renormalize here, because torch.distributions.Categorical
        will renormalize internally. Note: during training we don't use Categorical,
        so when taking the probability of the executed action, renormalization is needed.
        """

        if epsilon == 0 and evaluate:
            # During evaluation, directly take the maximum
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = list()
        inputs.append(obs)
        # Append the last action and the agent id to obs
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat([x for x in inputs], dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
        weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
        weights = weights.reshape(self.args.n_agents, self.args.n_actions)
        return weights.cpu()

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

    def train(self, batch, train_step, epsilon=None):  # During COMA training, epsilon is also needed to compute action execution probabilities
        # When learning each time, the lengths of episodes differ, so we take
        # the longest episode length among them as the effective length
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
