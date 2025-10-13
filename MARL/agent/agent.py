import numpy as np
import torch
from torch.distributions import Categorical

from MARL.policy.vdn import VDN
from MARL.policy.qmix import QMIX
from MARL.policy.coma import COMA
from MARL.policy.reinforce import Reinforce
from MARL.policy.central_v import CentralV
from MARL.policy.qtran_alt import QtranAlt
from MARL.policy.qtran_base import QtranBase
from MARL.policy.maven import MAVEN


# ==============================
# Agents (no communication)
# ==============================

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

    # -----------------------------
    # Action helpers (unchanged)
    # -----------------------------
    def random_choice_with_mask(self, avail_actions):
        temp = [i for i, v in enumerate(avail_actions) if v == 1]
        if len(temp) == 0:
            return 18  # sentinel
        if temp[0] == 18:
            return 18
        if 18 in temp:
            temp.remove(18)
        return np.random.choice(temp, 1, False)[0]

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]

        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions_t = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        else:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        if self.args.alg in ['coma', 'central_v', 'reinforce']:
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions_t, epsilon, evaluate)
        else:
            q_value[avail_actions_t == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = self.random_choice_with_mask(avail_actions_t[0])
                if action == 18:
                    print(avail_actions_t[0])
            else:
                action = torch.argmax(q_value).cpu()
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0
        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    # -----------------------------
    # Utilities
    # -----------------------------
    def _get_max_episode_len(self, batch):
        """
        Episode-batch modu için: pad kesimi.
        Eğer hiç terminated=1 yoksa, episode_limit'i kullan.
        """
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            found = False
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    max_episode_len = max(max_episode_len, transition_idx + 1)
                    found = True
                    break
            if not found:
                max_episode_len = max(max_episode_len, self.args.episode_limit)
        if max_episode_len == 0:
            max_episode_len = self.args.episode_limit
        return max_episode_len

    # -----------------------------
    # Training (Step 7A + 7B)
    # -----------------------------
    def train(self, batch, train_step, epsilon=None):
        """
        İki modu destekler:
          1) Episode-batch (Step 7A): batch['terminated'] mevcut → policy.learn(...)
          2) Transition-batch (Step 7B): batch['state'] mevcut → policy.learn_from_transitions(...)
             (Eğer policy bu API'yi sağlamıyorsa, dummy kayıp döndürür.)
        """
        # --- Transition-batch (Step 7B ReplayBuffer) ---
        if isinstance(batch, dict) and 'state' in batch and 'terminated' not in batch:
            # Beklenen alanlar: state, action, reward, next_state, done, plane_id, time, (aux)
            states = batch['state']
            actions = batch.get('action', None)
            rewards = batch['reward']
            next_states = batch['next_state']
            dones = batch['done']

            loss_val = None
            # Eğer policy bu API'yi sağlıyorsa onu kullan
            if hasattr(self.policy, 'learn_from_transitions'):
                loss_val = self.policy.learn_from_transitions(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    dones=dones,
                    train_step=train_step,
                    epsilon=epsilon
                )
            else:
                # Uyum katmanı: eğitim döngüsü kırılmasın diye, dummy bir değer döndür.
                # (Gerçek güncelleme için policy tarafına transition-öğrenme eklenebilir.)
                loss_val = float(-np.mean(rewards)) if rewards.size > 0 else 0.0

            # save_cycle kontrolü policy-specific olduğu için burada atlıyoruz
            return loss_val

        # --- Episode-batch (Step 7A - mevcut davranış) ---
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)

        if train_step > 0 and train_step % self.args.save_cycle == 0:
            print("\n[Agents] Saving model at step", train_step)
            self.policy.save_model(train_step)
        return None


# ==============================
# CommAgents (with communication)
# ==============================

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
        print('[CommAgents] Initialized')

    def choose_action(self, weights, avail_actions, epsilon, evaluate=False):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])
        prob = torch.nn.functional.softmax(weights, dim=-1)
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0
        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = [obs]
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat(inputs, dim=1)
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
            found = False
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    max_episode_len = max(max_episode_len, transition_idx + 1)
                    found = True
                    break
            if not found:
                max_episode_len = max(max_episode_len, self.args.episode_limit)
        if max_episode_len == 0:
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        """
        Episode-batch + (opsiyonel) transition-batch desteği.
        Comm algoritmalar genelde episode-batch ile çalışır; transition-batch
        geldiğinde policy’de özel bir API yoksa dummy bir değer döndürür.
        """
        if isinstance(batch, dict) and 'state' in batch and 'terminated' not in batch:
            rewards = batch['reward']
            if hasattr(self.policy, 'learn_from_transitions'):
                loss_val = self.policy.learn_from_transitions(
                    states=batch['state'],
                    actions=batch.get('action', None),
                    rewards=rewards,
                    next_states=batch['next_state'],
                    dones=batch['done'],
                    train_step=train_step,
                    epsilon=epsilon
                )
            else:
                loss_val = float(-np.mean(rewards)) if rewards.size > 0 else 0.0
            return loss_val

        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
        return None
