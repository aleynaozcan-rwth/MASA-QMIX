# MARL/network/base_net.py
# Step 8A.6.2 – Learning-Active Base Network
# ------------------------------------------------
# Simplified input (10 features per agent) for environment test
# Fully compatible with rollout 8A.6-QMix version
# ------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    """
    RNN-based agent network for QMIX
    Step 8A.6.2 version (10-dim obs input)
    -------------------------------------
    - Input: observation (size 10)
    - Hidden: 64 units
    - Output: Q-values over n_actions
    """

    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(10, 64)  # 🔸 CHANGED from 43 → 10
        self.rnn = nn.GRUCell(64, 64)
        self.fc2 = nn.Linear(64, args.n_actions)

    def forward(self, obs, hidden_state):
        """
        obs: (n_agents, obs_shape)
        hidden_state: (n_agents, hidden_dim)
        """
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, 64)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class BasicCritic(nn.Module):
    """
    Optional centralized critic (not used in this QMIX test)
    """

    def __init__(self, input_shape, args):
        super(BasicCritic, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 🔧 Compatibility aliases for older policies
RNN = RNNAgent
Critic = BasicCritic

