# MARL/network/base_net.py
# Step 8A.7 – Learning-Active Base Network (MASA-QMIX)
# ----------------------------------------------------
# Dynamically adapts to full input (obs + last_action + agent_ID)
# Fully compatible with replay-aware QMIX rollout (Step 8A.6.6+)

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    """
    RNN-based agent network for QMIX (MASA-QMIX version)
    ----------------------------------------------------
    • Input: dynamically determined by QMIX setup
      (obs_dim + n_actions if last_action=True + n_agents if reuse_network=True)
    • Hidden: 64 units (args.rnn_hidden_dim)
    • Output: Q-values over available actions
    """

    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        # ✅ Fully dynamic input dimension (matches QMIX input builder)
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        """
        obs: (n_agents, input_shape)
        hidden_state: (n_agents, hidden_dim)
        """
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        q = self.fc2(h_out)
        return q, h_out

    def eval_rnn(self, obs, hidden_state):
        """Evaluation-only forward (no gradient)."""
        with torch.no_grad():
            return self.forward(obs, hidden_state)


class BasicCritic(nn.Module):
    """
    Optional centralized critic (not used in QMIX baseline)
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


# Compatibility aliases
RNN = RNNAgent
Critic = BasicCritic
