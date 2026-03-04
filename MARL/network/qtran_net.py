import torch
import torch.nn as nn
import torch.nn.functional as F


# Counterfactual joint networks: inputs state, all agents' hidden_states, other agents' actions, own agent ID; outputs joint Q-values for all own actions
class QtranQAlt(nn.Module):
    def __init__(self, args):
        super(QtranQAlt, self).__init__()
        self.args = args

        # Encode each agent's action
        self.action_encoding = nn.Sequential(nn.Linear(self.args.n_actions, self.args.n_actions),
                                             nn.ReLU(),
                                             nn.Linear(self.args.n_actions, self.args.n_actions))

        # Encode each agent's hidden_state
        self.hidden_encoding = nn.Sequential(nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim))

        # After encoding and summing, input: state, sum of all agents' hidden_states, sum of other agents' actions; state includes current agent ID
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim + self.args.n_agents
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.n_actions))

    # Since all agents' hidden_states at all timesteps are already computed, joint Q-values can be calculated for all transitions at once, no need one-by-one.
    def forward(self, state, hidden_states, actions):  # (episode_num, max_episode_len, n_agents, n_actions)
        # state shape is (episode_num, max_episode_len, n_agents, state_shape+n_agents), includes current agent ID
        episode_num, max_episode_len, n_agents, n_actions = actions.shape

        # Encode each agent's action
        action_encoding = self.action_encoding(actions.reshape(-1, n_actions))
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, n_agents, n_actions)

        # Encode each agent's hidden_state
        hidden_encoding = self.hidden_encoding(hidden_states.reshape(-1, self.args.rnn_hidden_dim))
        hidden_encoding = hidden_encoding.reshape(episode_num, max_episode_len, n_agents, self.args.rnn_hidden_dim)

        # Sum all agents' hidden_encoding
        hidden_encoding = hidden_encoding.sum(dim=-2)  # (episode_num, max_episode_len, rnn_hidden_dim)
        hidden_encoding = hidden_encoding.unsqueeze(-2).expand(-1, -1, n_agents, -1)  # (episode_num, max_episode_len, n_agents, rnn_hidden_dim)

        # For each agent, sum other agents' action_encoding
        # First make last dimension contain all agents' actions
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, 1, n_agents * n_actions)
        action_encoding = action_encoding.repeat(1, 1, n_agents, 1)  # Now each agent has all agents' actions
        # Set each agent's own action to 0
        action_mask = (1 - torch.eye(n_agents))  # torch.eye() generates a 2D diagonal matrix
        action_mask = action_mask.view(-1, 1).repeat(1, n_actions).view(n_agents, -1)
        if self.args.cuda:
            action_mask = action_mask.cuda()
        action_encoding = action_encoding * action_mask.unsqueeze(0).unsqueeze(0)
        # Since all agents' actions are now in the last dimension, cannot add directly. So expand one dimension, sum, then remove
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, n_agents, n_agents, n_actions)
        action_encoding = action_encoding.sum(dim=-2)  # (episode_num, max_episode_len, n_agents, rnn_hidden_dim)

        inputs = torch.cat([state, hidden_encoding, action_encoding], dim=-1)
        q = self.q(inputs)
        return q


# Joint action-value network: inputs state, all agents' hidden_states, all agents' actions; outputs corresponding joint Q-value
class QtranQBase(nn.Module):
    def __init__(self, args):
        super(QtranQBase, self).__init__()
        self.args = args
        # action_encoding encodes each agent's hidden_state and action, then sums all agents' hidden_states and actions to get approximate joint hidden_state and action
        ae_input = self.args.rnn_hidden_dim + self.args.n_actions
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                             nn.ReLU(),
                                             nn.Linear(ae_input, ae_input))

        # After encoding and summing, input: state, sum of all agents' hidden_states and actions
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))

    # Since all agents' hidden_states at all timesteps are already computed, joint Q-values can be calculated for all transitions at once, no need one-by-one.
    def forward(self, state, hidden_states, actions):  # (episode_num, max_episode_len, n_agents, n_actions)
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([hidden_states, actions], dim=-1)
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)  # Reshape back to n_agents dimension for summing
        hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2)

        inputs = torch.cat([state.reshape(episode_num * max_episode_len, -1), hidden_actions_encoding], dim=-1)
        q = self.q(inputs)
        return q


# Input current state and all agents' hidden_states, output V-value
class QtranV(nn.Module):
    def __init__(self, args):
        super(QtranV, self).__init__()
        self.args = args

        # hidden_encoding encodes each agent's hidden_state, then sums all agents' hidden_states to get approximate joint hidden_state
        hidden_input = self.args.rnn_hidden_dim
        self.hidden_encoding = nn.Sequential(nn.Linear(hidden_input, hidden_input),
                                             nn.ReLU(),
                                             nn.Linear(hidden_input, hidden_input))

        # After encoding and summing, input: state, sum of all agents' hidden_states
        v_input = self.args.state_shape + self.args.rnn_hidden_dim
        self.v = nn.Sequential(nn.Linear(v_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))

    def forward(self, state, hidden):
        episode_num, max_episode_len, n_agents, _ = hidden.shape
        state = state.reshape(episode_num * max_episode_len, -1)
        hidden_encoding = self.hidden_encoding(hidden.reshape(-1, self.args.rnn_hidden_dim))
        hidden_encoding = hidden_encoding.reshape(episode_num * max_episode_len, n_agents, -1).sum(dim=-2)
        inputs = torch.cat([state, hidden_encoding], dim=-1)
        v = self.v(inputs)
        return v
