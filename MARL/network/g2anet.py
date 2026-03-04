import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np



class G2ANet(nn.Module):
    def __init__(self, input_shape, args):
        super(G2ANet, self).__init__()
        # store args early and use self.args consistently (Option-B)
        self.args = args

        # Encoding
        self.encoding = nn.Linear(input_shape, self.args.rnn_hidden_dim)  # Decode observations for all agents
        self.h = nn.GRUCell(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)  # Each agent encodes its observation to get hidden_state for memorizing previous observations

        # Hard
        
        self.hard_bi_GRU = nn.GRU(self.args.rnn_hidden_dim * 2, self.args.rnn_hidden_dim, bidirectional=True)
        # Analyze h_j to get agent j's weight for agent i, output 2 dimensions, after gumbel_softmax take one dimension: 0=don't consider agent j, 1=consider agent j
        self.hard_encoding = nn.Linear(self.args.rnn_hidden_dim * 2, 2)  # Multiply by 2 because bidirectional GRU, hidden_state dimension is 2 * hidden_dim

        # Soft
        self.q = nn.Linear(self.args.rnn_hidden_dim, self.args.attention_dim, bias=False)
        self.k = nn.Linear(self.args.rnn_hidden_dim, self.args.attention_dim, bias=False)
        self.v = nn.Linear(self.args.rnn_hidden_dim, self.args.attention_dim)

        # Decoding: input own h_i and x_i, output probability distribution of own action
        self.decoding = nn.Linear(self.args.rnn_hidden_dim + self.args.attention_dim, self.args.n_actions)
        self.input_shape = input_shape

    def forward(self, obs, hidden_state):
        size = obs.shape[0]  # batch_size * n_agents
        # First encode observations
        obs_encoding = f.relu(self.encoding(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        # Get h through own GRU
        h_out = self.h(obs_encoding, h_in)  # (batch_size * n_agents, args.rnn_hidden_dim)

        # Hard Attention: GRU and GRUCell are different, input dimension is (sequence_length, batch_size, dim)
        if self.args.hard:
            # Preparation before Hard Attention
            h = h_out.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)  # Transform h to n_agents dimension: (batch_size, n_agents, rnn_hidden_dim)
            input_hard = []
            for i in range(self.args.n_agents):
                h_i = h[:, i]  # (batch_size, rnn_hidden_dim)
                h_hard_i = []
                for j in range(self.args.n_agents):  # For agent i, concatenate own h_i with each other agent's h separately
                    if j != i:
                        h_hard_i.append(torch.cat([h_i, h[:, j]], dim=-1))
                # After j loop: h_hard_i is a list containing (n_agents - 1) tensors of dimension (batch_size, rnn_hidden_dim * 2)
                h_hard_i = torch.stack(h_hard_i, dim=0)
                input_hard.append(h_hard_i)
            # After i loop: input_hard is a list containing n_agents tensors of dimension (n_agents - 1, batch_size, rnn_hidden_dim * 2)
            input_hard = torch.stack(input_hard, dim=-2)
            # Finally get dimension (n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2), ready for input
            input_hard = input_hard.view(self.args.n_agents - 1, -1, self.args.rnn_hidden_dim * 2)

            h_hard = torch.zeros((2 * 1, size, self.args.rnn_hidden_dim))  # Because bidirectional GRU with single layer, first dimension is 2 * 1
            if self.args.cuda:
                h_hard = h_hard.cuda()
            h_hard, _ = self.hard_bi_GRU(input_hard, h_hard)  # (n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2)
            h_hard = h_hard.permute(1, 0, 2)  # (batch_size * n_agents, n_agents - 1, rnn_hidden_dim * 2)
            h_hard = h_hard.reshape(-1, self.args.rnn_hidden_dim * 2)  # (batch_size * n_agents * (n_agents - 1), rnn_hidden_dim * 2)

            # Get hard weights: (n_agents, batch_size, 1, n_agents - 1), extra dimension is used for weighted sum below
            hard_weights = self.hard_encoding(h_hard)
            hard_weights = f.gumbel_softmax(hard_weights, tau=0.01)
            # print(hard_weights)
            hard_weights = hard_weights[:, 1].view(-1, self.args.n_agents, 1, self.args.n_agents - 1)
            hard_weights = hard_weights.permute(1, 0, 2, 3)

        else:
            hard_weights = torch.ones((self.args.n_agents, size // self.args.n_agents, 1, self.args.n_agents - 1))
            if self.args.cuda:
                hard_weights = hard_weights.cuda()

        # Soft Attention
        q = self.q(h_out).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        k = self.k(h_out).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        v = f.relu(self.v(h_out)).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        x = []
        for i in range(self.args.n_agents):
            q_i = q[:, i].view(-1, 1, self.args.attention_dim)  # q of agent i: (batch_size, 1, args.attention_dim)
            k_i = [k[:, j] for j in range(self.args.n_agents) if j != i]  # For agent i, k of other agents
            v_i = [v[:, j] for j in range(self.args.n_agents) if j != i]  # For agent i, v of other agents

            k_i = torch.stack(k_i, dim=0)  # (n_agents - 1, batch_size, args.attention_dim)
            k_i = k_i.permute(1, 2, 0)  # Swap dimensions to get (batch_size, args.attention_dim, n_agents - 1)
            v_i = torch.stack(v_i, dim=0)
            v_i = v_i.permute(1, 2, 0)

            # (batch_size, 1, attention_dim) * (batch_size, attention_dim, n_agents - 1) = (batch_size, 1, n_agents - 1)
            score = torch.matmul(q_i, k_i)

            # Normalization
            scaled_score = score / np.sqrt(self.args.attention_dim)

            # Get weights through softmax
            soft_weight = f.softmax(scaled_score, dim=-1)  # (batch_size, 1, n_agents - 1)

            # Weighted sum: note that last dimension of three matrices is n_agents - 1, result is (batch_size, args.attention_dim)
            x_i = (v_i * soft_weight * hard_weights[i]).sum(dim=-1)
            x.append(x_i)

        # Merge h and x of each agent
        x = torch.stack(x, dim=1).reshape(-1, self.args.attention_dim)  # (batch_size * n_agents, args.attention_dim)
        final_input = torch.cat([h_out, x], dim=-1)
        output = self.decoding(final_input)

        return output, h_out

