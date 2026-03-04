import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixNet(nn.Module):
    """
    Robust QMIX mixer with safe getattr defaults for missing args.
    Tolerant to different arg naming (qmix_hidden_dim vs mix_embed_dim, etc.).
    """

    def __init__(self, args):
        super().__init__()
        # safe/compatible defaults and aliases
        self.n_agents = int(getattr(args, "n_agents", 1))
        self.state_dim = int(getattr(args, "state_shape", getattr(args, "state_dim", 64)))
        self.embed_dim = int(getattr(args, "qmix_hidden_dim", getattr(args, "mix_embed_dim", 32)))
        self.two_hyper_layers = bool(getattr(args, "two_hyper_layers", False))

        # hypernet embedding size alias
        hypernet_embed = int(getattr(args, "hyper_hidden_dim", getattr(args, "hypernet_embed", max(self.embed_dim, 64))))

        # first hypernet (state -> weights for first layer)
        if self.two_hyper_layers:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
        else:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)

        # bias for first layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # second hypernet (state -> weights for second layer)
        if self.two_hyper_layers:
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        else:
            self.hyper_w_2 = nn.Linear(self.state_dim, self.embed_dim)

        # final bias producing scalar
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

        # small initializer safety
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, agent_qs, states):
        """
        agent_qs: Tensor shape (B, T, n_agents) or (B, n_agents)
        states:   Tensor shape (B, T, state_dim) or (B, state_dim)
        returns:  q_total shape (B, T, 1) or (B, 1)
        """
        squeeze_time = False
        if agent_qs.dim() == 2:
            agent_qs = agent_qs.unsqueeze(1)
            states = states.unsqueeze(1)
            squeeze_time = True

        B, T, N = agent_qs.shape
        if N != self.n_agents:
            raise AssertionError(f"QMixNet expected {self.n_agents} agents, got {N}")

        agent_qs_flat = agent_qs.view(B * T, 1, N)
        states_flat = states.view(B * T, -1)

        # first layer
        w1 = self.hyper_w_1(states_flat)
        # ensure non-negative mixing weights if desired (common practice)
        w1 = torch.abs(w1)
        b1 = self.hyper_b_1(states_flat)

        w1 = w1.view(-1, N, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        hidden = torch.bmm(agent_qs_flat, w1).squeeze(1) + b1.squeeze(1)
        hidden = F.elu(hidden)

        # second layer
        w2 = self.hyper_w_2(states_flat)
        w2 = torch.abs(w2)
        b2 = self.hyper_b_2(states_flat)

        if w2.dim() == 1:
            w2 = w2.unsqueeze(-1)
        w2 = w2.view(-1, self.embed_dim, 1)
        b2 = b2.view(-1, 1, 1)

        y = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2.squeeze(1)
        q_total = y.view(B, T, 1)

        if squeeze_time:
            return q_total.squeeze(1)
        return q_total
        #--------------------------------------------------------------