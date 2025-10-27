import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    """
    Minimal, robust RNN agent base used by QMIX/others.
    Uses getattr(args, 'rnn_hidden_dim', 64) when args may not provide it.
    Accepts input_shape (int) and args (Namespace-like).
    """

    def __init__(self, input_shape: int, args):
        super().__init__()
        hidden_dim = int(getattr(args, "rnn_hidden_dim", 64))
        self.input_shape = int(input_shape)
        self.hidden_size = hidden_dim

        # small input encoder -> recurrent core
        self.fc1 = nn.Linear(self.input_shape, hidden_dim)
        # single-layer GRU for sequence/hidden handling
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # final action-value head (per-agent)
        # Output action-values. If args provides n_actions use it, otherwise fall back to hidden_dim
        out_dim = int(getattr(args, "n_actions", hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, h_in=None):
        """
        x: tensor shape (..., input_shape) or (batch, seq, input_shape)
        h_in: optional hidden state (1, batch, hidden_dim)
        Returns: (out, h_out)
        """
        # normalize shape to (batch, seq, input)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, seq, _ = x.shape
        x_enc = F.relu(self.fc1(x.view(b * seq, -1))).view(b, seq, -1)

        if h_in is None:
            h0 = torch.zeros(1, b, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h0 = h_in

        rnn_out, h_out = self.rnn(x_enc, h0)  # rnn_out: (b, seq, hidden)
        out = F.relu(self.fc_out(rnn_out))
        # if original input was 2D, squeeze seq dim
        if out.shape[1] == 1:
            return out[:, 0, :], h_out
        return out, h_out


class BasicCritic(nn.Module):
    """
    Simple critic network mapping global state -> scalar value(s).
    Uses safe defaults if args missing.
    """

    def __init__(self, state_dim: int, args):
        super().__init__()
        hid = int(getattr(args, "critic_hidden_dim", getattr(args, "rnn_hidden_dim", 64)))
        self.fc1 = nn.Linear(int(state_dim), hid)
        self.fc2 = nn.Linear(hid, hid)
        self.out = nn.Linear(hid, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)


# Compatibility aliases
RNN = RNNAgent
Critic = BasicCritic