# MARL/policy/qmix.py
# Step 8A.7 – Replay-aware QMIX (MASA-QMIX version)
# -------------------------------------------------
# Compatible with:
#   • MASAEnv (obs: 11-dim, state: 64-dim)
#   • ReplayBuffer + RolloutWorker 8A.6.6
# Notes:
#   - Input to RNN = obs_dim [+ n_actions if last_action] [+ n_agents if reuse_network]
#   - Uses QMixNet (aliased as QMixer) for state-conditioned mixing

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from MARL.network.base_net import RNNAgent as RNN
from MARL.network.qmix_net import QMixNet as QMixer


class QMIX:
    """
    QMIX policy implementation adapted for MASA-QMIX.
    • Input to RNN: obs (+ last_action, + agent_ID one-hot) depending on args
    • Hidden: args.rnn_hidden_dim (default: 64)
    • Output: per-agent Q-values, mixed to joint Q via QMixNet
    """

    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.device = torch.device("cuda" if args.cuda else "cpu")

        # ===== RNN input dimension (MUST match _get_inputs_t) =====
        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        print(f"[QMIX Init] RNN input shape = {input_shape} (obs={self.obs_shape}, "
              f"last_action={args.last_action}, reuse_network={args.reuse_network})")

        # ----- Networks -----
        self.eval_rnn = RNN(input_shape, args).to(self.device)
        self.target_rnn = RNN(input_shape, args).to(self.device)
        self.eval_mixer = QMixer(args).to(self.device)
        self.target_mixer = QMixer(args).to(self.device)

        # Params & optimizer (keep names consistent)
        self.eval_parameters = list(self.eval_rnn.parameters()) + list(self.eval_mixer.parameters())
        if args.optimizer.upper() == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)

        # Hidden states
        self.eval_hidden = None
        self.target_hidden = None

        # Model I/O
        self.model_dir = os.path.join(args.model_dir, args.alg, args.map)
        if self.args.load_model:
            rnn_path = os.path.join(self.model_dir, "rnn_net_params.pkl")
            mix_path = os.path.join(self.model_dir, "qmix_net_params.pkl")
            if os.path.exists(rnn_path) and os.path.exists(mix_path):
                map_loc = "cuda:0" if self.args.cuda else "cpu"
                self.eval_rnn.load_state_dict(torch.load(rnn_path, map_location=map_loc))
                self.eval_mixer.load_state_dict(torch.load(mix_path, map_location=map_loc))
                self._update_target_networks()
                print(f"[QMIX] Loaded pretrained weights from {self.model_dir}")
            else:
                print("[QMIX] No pretrained model found — starting from scratch.")

        print("[QMIX] Policy initialized and moved to device:", self.device)

    # -----------------------------------------------------------
    # Forward pass (RNN evaluation)
    # -----------------------------------------------------------
    def forward(self, obs, hidden_state):
        q, h_out = self.eval_rnn(obs, hidden_state)
        return q, h_out

    # -----------------------------------------------------------
    # Target network synchronization
    # -----------------------------------------------------------
    def _update_target_networks(self):
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mixer.load_state_dict(self.eval_mixer.state_dict())

    # -----------------------------------------------------------
    # Replay-based training
    # -----------------------------------------------------------
    def learn(self, batch, train_step):
        """
        Learn from sampled batch (replay-aware).
        Expects keys: o, o_next, u, r, terminated, filled, state, state_next
                      (optional) u_onehot, avail_u_next
        Shapes:
          o, o_next:     (B, T, n_agents, obs_dim)
          u:             (B, T, n_agents, 1)
          r:             (B, T, 1)
          terminated:    (B, T, 1)
          filled:        (B, T, 1)
          state, state_next: (B, T, state_dim)
          u_onehot:      (B, T, n_agents, n_actions) [optional]
          avail_u_next:  (B, T, n_agents, n_actions) [optional]
        """
        req = ["o", "o_next", "u", "r", "terminated", "filled", "state", "state_next"]
        for k in req:
            if k not in batch:
                raise KeyError(f"[QMIX.learn] Missing key in batch: '{k}'")

        to_t = lambda x, dtype=torch.float32: torch.tensor(
            x, dtype=dtype, device=("cuda:0" if self.args.cuda else "cpu")
        )

        o       = to_t(batch["o"])                               # (B, T, n_agents, obs_dim)
        o_next  = to_t(batch["o_next"])
        u       = to_t(batch["u"], dtype=torch.long)             # (B, T, n_agents, 1)
        r       = to_t(batch["r"])                               # (B, T, 1)
        term    = to_t(batch["terminated"])                      # (B, T, 1)
        filled  = to_t(batch["filled"])                          # (B, T, 1)
        s       = to_t(batch["state"])                           # (B, T, state_dim)
        s_next  = to_t(batch["state_next"])

        avail_u_next = to_t(batch["avail_u_next"]) if "avail_u_next" in batch else None
        u_onehot     = to_t(batch["u_onehot"])   if "u_onehot"     in batch else None

        B, T, _, _ = o.shape

        # Init hidden states
        self.init_hidden(episode_num=B)

        # Per-timestep agent Q-values (eval & target)
        q_evals, q_targets = self._get_q_values_replay(o, o_next, u_onehot)  # (B, T, n_agents, n_actions)

        # Chosen action values
        q_eval_chosen = torch.gather(q_evals, dim=3, index=u).squeeze(3)     # (B, T, n_agents)

        # Mask invalid next actions (if provided)
        if avail_u_next is not None:
            q_targets = q_targets.clone()
            q_targets[avail_u_next == 0.0] = -1e9

        # Max-Q over next actions
        q_target_max = q_targets.max(dim=3)[0]                               # (B, T, n_agents)

        # Mix to joint Q
        q_total_eval   = self.eval_mixer(q_eval_chosen, s)                   # (B, T, 1)
        q_total_target = self.target_mixer(q_target_max, s_next)             # (B, T, 1)

        # TD targets & loss
        targets  = r + self.args.gamma * q_total_target * (1.0 - term)       # (B, T, 1)
        td_error = q_total_eval - targets.detach()                            # (B, T, 1)
        loss     = ((td_error * filled) ** 2).sum() / (filled.sum() + 1e-9)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        # Target sync
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self._update_target_networks()

        # Logging (optional)
        try:
            os.makedirs("./my_data_and_graph/historydata", exist_ok=True)
            with open("./my_data_and_graph/historydata/loss.txt", "a") as f:
                print(float(loss.item()), file=f)
            with open("./my_data_and_graph/historydata/td_error.txt", "a") as f:
                print(float(td_error.abs().mean().item()), file=f)
        except Exception:
            pass

        return {"loss": float(loss.item()), "td_error": float(td_error.abs().mean().item())}

    # ------------------------------------------------------------------
    # Build inputs & compute Q over replay mini-batch
    # ------------------------------------------------------------------
    def _get_inputs_t(self, obs_t, u_onehot_t_minus1, episode_num):
        """
        Build RNN inputs at a single timestep for all episodes:
          obs_t: (B, n_agents, obs_dim)
          u_onehot_t_minus1: (B, n_agents, n_actions) or None
        Returns:
          (B*n_agents, input_dim) where input_dim matches __init__ construction
        """
        parts = [obs_t]  # (B, n_agents, obs_dim)

        if self.args.last_action:
            if u_onehot_t_minus1 is None:
                zeros = torch.zeros(episode_num, self.n_agents, self.n_actions, device=obs_t.device)
                parts.append(zeros)
            else:
                parts.append(u_onehot_t_minus1)

        if self.args.reuse_network:
            eye = torch.eye(self.n_agents, device=obs_t.device).unsqueeze(0).expand(episode_num, -1, -1)
            parts.append(eye)

        x = torch.cat([p.reshape(episode_num * self.n_agents, -1) for p in parts], dim=1)
        return x

    def _get_q_values_replay(self, o, o_next, u_onehot=None):
        """
        Compute per-agent Q_evals and Q_targets across the replay batch.
        Returns:
          q_evals, q_targets: (B, T, n_agents, n_actions)
        """
        B, T, _, _ = o.shape
        q_evals, q_targets = [], []

        for t in range(T):
            obs_t      = o[:, t]        # (B, n_agents, obs_dim)
            obs_next_t = o_next[:, t]   # (B, n_agents, obs_dim)

            u_prev = None
            if self.args.last_action and (u_onehot is not None):
                u_prev = u_onehot[:, t - 1] if t > 0 else None

            inputs_eval = self._get_inputs_t(obs_t, u_prev, episode_num=B)
            inputs_tgt  = self._get_inputs_t(obs_next_t, (u_onehot[:, t] if u_onehot is not None else None), episode_num=B)

            if self.args.cuda:
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()

            q_eval,  self.eval_hidden   = self.eval_rnn(inputs_eval, self.eval_hidden)     # (B*n_agents, n_actions)
            q_tgt,   self.target_hidden = self.target_rnn(inputs_tgt, self.target_hidden)  # (B*n_agents, n_actions)

            q_eval = q_eval.view(B, self.n_agents, -1)
            q_tgt  = q_tgt.view(B, self.n_agents, -1)

            q_evals.append(q_eval)
            q_targets.append(q_tgt)

        q_evals   = torch.stack(q_evals,   dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    # ------------------------------------------------------------------
    # Hidden state utils & saving
    # ------------------------------------------------------------------
    def init_hidden(self, episode_num):
        self.eval_hidden   = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim), device=self.device)
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim), device=self.device)

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.eval_mixer.state_dict(), os.path.join(self.model_dir, f"{num}_qmix_net_params.pkl"))
        torch.save(self.eval_rnn.state_dict(),   os.path.join(self.model_dir, f"{num}_rnn_net_params.pkl"))
