# MARL/policy/qmix.py – Step 8A.6.5
# -------------------------------------------------------------
# True replay-based QMIX learning:
# - Learns from episodic mini-batches sampled from ReplayBuffer
# - Double-Q with target network & invalid-action masking
# - Returns {"loss": ..., "td_error": ...} for logging
#
# Assumes batch dict contains (from ReplayBuffer.sample):
#   o: (B, T, n_agents, obs_dim)
#   o_next: (B, T, n_agents, obs_dim)
#   u: (B, T, n_agents, 1)                 # action indices
#   u_onehot: (B, T, n_agents, n_actions)  # optional (if provided)
#   r: (B, T, 1)
#   terminated: (B, T, 1)
#   filled: (B, T, 1)
#   avail_u_next: (B, T, n_agents, n_actions)   # optional
#   state: (B, T, state_dim)
#   state_next: (B, T, state_dim)
#
# Notes:
# - If u_onehot is missing and args.last_action=True, we handle t=0 with zeros.
# - If avail_u_next is missing, we skip masking (not recommended).
# - CUDA usage controlled by args.cuda

import os
import torch
import torch.nn as nn

from MARL.network.base_net import RNN
from MARL.network.qmix_net import QMixNet


class QMIX:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        # Input to agent RNN
        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # Networks
        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)
        self.target_qmix_net = QMixNet(args)

        # Device
        if self.args.cuda:
            self.eval_rnn.cuda(); self.target_rnn.cuda()
            self.eval_qmix_net.cuda(); self.target_qmix_net.cuda()

        # Model dir & (optional) load
        self.model_dir = os.path.join(args.model_dir, args.alg, args.map)
        if self.args.load_model:
            rnn_path = os.path.join(self.model_dir, "rnn_net_params.pkl")
            qmix_path = os.path.join(self.model_dir, "qmix_net_params.pkl")
            if os.path.exists(rnn_path) and os.path.exists(qmix_path):
                map_location = "cuda:0" if self.args.cuda else "cpu"
                self.eval_rnn.load_state_dict(torch.load(rnn_path, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(qmix_path, map_location=map_location))
                self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
                self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
                print(f"[QMIX] Loaded pretrained weights from {self.model_dir}")
            else:
                print("[QMIX] No pretrained model found — starting from scratch.")

        # Optimizer
        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer.upper() == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)

        # Hidden states
        self.eval_hidden = None
        self.target_hidden = None
        print("[Init] QMIX (Step 8A.6.5, replay-based)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def learn(self, batch: dict, train_step: int):
        """
        Replay-based QMIX learning from sampled episodic mini-batch.
        Returns a dict with {"loss": float, "td_error": float}.
        """
        # Required fields
        required = ["o", "o_next", "u", "r", "terminated", "filled", "state", "state_next"]
        for k in required:
            if k not in batch:
                raise KeyError(f"[QMIX.learn] Missing key in batch: '{k}'")

        # Convert to tensors
        to_t = lambda x, dtype=torch.float32: torch.tensor(x, dtype=dtype, device=("cuda:0" if self.args.cuda else "cpu"))
        o = to_t(batch["o"])                              # (B, T, n_agents, obs_dim)
        o_next = to_t(batch["o_next"])
        u = to_t(batch["u"], dtype=torch.long)            # (B, T, n_agents, 1)
        r = to_t(batch["r"])                              # (B, T, 1)
        terminated = to_t(batch["terminated"])            # (B, T, 1)
        filled = to_t(batch["filled"])                    # (B, T, 1)
        s = to_t(batch["state"])                          # (B, T, state_dim)
        s_next = to_t(batch["state_next"])

        avail_u_next = None
        if "avail_u_next" in batch:
            avail_u_next = to_t(batch["avail_u_next"])

        u_onehot = None
        if "u_onehot" in batch:
            u_onehot = to_t(batch["u_onehot"])

        B, T, _, _ = o.shape

        # Init hidden states
        self.init_hidden(episode_num=B)

        # Compute per-agent Q over time
        q_evals, q_targets = self._get_q_values_replay(
            o, o_next, u_onehot=u_onehot
        )  # (B, T, n_agents, n_actions) each

        # Q for executed actions
        q_eval_chosen = torch.gather(q_evals, dim=3, index=u).squeeze(3)  # (B, T, n_agents)

        # Mask invalid next actions
        if avail_u_next is not None:
            q_targets = q_targets.clone()
            q_targets[avail_u_next == 0.0] = -1e9

        # Max over next actions
        q_target_max = q_targets.max(dim=3)[0]  # (B, T, n_agents)

        # Mix
        q_total_eval = self.eval_qmix_net(q_eval_chosen, s)       # (B, T, 1) or (B, T)
        q_total_target = self.target_qmix_net(q_target_max, s_next)

        # Targets & loss
        targets = r + self.args.gamma * q_total_target * (1.0 - terminated)  # (B, T, 1)
        td_error = q_total_eval - targets.detach()                            # (B, T, 1)
        mask = filled                                                         # (B, T, 1)
        loss = ((td_error * mask) ** 2).sum() / (mask.sum() + 1e-9)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        # Target sync
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        # Logging
        try:
            os.makedirs("./my_data_and_graph/historydata", exist_ok=True)
            with open("./my_data_and_graph/historydata/loss.txt", "a") as f:
                print(float(loss.item()), file=f)
            with open("./my_data_and_graph/historydata/td_error.txt", "a") as f:
                mean_td = float(td_error.abs().mean().item())
                print(mean_td, file=f)
        except Exception:
            pass

        return {"loss": float(loss.item()), "td_error": float(td_error.abs().mean().item())}

    # ------------------------------------------------------------------
    # Q computation over replay mini-batch
    # ------------------------------------------------------------------
    def _get_inputs_t(self, obs_t, u_onehot_t_minus1, episode_num):
        """
        Build RNN inputs at a single timestep for all episodes:
          obs_t: (B, n_agents, obs_dim)
          u_onehot_t_minus1: (B, n_agents, n_actions) or None
        Returns: inputs flattened to (B*n_agents, input_dim)
        """
        parts = [obs_t]  # (B, n_agents, obs_dim)

        # last action feature
        if self.args.last_action:
            if u_onehot_t_minus1 is None:
                zeros = torch.zeros(episode_num, self.n_agents, self.n_actions, device=obs_t.device)
                parts.append(zeros)
            else:
                parts.append(u_onehot_t_minus1)

        # reuse network: add agent ID one-hot
        if self.args.reuse_network:
            eye = torch.eye(self.n_agents, device=obs_t.device).unsqueeze(0).expand(episode_num, -1, -1)
            parts.append(eye)

        x = torch.cat([p.reshape(episode_num * self.n_agents, -1) for p in parts], dim=1)
        return x

    def _get_q_values_replay(self, o, o_next, u_onehot=None):
        """
        Compute Q_evals and Q_targets over replay batch.
          o, o_next: (B, T, n_agents, obs_dim)
          u_onehot: (B, T, n_agents, n_actions) or None
        Returns:
          q_evals, q_targets: (B, T, n_agents, n_actions)
        """
        B, T, _, _ = o.shape
        q_evals = []
        q_targets = []

        for t in range(T):
            obs_t = o[:, t]         # (B, n_agents, obs_dim)
            obs_next_t = o_next[:, t]

            # Get u_onehot(t-1) and u_onehot(t)
            u_prev = None
            if self.args.last_action:
                if u_onehot is not None:
                    u_prev = u_onehot[:, t - 1] if t > 0 else None

            # Prepare inputs for eval and target
            inputs_eval = self._get_inputs_t(obs_t, u_prev, episode_num=B)
            inputs_tgt = self._get_inputs_t(obs_next_t, (u_onehot[:, t] if (u_onehot is not None) else None), episode_num=B)

            # Move hidden to device
            if self.args.cuda:
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()

            # Forward
            q_eval, self.eval_hidden = self.eval_rnn(inputs_eval, self.eval_hidden)     # (B*n_agents, n_actions)
            q_tgt, self.target_hidden = self.target_rnn(inputs_tgt, self.target_hidden) # (B*n_agents, n_actions)

            # Reshape back
            q_eval = q_eval.view(B, self.n_agents, -1)
            q_tgt = q_tgt.view(B, self.n_agents, -1)

            q_evals.append(q_eval)
            q_targets.append(q_tgt)

        q_evals = torch.stack(q_evals, dim=1)   # (B, T, n_agents, n_actions)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    # ------------------------------------------------------------------
    # Hidden state utils & saving
    # ------------------------------------------------------------------
    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        if self.args.cuda:
            self.eval_hidden = self.eval_hidden.cuda()
            self.target_hidden = self.target_hidden.cuda()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.eval_qmix_net.state_dict(), os.path.join(self.model_dir, f"{num}_qmix_net_params.pkl"))
        torch.save(self.eval_rnn.state_dict(), os.path.join(self.model_dir, f"{num}_rnn_net_params.pkl"))
