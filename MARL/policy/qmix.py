# =============================================================
# MARL/policy/qmix.py
# Step 8A.7.6 – MASA-QMIX Replay-Aware Learning + Safe AutoSave
# -------------------------------------------------------------
# - Compatible with MASAEnv (11D obs, 64D state)
# - Reward normalization for stable updates
# - Double-Q with target sync
# - Guaranteed checkpoint save to ./MARL/model/qmix/masa_schedule/
# ------------------------------------------------------------

import os
import torch
import torch.nn.functional as F
from MARL.network.base_net import RNNAgent
from MARL.network.qmix_net import QMixNet as QMixer


class QMIX:
    """Replay-based QMIX policy for MASA-QMIX environment."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")

        # --- Network dimensions ---
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        self.state_shape = args.state_shape

        # --- Input (obs + last_action + agent_ID) ---
        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        print(f"[QMIX Init] RNN input shape = {input_shape} "
              f"(obs={self.obs_shape}, last_action={args.last_action}, reuse_network={args.reuse_network})")

        # --- Networks ---
        self.eval_rnn = RNNAgent(input_shape, args).to(self.device)
        self.target_rnn = RNNAgent(input_shape, args).to(self.device)
        self.eval_mixer = QMixer(args).to(self.device)
        self.target_mixer = QMixer(args).to(self.device)

        # --- Optimizer ---
        self.params = list(self.eval_rnn.parameters()) + list(self.eval_mixer.parameters())
        self.optimizer = torch.optim.RMSprop(self.params, lr=args.lr)
        self.train_step = 0

        # =====================================================
        # === Canonical Model Save Directory (always same) ====
        # =====================================================
        base_dir = os.path.abspath("./MARL/model/qmix/masa_schedule")
        os.makedirs(base_dir, exist_ok=True)
        self.model_dir = base_dir
        print(f"[QMIX] Checkpoints dir: {self.model_dir}")
        print(f"[QMIX] Policy initialized and moved to device: {self.device}")

    # ==========================================================
    # === Main Learning Function ===============================
    # ==========================================================
    def learn(self, batch, train_step):
        """Learn from a batch of replayed episodes."""
        required_keys = ["o", "o_next", "u", "r", "terminated", "filled", "state", "state_next"]
        for k in required_keys:
            if k not in batch:
                raise KeyError(f"[QMIX.learn] Missing key in batch: '{k}'")

        to_t = lambda x, dtype=torch.float32: torch.tensor(x, dtype=dtype, device=self.device)

        o = to_t(batch["o"])
        o_next = to_t(batch["o_next"])
        u = to_t(batch["u"], dtype=torch.long)
        r = to_t(batch["r"]) / 10.0  # normalize reward
        terminated = to_t(batch["terminated"])
        filled = to_t(batch["filled"])
        s = to_t(batch["state"])
        s_next = to_t(batch["state_next"])

        avail_u_next = to_t(batch["avail_u_next"]) if "avail_u_next" in batch else None
        u_onehot = to_t(batch["u_onehot"]) if "u_onehot" in batch else None

        B, T, _, _ = o.shape
        self.init_hidden(episode_num=B)

        # --- Compute Q-values ---
        q_evals, q_targets = self._get_q_values_replay(o, o_next, u_onehot)
        q_eval_chosen = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        if avail_u_next is not None:
            q_targets = q_targets.clone()
            q_targets[avail_u_next == 0.0] = -1e9
        q_target_max = q_targets.max(dim=3)[0]

        # --- Mix ---
        q_total_eval = self.eval_mixer(q_eval_chosen, s)
        q_total_target = self.target_mixer(q_target_max, s_next)

        # --- TD target and loss ---
        targets = r + self.args.gamma * q_total_target * (1.0 - terminated)
        td_error = q_total_eval - targets.detach()
        mask = filled
        loss = ((td_error * mask) ** 2).sum() / (mask.sum() + 1e-9)

        # --- Backpropagation ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()

        # --- Target update ---
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self._update_target_networks()

        # ======================================================
        # === Safe Auto-Save Checkpoint (every 100 steps) ======
        # ======================================================
        if (train_step + 1) % 100 == 0 or train_step == (self.args.train_steps - 1):
            try:
                rnn_path = os.path.join(self.model_dir, "rnn_net_params.pkl")
                mix_path = os.path.join(self.model_dir, "qmix_net_params.pkl")
                torch.save(self.eval_rnn.state_dict(), rnn_path)
                torch.save(self.eval_mixer.state_dict(), mix_path)
                print(f"[QMIX] ✅ Checkpoint saved @ step {train_step + 1}\n"
                      f"   RNN  → {rnn_path}\n   MIX  → {mix_path}")
            except Exception as e:
                print(f"[QMIX WARNING] Failed to save checkpoint: {e}")

        # --- Logging ---
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

    # ==========================================================
    # === Helper Functions =====================================
    # ==========================================================
    def _get_inputs_t(self, obs_t, u_onehot_t_minus1, episode_num):
        parts = [obs_t]
        if self.args.last_action:
            parts.append(u_onehot_t_minus1 if u_onehot_t_minus1 is not None
                         else torch.zeros(episode_num, self.n_agents, self.n_actions, device=obs_t.device))
        if self.args.reuse_network:
            eye = torch.eye(self.n_agents, device=obs_t.device).unsqueeze(0).expand(episode_num, -1, -1)
            parts.append(eye)
        return torch.cat([p.reshape(episode_num * self.n_agents, -1) for p in parts], dim=1)

    def _get_q_values_replay(self, o, o_next, u_onehot=None):
        B, T, _, _ = o.shape
        q_evals, q_targets = [], []
        for t in range(T):
            obs_t = o[:, t]
            obs_next_t = o_next[:, t]
            u_prev = (u_onehot[:, t - 1] if (self.args.last_action and u_onehot is not None and t > 0) else None)
            inputs_eval = self._get_inputs_t(obs_t, u_prev, episode_num=B)
            inputs_tgt = self._get_inputs_t(obs_next_t,
                                            (u_onehot[:, t] if u_onehot is not None else None),
                                            episode_num=B)

            if self.args.cuda:
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()

            q_eval, self.eval_hidden = self.eval_rnn(inputs_eval, self.eval_hidden)
            q_tgt, self.target_hidden = self.target_rnn(inputs_tgt, self.target_hidden)

            q_evals.append(q_eval.view(B, self.n_agents, -1))
            q_targets.append(q_tgt.view(B, self.n_agents, -1))
        return torch.stack(q_evals, dim=1), torch.stack(q_targets, dim=1)

    def _update_target_networks(self):
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mixer.load_state_dict(self.eval_mixer.state_dict())

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim), device=self.device)
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim), device=self.device)
