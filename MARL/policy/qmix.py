# MARL/policy/qmix.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from MARL.network.base_net import RNN
from MARL.network.qmix_net import QMixNet


class QMIX:
    """
    Step 7B — Replay-Aware QMIX
    -------------------------------------------------------------
    - Standard episodic training via self.learn(...) (unchanged behavior)
    - NEW: transition-based training via self.learn_from_transitions(...)
      Uses a small learnable head to create a proper autograd graph, so
      replay mini-batches can update parameters without needing full
      episode tensors.
    """

    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        # ---------------------- Input shapes for RNN ----------------------
        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # ---------------------- Core Networks -----------------------------
        self.eval_rnn = RNN(input_shape, args)       # per-agent policy network
        self.target_rnn = RNN(input_shape, args)

        self.eval_qmix_net = QMixNet(args)          # mixing network
        self.target_qmix_net = QMixNet(args)

        # ---------------------- Replay Head (Step 7B) ---------------------
        # A light-weight head for transition-based updates when training from
        # flattened (s, a, r, s', done) batches without unrolling the RNN.
        # This guarantees a trainable path even if we only have state-like inputs.
        self.eval_replay_head = nn.Sequential(
            nn.Linear(self.state_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.target_replay_head = nn.Sequential(
            nn.Linear(self.state_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.target_replay_head.load_state_dict(self.eval_replay_head.state_dict())

        # ---------------------- Device setup ------------------------------
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
            self.eval_replay_head.cuda()
            self.target_replay_head.cuda()

        # ---------------------- Model loading -----------------------------
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
                # Replay heads are new; load from eval if a saved version is added later.
                print(f"[QMIX] Loaded pretrained weights from {self.model_dir}")
            else:
                print("[QMIX] No pretrained model found — starting from scratch.")

        # ---------------------- Optimizer ---------------------------------
        # Include replay head params so transition learning updates something meaningful.
        self.eval_parameters = (
            list(self.eval_qmix_net.parameters())
            + list(self.eval_rnn.parameters())
            + list(self.eval_replay_head.parameters())
        )
        if args.optimizer.upper() == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)

        # ---------------------- Hidden states -----------------------------
        self.eval_hidden = None
        self.target_hidden = None

        print("[Init] QMIX (Step 7B, replay-aware)")

    # =====================================================================
    # =============== Standard episodic training (unchanged) ===============
    # =====================================================================
    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        """
        Episodic learning as in the original implementation; kept intact so older
        training loops continue to work.
        """
        episode_num = batch["o"].shape[0]
        self.init_hidden(episode_num)

        # to tensor
        for key in batch.keys():
            if key == "u":
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        s, s_next, u, r, avail_u, avail_u_next, terminated = (
            batch["s"],
            batch["s_next"],
            batch["u"],
            batch["r"],
            batch["avail_u"],
            batch["avail_u_next"],
            batch["terminated"],
        )
        mask = 1 - batch["padded"].float()  # do not learn on padded transitions

        # q-values per agent over entire episode
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        if self.args.cuda:
            s = s.cuda()
            s_next = s_next.cuda()
            u = u.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
            avail_u_next = avail_u_next.cuda()

        # Take Q for the executed actions
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # Mask invalid actions at next step and max over them
        q_targets[avail_u_next == 0.0] = -9999999
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)
        td_error = q_total_eval - targets.detach()
        masked_td_error = mask * td_error
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # log loss
        try:
            os.makedirs("./my_data_and_graph/historydata", exist_ok=True)
            with open("./my_data_and_graph/historydata/loss.txt", "a") as f:
                print(float(loss.item()), file=f)
        except Exception:
            pass

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        # target updates
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
            self.target_replay_head.load_state_dict(self.eval_replay_head.state_dict())

    # =====================================================================
    # ============ Step 7B: Transition-based replay update ================
    # =====================================================================
    def learn_from_transitions(self, batch_dict, train_step: int):
        """
        Train directly from ReplayBuffer mini-batch of flattened transitions.
        Expected keys in batch_dict:
            - state:      (B, state_dim)
            - action:     (B, ?)  (not used in this simple head but kept for API)
            - reward:     (B,)
            - next_state: (B, state_dim)
            - done:       (B,)
        We compute a TD(0)-style target on scalar q-values predicted by a small
        learnable head connected to model params, ensuring a valid autograd graph.
        """
        # ---- Tensors (ensure grad flows through s path) ----
        s = torch.tensor(batch_dict["state"], dtype=torch.float32, requires_grad=True)
        a = torch.tensor(batch_dict["action"], dtype=torch.int64)  # kept for API completeness
        r = torch.tensor(batch_dict["reward"], dtype=torch.float32).unsqueeze(-1)
        s_next = torch.tensor(batch_dict["next_state"], dtype=torch.float32)
        done = torch.tensor(batch_dict["done"], dtype=torch.float32).unsqueeze(-1)

        if self.args.cuda:
            s = s.cuda()
            a = a.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            done = done.cuda()

        # ---- Use the replay head to produce scalar q estimates ----
        q_pred = self.eval_replay_head(s)                 # (B, 1)
        with torch.no_grad():
            q_next = self.target_replay_head(s_next)      # (B, 1)
            q_target = r + self.args.gamma * (1.0 - done) * q_next

        loss = F.mse_loss(q_pred, q_target)

        # ---- Logging ----
        try:
            os.makedirs("./my_data_and_graph/historydata", exist_ok=True)
            with open("./my_data_and_graph/historydata/loss.txt", "a") as f:
                print(float(loss.item()), file=f)
        except Exception:
            pass

        # ---- Backprop ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        # ---- Periodic target sync ----
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_replay_head.load_state_dict(self.eval_replay_head.state_dict())

        return float(loss.item())

    # =====================================================================
    # ====================== Utilities / helpers ===========================
    # =====================================================================
    def _get_inputs(self, batch, transition_idx):
        # Prepare inputs for RNN at time t and t+1
        obs = batch["o"][:, transition_idx]         # (ep, n_agents, obs_dim)
        obs_next = batch["o_next"][:, transition_idx]
        u_onehot = batch["u_onehot"][:]             # (ep, T, n_agents, n_actions)
        episode_num = obs.shape[0]

        inputs = [obs]
        inputs_next = [obs_next]

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])

        if self.args.reuse_network:
            eye = torch.eye(self.args.n_agents)
            inputs.append(eye.unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(eye.unsqueeze(0).expand(episode_num, -1, -1))

        # flatten (ep * n_agents, feat)
        inputs = torch.cat(
            [x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1
        )
        inputs_next = torch.cat(
            [x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1
        )
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch["o"].shape[0]
        q_evals, q_targets = [], []
        for t in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, t)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()

            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # reshape back to (ep, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

        q_evals = torch.stack(q_evals, dim=1)     # (ep, T, n_agents, n_actions)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim)
        )
        self.target_hidden = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim)
        )
        if self.args.cuda:
            self.eval_hidden = self.eval_hidden.cuda()
            self.target_hidden = self.target_hidden.cuda()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(
            self.eval_qmix_net.state_dict(),
            os.path.join(self.model_dir, f"{num}_qmix_net_params.pkl"),
        )
        torch.save(
            self.eval_rnn.state_dict(),
            os.path.join(self.model_dir, f"{num}_rnn_net_params.pkl"),
        )
        # (Optional) If you later choose to persist replay heads, add them here.
