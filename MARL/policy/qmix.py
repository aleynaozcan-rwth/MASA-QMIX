import os
import numpy as np
import torch
from MARL.network.base_net import RNN
from MARL.network.qmix_net import QMixNet


class QMIX:
    """
    Step 7B — Replay-Aware QMIX
    -------------------------------------------------------------
    Adds:  learn_from_transitions()  → allows training directly
           on flattened replay batches (s, a, r, s′, done)
    Fully backward-compatible with episodic training.
    """

    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # --- Networks -------------------------------------------------
        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)
        self.target_qmix_net = QMixNet(args)

        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()

        # --- Model loading --------------------------------------------
        self.model_dir = os.path.join(args.model_dir, args.alg, args.map)
        rnn_path = os.path.join(self.model_dir, "rnn_net_params.pkl")
        qmix_path = os.path.join(self.model_dir, "qmix_net_params.pkl")
        if self.args.load_model and os.path.exists(rnn_path) and os.path.exists(qmix_path):
            map_location = "cuda:0" if self.args.cuda else "cpu"
            self.eval_rnn.load_state_dict(torch.load(rnn_path, map_location=map_location))
            self.eval_qmix_net.load_state_dict(torch.load(qmix_path, map_location=map_location))
            print(f"[QMIX] Loaded pretrained weights from {self.model_dir}")

        # target = eval (hard copy)
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        # --- Optimizer ------------------------------------------------
        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # hidden states
        self.eval_hidden = None
        self.target_hidden = None
        print("[Init] QMIX (Step 7B, replay-aware)")

    # ================================================================
    # === Standard episodic training (unchanged) =====================
    # ================================================================
    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        """
        batch keys:
          o, s, u, r, avail_u, o_next, s_next, avail_u_next, u_onehot, padded, terminated
        Shapes follow: (episode, T, n_agents, ...) except s, s_next which are (episode, T, state_dim)
        """
        episode_num = batch["o"].shape[0]
        self.init_hidden(episode_num)

        # to torch tensors
        for key in batch.keys():
            if key == "u":
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        s, s_next, u, r, avail_u, avail_u_next, terminated = (
            batch["s"], batch["s_next"], batch["u"], batch["r"],
            batch["avail_u"], batch["avail_u_next"], batch["terminated"]
        )
        mask = 1 - batch["padded"].float()

        # Q-values per agent
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        if self.args.cuda:
            s, s_next, u, r, terminated, mask = [
                x.cuda() for x in [s, s_next, u, r, terminated, mask]
            ]

        # pick taken-action Q
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # max over next actions (mask invalid)
        q_targets[avail_u_next == 0.0] = -9999999
        q_targets = q_targets.max(dim=3)[0]

        # mix
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

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    # ================================================================
    # === Step 7B: Transition-based replay update ====================
    # ================================================================
    def learn_from_transitions(self, *args, **kwargs):
        """
        Flexible adapter to support Step 7B transition batches.
        Accepts either:
          - learn_from_transitions(batch_dict, train_step)
          - learn_from_transitions(states=..., actions=..., rewards=..., next_states=..., dones=..., train_step=...)
        Converts the transition batch into a 1-step episodic batch and calls self.learn(...)
        so that the full QMIX (RNN + mixer) pipeline remains intact.
        """
        # ---- Parse inputs flexibly ---------------------------------
        train_step = kwargs.pop("train_step", 0)
        if len(args) >= 1 and isinstance(args[0], dict):
            batch_dict = args[0]
        else:
            # keyword style
            batch_dict = {
                "state": kwargs["states"] if "states" in kwargs and kwargs["states"] is not None else kwargs.get("state"),
                "action": kwargs["actions"] if "actions" in kwargs and kwargs["actions"] is not None else kwargs.get("action"),
                "reward": kwargs["rewards"] if "rewards" in kwargs and kwargs["rewards"] is not None else kwargs.get("reward"),
                "next_state": kwargs["next_states"] if "next_states" in kwargs and kwargs["next_states"] is not None else kwargs.get("next_state"),
                "done": kwargs["dones"] if "dones" in kwargs and kwargs["dones"] is not None else kwargs.get("done"),
            }
        required = ["state", "action", "reward", "next_state", "done"]
        for k in required:
            if batch_dict.get(k) is None:
                raise ValueError(f"[QMIX.learn_from_transitions] Missing key '{k}' in inputs.")

        # to numpy (B, dim)
        s = np.asarray(batch_dict["state"])
        a = np.asarray(batch_dict["action"])
        r = np.asarray(batch_dict["reward"]).reshape(-1, 1)
        s_next = np.asarray(batch_dict["next_state"])
        done = np.asarray(batch_dict["done"]).reshape(-1, 1)

        B = s.shape[0]
        T = 1  # single-step pseudo-episode

        # ---- Build pseudo episodic batch with correct shapes -------
        # observations are not available from transitions; use zeros as placeholder
        o = np.zeros((B, T, self.n_agents, self.obs_shape), dtype=np.float32)
        o_next = np.zeros_like(o)

        # global states
        # if incoming state dim differs, we will fit/crop
        def fit_state(x, target_dim):
            x = np.asarray(x, dtype=np.float32)
            if x.shape[1] == target_dim:
                return x
            out = np.zeros((x.shape[0], target_dim), dtype=np.float32)
            cols = min(x.shape[1], target_dim)
            out[:, :cols] = x[:, :cols]
            return out

        s_fit = fit_state(s, self.state_shape)
        s_next_fit = fit_state(s_next, self.state_shape)
        s_batched = s_fit.reshape(B, T, self.state_shape)
        s_next_batched = s_next_fit.reshape(B, T, self.state_shape)

        # actions: put into agent 0 slot; others zero
        u = np.zeros((B, T, self.n_agents, 1), dtype=np.int64)
        a_int = a.reshape(-1).astype(np.int64)
        a_int = np.mod(a_int, self.n_actions)  # be safe
        u[:, 0, 0, 0] = a_int  # agent 0 executes

        # onehot (not used in loss, but required by learn() flow)
        u_onehot = np.zeros((B, T, self.n_agents, self.n_actions), dtype=np.float32)
        u_onehot[np.arange(B), 0, 0, a_int] = 1.0

        # rewards / done (keep shapes)
        r_batched = r.reshape(B, T, 1).astype(np.float32)
        terminated = done.reshape(B, T, 1).astype(np.float32)

        # availability masks → ones (all actions available)
        avail_u = np.ones((B, T, self.n_agents, self.n_actions), dtype=np.float32)
        avail_u_next = np.ones_like(avail_u)

        # padded flags → zeros
        padded = np.zeros((B, T, 1), dtype=np.float32)

        batch_epi = dict(
            o=o,
            s=s_batched,
            u=u,
            r=r_batched,
            avail_u=avail_u,
            o_next=o_next,
            s_next=s_next_batched,
            avail_u_next=avail_u_next,
            u_onehot=u_onehot,
            padded=padded,
            terminated=terminated,
        )

        # call standard episodic learn with T=1
        try:
            self.learn(batch_epi, max_episode_len=1, train_step=train_step)
        except Exception as e:
            print("[ERROR] QMIX.learn_from_transitions adapter failed:", e)
            raise
        return 0.0  # optional: return last loss if desired

    # ================================================================
    # === Helper functions (unchanged) ===============================
    # ================================================================
    def _get_inputs(self, batch, transition_idx):
        obs, obs_next, u_onehot = (
            batch["o"][:, transition_idx],
            batch["o_next"][:, transition_idx],
            batch["u_onehot"][:],
        )
        episode_num = obs.shape[0]
        inputs, inputs_next = [obs], [obs_next]
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
        inputs = torch.cat(
            [x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1
        )
        inputs_next = torch.cat(
            [x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1
        )
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch["o"].shape[0]
        q_evals, q_targets = [], []
        for t in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, t)
            if self.args.cuda:
                inputs, inputs_next = inputs.cuda(), inputs_next.cuda()
                self.eval_hidden, self.target_hidden = (
                    self.eval_hidden.cuda(),
                    self.target_hidden.cuda(),
                )
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        return torch.stack(q_evals, dim=1), torch.stack(q_targets, dim=1)

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim)
        )
        self.target_hidden = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim)
        )

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
