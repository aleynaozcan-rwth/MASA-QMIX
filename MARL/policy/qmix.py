# Numpy import for fail-safe action_idx handling
import numpy as np
# =============================================================
# MARL/policy/qmix.py
# Step 8A.7.6 – MASA-QMIX Replay-Aware Learning + Safe AutoSave
# -------------------------------------------------------------
# - Compatible with MASAEnv (6D obs, 64D state)
# - Double-Q with target sync
# - Guaranteed checkpoint save to ./MARL/model/qmix/masa_schedule/
# ------------------------------------------------------------

import os
import torch
import torch.nn.functional as F
import time
import threading  # [PHASE7] For thread-safe target update counter
from MARL.network.base_net import RNNAgent
from MARL.network.qmix_net import QMixNet as QMixer


class QMIX:
    """Replay-based QMIX policy for MASA-QMIX environment."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if self.args.cuda else "cpu")

        # --- Network dimensions ---
        self.n_agents = self.args.n_agents
        self.n_actions = self.args.n_actions
        self.obs_shape = self.args.obs_shape
        self.state_shape = self.args.state_shape

        # --- Input (obs + last_action + agent_ID) ---
        input_shape = self.obs_shape
        if self.args.last_action:
            input_shape += self.n_actions
        if self.args.reuse_network:
            input_shape += self.n_agents

        # [PHASE4-FIX] Validate network dimensions are positive
        if input_shape <= 0:
            raise ValueError(
                f"[PHASE4] Invalid input_shape: {input_shape}. "
                f"obs_shape={self.obs_shape}, n_actions={self.n_actions}, n_agents={self.n_agents}"
            )
        if self.state_shape <= 0:
            raise ValueError(f"[PHASE4] Invalid state_shape: {self.state_shape}")
        if self.n_agents <= 0:
            raise ValueError(f"[PHASE4] Invalid n_agents: {self.n_agents}")
        if self.n_actions <= 0:
            raise ValueError(f"[PHASE4] Invalid n_actions: {self.n_actions}")

        print(f"[QMIX Init] RNN input shape = {input_shape} "
              f"(obs={self.obs_shape}, last_action={self.args.last_action}, reuse_network={self.args.reuse_network})")

        # --- Networks ---
        self.eval_rnn = RNNAgent(input_shape, self.args).to(self.device)
        self.target_rnn = RNNAgent(input_shape, self.args).to(self.device)
        self.eval_mixer = QMixer(self.args).to(self.device)
        self.target_mixer = QMixer(self.args).to(self.device)

        # --- Optimizer ---
        self.params = list(self.eval_rnn.parameters()) + list(self.eval_mixer.parameters())
        self.optimizer = torch.optim.RMSprop(self.params, lr=self.args.lr)
        self.train_step = 0
        # [PHASE7] Task 7.2: Thread-safe target update counter
        self.target_update_count = 0
        self._target_update_lock = threading.Lock()

        # =====================================================
        # === Canonical Model Save Directory (always same) ====
        # =====================================================
        base_dir = os.path.abspath("./MARL/model/qmix/masa_schedule")
        os.makedirs(base_dir, exist_ok=True)
        self.model_dir = base_dir
        print(f"[QMIX] Checkpoints dir: {self.model_dir}")
        print(f"[QMIX] Policy initialized and moved to device: {self.device}")
        
    def _grad_norm(self):
        """Compute global L2 norm of all gradients in self.params.

        NOTE: clip_grad_norm_ returns the norm *before* clipping,
        so we use this helper to measure grad_before and grad_after correctly.
        """
        total = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(2)
            total += param_norm.item() ** 2
        return total ** 0.5

    # ==========================================================
    # === Main Learning Function ===============================
    # ==========================================================
    def learn(self, batch, train_step, epsilon=None):
        """Learn from a batch of replayed episodes.
        
        Args:
            batch: Replay buffer batch
            train_step: Current training step
            epsilon: Current exploration rate (for adaptive LR scheduling)
        """
        required_keys = ["o", "o_next", "u", "r", "terminated", "filled", "state", "state_next"]
        for k in required_keys:
            if k not in batch:
                raise KeyError(f"[QMIX.learn] Missing key in batch: '{k}'")

        to_t = lambda x, dtype=torch.float32: torch.tensor(x, dtype=dtype, device=self.device)

        o = to_t(batch["o"])
        o_next = to_t(batch["o_next"])
        # Fail-safe: action_idx=-1 olanları 0'a çevir (torch.gather out-of-bounds hatasını engelle)
        u_np = batch["u"]
        if isinstance(u_np, torch.Tensor):
            u_np = u_np.cpu().numpy()
        u_np_safe = np.where(np.array(u_np) == -1, 0, np.array(u_np))
        u = to_t(u_np_safe, dtype=torch.long)
        r = to_t(batch["r"])
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

        # [PHASE1-FIX] Apply mask only to q_targets for max operation, NOT to q_evals
        # Masking q_evals would incorrectly penalize actually-taken actions
        q_target_max_input = q_targets.clone()
        if avail_u_next is not None:
            q_target_max_input[avail_u_next == 0.0] = -1e9
        q_target_max = q_target_max_input.max(dim=3)[0]

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

        # =================================================================
        # === Epsilon-Based Adaptive Learning Rate Scheduling =============
        # =================================================================
        # [v8] Dynamic LR adjustment based on exploration rate (epsilon)
        # Rationale: High exploration → high LR (large updates), low exploration → low LR (fine-tuning)
        # This approach is setting-agnostic and automatically adapts to different epsilon decay schedules
        # 
        # Benefits over fixed milestones:
        # Fixed learning rate (no adaptive decay)
        # Adaptive LR experiments (0.1, 0.5, 0.25 min) all caused divergence
        # Reverting to stable fixed LR approach

        # [PHASE6-FIX] Task 6.6: Use clip_grad_norm_ return value for grad norm computation
        # This is more efficient (computed internally) and avoids manual loops with .item() calls
        #grad_norm_before = torch.nn.utils.clip_grad_norm_(self.params, float('inf'))  # Compute norm without clipping
        #grad_norm_before = float(grad_norm_before)  # Convert to Python float once
        
        # [C1] Gradient clipping is CRITICAL for training stability - must not fail silently
        #grad_norm_after = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        #grad_norm_after = float(grad_norm_after)  # Convert to Python float once
        # NOTE: clip_grad_norm_ returns the pre-clipping gradient norm, 
        # so we must compute grad_before/after manually. The first call 
        # does NOT give the clipped norm. We use a custom grad_norm() 
        # function to correctly measure both before and after clipping.

        # 1) Gradient norm BEFORE clipping
        grad_norm_before = self._grad_norm()
        # 2) Apply gradient clipping (modifies gradients in-place)
        torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        # 3) Gradient norm AFTER clipping
        grad_norm_after = self._grad_norm()
        # 4) Optimizer update
        self.optimizer.step()

        # --- Target update ---
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self._update_target_networks()

        # ======================================================
        # === Safe Auto-Save Checkpoint (every 100 steps) ======
        # ======================================================
        # [C1] Checkpoint saves are CRITICAL - fail-fast if save fails
        # [PHASE6-FIX] Task 6.5: Atomic checkpoint save with validation
        if (train_step + 1) % 100 == 0 or train_step == (self.args.train_steps - 1):
            rnn_path = os.path.join(self.model_dir, "rnn_net_params.pkl")
            mix_path = os.path.join(self.model_dir, "qmix_net_params.pkl")
            
            # Save to temp files first
            rnn_temp = rnn_path + '.tmp'
            mix_temp = mix_path + '.tmp'
            
            try:
                torch.save(self.eval_rnn.state_dict(), rnn_temp)
                torch.save(self.eval_mixer.state_dict(), mix_temp)
                
                # [PHASE6-FIX] Verify files were written correctly
                if not os.path.exists(rnn_temp) or os.path.getsize(rnn_temp) == 0:
                    raise RuntimeError(f"[PHASE6] RNN checkpoint file empty or missing: {rnn_temp}")
                if not os.path.exists(mix_temp) or os.path.getsize(mix_temp) == 0:
                    raise RuntimeError(f"[PHASE6] Mixer checkpoint file empty or missing: {mix_temp}")
                
                # [PHASE6-FIX] Atomic rename (overwrites old checkpoint only if new one is valid)
                os.replace(rnn_temp, rnn_path)
                os.replace(mix_temp, mix_path)
                
                print(f"[QMIX] ✅ Checkpoint saved @ step {train_step + 1}\n"
                      f"   RNN  → {rnn_path}\n   MIX  → {mix_path}")
            except Exception as e:
                # Clean up temp files on error
                try:
                    if os.path.exists(rnn_temp):
                        os.remove(rnn_temp)
                    if os.path.exists(mix_temp):
                        os.remove(mix_temp)
                except Exception:
                    pass
                # Re-raise original error
                raise RuntimeError(f"[PHASE6] Checkpoint save failed at step {train_step + 1}: {e}") from e

        # --- Logging ---
        # [C1] Loss/TD logging - best effort (Rule 3), diagnostics are informational only
        # Compute avg_q and avg_reward for return value
        avg_q = float(q_total_eval.mean().detach().cpu().item())
        # Compute average reward from this batch (for training metrics)
        avg_reward = float(r.mean().detach().cpu().item())
        
        try:
            os.makedirs("./my_data_and_graph/historydata", exist_ok=True)
            # NOTE: loss.txt and td_error.txt are written from runner at epoch-end
            # We don't write them here to avoid format conflicts
            
            # append diagnostics line (lightweight, rate-limited)
            diagnostics_every = int(getattr(self.args, "diagnostics_every", 20) or 20)
            if diagnostics_every > 0 and (train_step % diagnostics_every == 0):
                ts = time.time()
                # [PHASE7] Task 7.2: Use lock when reading target_update_count for logging
                with self._target_update_lock:
                    target_update_count_snapshot = self.target_update_count
                diag_path = "./my_data_and_graph/historydata/diagnostics_log.txt"
                with open(diag_path, "a") as df:
                    # CSV: ts,train_step,avg_q,grad_norm_before,grad_norm_after,target_updates,loss,td_error
                    df.write(f"{ts},{train_step},{avg_q},{grad_norm_before},{grad_norm_after},{target_update_count_snapshot},{float(loss.item())},{float(td_error.abs().mean().item())}\n")
                # also print a concise console line
                print(f"[QMIX Diagnostics] step={train_step} avg_q={avg_q:.4f} grad_before={grad_norm_before} grad_after={grad_norm_after} target_updates={target_update_count_snapshot}")
        except Exception as e:
            # [C1] Logging failures should not crash training (Rule 3)
            import logging
            logging.getLogger(__name__).warning(f"[C1] Diagnostics logging failed (train_step={train_step}): {e}")

        return {
            "loss": float(loss.item()), 
            "td_error": float(td_error.abs().mean().item()), 
            "q_value": avg_q,
            "reward": avg_reward
        }

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
        # [C1] Target update counter is best-effort diagnostics (Rule 3)
        # [PHASE7] Task 7.2: Use lock to protect counter increment
        try:
            with self._target_update_lock:
                self.target_update_count += 1
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[C1] Failed to increment target_update_count (diagnostics only): {e}")

    def init_hidden(self, episode_num):
        # GRU expects hidden shape (num_layers * num_directions, batch, hidden_size)
        # Our RNNAgent.forward packs inputs as (episode_num * n_agents, ...), so batch should be episode_num * n_agents
        batch_size = int(episode_num * self.n_agents)
        h_shape = (1, batch_size, int(getattr(self.args, "rnn_hidden_dim", 64)))
        self.eval_hidden = torch.zeros(h_shape, device=self.device)
        self.target_hidden = torch.zeros(h_shape, device=self.device)

    # --------------------------------------------------
    def select_actions(self, obs_batch, avail_batch=None, evaluate=False, epsilon=None, agent_masks=None):
        """Select actions for a batch of per-agent observations using the eval_rnn Q-values.

        obs_batch: list or array of shape (n_agents, obs_dim) or list-of-arrays
        avail_batch: optional list of per-agent availability vectors (list or np.array)
        epsilon: exploration rate (required for training, optional for evaluation)
        agent_masks: optional binary mask (1=real agent, 0=padded), shape (n_agents,)
        Returns: list of integer actions (one per agent)
        """
        import numpy as _np
        to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=self.device)

        # [TIME-BASED EPSILON] Epsilon must be provided explicitly
        if epsilon is None:
            raise ValueError("epsilon must be provided explicitly to select_actions")

        # Normalize obs_batch to tensor shape (1, n_agents, obs_dim)
        obs_arr = _np.asarray(obs_batch, dtype=_np.float32)
        
        # Validate batch size matches n_agents (after padding)
        if obs_arr.shape[0] != self.n_agents:
            raise ValueError(
                f"[FIXED_AGENT_BATCH] Observation batch size mismatch: "
                f"got {obs_arr.shape[0]}, expected {self.n_agents}. "
                f"Shape: {obs_arr.shape}"
            )
        
        if obs_arr.ndim == 1:
            raise ValueError(
                f"[FIXED_AGENT_BATCH] Received 1D observation batch. "
                f"Expected 2D (n_agents, obs_dim) after padding. Shape: {obs_arr.shape}"
            )
        if obs_arr.ndim == 2:
            # assume (n_agents, obs_dim)
            obs_t = to_t(obs_arr).unsqueeze(0)
        elif obs_arr.ndim == 3:
            obs_t = to_t(obs_arr)
        else:
            obs_t = to_t(obs_arr).unsqueeze(0)

        # Ensure hidden state is initialized for a single-step batch
        self.init_hidden(episode_num=1)

        # Build inputs using the same helper as training
        u_onehot = None
        inputs = self._get_inputs_t(obs_t, None, episode_num=1)

        # Forward through eval_rnn
        q_vals, _ = self.eval_rnn(inputs, self.eval_hidden)
        # q_vals shape: (episode_num * n_agents, n_actions)
        q_vals = q_vals.view(1, self.n_agents, -1).squeeze(0).detach().cpu().numpy()
        
        # Mask Q-values for padded agents (set to large negative)
        if agent_masks is not None:
            agent_masks_arr = _np.array(agent_masks, dtype=_np.float32)
            if agent_masks_arr.shape[0] != self.n_agents:
                raise ValueError(
                    f"[FIXED_AGENT_BATCH] Agent mask size mismatch: "
                    f"got {agent_masks_arr.shape[0]}, expected {self.n_agents}"
                )
            # Broadcast mask to Q-values shape and apply
            mask = agent_masks_arr[:, _np.newaxis]  # (n_agents, 1)
            q_vals = q_vals * mask + (1 - mask) * (-1e10)  # Mask out padded agents

        actions = []
        # A1: Fail-fast validation - avail_batch must match n_agents if provided
        if avail_batch is not None:
            if len(avail_batch) != q_vals.shape[0]:
                raise ValueError(
                    f"avail_batch length mismatch: got {len(avail_batch)}, "
                    f"expected {q_vals.shape[0]} (n_agents)"
                )
        
        for a_idx in range(q_vals.shape[0]):
            q_row = q_vals[a_idx]
            # apply availability mask if provided
            allowed = None
            if avail_batch is not None:
                # Fail-fast: if avail_batch is provided, conversion must succeed
                allowed = list(_np.asarray(avail_batch[a_idx], dtype=_np.int32))

            if allowed is not None:
                # mask unavailable actions by setting very low Q
                mask = _np.asarray(allowed, dtype=_np.int32)
                masked_q = _np.where(mask, q_row, -1e9)
            else:
                masked_q = q_row

            # epsilon-greedy
            if (not evaluate) and (float(_np.random.rand()) < float(epsilon)):
                # choose uniformly among allowed actions
                if allowed is None:
                    act = int(_np.argmax(masked_q))
                else:
                    allowed_inds = [i for i, v in enumerate(mask) if int(v)]
                    if allowed_inds:
                        act = int(_np.random.choice(allowed_inds))
                    else:
                        act = -1
            else:
                # greedy
                if allowed is not None:
                    allowed_inds = [i for i, v in enumerate(mask) if int(v)]
                    if allowed_inds:
                        # argmax only among allowed
                        act = allowed_inds[int(_np.argmax([masked_q[i] for i in allowed_inds]))]
                    else:
                        act = -1
                else:
                    act = int(_np.argmax(masked_q))

            actions.append(act)

        return actions
