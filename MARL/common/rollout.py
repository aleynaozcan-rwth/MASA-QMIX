import numpy as np
import time
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

# [C1] torch is REQUIRED for RNN hidden handling - fail-fast if not available
import torch

# [C1] mask utilities are REQUIRED for action masking - fail-fast if not available
from MARL.common.mask_utils import build_machine_major_mask

# gantt helpers (selection logging) - best effort
try:
    from utils import gantt as gantt_utils
except Exception as e:
    logging.getLogger(__name__).warning(f"[C1] Gantt utilities not available (visualization only): {e}")
    gantt_utils = None

# [C1] io_control is REQUIRED for history writes control - fail-fast if not available
from utils.io_control import allow_history_writes


class RolloutWorker:
    """
    Step 8A.7.3 – Final RolloutWorker for MASA-QMIX (complete fallback-friendly)
    • Works as legacy step-based rollout and provides decide_batch / collect_gantt_from_batch
      helpers so Runner can drive SimPy event-driven MASAEnv.
    """

    def __init__(
        self,
        env,
        agents,
        buffer=None,
        args=None,
        episode_limit: Optional[int] = None,
        epsilon_start: Optional[float] = None,
        epsilon_end: Optional[float] = None,
        device: Optional[str] = None,
        log_prefix: str = "8A.7.3",
    ):
        self.env = env
        self.agents = agents
        self.buffer = buffer
        self.args = args or type("A", (), {})()
        # [C1] Use episode_limit from runner-provided args when available (fail-fast if invalid)
        if hasattr(self.args, "episode_limit") and getattr(self.args, "episode_limit") is not None:
            self.episode_limit = int(getattr(self.args, "episode_limit"))
        elif episode_limit is not None:
            self.episode_limit = int(episode_limit)
        else:
            # Co-Pilot Rule compliance: episode_limit must be explicit
            if not hasattr(self.args, 'episode_limit'):
                raise ValueError(
                    "[FIXED_AGENT_BATCH] args.episode_limit is required but missing. "
                    "This must be set explicitly in configuration."
                )
            self.episode_limit = int(self.args.episode_limit)

        # [C1] Device configuration must be valid (fail-fast if invalid)
        if hasattr(self.args, "device") and getattr(self.args, "device") is not None:
            self.device = getattr(self.args, "device")
        elif device is not None:
            self.device = device
        else:
            self.device = getattr(self.args, 'device', 'cpu')
        self.log_prefix = log_prefix

        # [C1] Epsilon schedule configuration must be valid (fail-fast if invalid)
        if epsilon_start is not None:
            self.epsilon_start = float(epsilon_start)
        else:
            self.epsilon_start = float(self.args.epsilon_start)

        if epsilon_end is not None:
            self.epsilon_end = float(epsilon_end)
        else:
            self.epsilon_end = float(self.args.epsilon_end)

        self.epsilon = float(self.epsilon_start)

        # runtime placeholders
        self.eval_hidden = None
        self.episode_duration = 0

        # [C1] RNG initialization is CRITICAL for reproducibility (fail-fast if invalid)
        seed = getattr(self.args, "seed", None)
        self.rng = np.random.RandomState(seed if seed is not None else 0)

        # [C1] Step counter and diagnostics - configuration must be valid (fail-fast)
        self.step_counter = int(getattr(self.args, 'start_step_counter', 0) or 0)
        self.epsilon_log_every = int(getattr(self.args, 'epsilon_diagnostics_every', 50) or 50)

        # log
        print(f"[RolloutWorker] init | episode_limit={self.episode_limit} | device={self.device}")

    # -------------------------
    # Legacy step-based rollout
    # -------------------------
    # Legacy non-SimPy step-based episode execution removed.
    # The repository now exclusively supports SimPy event-driven episodes via
    # `run_event_driven_episode`. The previous `generate_episode` step-based
    # runner was deleted to simplify runtime paths and enforce a single
    # execution model.

    # ------------------------------------------------
    # Action selection helpers used by Runner fallback
    # ------------------------------------------------
    def _select_actions(self, obs_batch: List[Any], avail_batch: Optional[List[Any]], evaluate: bool = False, epsilon: float = None, agent_masks: Optional[List[int]] = None) -> Tuple[List[Any], Any]:
        """
        Return (actions_list, hidden_state) where actions_list is a list of int actions
        This wrapper tries common agent APIs then falls back to deterministic/random picks.
        
        Args:
            obs_batch: List of observations
            avail_batch: List of available actions
            evaluate: Whether in evaluation mode
            epsilon: Epsilon for exploration (required)
            agent_masks: Binary mask (1=real agent, 0=padded)
        """
        # Epsilon must be provided explicitly
        if epsilon is None:
            raise ValueError("epsilon must be provided explicitly to _select_actions")
        
        # [C1] Try batch API on agents - fail-fast if select_actions/choose_actions fail
        if hasattr(self.agents, "select_actions"):
            # Try to pass epsilon and agent_masks if the method accepts them
            try:
                return self.agents.select_actions(obs_batch, avail_batch, evaluate=evaluate, epsilon=epsilon, agent_masks=agent_masks), None
            except TypeError:
                # Fallback if method doesn't accept agent_masks - try without it
                try:
                    return self.agents.select_actions(obs_batch, avail_batch, evaluate=evaluate, epsilon=epsilon), None
                except TypeError:
                    # Final fallback if method doesn't accept epsilon
                    return self.agents.select_actions(obs_batch, avail_batch, evaluate=evaluate), None
        if hasattr(self.agents, "choose_actions"):
            return self.agents.choose_actions(obs_batch, avail_batch, evaluate=evaluate), None

        # [C1] Try per-observation act method - fail-fast if fails
        if hasattr(self.agents, "act"):
            actions = []
            for ob in obs_batch:
                a = self.agents.act(ob, None, evaluate=evaluate)
                actions.append(a)
            return actions, None

            # Fallback: pick first allowed or random
        actions = []
        for ob in obs_batch:
            # try to extract allowed_machine_indices from ob if present
            allowed = None
            if isinstance(ob, dict):
                allowed = ob.get("allowed_machine_indices", None) or ob.get("avail_row", None)
            if allowed is None:
                # no info — pick 0
                actions.append(0)
            else:
                # [C1] Fallback action selection - fail-fast if conversion fails
                allowed_list = list(allowed)
                if len(allowed_list) == 0:
                    actions.append(None)
                else:
                    if evaluate:
                        actions.append(int(allowed_list[0]))
                    else:
                        actions.append(int(self.rng.choice(allowed_list)))
        return actions, None

    # ------------------------------------------------------------------
    # Event-driven / Runner helpers (used as fallback by Runner)
    # ------------------------------------------------------------------
    def decide_batch(self, batch, evaluate: bool = False):
        """
        Provide actions for a batch of decision-items coming from MASAEnv.
        Best-effort: reuse RolloutWorker._select_actions by treating the batch
        as a simultaneous multi-agent observation list.
        """
        if not batch:
            return []
        obs_list = [item.get("obs") for item in batch]
        # [C1] Avail extraction is CRITICAL - fail-fast if batch access fails
        avail = [item.get("avail_row") for item in batch]

        # [C1] Ensure every decision item has a machine-major availability row.
        # Prefer existing `item['avail_row']` populated by the environment; when
        # missing, compute using the canonical helper `build_machine_major_mask`.
        for item in batch:
            if item is None:
                continue
            if item.get('avail_row') is None and build_machine_major_mask is not None:
                jid = item.get('job_id')
                try:
                    row = build_machine_major_mask(self.env, jid)
                    if row:
                        item['avail_row'] = row
                except Exception as e:
                    # [C1] Mask computation is best-effort (Rule 3)
                    logging.getLogger(__name__).warning(f"[C1] Failed to compute mask for job {jid}: {e}")

        # [C1] Action selection is CRITICAL - fail-fast if _select_actions fails
        actions, _ = self._select_actions(obs_list, avail, evaluate=evaluate)
        return actions

    def collect_gantt_from_batch(self, batch, sim_time):
        """Optional hook to build gantt records from a decision batch. Default: empty."""
        return []

    def process_and_apply_actions(self, batch, raw_actions, sim_time, runner_args=None):
        """Process raw agent outputs into finalized actions and apply them via the
        decision item's resume callable.

        Returns (processed_actions, processed_machine_names)
        """
        args = runner_args if runner_args is not None else self.args
        processed_actions = []
        processed_machine_names = []

        for item, act in zip(batch, (raw_actions or [])):
            chosen = act
            chosen_machine_name = None
            # [C1] Machine action conversion is CRITICAL - fail-fast (Rule 1)
            if bool(getattr(args, 'use_machine_actions', False)):
                mlist = getattr(self.env.workcenters_meta, 'machine_list', []) or []
                ops = int(getattr(self.env, 'num_ops', 1))
                allowed_m_inds = item.get('allowed_machine_indices') or []

                chosen_i = int(act) if act is not None else None

                if chosen_i is None or chosen_i < 0 or chosen_i >= len(mlist):
                    if allowed_m_inds:
                        chosen_i = int(self.rng.choice(allowed_m_inds))
                    else:
                        if len(mlist) > 0:
                            chosen_i = max(0, min(len(mlist) - 1, (chosen_i or 0)))
                        else:
                            chosen_i = 0

                # Prefer per-machine availability (avail_row) for validation.
                per_machine = item.get('avail_row') if item.get('avail_row') is not None else None

                validated = False
                if per_machine is not None:
                    if 0 <= int(chosen_i) < len(per_machine) and int(bool(per_machine[int(chosen_i)])):
                        validated = True
                    else:
                        # try to find a permitted machine that is available
                        found = None
                        for cand in (allowed_m_inds or list(range(len(mlist)))):
                            if 0 <= int(cand) < len(per_machine) and int(bool(per_machine[int(cand)])):
                                found = int(cand)
                                break
                        if found is not None:
                            chosen_i = found
                            validated = True
                        elif allowed_m_inds:
                            chosen_i = int(allowed_m_inds[0])
                            validated = False

                if not validated:
                    # [C1] Mask recomputation is best-effort (Rule 3)
                    if build_machine_major_mask is not None:
                        jid = item.get('job_id')
                        try:
                            new_row = build_machine_major_mask(self.env, jid)
                            if new_row is not None:
                                if 0 <= int(chosen_i) < len(new_row) and int(bool(new_row[int(chosen_i)])):
                                    validated = True
                                else:
                                    found = None
                                    for cand in (allowed_m_inds or list(range(len(mlist)))):
                                        if 0 <= int(cand) < len(new_row) and int(bool(new_row[int(cand)])):
                                            found = int(cand); break
                                    if found is not None:
                                        chosen_i = found; validated = True
                                    elif allowed_m_inds:
                                        chosen_i = int(allowed_m_inds[0])
                        except Exception as e:
                            logging.getLogger(__name__).warning(f"[C1] Mask recomputation failed for job {jid}: {e}")
                    # final fallback: choose first allowed or clamp
                    if not validated:
                        if allowed_m_inds:
                            chosen_i = int(allowed_m_inds[0])

                if not (0 <= chosen_i < len(mlist)) and len(mlist) > 0:
                    chosen_i = max(0, min(len(mlist) - 1, chosen_i if chosen_i is not None else 0))

                chosen = int(chosen_i)
                chosen_machine_name = mlist[chosen]
            else:
                chosen = int(act) if act is not None else None
            # [C1] Machine action conversion complete - fail-fast design            # [C1] Human-readable machine name lookup - best-effort logging (Rule 3)
            try:
                if chosen_machine_name is None and bool(getattr(args, 'use_machine_actions', False)):
                    mlist = getattr(self.env.workcenters_meta, 'machine_list', []) or []
                    if 0 <= chosen < len(mlist):
                        chosen_machine_name = mlist[chosen]
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Failed to lookup machine name for chosen={chosen}: {e}")

            # [C1] Action selection logging - best-effort (Rule 3)
            try:
                if bool(getattr(args, 'use_machine_actions', False)) and chosen_machine_name is not None:
                    wc_for_m = int(self.env.workcenters_meta.machine_registry.get(chosen_machine_name, {}).get('workcenter', -1))
                    msg = f"[JobAgent {item.get('job_id')}] selected action={act} → Machine={chosen_machine_name} (WC{wc_for_m})"
                else:
                    msg = f"[JobAgent {item.get('job_id')}] selected action={chosen}"
                print(msg)
                logging.getLogger(__name__).info(msg)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Failed to log action selection for job {item.get('job_id')}: {e}")

            # append scheduling trace line (best-effort) — delegate to utils.gantt
            try:
                history_dir = getattr(self, 'history_dir', None) or getattr(self.env, 'history_dir', None) or './my_data_and_graph/historydata/'
                os.makedirs(history_dir, exist_ok=True)
                sched_path = os.path.join(history_dir, 'scheduling_trace.csv')
                job_id = item.get('job_id')
                allowed_machine_indices = item.get('allowed_machine_indices', [])
                # use canonical machine-major availability
                avail_actions = item.get('avail_row')
                # [C1] chosen_idx conversion is CRITICAL - fail-fast if invalid
                chosen_idx = int(chosen) if chosen is not None else None
                chosen_name = chosen_machine_name

                # [C1] Availability check for reason logging - best-effort (Rule 3)
                reason = ''
                try:
                    if avail_actions is not None and chosen_idx is not None:
                        arr = np.asarray(avail_actions)
                        if arr.size > int(chosen_idx):
                            if int(arr[int(chosen_idx)]) == 0:
                                reason = 'machine_not_available'
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Failed availability check for chosen_idx={chosen_idx}: {e}")

                # [C1] Scheduling trace file writes - best-effort I/O (Rule 3)
                if not allow_history_writes():
                    pass  # history writes disabled; skip logging
                else:
                    if gantt_utils is not None:
                        try:
                            gantt_utils.append_selection_log(sched_path, sim_time, job_id, allowed_machine_indices, avail_actions, chosen_idx, chosen_name, reason)
                        except Exception as e:
                            logging.getLogger(__name__).warning(f"[C1] Failed to append selection log for job {job_id}: {e}")
                    else:
                        try:
                            header_needed = not os.path.exists(sched_path)
                            with open(sched_path, 'a') as sf:
                                if header_needed:
                                    sf.write('time,job_id,allowed_machine_indices,avail_actions,chosen_machine_idx,chosen_machine_name,reason\n')
                                sf.write(f"{sim_time},{job_id},{allowed_machine_indices},{avail_actions},{chosen_idx},{repr(chosen_name)},{reason}\n")
                        except Exception as e:
                            logging.getLogger(__name__).warning(f"[C1] Failed to write scheduling trace for job {job_id}: {e}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Failed scheduling trace block for job {item.get('job_id')}: {e}")

            processed_actions.append(chosen)
            processed_machine_names.append(chosen_machine_name)

            # [C1] Resume decision event/callback - best-effort SimPy integration (Rule 3)
            # [PHASE5-FIX] Task 5.3: Check if event already triggered to prevent EventProcessed error
            try:
                resume_evt = item.get('resume_evt')
                if resume_evt is not None:
                    # [PHASE5-FIX] Check if event already triggered before calling succeed()
                    if not resume_evt.triggered:
                        try:
                            resume_evt.succeed(chosen)
                        except Exception as e:
                            try:
                                resume_evt.succeed(int(chosen))
                            except Exception as e:
                                try:
                                    resume_evt.succeed(chosen_machine_name)
                                except Exception as e:
                                    logging.getLogger(__name__).warning(f"[C1] Failed to resume decision event for job {item.get('job_id')}: {e}")
                    else:
                        logging.getLogger(__name__).warning(
                            f"[PHASE5] Resume event already triggered for job {item.get('job_id')}, skipping. "
                            f"This may indicate a race condition or duplicate event handling."
                        )
                else:
                    # backward compatibility: old resume callable
                    try:
                        item.get('resume')(chosen)
                    except Exception as e1:
                        try:
                            item.get('resume')(int(chosen))
                        except Exception as e2:
                            try:
                                item.get('resume')(chosen_machine_name)
                            except Exception as e3:
                                logging.getLogger(__name__).warning(
                                    f"[C1] Failed to resume decision with chosen={chosen}, "
                                    f"int(chosen)={int(chosen) if chosen is not None else None}, "
                                    f"machine_name={chosen_machine_name}: {e1}, {e2}, {e3}"
                                )
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Failed resume decision block for job {item.get('job_id')}: {e}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception when resuming decision (item processing): {e}")

        return processed_actions, processed_machine_names

    # -------------------------
    # Utility: ensure hidden
    # -------------------------
    def _ensure_hidden(self, n_agents: int):
        """Ensure RNN hidden state exists (best-effort).
        
        [PHASE2-FIX] Support multi-layer RNNs by creating shape (n_layers, n_agents, hdim)
        instead of just (n_agents, hdim). Single-layer networks still work since
        n_layers defaults to 1.
        """
        hdim = getattr(self.args, "rnn_hidden_dim", 64)
        n_layers = getattr(self.args, "rnn_num_layers", 1)  # [PHASE2-FIX] Support multi-layer
        if torch is None:
            self.eval_hidden = None
            return
        # [PHASE2-FIX] Create (n_layers, n_agents, hdim) for multi-layer RNN support
        self.eval_hidden = torch.zeros((n_layers, n_agents, hdim), dtype=torch.float32, device=self.device)

    def build_transitions_from_decision_batch(self, batch, processed_actions, processed_machine_names, r, s_before=None, s_after=None, avail_after=None, runner_args=None):
        """Build list-of-transition dicts from a Runner decision batch.

        This extracts and consolidates the logic used by Runner._run_event_driven_episode
        to produce transitions compatible with ReplayBuffer.store_episode().

        Parameters
        - batch: list of decision items
        - processed_actions: list of integer actions selected (one per item)
        - processed_machine_names: list of machine names (or None) for each action
        - r: scalar reward for this decision boundary
        - s_before/s_after: global state vectors (or None)
        - avail_after: availability matrix after decision (or None)
        - runner_args: optional args namespace (fallback to self.args)

        Returns: list of transition dicts (one per item in batch)
        """
        args = runner_args if runner_args is not None else self.args
        tr_list = []
        # replicate Runner behavior: build obs_batch and avail masks then create tr for each decision
        try:
            obs_batch = [item.get('obs') for item in batch]
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
            obs_batch = None

        # Determine granularity
        try:
            use_gran = bool(getattr(args, 'use_granular_actions', False))
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
            use_gran = False

        # Build avail_batch similar to Runner
        avail_batch = []
        if use_gran:
            # Compatibility: when operator-granular actions are requested we
            # deterministically expand the canonical per-machine 'avail_row'
            # into a flattened per-(machine×operator) vector by repeating each
            # machine slot `num_ops` times. This preserves deterministic
            # behavior while avoiding operator-selection during mask build.
            try:
                ops = int(getattr(self.env, 'num_ops', 1))
            except Exception as e:
                ops = 1
            try:
                num_m = int(len(getattr(self.env.workcenters_meta, 'machine_list', []) or []))
                if num_m == 0:
                    num_m = int(getattr(self.env, 'num_wcs', 0))
            except Exception as e:
                num_m = int(getattr(self.env, 'num_wcs', 0))

            for item in batch:
                try:
                    # prefer canonical per-machine row
                    row = item.get('avail_row')
                    if row is not None:
                        r = np.asarray(row, dtype=np.int32)
                        # ensure length matches known machines
                        if r.size < num_m:
                            # pad permissively
                            pad = np.ones((num_m - r.size,), dtype=np.int32)
                            r = np.concatenate([r, pad], axis=0)
                        expanded = np.repeat(r.astype(np.int32), ops)
                        avail_batch.append(expanded.tolist())
                        continue

                    # fallback: if avail_arr is present, expand that row
                    try:
                        if avail_arr is not None:
                            row = avail_arr[len(avail_batch)]
                            r = np.asarray(row, dtype=np.int32)
                            expanded = np.repeat(r.astype(np.int32), ops)
                            avail_batch.append(expanded.tolist())
                            continue
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Failed to expand avail row (using permissive fallback): {e}")

                    # final fallback: all-ones (permissive)
                    avail_batch.append([1] * (max(1, num_m) * max(1, ops)))
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Failed to build avail for item (using None): {e}")
                    avail_batch.append(None)
        else:
            # [C1] Extract avail_row from batch items - best-effort (Rule 3)
            try:
                avail_batch = [item.get('avail_row') for item in batch]
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Failed to extract avail_row from batch: {e}")
                avail_batch = None

        # Build u_list and u_machine mappings similar to Runner
        # [C1] Action list construction - best-effort (Rule 3)
        try:
            u_list = []
            u_machine_list = []
            for i, a in enumerate(processed_actions):
                try:
                    u_list.append(int(a) if a is not None else 0)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Failed to convert action {a} to int: {e}")
                    u_list.append(0)
                try:
                    uname = processed_machine_names[i] if i < len(processed_machine_names) else None
                    if uname is not None:
                        u_machine_list.append(int(getattr(self.env.workcenters_meta, 'machine_index', {}).get(uname, -1)))
                    else:
                        u_machine_list.append(-1)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Failed to map machine name {uname if 'uname' in locals() else 'N/A'}: {e}")
                    u_machine_list.append(-1)
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Failed to build u_list (using fallback): {e}")
            u_list = [int(a) if a is not None else 0 for a in (processed_actions or [])]
            u_machine_list = [-1] * len(u_list)

        # pad/truncate to args.n_agents - must be explicitly defined, no fallback
        if not hasattr(args, 'n_agents'):
            raise ValueError(
                "[FIXED_AGENT_BATCH] args.n_agents is required but missing. "
                "This must be set explicitly in configuration."
            )
        n_agents = int(args.n_agents)
        if len(u_list) < n_agents:
            u_list = u_list + [0] * (n_agents - len(u_list))
            u_machine_list = u_machine_list + [-1] * (n_agents - len(u_machine_list))
        elif len(u_list) > n_agents:
            u_list = u_list[:n_agents]
            u_machine_list = u_machine_list[:n_agents]

        # Prepare obs array
        obs_dim = getattr(args, 'obs_shape', None)
        if obs_batch is None:
            o_arr = np.zeros((n_agents, obs_dim if obs_dim is not None else 1), dtype=np.float32)
        else:
            try:
                o_tmp = np.asarray(obs_batch, dtype=np.float32)
                if o_tmp.ndim == 1:
                    o_tmp = o_tmp.reshape(1, -1)
                if obs_dim is None:
                    obs_dim = o_tmp.shape[1]
                if o_tmp.shape[0] < n_agents:
                    pad_rows = np.zeros((n_agents - o_tmp.shape[0], obs_dim), dtype=np.float32)
                    if o_tmp.shape[1] < obs_dim:
                        col_pad = np.zeros((o_tmp.shape[0], obs_dim - o_tmp.shape[1]), dtype=np.float32)
                        o_tmp = np.concatenate([o_tmp, col_pad], axis=1)
                    o_arr = np.concatenate([o_tmp, pad_rows], axis=0)
                else:
                    o_arr = o_tmp[:n_agents, :obs_dim]
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                o_arr = np.zeros((n_agents, obs_dim if obs_dim is not None else 1), dtype=np.float32)

        # Prepare avail array
        n_actions = getattr(args, 'n_actions', None)
        if avail_batch is None or n_actions is None:
            avail_arr = None
        else:
            try:
                a_tmp = np.asarray(avail_batch, dtype=np.float32)
                if a_tmp.ndim == 1:
                    a_tmp = a_tmp.reshape(1, -1)
                if a_tmp.shape[0] < n_agents:
                    pad_rows = np.zeros((n_agents - a_tmp.shape[0], n_actions), dtype=np.float32)
                    if a_tmp.shape[1] < n_actions:
                        col_pad = np.zeros((a_tmp.shape[0], n_actions - a_tmp.shape[1]), dtype=np.float32)
                        a_tmp = np.concatenate([a_tmp, col_pad], axis=1)
                    avail_arr = np.concatenate([a_tmp, pad_rows], axis=0)
                else:
                    avail_arr = a_tmp[:n_agents, :n_actions]
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                avail_arr = None

        # Build per-item transition dicts
        for i_item, item in enumerate(batch):
            tr = {}
            tr['o'] = o_arr
            tr['u'] = u_list
            tr['u_machine'] = [(-1 if x is None else int(x)) for x in u_machine_list]
            try:
                tr['u_machine_name'] = [(None if x is None else str(x)) for x in processed_machine_names]
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                tr['u_machine_name'] = [None] * len(u_list)
            tr['r'] = r
            # attach avail_a / avail_a_next
            try:
                if bool(getattr(args, 'use_granular_actions', False)):
                    try:
                        ops = int(self.env.num_ops)
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                        ops = None
                    n_agents_local = len(batch)
                    if ops is not None and hasattr(self.env, 'num_wcs'):
                        n_actions_local = int(self.env.num_wcs) * ops
                    else:
                        n_actions_local = None
                    avail_flat = []
                    for j, it in enumerate(batch):
                            # prefer canonical per-machine row if present
                            try:
                                row = it.get('avail_row')
                                if row is not None:
                                    r = np.asarray(row, dtype=np.float32)
                                    if ops is not None:
                                        expanded = np.repeat(r.astype(np.float32), ops)
                                        if n_actions_local is not None:
                                            expanded = expanded[:n_actions_local]
                                        avail_flat.append(expanded)
                                        continue
                                    else:
                                        avail_flat.append(r)
                                        continue
                            except Exception as e:
                                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")

                            try:
                                # fallback: avail_arr (post-decision availability) if present
                                if avail_arr is not None:
                                    row = np.asarray(avail_arr[j], dtype=np.float32)
                                    if ops is not None:
                                        expanded = np.repeat(row.astype(np.float32), ops)
                                        if n_actions_local is not None:
                                            expanded = expanded[:n_actions_local]
                                        avail_flat.append(expanded)
                                        continue
                                    else:
                                        avail_flat.append(row)
                                        continue
                            except Exception as e:
                                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")

                            # final fallback: zeros
                            if n_actions_local is not None:
                                avail_flat.append(np.zeros((n_actions_local,), dtype=np.float32))
                            else:
                                avail_flat.append(np.zeros((len(u_list),), dtype=np.float32))

                    tr['avail_a'] = np.asarray(avail_flat, dtype=np.float32)
                else:
                    if avail_arr is not None:
                        tr['avail_a'] = avail_arr
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                pass

            # avail_a_next
            try:
                if avail_after is not None and avail_arr is not None:
                    n_agents_local = avail_arr.shape[0]
                    n_actions_local = avail_arr.shape[1]
                    avail_next_arr = np.zeros((n_agents_local, n_actions_local), dtype=np.float32)
                    for idx_item, it in enumerate(batch):
                        job_id = it.get('job_id')
                        if job_id is None:
                            continue
                        try:
                            row = avail_after[int(job_id)]
                            row = np.asarray(row, dtype=np.float32)
                            if row.ndim == 1 and hasattr(self.env, 'num_ops'):
                                ops = int(self.env.num_ops)
                                expanded = np.repeat(row.astype(np.float32), ops)
                                avail_next_arr[idx_item, :] = expanded[:n_actions_local]
                            elif row.ndim == 1:
                                avail_next_arr[idx_item, :] = row[:n_actions_local]
                        except Exception as e:
                            logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                            pass
                    tr['avail_a_next'] = avail_next_arr
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                pass

            if s_before is not None:
                tr['s'] = s_before
            if s_after is not None:
                tr['s_next'] = s_after
            tr['done'] = getattr(self.env, 'done', False)

            tr_list.append(tr)
        # -------------------------
        # [PHASE5-FIX] Task 5.5: Shape validation always enabled (not just in debug mode)
        # Critical shape mismatches should always raise errors, not just warnings
        try:
            # Co-Pilot Rule compliance: No fallback defaults for n_agents
            if not hasattr(args, 'n_agents'):
                raise ValueError(
                    "[FIXED_AGENT_BATCH] args.n_agents is required but missing. "
                    "This must be set explicitly in configuration."
                )
            n_agents_expected = int(args.n_agents)
            obs_dim_expected = getattr(args, 'obs_shape', None) or getattr(args, 'obs_dim_agent', None) or None
            if obs_dim_expected is not None:
                obs_dim_expected = int(obs_dim_expected)

            for idx_tr, tr in enumerate(tr_list):
                # [PHASE5-FIX] Check observation shape - ALWAYS validate, not optional
                if 'o' in tr and n_agents_expected is not None and obs_dim_expected is not None:
                    o_arr = np.asarray(tr['o'], dtype=np.float32)
                    if o_arr.ndim == 2:
                        if o_arr.shape[0] != n_agents_expected:
                            raise ValueError(
                                f"[PHASE5] Transition[{idx_tr}]['o'] shape mismatch: "
                                f"got {o_arr.shape[0]} agents, expected {n_agents_expected}. "
                                f"Check environment decision batch consistency."
                            )
                        if o_arr.shape[1] != obs_dim_expected:
                            raise ValueError(
                                f"[PHASE5] Transition[{idx_tr}]['o'] obs_dim mismatch: "
                                f"got {o_arr.shape[1]}, expected {obs_dim_expected}. "
                                f"Check observation builder."
                            )

                # [PHASE5-FIX] Check action vector length - ALWAYS validate
                if 'u' in tr and n_agents_expected is not None:
                    u_val = tr['u']
                    if isinstance(u_val, (list, tuple, np.ndarray)):
                        if len(u_val) != n_agents_expected:
                            raise ValueError(
                                f"[PHASE5] Transition[{idx_tr}]['u'] length mismatch: "
                                f"got {len(u_val)}, expected n_agents={n_agents_expected}"
                            )
        except ValueError:
            # [PHASE5-FIX] Re-raise ValueError (validation errors should propagate)
            raise
        except Exception as e:
            # Only catch unexpected errors, log and continue
            logging.getLogger(__name__).warning(f"[C1] Unexpected error in shape validation: {e}")
            logging.getLogger(__name__).debug("Shape validation encountered unexpected error", exc_info=True)

        return tr_list

    def run_event_driven_episode(self, global_ep_idx, evaluate=False, runner_args=None):
        """
        Drive a SimPy event-driven episode using the worker's env and helpers.

        Returns the same tuple as Runner._run_event_driven_episode:
          (episode_dict, ep_reward, win_tag, gantt_list)

        Parameters:
        - global_ep_idx: index of the global episode (for bookkeeping/logging)
        - evaluate: whether this is an evaluation episode
        - runner_args: optional args namespace to override self.args
        """
        args = runner_args if runner_args is not None else self.args

        # start/reset environment (env.reset returns initial obs/info)
        # Lifecycle ordering:
        # 1) Close any previous lifecycle block (END for prior episode)
        # 2) Assign the new numeric episode id to env.episode_id
        # 3) Write START lifecycle header for the new episode
        # 4) Call env.reset() to perform environment reset
        try:
            # Close previous lifecycle block if present and episode_id known
            # [C1] Handle lifecycle traces - best-effort logging (Rule 3)
            # Only close a previous lifecycle block when the env reports a
            # different episode id than the one we're about to run. This
            # avoids writing an END for episode 0 at startup when the env's
            # default episode_id is 0 (causing duplicate ENDs).
            try:
                if hasattr(self.env, "end_lifecycle_trace") and hasattr(self.env, 'episode_id'):
                    prev_ep = getattr(self.env, 'episode_id')
                    try:
                        prev_ep_int = int(prev_ep) if prev_ep is not None else None
                        new_ep_int = int(global_ep_idx)
                        # Only end previous if it's a different episode value
                        if prev_ep_int is not None and prev_ep_int != new_ep_int:
                            self.env.end_lifecycle_trace(prev_ep_int)
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Failed to end lifecycle trace for prev episode {prev_ep}: {e}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Failed lifecycle trace check: {e}")

            # Assign new episode id (numeric preferred)
            try:
                self.env.episode_id = int(global_ep_idx)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Failed to assign episode_id={global_ep_idx}: {e}")

            # [C1] Write lifecycle START for the new episode BEFORE reset() - best-effort (Rule 3)
            try:
                if hasattr(self.env, "start_lifecycle_trace"):
                    ep_to_start = int(global_ep_idx)
                    self.env.start_lifecycle_trace(ep_to_start)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Failed to start lifecycle trace for episode {global_ep_idx}: {e}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Failed lifecycle/episode setup: {e}")
        
        # [C1] Ensure the environment has a history_dir for timeline writes - best-effort (Rule 3)
        try:
            if not hasattr(self.env, 'history_dir') or getattr(self.env, 'history_dir', None) is None:
                import os
                default_hist = os.path.join(os.getcwd(), 'my_data_and_graph', 'historydata')
                self.env.history_dir = default_hist
                os.makedirs(self.env.history_dir, exist_ok=True)
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Failed to setup history_dir: {e}")
            pass
        try:
            obs_init, info = self.env.reset()
        except TypeError:
            # some reset signatures may return only obs
            obs_init = self.env.reset()
            info = {}

        # Stamp the env with the current episode index so environment-level
        # appenders can record which episode a gantt record belongs to.
        try:
            # [C1] Episode ID assignment - best-effort metadata (Rule 3)
            try:
                self.env.current_episode = int(global_ep_idx)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Failed to set current_episode as int, using raw value: {e}")
                setattr(self.env, 'current_episode', global_ep_idx)
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Failed to assign episode metadata: {e}")

        episode = {"r": []}
        gantt = []
        # collect per-decision transitions for replay
        ep_transitions = []

        # run until environment signals done
        while True:
            batch, sim_time = self.env.wait_for_decisions()
            # empty batch may mean done
            if not batch:
                break

            # capture global state before decision (if available)
            try:
                s_before = np.asarray(self.env._build_state_vector(), dtype=np.float32)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                s_before = None

            # get actions for the batch via the worker's agent wrapper
            obs_list = [item.get('obs') for item in batch]
            try:
                avail = [item.get('avail_row') for item in batch]
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                avail = None
            
            # Extract agent masks from batch items
            agent_masks = [item.get('agent_mask', 1) for item in batch]
            
            # Validate batch size matches n_agents (after padding)
            if not hasattr(self.args, 'n_agents'):
                raise ValueError(
                    "[FIXED_AGENT_BATCH] args.n_agents is required but missing. "
                    "This must be set explicitly in configuration."
                )
            n_agents = int(self.args.n_agents)
            if len(obs_list) != n_agents:
                raise ValueError(
                    f"[FIXED_AGENT_BATCH] Batch size mismatch: got {len(obs_list)}, "
                    f"expected {n_agents}. Environment padding may have failed."
                )
            
            try:
                # _select_actions returns (actions_list, hidden_state).
                # Unpack the pair so callers receive the actions list.
                # Pass current epsilon and agent masks for exploration
                actions, _ = self._select_actions(obs_list, avail, evaluate=evaluate, epsilon=self.epsilon, agent_masks=agent_masks)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                actions = []

            # [TIME-BASED EPSILON DECAY] Update epsilon based on simulation time
            if not evaluate:
                current_t = float(self.env.env.now)
                limit_t = float(self.episode_limit)
                fraction = min(1.0, current_t / limit_t)
                self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)

            # Filter actions for real agents only - apply only to non-padded agents
            real_batch = [item for item in batch if item.get('agent_mask', 1) == 1]
            real_actions = [actions[i] for i in range(len(batch)) if batch[i].get('agent_mask', 1) == 1]
            
            # Process and apply actions via RolloutWorker helper (centralized logic)
            try:
                processed_actions, processed_machine_names = self.process_and_apply_actions(real_batch, real_actions, sim_time, runner_args=args)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                processed_actions = actions or []
                processed_machine_names = [None] * len(processed_actions)

            actions = processed_actions

            # [PHASE1-FIX] Epsilon decay moved to per-episode level in runner.py
            # No longer decay per decision step - this caused circular dependency
            # where epsilon schedule depended on number of decisions made
            
            # --- Epsilon diagnostics (periodic, non-fatal) ---
            try:
                try:
                    self.step_counter += 1
                except Exception as e:
                    self.step_counter = getattr(self, 'step_counter', 0) + 1

                # [C1] Epsilon decay logging - best-effort diagnostics (Rule 3)
                if getattr(self, 'epsilon_log_every', 0) > 0 and (self.step_counter % int(self.epsilon_log_every) == 0):
                    try:
                        msg = f"[EPSILON_DECAY] step={self.step_counter}, epsilon={self.epsilon:.4f}"
                    except Exception as e:
                        try:
                            msg = f"[EPSILON_DECAY] step={self.step_counter}, epsilon={float(self.epsilon)}"
                        except Exception as e2:
                            msg = f"[EPSILON_DECAY] step={self.step_counter}, epsilon={getattr(self, 'epsilon', 'NA')}"
                    try:
                        print(msg)
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Failed to print epsilon decay message: {e}")
                    try:
                        import os as os_module
                        history_dir = getattr(self, 'history_dir', None) or getattr(self.env, 'history_dir', None) or './my_data_and_graph/historydata/'
                        os_module.makedirs(history_dir, exist_ok=True)
                        diag_path = os_module.path.join(history_dir, 'diagnostics_log.txt')
                        with open(diag_path, 'a', encoding='utf-8') as df:
                            df.write(msg + '\n')
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Failed to write epsilon decay to diagnostics log: {e}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in epsilon decay diagnostics: {e}")

            # after resuming processes, collect reward accumulated since last decision boundary
            try:
                r = float(self.env.pop_decision_reward())
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                r = 0.0
            episode["r"].append(r)

            # capture global state after decision (if available)
            try:
                s_after = np.asarray(self.env._build_state_vector(), dtype=np.float32)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                s_after = None

            # capture availabilities after decision (for avail_a_next)
            try:
                avail_after = None
                if hasattr(self.env, '_build_avail_actions'):
                    avail_after = self.env._build_avail_actions()
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                avail_after = None

            # --- build a replay transition for this decision boundary ---
            try:
                tr_list = self.build_transitions_from_decision_batch(
                    batch,
                    actions,
                    processed_machine_names,
                    r,
                    s_before=s_before,
                    s_after=s_after,
                    avail_after=avail_after,
                    runner_args=args,
                )
                for tr in tr_list:
                    ep_transitions.append(tr)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                pass

            # collect possible lightweight gantt info if present on env or items
            try:
                gantt.extend(self.collect_gantt_from_batch(batch, sim_time))
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                pass

            # stop if env signals done
            if getattr(self.env, "done", False):
                break

        ep_reward = float(np.sum(episode.get("r", [])))
        try:
            win_tag = all(j.finished for j in getattr(self.env, "jobs", []))
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
            win_tag = False

        # Optional per-episode reward components logging (append-only)
        try:
            try:
                do_log = bool(getattr(args, 'reward_log_components', False))
            except Exception as e:
                do_log = False

            if do_log and allow_history_writes():
                try:
                    hist_dir = getattr(self, 'history_dir', None) or getattr(self.env, 'history_dir', None) or './my_data_and_graph/historydata/'
                    os.makedirs(hist_dir, exist_ok=True)
                    out_path = os.path.join(hist_dir, 'reward_components_log.txt')

                    comps = getattr(self.env, 'last_reward_components', {}) or {}
                    # [C1] Reward component parsing - best-effort data extraction (Rule 3)
                    def _getc(keys):
                        for k in keys:
                            if k in comps:
                                try:
                                    return float(comps.get(k))
                                except Exception as e:
                                    logging.getLogger(__name__).warning(f"[C1] Failed to convert reward component {k} to float: {e}")
                                    return comps.get(k)
                        return ''

                    r_global = _getc(['R_global', 'r_global', 'Rglobal'])
                    r_local = _getc(['R_local_mean', 'r_local_mean', 'Rlocal_mean', 'R_local'])
                    r_total = _getc(['R_total', 'r_total', 'Rtotal'])

                    header_needed = not os.path.exists(out_path)
                    with open(out_path, 'a', encoding='utf-8') as rf:
                        if header_needed:
                            rf.write('episode,R_global,R_local_mean,R_total\n')
                        try:
                            ep_write = int(getattr(self.env, 'episode_id', global_ep_idx))
                        except Exception as e:
                            ep_write = global_ep_idx
                        rf.write(f"{ep_write},{r_global},{r_local},{r_total}\n")
                except Exception as e:
                    logging.getLogger(__name__).exception("Failed to write reward_components_log", exc_info=True)
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in rollout: {e}")

        # store episode into replay buffer if available
        try:
            if self.buffer is not None and len(ep_transitions) > 0:
                try:
                    self.buffer.store_episode(ep_transitions)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
                    pass
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
            pass

        # include environment-level gantt records if present
        try:
            if hasattr(self.env, "gantt_records") and isinstance(self.env.gantt_records, (list, tuple)):
                gantt.extend(list(self.env.gantt_records))
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in rollout batch processing: {e}")
            pass

        # Clear the temporary episode stamp so other callers are not confused
        try:
            if hasattr(self.env, 'current_episode'):
                try:
                    delattr(self.env, 'current_episode')
                except Exception as e:
                    try:
                        del self.env.current_episode
                    except Exception as e:
                        setattr(self.env, 'current_episode', None)
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in rollout: {e}")

        # Ensure we close the lifecycle trace for this episode after it finishes
        if hasattr(self.env, 'end_lifecycle_trace'):
            ended_ep = getattr(self.env, 'episode_id', None)

            # Prepare cleaned gantt records and metadata before ending the
            # lifecycle block. We deliberately do NOT call the generator yet;
            # instead we end the lifecycle trace first so the lifecycle START/END
            # pair is fully written and then invoke generation so TIMELINE and
            # SUMMARY appear after the lifecycle block (as requested).
            cleaned = list(gantt or [])
            ep_int = ended_ep
            history_dir = getattr(self, 'history_dir', None) or getattr(self.env, 'history_dir', None) or './my_data_and_graph/historydata/'
            try:
                import os
                os.makedirs(history_dir, exist_ok=True)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in rollout: {e}")

            if ended_ep is not None:
                try:
                    try:
                        ep_int = int(ended_ep)
                    except Exception as e:
                        ep_int = ended_ep

                    # Unwrap wrapper-shaped gantt records when present so the
                    # generator receives the canonical per-op dicts it expects.
                    try:
                        tmp = []
                        for rec in (gantt or []):
                            try:
                                if isinstance(rec, dict) and 'record' in rec and isinstance(rec.get('record'), dict):
                                    inner = dict(rec.get('record') or {})
                                    try:
                                        if inner.get('episode') is None and rec.get('episode') is not None:
                                            inner['episode'] = rec.get('episode')
                                    except Exception as e:
                                        logging.getLogger(__name__).warning(f"[C1] Exception in rollout: {e}")
                                    tmp.append(inner)
                                else:
                                    tmp.append(rec)
                            except Exception as e:
                                tmp.append(rec)
                        cleaned = tmp
                    except Exception as e:
                        cleaned = list(gantt or [])
                except Exception as e:
                    cleaned = list(gantt or [])

            # end lifecycle (best-effort) — write the END footer first so the
            # lifecycle START/END pair is closed before we append TIMELINE and
            # SUMMARY. This ensures the generator runs after the episode footer
            # has been written and can still force the presence of the EPISODE
            # header immediately before the TIMELINE/SUMMARY blocks.
            try:
                self.env.end_lifecycle_trace(ended_ep)
            except Exception as e:
                # swallow to avoid breaking runner flow
                pass

            # Invoke the generator so TIMELINE and SUMMARY are written after
            # the lifecycle END footer. All I/O remains best-effort.
            if ended_ep is not None:
                try:
                    from utils.gantt import generate_scheduling_timeline
                except Exception as e:
                    generate_scheduling_timeline = None
                if generate_scheduling_timeline is not None:
                    try:
                        # Lightweight console diagnostics
                        try:
                            print(f"[DEBUG] generate_scheduling_timeline called with len(gantt)={len(gantt)} cleaned_len={len(cleaned)}")
                        except Exception as e:
                            logging.getLogger(__name__).warning(f"[C1] Exception in rollout: {e}")
                        try:
                            print(f"[DEBUG] allow_history_writes()={allow_history_writes()}")
                        except Exception as e:
                            logging.getLogger(__name__).warning(f"[C1] Exception in rollout: {e}")

                        try:
                            generate_scheduling_timeline(
                                self.env,
                                episode_id=ep_int,
                                episode_reward=ep_reward if 'ep_reward' in locals() else None,
                                write_if_allowed=True,
                                skip_header=True,
                                out_dir=history_dir,
                                records=cleaned,
                            )
                        except Exception as e:
                            # non-fatal; timeline generation best-effort
                            pass
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Exception in rollout: {e}")

        return episode, ep_reward, bool(win_tag), gantt


# Backwards-compatibility alias: older code imported CommRolloutWorker
# Ensure such imports continue to work until callers are updated.
CommRolloutWorker = RolloutWorker