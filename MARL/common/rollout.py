import numpy as np
import time
import os
import logging
from typing import Any, Dict, List, Optional, Tuple

# try to import torch for RNN hidden handling; degrade gracefully if not available
try:
    import torch
except Exception as e:
    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
    torch = None

# mask utilities
try:
    from MARL.common.mask_utils import build_index_map, build_mask_for_job
except Exception as e:
    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
    build_index_map = None
    build_mask_for_job = None
# gantt helpers (selection logging) and IO control
try:
    from utils import gantt as gantt_utils
except Exception as e:
    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
    gantt_utils = None
try:
    # io_control provides central allow flag
    from utils.io_control import allow_history_writes
except Exception:
    def allow_history_writes():
        return False


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
        epsilon_anneal_steps: Optional[int] = None,
        device: Optional[str] = None,
        log_prefix: str = "8A.7.3",
    ):
        self.env = env
        self.agents = agents
        self.buffer = buffer
        self.args = args or type("A", (), {})()
        # Use episode_limit coming from runner-provided args when available.
        # Fall back to the explicit constructor value or 300 as a safe default.
        try:
            if hasattr(self.args, "episode_limit") and getattr(self.args, "episode_limit") is not None:
                self.episode_limit = int(getattr(self.args, "episode_limit"))
            elif episode_limit is not None:
                self.episode_limit = int(episode_limit)
            else:
                # extend default rollout episode_limit to match environment
                # default (300) so short runs include initial arrivals.
                self.episode_limit = int(getattr(self.args, 'episode_limit', 300))
        except Exception as e:
            logging.getLogger(__name__).exception("RolloutWorker init: failed to determine episode_limit", exc_info=True)
            self.episode_limit = 300

        if hasattr(self.args, "device") and getattr(self.args, "device") is not None:
            self.device = getattr(self.args, "device")
        elif device is not None:
            self.device = device
        else:
            # fallback to a sensible default
            try:
                self.device = getattr(self.args, 'device', 'cpu')
            except Exception:
                self.device = 'cpu'
        self.log_prefix = log_prefix

        # epsilon schedule: prefer centralized args values, then constructor
        # params, then the global defaults from arguments.py.
        try:
            if hasattr(self.args, 'epsilon_start') and getattr(self.args, 'epsilon_start') is not None:
                self.epsilon_start = float(getattr(self.args, 'epsilon_start'))
            elif epsilon_start is not None:
                self.epsilon_start = float(epsilon_start)
            else:
                self.epsilon_start = float(getattr(self.args, 'epsilon_start', 1.0))
        except Exception:
            self.epsilon_start = 1.0

        try:
            if hasattr(self.args, 'epsilon_end') and getattr(self.args, 'epsilon_end') is not None:
                self.epsilon_end = float(getattr(self.args, 'epsilon_end'))
            elif epsilon_end is not None:
                self.epsilon_end = float(epsilon_end)
            else:
                self.epsilon_end = float(getattr(self.args, 'epsilon_end', 0.05))
        except Exception:
            self.epsilon_end = 0.05

        try:
            if hasattr(self.args, 'epsilon_anneal_steps') and getattr(self.args, 'epsilon_anneal_steps') is not None:
                self.epsilon_anneal_steps = int(getattr(self.args, 'epsilon_anneal_steps'))
            elif epsilon_anneal_steps is not None:
                self.epsilon_anneal_steps = int(epsilon_anneal_steps)
            else:
                self.epsilon_anneal_steps = int(getattr(self.args, 'epsilon_anneal_steps', 50000))
        except Exception:
            self.epsilon_anneal_steps = 50000

        self.epsilon = float(self.epsilon_start)
        self._eps_decay = (self.epsilon_start - self.epsilon_end) / max(1, self.epsilon_anneal_steps)

        # runtime placeholders
        self.eval_hidden = None
        self.episode_duration = 0

        # RNG for Runner fallback
        seed = getattr(self.args, "seed", None)
        try:
            self.rng = np.random.RandomState(seed if seed is not None else 0)
        except Exception as e:
            logging.getLogger(__name__).exception("RolloutWorker init: RNG init failed")
            self.rng = np.random.RandomState(0)

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
    def _select_actions(self, obs_batch: List[Any], avail_batch: Optional[List[Any]], evaluate: bool = False) -> Tuple[List[Any], Any]:
        """
        Return (actions_list, hidden_state) where actions_list is a list of int actions
        This wrapper tries common agent APIs then falls back to deterministic/random picks.
        """
        # Try batch API on agents
        try:
            if hasattr(self.agents, "select_actions"):
                return self.agents.select_actions(obs_batch, avail_batch, evaluate=evaluate), None
            if hasattr(self.agents, "choose_actions"):
                return self.agents.choose_actions(obs_batch, avail_batch, evaluate=evaluate), None
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass

        # Try per-observation act method
        try:
            if hasattr(self.agents, "act"):
                actions = []
                for ob in obs_batch:
                    a = self.agents.act(ob, None, evaluate=evaluate)
                    actions.append(a)
                return actions, None
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass

        # Fallback: pick first allowed or random
        actions = []
        for ob in obs_batch:
            # try to extract allowed_wcs from ob if present
            allowed = None
            if isinstance(ob, dict):
                allowed = ob.get("allowed_wcs", None) or ob.get("avail_row", None)
            if allowed is None:
                # no info — pick 0
                actions.append(0)
            else:
                try:
                    allowed_list = list(allowed)
                    if len(allowed_list) == 0:
                        actions.append(None)
                    else:
                        if evaluate:
                            actions.append(int(allowed_list[0]))
                        else:
                            actions.append(int(self.rng.choice(allowed_list)))
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    actions.append(0)
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
        try:
            avail = [item.get("avail_row") for item in batch]
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            avail = None

        try:
            # Best-effort: compute detailed avail masks and attach to batch entries
            if build_index_map is not None and hasattr(self.env, 'num_ops'):
                # determine if we're using machine-level actions (global machine list)
                machine_list = getattr(self.env.workcenters_meta, 'machine_list', None)
                if bool(getattr(self.args, 'use_machine_actions', False)) and machine_list is not None:
                    num_m = int(len(machine_list))
                    # op_to_m maps operator -> list of machine indices
                    op_to_m = {p: [] for p in range(int(self.env.num_ops))}
                    try:
                        # build by scanning machine_registry capabilities
                        mreg = getattr(self.env.workcenters_meta, 'machine_registry', {})
                        mindex = getattr(self.env.workcenters_meta, 'machine_index', {})
                        for mname, mdata in mreg.items():
                            caps = list(mdata.get('capabilities', []))
                            for p in caps:
                                op_to_m.setdefault(int(p), []).append(int(mindex.get(mname, 0)))
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        op_to_m = {p: [] for p in range(int(self.env.num_ops))}
                    # machine resource free state: prefer per-machine resources
                    machine_free = []
                    try:
                        if getattr(self.env, 'machine_resources', None):
                            mindex = getattr(self.env.workcenters_meta, 'machine_index', {})
                            for mname in machine_list:
                                try:
                                    mi = int(mindex.get(mname, 0))
                                    machine_free.append(self.env._resource_free(self.env.machine_resources[mi]))
                                except Exception as e:
                                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                    machine_free.append(True)
                        else:
                            mreg = getattr(self.env.workcenters_meta, 'machine_registry', {})
                            for mname in machine_list:
                                try:
                                    wc_i = int(mreg.get(mname, {}).get('workcenter', 0))
                                    machine_free.append(self.env._resource_free(self.env.wc_resources[wc_i]))
                                except Exception as e:
                                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                    machine_free.append(True)
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        machine_free = [True] * num_m
                    try:
                        operator_free = [self.env._resource_free(self.env.operator_groups[p]) for p in range(int(self.env.num_ops))]
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        operator_free = [True] * int(self.env.num_ops)
                    num_p = int(self.env.num_ops)
                    idx_map = build_index_map(num_m, num_p)
                else:
                    # fallback: use workcenter-level mapping as before
                    num_m = int(self.env.num_wcs)
                    num_p = int(self.env.num_ops)
                    idx_map = build_index_map(num_m, num_p)
                    # build operator->workcenter mapping
                    op_to_m = {p: [] for p in range(num_p)}
                    try:
                        groups_map = getattr(self.env.workcenters_meta, 'eligible_operator_groups_by_wc', {})
                        for wc_idx, groups in groups_map.items():
                            for g in groups:
                                if g in op_to_m:
                                    op_to_m[g].append(int(wc_idx))
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        # fallback to config mapping
                        if getattr(self.env, 'config', None):
                            ops_cfg = self.env.config.get('operators', [])
                            for p_idx, opconf in enumerate(ops_cfg):
                                q = opconf.get('qualified_machines', [])
                                mapped = []
                                for mname in q:
                                    for mid in getattr(self.env.workcenters_meta, 'machine_registry', {}).keys():
                                        if mname in mid or mname == mid:
                                            try:
                                                wc_i = int(self.env.workcenters_meta.machine_registry[mid]['workcenter'])
                                                mapped.append(wc_i)
                                            except Exception as e:
                                                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                                continue
                                op_to_m[p_idx] = mapped

                    try:
                        if getattr(self.env, 'machine_resources', None):
                            machine_free = [self.env._resource_free(self.env.machine_resources[m]) for m in range(num_m)]
                        else:
                            machine_free = [self.env._resource_free(self.env.wc_resources[m]) for m in range(num_m)]
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        machine_free = [True] * num_m
                    try:
                        operator_free = [self.env._resource_free(self.env.operator_groups[p]) for p in range(num_p)]
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        operator_free = [True] * num_p
            else:
                idx_map = None
                op_to_m = {}
                machine_free = []
                operator_free = []

            for item in batch:
                allowed = item.get('allowed_wcs', [])
                try:
                    if idx_map is not None:
                        mask = build_mask_for_job(idx_map, allowed, op_to_m, machine_free, operator_free)
                        # attach granular (machine×op) mask
                        item['avail_mask'] = mask.tolist()
                        # also attach per-machine availability (n_actions) by OR-ing operators
                        num_ops = int(self.env.num_ops) if hasattr(self.env, 'num_ops') else 0
                        # determine num_m: if machine_list present and using machine actions, use that
                        if getattr(self.args, 'use_machine_actions', False) and getattr(self.env.workcenters_meta, 'machine_list', None) is not None:
                            num_m = int(len(self.env.workcenters_meta.machine_list))
                        else:
                            num_m = int(getattr(self.env, 'num_wcs', 0))
                        per_machine = []
                        for m in range(num_m):
                            start = m * num_ops
                            end = start + num_ops
                            try:
                                per_machine.append(int(bool(mask[start:end].any())))
                            except Exception as e:
                                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                per_machine.append(0)
                        # prefer to set 'avail_row' so Runner will pick it up for storing into replay
                        item['avail_row'] = per_machine
                    else:
                        item['avail_mask'] = None
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    item['avail_mask'] = None
                    # leave avail_row untouched if we can't compute mask

            actions, _ = self._select_actions(obs_list, avail, evaluate=evaluate)
            return actions
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            # fallback deterministic/random pick
            outs = []
            for item in batch:
                allowed = item.get("allowed_wcs", [])
                if not allowed:
                    outs.append(None)
                else:
                    if evaluate:
                        outs.append(int(allowed[0]))
                    else:
                        try:
                            outs.append(int(self.rng.choice(allowed)))
                        except Exception as e:
                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                            outs.append(int(np.random.choice(allowed)))
            return outs

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
            try:
                if bool(getattr(args, 'use_machine_actions', False)):
                    mlist = getattr(self.env.workcenters_meta, 'machine_list', []) or []
                    ops = int(getattr(self.env, 'num_ops', 1))
                    allowed_m_inds = item.get('allowed_machine_indices') or []

                    try:
                        chosen_i = int(act) if act is not None else None
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        chosen_i = None

                    if chosen_i is None or chosen_i < 0 or chosen_i >= len(mlist):
                        if allowed_m_inds:
                            chosen_i = int(self.rng.choice(allowed_m_inds))
                        else:
                            if len(mlist) > 0:
                                chosen_i = max(0, min(len(mlist) - 1, (chosen_i or 0)))
                            else:
                                chosen_i = 0

                    # validate against granular avail_mask if present
                    mask = None
                    try:
                        mask = np.asarray(item.get('avail_mask')) if item.get('avail_mask') is not None else None
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        mask = None

                    if mask is not None:
                        start = chosen_i * ops
                        end = start + ops
                        if end <= mask.size and not bool(mask[start:end].any()):
                            found = None
                            for cand in (allowed_m_inds or list(range(len(mlist)))):
                                s = int(cand) * ops
                                if s + ops <= mask.size and bool(mask[s:s+ops].any()):
                                    found = int(cand)
                                    break
                            if found is not None:
                                chosen_i = found
                            elif allowed_m_inds:
                                chosen_i = int(allowed_m_inds[0])

                    if not (0 <= chosen_i < len(mlist)) and len(mlist) > 0:
                        chosen_i = max(0, min(len(mlist) - 1, chosen_i if chosen_i is not None else 0))

                    chosen = int(chosen_i)
                    try:
                        chosen_machine_name = mlist[chosen]
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        chosen_machine_name = None
                else:
                    chosen = int(act) if act is not None else None
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                try:
                    chosen = int(act) if act is not None else 0
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    chosen = 0

            # human-readable logging message (best-effort)
            try:
                if chosen_machine_name is None and bool(getattr(args, 'use_machine_actions', False)):
                    mlist = getattr(self.env.workcenters_meta, 'machine_list', []) or []
                    if 0 <= chosen < len(mlist):
                        chosen_machine_name = mlist[chosen]
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            try:
                if bool(getattr(args, 'use_machine_actions', False)) and chosen_machine_name is not None:
                    wc_for_m = int(self.env.workcenters_meta.machine_registry.get(chosen_machine_name, {}).get('workcenter', -1))
                    msg = f"[JobAgent {item.get('job_id')}] selected action={act} → Machine={chosen_machine_name} (WC{wc_for_m})"
                else:
                    msg = f"[JobAgent {item.get('job_id')}] selected action={chosen}"
                print(msg)
                logging.getLogger(__name__).info(msg)
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            # append scheduling trace line (best-effort) — delegate to utils.gantt
            try:
                history_dir = getattr(self, 'history_dir', None) or getattr(self.env, 'history_dir', None) or './my_data_and_graph/historydata/'
                os.makedirs(history_dir, exist_ok=True)
                sched_path = os.path.join(history_dir, 'scheduling_trace.csv')
                job_id = item.get('job_id')
                allowed_wcs = item.get('allowed_wcs') or item.get('allowed_machine_indices') or []
                avail_mask = item.get('avail_mask') if item.get('avail_mask') is not None else item.get('avail_row')
                try:
                    chosen_idx = int(chosen) if chosen is not None else None
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    chosen_idx = None
                chosen_name = chosen_machine_name

                reason = ''
                try:
                    if avail_mask is not None and chosen_idx is not None:
                        arr = np.asarray(avail_mask)
                        if arr.size > 0:
                            if hasattr(self.env, 'num_wcs') and hasattr(self.env, 'num_ops'):
                                ops = int(self.env.num_ops)
                                mcnt = int(self.env.num_wcs)
                                if arr.size >= mcnt * ops:
                                    start = chosen_idx * ops
                                    end = start + ops
                                    if end <= arr.size and not bool(arr[start:end].any()):
                                        reason = 'no_operator_free'
                            else:
                                if hasattr(self.env, 'num_wcs') and arr.size == int(self.env.num_wcs):
                                    if int(arr[chosen_idx]) == 0:
                                        reason = 'machine_not_available'
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    reason = reason or ''

                if not allow_history_writes():
                    # history writes disabled; skip logging
                    pass
                else:
                    if gantt_utils is not None:
                        try:
                            gantt_utils.append_selection_log(sched_path, sim_time, job_id, allowed_wcs, avail_mask, chosen_idx, chosen_name, reason)
                        except Exception as e:
                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                            pass
                    else:
                        try:
                            header_needed = not os.path.exists(sched_path)
                            with open(sched_path, 'a') as sf:
                                if header_needed:
                                    sf.write('time,job_id,allowed_wcs,avail_mask,chosen_machine_idx,chosen_machine_name,reason\n')
                                sf.write(f"{sim_time},{job_id},{allowed_wcs},{avail_mask},{chosen_idx},{repr(chosen_name)},{reason}\n")
                        except Exception as e:
                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                            pass
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            processed_actions.append(chosen)
            processed_machine_names.append(chosen_machine_name)

            # finally, attempt to resume the decision with chosen value
            try:
                item.get('resume')(chosen)
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                try:
                    item.get('resume')(int(chosen))
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    try:
                        item.get('resume')(chosen_machine_name)
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        pass

        return processed_actions, processed_machine_names

    # -------------------------
    # Utility: ensure hidden
    # -------------------------
    def _ensure_hidden(self, n_agents: int):
        """Ensure RNN hidden state exists (best-effort)."""
        hdim = getattr(self.args, "rnn_hidden_dim", 64)
        if torch is None:
            self.eval_hidden = None
            return
        self.eval_hidden = torch.zeros((n_agents, hdim), dtype=torch.float32, device=self.device)

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
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            obs_batch = None

        # Determine granularity
        try:
            use_gran = bool(getattr(args, 'use_granular_actions', False))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            use_gran = False

        # Build avail_batch similar to Runner
        avail_batch = []
        if use_gran:
            # try to import mask utils
            try:
                from MARL.common.mask_utils import build_index_map, build_mask_for_job
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                build_index_map = None
                build_mask_for_job = None

            # build op->machines mapping
            op_to_m = {}
            try:
                groups_map = getattr(self.env.workcenters_meta, 'eligible_operator_groups_by_wc', {})
                for wc_idx, ops in groups_map.items():
                    for p in ops:
                        op_to_m.setdefault(int(p), []).append(int(wc_idx))
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                op_to_m = {}

            try:
                num_m = int(len(getattr(self.env.workcenters_meta, 'machine_list', []) or []))
                if num_m == 0:
                    num_m = int(getattr(self.env, 'num_wcs', 0))
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                num_m = int(getattr(self.env, 'num_wcs', 0))
            num_p = int(getattr(self.env, 'num_ops', 0))

            try:
                if getattr(self.env, 'machine_resources', None):
                    machine_free = [self.env._resource_free(self.env.machine_resources[m]) for m in range(num_m)]
                else:
                    machine_free = [self.env._resource_free(self.env.wc_resources[m]) for m in range(num_m)]
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                machine_free = [True] * num_m
            try:
                operator_free = [self.env._resource_free(self.env.operator_groups[p]) for p in range(num_p)]
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                operator_free = [True] * num_p

            for item in batch:
                try:
                    if item.get('avail_mask') is not None:
                        mask = item.get('avail_mask')
                        avail_batch.append(list(mask))
                        continue

                    allowed = item.get('allowed_wcs', [])
                    if build_index_map is not None and build_mask_for_job is not None and num_m > 0 and num_p > 0:
                        idx_map = build_index_map(num_m, num_p)
                        try:
                            msk = build_mask_for_job(idx_map, allowed, op_to_m, machine_free, operator_free)
                            avail_batch.append(msk.tolist())
                            item['avail_mask'] = msk.tolist()
                            continue
                        except Exception as e:
                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                            pass

                    row = item.get('avail_row') or []
                    flat = []
                    for m in range(num_m):
                        v = 1 if (m < len(row) and int(bool(row[m]))) else 0
                        flat.extend([int(v)] * max(1, num_p))
                    avail_batch.append(flat)
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    avail_batch.append(None)
        else:
            try:
                avail_batch = [item.get('avail_row') for item in batch]
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                avail_batch = None

        # Build u_list and u_machine mappings similar to Runner
        try:
            u_list = []
            u_machine_list = []
            for i, a in enumerate(processed_actions):
                try:
                    u_list.append(int(a) if a is not None else 0)
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    u_list.append(0)
                try:
                    uname = processed_machine_names[i] if i < len(processed_machine_names) else None
                    if uname is not None:
                        u_machine_list.append(int(getattr(self.env.workcenters_meta, 'machine_index', {}).get(uname, -1)))
                    else:
                        u_machine_list.append(-1)
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    u_machine_list.append(-1)
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            u_list = [int(a) if a is not None else 0 for a in (processed_actions or [])]
            u_machine_list = [-1] * len(u_list)

        # pad/truncate to args.n_agents
        n_agents = getattr(args, 'n_agents', len(u_list) if u_list else 1)
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
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                tr['u_machine_name'] = [None] * len(u_list)
            tr['r'] = r
            # attach avail_a / avail_a_next
            try:
                if bool(getattr(args, 'use_granular_actions', False)):
                    try:
                        ops = int(self.env.num_ops)
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        ops = None
                    n_agents_local = len(batch)
                    if ops is not None and hasattr(self.env, 'num_wcs'):
                        n_actions_local = int(self.env.num_wcs) * ops
                    else:
                        n_actions_local = None
                    avail_flat = []
                    for j, it in enumerate(batch):
                        mask = it.get('avail_mask')
                        if mask is not None:
                            try:
                                arr = np.asarray(mask, dtype=np.float32)
                                if n_actions_local is None or arr.size >= n_actions_local:
                                    if n_actions_local is not None:
                                        s = arr.size
                                        if s < n_actions_local:
                                            pad = np.zeros((n_actions_local - s,), dtype=np.float32)
                                            arr = np.concatenate([arr, pad], axis=0)
                                        arr = arr[:n_actions_local]
                                    avail_flat.append(arr.astype(np.float32))
                                    continue
                            except Exception as e:
                                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                pass
                        try:
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
                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                            pass
                        if n_actions_local is not None:
                            avail_flat.append(np.zeros((n_actions_local,), dtype=np.float32))
                        else:
                            avail_flat.append(np.zeros((len(u_list),), dtype=np.float32))

                    tr['avail_a'] = np.asarray(avail_flat, dtype=np.float32)
                else:
                    if avail_arr is not None:
                        tr['avail_a'] = avail_arr
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                            pass
                    tr['avail_a_next'] = avail_next_arr
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            if s_before is not None:
                tr['s'] = s_before
            if s_after is not None:
                tr['s_next'] = s_after
            tr['done'] = getattr(self.env, 'done', False)

            tr_list.append(tr)
        # -------------------------
        # Dev-only runtime shape assertions
        # If `args.debug_assert_shapes` is True the code will raise AssertionError
        # on mismatches; otherwise it logs a warning. This helps surface
        # silent failures caused by broad exception swallowing during refactors.
        try:
            debug_assert = bool(getattr(args, 'debug_assert_shapes', False))
            n_agents_expected = getattr(args, 'n_agents', None)
            obs_dim_expected = getattr(args, 'obs_shape', None) or getattr(args, 'obs_dim_agent', None) or None
            if n_agents_expected is not None:
                n_agents_expected = int(n_agents_expected)
            if obs_dim_expected is not None:
                obs_dim_expected = int(obs_dim_expected)

            for idx_tr, tr in enumerate(tr_list):
                # check observation shape
                try:
                    if 'o' in tr and n_agents_expected is not None and obs_dim_expected is not None:
                        o_arr = np.asarray(tr['o'], dtype=np.float32)
                        if o_arr.ndim == 2:
                            if o_arr.shape[0] != n_agents_expected:
                                msg = f"Transition[{idx_tr}]['o'] has {o_arr.shape[0]} agents, expected {n_agents_expected}"
                                if debug_assert:
                                    raise AssertionError(msg)
                                else:
                                    logging.getLogger(__name__).warning(msg)
                            if o_arr.shape[1] != obs_dim_expected:
                                msg = f"Transition[{idx_tr}]['o'] obs_dim {o_arr.shape[1]} != expected {obs_dim_expected}"
                                if debug_assert:
                                    raise AssertionError(msg)
                                else:
                                    logging.getLogger(__name__).warning(msg)
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    # don't break production flow; warnings already emitted above
                    logging.getLogger(__name__).debug("Failed to validate transition['o'] shape", exc_info=True)

                # check action vector length
                try:
                    if 'u' in tr and n_agents_expected is not None:
                        u_val = tr['u']
                        if isinstance(u_val, (list, tuple, np.ndarray)):
                            if len(u_val) != n_agents_expected:
                                msg = f"Transition[{idx_tr}]['u'] length {len(u_val)} != expected n_agents {n_agents_expected}"
                                if debug_assert:
                                    raise AssertionError(msg)
                                else:
                                    logging.getLogger(__name__).warning(msg)
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    logging.getLogger(__name__).debug("Failed to validate transition['u'] length", exc_info=True)
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            # Best-effort: don't let assertion scaffolding break normal runs
            logging.getLogger(__name__).debug("Transition assertions encountered an unexpected error", exc_info=True)

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
        try:
            obs_init, info = self.env.reset()
        except TypeError:
            # some reset signatures may return only obs
            obs_init = self.env.reset()
            info = {}

        # Stamp the env with the current episode index so environment-level
        # appenders can record which episode a gantt record belongs to.
        try:
            try:
                self.env.current_episode = int(global_ep_idx)
            except Exception:
                # best-effort: if conversion fails, still attach raw value
                setattr(self.env, 'current_episode', global_ep_idx)
        except Exception:
            pass

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
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                s_before = None

            # get actions for the batch via the worker's agent wrapper
            obs_list = [item.get('obs') for item in batch]
            try:
                avail = [item.get('avail_row') for item in batch]
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                avail = None
            try:
                # _select_actions returns (actions_list, hidden_state).
                # Unpack the pair so callers receive the actions list.
                actions, _ = self._select_actions(obs_list, avail, evaluate=evaluate)
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                actions = []

            # Process and apply actions via RolloutWorker helper (centralized logic)
            try:
                processed_actions, processed_machine_names = self.process_and_apply_actions(batch, actions, sim_time, runner_args=args)
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                processed_actions = actions or []
                processed_machine_names = [None] * len(processed_actions)

            actions = processed_actions

            # after resuming processes, collect reward accumulated since last decision boundary
            try:
                r = float(self.env.pop_decision_reward())
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                r = 0.0
            episode["r"].append(r)

            # capture global state after decision (if available)
            try:
                s_after = np.asarray(self.env._build_state_vector(), dtype=np.float32)
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                s_after = None

            # capture availabilities after decision (for avail_a_next)
            try:
                avail_after = None
                if hasattr(self.env, '_build_avail_actions'):
                    avail_after = self.env._build_avail_actions()
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            # collect possible lightweight gantt info if present on env or items
            try:
                gantt.extend(self.collect_gantt_from_batch(batch, sim_time))
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            # stop if env signals done
            if getattr(self.env, "done", False):
                break
            # safety: break if time limit reached
            if getattr(self.env, "t", 0.0) >= getattr(self.env, "episode_limit", getattr(args, 'n_steps', 1e9)):
                break

        ep_reward = float(np.sum(episode.get("r", [])))
        try:
            win_tag = all(j.finished for j in getattr(self.env, "jobs", []))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            win_tag = False

        # store episode into replay buffer if available
        try:
            if self.buffer is not None and len(ep_transitions) > 0:
                try:
                    self.buffer.store_episode(ep_transitions)
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    pass
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass

        # include environment-level gantt records if present
        try:
            if hasattr(self.env, "gantt_records") and isinstance(self.env.gantt_records, (list, tuple)):
                gantt.extend(list(self.env.gantt_records))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass

        # Clear the temporary episode stamp so other callers are not confused
        try:
            if hasattr(self.env, 'current_episode'):
                try:
                    delattr(self.env, 'current_episode')
                except Exception:
                    try:
                        del self.env.current_episode
                    except Exception:
                        setattr(self.env, 'current_episode', None)
        except Exception:
            pass

        return episode, ep_reward, bool(win_tag), gantt


# Backwards-compatibility alias: older code imported CommRolloutWorker
# Ensure such imports continue to work until callers are updated.
CommRolloutWorker = RolloutWorker