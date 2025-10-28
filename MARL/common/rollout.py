import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple

# try to import torch for RNN hidden handling; degrade gracefully if not available
try:
    import torch
except Exception:
    torch = None

# mask utilities
try:
    from MARL.common.mask_utils import build_index_map, build_mask_for_job
except Exception:
    build_index_map = None
    build_mask_for_job = None


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
        episode_limit: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_anneal_steps: int = 50000,
        device: str = "cpu",
        log_prefix: str = "8A.7.3",
    ):
        self.env = env
        self.agents = agents
        self.buffer = buffer
        self.args = args or type("A", (), {})()
        self.episode_limit = getattr(self.args, "episode_limit", episode_limit)
        self.device = getattr(self.args, "device", device)
        self.log_prefix = log_prefix

        # epsilon schedule
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_anneal_steps = int(epsilon_anneal_steps)
        self._eps_decay = (self.epsilon_start - self.epsilon_end) / max(1, self.epsilon_anneal_steps)

        # runtime placeholders
        self.eval_hidden = None
        self.episode_duration = 0

        # RNG for Runner fallback
        seed = getattr(self.args, "seed", None)
        try:
            self.rng = np.random.RandomState(seed if seed is not None else 0)
        except Exception:
            self.rng = np.random.RandomState(0)

        # log
        print(f"[RolloutWorker] init | episode_limit={self.episode_limit} | device={self.device}")

    # -------------------------
    # Legacy step-based rollout
    # -------------------------
    def generate_episode(self, global_ep_idx: int = 0, evaluate: bool = False) -> Tuple[Dict[str, List[Any]], float, bool, List]:
        """
        Run a full episode using legacy env.step interface.
        Returns: (episode_dict, episode_reward_sum, win_flag, gantt_list)
        Minimal episode_dict contains key 'r' (list of per-decision rewards).
        """
        episode = {"r": []}
        gantt = []

        # reset environment (support both obs or (obs,info) signatures)
        try:
            reset_ret = self.env.reset()
            if isinstance(reset_ret, tuple) and len(reset_ret) >= 1:
                obs = reset_ret[0]
                info = reset_ret[1] if len(reset_ret) > 1 else {}
            else:
                obs = reset_ret
                info = {}
        except Exception:
            # fallback: try calling without args
            obs = self.env.reset()
            info = {}

        done = False
        t = 0

        # step loop (best-effort generic)
        while not done and t < self.episode_limit:
            # build obs_batch (if env returns per-agent obs)
            obs_batch = obs if isinstance(obs, (list, tuple)) else [obs]

            # choose actions
            actions = self._select_actions(obs_batch, None, evaluate=evaluate)

            # try stepping the environment with actions
            try:
                step_ret = self.env.step(actions)
            except Exception:
                # env may expect single action for single-agent
                try:
                    step_ret = self.env.step(actions[0] if isinstance(actions, (list, tuple)) and len(actions) > 0 else actions)
                except Exception:
                    break

            # Interpret common return signatures
            obs_next = None
            reward = 0.0
            done = False
            info = {}
            if isinstance(step_ret, tuple):
                if len(step_ret) == 4:
                    obs_next, reward, done, info = step_ret
                elif len(step_ret) == 3:
                    obs_next, reward, done = step_ret
                    info = {}
                elif len(step_ret) == 2:
                    # (reward, info) or (obs_next, info) — best guess
                    if isinstance(step_ret[0], (int, float, list, np.ndarray)):
                        reward = step_ret[0]
                        info = step_ret[1]
                    else:
                        obs_next, info = step_ret
                else:
                    # unknown tuple shape — attempt to assign first elements
                    obs_next = step_ret[0] if len(step_ret) > 0 else None
                    reward = step_ret[1] if len(step_ret) > 1 and isinstance(step_ret[1], (int, float)) else 0.0
                    done = bool(step_ret[2]) if len(step_ret) > 2 else False
            else:
                # single scalar or object returned — unlikely; treat as reward
                if isinstance(step_ret, (int, float)):
                    reward = float(step_ret)
                else:
                    # cannot interpret; break
                    break

            # normalize reward to float (if vector, sum)
            if isinstance(reward, (list, tuple, np.ndarray)):
                try:
                    r_val = float(np.sum(reward))
                except Exception:
                    r_val = float(reward[0]) if len(reward) > 0 else 0.0
            else:
                try:
                    r_val = float(reward)
                except Exception:
                    r_val = 0.0

            episode["r"].append(r_val)
            t += 1
            self.episode_duration = t

            # advance
            obs = obs_next

        ep_reward = float(np.sum(episode.get("r", [])))
        # try to detect win: environment-specific; fallback False
        try:
            win_tag = all(j.finished for j in getattr(self.env, "jobs", []))
        except Exception:
            win_tag = False

        return episode, ep_reward, win_tag, gantt

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
        except Exception:
            pass

        # Try per-observation act method
        try:
            if hasattr(self.agents, "act"):
                actions = []
                for ob in obs_batch:
                    a = self.agents.act(ob, None, evaluate=evaluate)
                    actions.append(a)
                return actions, None
        except Exception:
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
                except Exception:
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
        except Exception:
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
                    except Exception:
                        op_to_m = {p: [] for p in range(int(self.env.num_ops))}
                    # machine resource free state: map each machine to its WC resource
                    machine_free = []
                    try:
                        mreg = getattr(self.env.workcenters_meta, 'machine_registry', {})
                        for mname in machine_list:
                            try:
                                wc_i = int(mreg.get(mname, {}).get('workcenter', 0))
                                machine_free.append(self.env._resource_free(self.env.wc_resources[wc_i]))
                            except Exception:
                                machine_free.append(True)
                    except Exception:
                        machine_free = [True] * num_m
                    try:
                        operator_free = [self.env._resource_free(self.env.operator_groups[p]) for p in range(int(self.env.num_ops))]
                    except Exception:
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
                    except Exception:
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
                                            except Exception:
                                                continue
                                op_to_m[p_idx] = mapped

                    machine_free = [self.env._resource_free(self.env.wc_resources[m]) for m in range(num_m)]
                    try:
                        operator_free = [self.env._resource_free(self.env.operator_groups[p]) for p in range(num_p)]
                    except Exception:
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
                            except Exception:
                                per_machine.append(0)
                        # prefer to set 'avail_row' so Runner will pick it up for storing into replay
                        item['avail_row'] = per_machine
                    else:
                        item['avail_mask'] = None
                except Exception:
                    item['avail_mask'] = None
                    # leave avail_row untouched if we can't compute mask

            actions, _ = self._select_actions(obs_list, avail, evaluate=evaluate)
            return actions
        except Exception:
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
                        except Exception:
                            outs.append(int(np.random.choice(allowed)))
            return outs

    def collect_gantt_from_batch(self, batch, sim_time):
        """Optional hook to build gantt records from a decision batch. Default: empty."""
        return []

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