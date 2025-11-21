import numpy as np
import torch
from MARL.policy.vdn import VDN
from MARL.policy.qmix import QMIX
from MARL.policy.coma import COMA
from MARL.policy.reinforce import Reinforce
from MARL.policy.central_v import CentralV
from MARL.policy.qtran_alt import QtranAlt
from MARL.policy.qtran_base import QtranBase
from MARL.policy.maven import MAVEN


class Agents:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape

        # --- Policy selection ---
        alg = self.args.alg.lower()
        if alg == "vdn":
            self.policy = VDN(self.args)
        elif alg == "qmix":
            self.policy = QMIX(self.args)
        elif alg == "coma":
            self.policy = COMA(self.args)
        elif alg == "qtran_alt":
            self.policy = QtranAlt(self.args)
        elif alg == "qtran_base":
            self.policy = QtranBase(self.args)
        elif alg == "maven":
            self.policy = MAVEN(self.args)
        elif alg == "central_v":
            self.policy = CentralV(self.args)
        elif alg == "reinforce":
            self.policy = Reinforce(self.args)
        else:
            raise Exception(f"Unknown algorithm: {self.args.alg}")

        print(f"[Agents] Initialized ({self.args.alg.upper()})")

    # ============================================================
    # Step 8A.7 – Replay-aware learning (QMIX / QTRAN / MAVEN)
    # ============================================================
    def learn_from_replay(self, batch: dict, train_step: int) -> dict:
        """Replay-based MARL learning path."""
        if not hasattr(self.policy, "learn"):
            raise AttributeError("Current policy does not implement learn().")

        output = self.policy.learn(batch, train_step)

        # Optional log
        if isinstance(output, dict):
            loss = output.get("loss")
            td_error = output.get("td_error")
            if loss is not None:
                if td_error is not None:
                    print(f"[Agents] Step {train_step} | Loss={loss:.4f} | TD-Error={td_error:.4f}")
                else:
                    print(f"[Agents] Step {train_step} | Loss={loss:.4f}")
        return output

    # ============================================================
    # Step 8A.5 – Episodic learning (COMA / VDN / Reinforce)
    # ============================================================
    def _get_max_episode_len(self, batch):
        terminated = batch["terminated"]
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for e_idx in range(episode_num):
            for t_idx in range(self.args.episode_limit):
                if terminated[e_idx, t_idx, 0] == 1:
                    max_episode_len = max(max_episode_len, t_idx + 1)
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        """
        Unified training entry for all algorithms.
        Automatically detects replay-based vs episodic learning mode.
        """
        alg = self.args.alg.lower()

        # --- Replay-based algorithms (QMIX / QTRAN / MAVEN) ---
        if alg in ["qmix", "qtran_alt", "qtran_base", "maven"]:
            return self.learn_from_replay(batch, train_step)

        # --- Episodic algorithms (VDN, COMA, CentralV, Reinforce) ---
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != "z":
                batch[key] = batch[key][:, :max_episode_len]

        # Determine learn signature safely
        learn_args = self.policy.learn.__code__.co_varnames
        if len(learn_args) >= 4:
            self.policy.learn(batch, max_episode_len, train_step, epsilon)
        else:
            self.policy.learn(batch, train_step)

        # Optional checkpoint
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            if hasattr(self.policy, "save_model"):
                print(f"\n[Agents] Saving model checkpoint at step {train_step}")
                self.policy.save_model(train_step)
        return None

    # -----------------------------------------------------------------
    # Batch-action helper for Runner / RolloutWorker compatibility
    # -----------------------------------------------------------------
    def select_actions(self, obs_batch, avail_batch=None, evaluate=False):
        """
        Return list of actions for obs_batch.
        Tries policy.select_actions first, falls back to per-observation policy.act.
        """
        # [C1] Try policy-level batch API - fail-fast if select_actions fails
        if hasattr(self.policy, "select_actions"):
            return self.policy.select_actions(obs_batch, avail_batch, evaluate=evaluate)

        # [C1] Fallback: call per-observation act() if available - fail-fast if fails
        actions = []
        if hasattr(self.policy, "act"):
            for ob, avail in zip(obs_batch, (avail_batch or [None] * len(obs_batch))):
                a = self.policy.act(ob, avail, evaluate=evaluate)
                actions.append(int(a) if a is not None else None)
            return actions

        # [C1] Last resort: deterministic/random pick from avail_batch - fail-fast if fails
        for avail in (avail_batch or [None] * len(obs_batch)):
            if avail is None:
                actions.append(0)
            else:
                # [C1] Action extraction from availability mask - fail-fast if fails
                allowed = [i for i, v in enumerate(avail) if int(v)]
                actions.append(int(allowed[0]) if allowed else 0)
        return actions


# ============================================================
# Communication-based algorithms (COMMNET / G2ANet)
# ============================================================
class CommAgents:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        alg = self.args.alg.lower()

        if "reinforce" in alg:
            self.policy = Reinforce(self.args)
        elif "coma" in alg:
            self.policy = COMA(self.args)
        elif "central_v" in alg:
            self.policy = CentralV(self.args)
        else:
            raise Exception(f"No CommAgent variant implemented for: {alg}")

        print(f"[CommAgents] Initialized ({alg.upper()})")

    def select_actions(self, obs_batch, avail_batch=None, evaluate=False):
        """Same batch helper for communication-based agents."""
        # [C1] Try policy-level batch API - fail-fast if select_actions fails
        if hasattr(self.policy, "select_actions"):
            return self.policy.select_actions(obs_batch, avail_batch, evaluate=evaluate)

        # [C1] Fallback: call per-observation act() - fail-fast if fails
        actions = []
        if hasattr(self.policy, "act"):
            for ob, avail in zip(obs_batch, (avail_batch or [None] * len(obs_batch))):
                a = self.policy.act(ob, avail, evaluate=evaluate)
                actions.append(int(a) if a is not None else None)
            return actions

        # [C1] Last resort: deterministic pick from avail_batch - fail-fast if fails
        for avail in (avail_batch or [None] * len(obs_batch)):
            if avail is None:
                actions.append(0)
            else:
                # [C1] Action extraction from availability mask - fail-fast if fails
                allowed = [i for i, v in enumerate(avail) if int(v)]
                actions.append(int(allowed[0]) if allowed else 0)
        return actions