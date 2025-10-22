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
from torch.distributions import Categorical


class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        if args.alg == 'vdn':
            self.policy = VDN(args)
        elif args.alg == 'qmix':
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            self.policy = Reinforce(args)
        else:
            raise Exception("No such algorithm")

        self.args = args
        print(f"[Agents] Initialized ({args.alg.upper()})")

    # ------------------------------------------------------------------
    # Step 8A.6.5 – Replay-aware Learning Path
    # ------------------------------------------------------------------
    def learn_from_replay(self, batch: dict, train_step: int) -> dict:
        """
        New learning path for replay-based MARL (QMIX-style).
        This directly forwards the sampled batch to the policy's learn() method.
        """
        if not hasattr(self.policy, "learn"):
            raise AttributeError("Current policy does not implement learn() method.")

        # Forward to QMIX or equivalent learner
        output = self.policy.learn(batch, train_step)

        # Optional debug
        if output is not None and isinstance(output, dict):
            loss = output.get("loss", None)
            td_error = output.get("td_error", None)
            if loss is not None:
                print(f"[Agents] Step {train_step} | Loss={loss:.4f} | TD-Error={td_error:.4f}" if td_error else
                      f"[Agents] Step {train_step} | Loss={loss:.4f}")

        return output

    # ------------------------------------------------------------------
    # Original episodic learning path (kept for backward compat.)
    # ------------------------------------------------------------------
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        """
        Unified training entry.
        - For replay-based learners (QMIX), call learn_from_replay().
        - For episodic learners (VDN, COMA, etc.), use original episodic path.
        """
        # Replay-aware QMIX path
        if isinstance(batch, dict) and 'state' in batch:
            return self.learn_from_replay(batch, train_step)

        # Episodic fallback (legacy)
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            print(f"\n[Agents] Saving model checkpoint at step {train_step}")
            self.policy.save_model(train_step)
        return None


class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print(f"[CommAgents] Initialized ({alg.upper()})")
