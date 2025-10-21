"""
environment.py
Step 8A.5.3 (Learning-Activated Revision)
------------------------------------------
- Keeps full 18-WorkCenter + Operator constraint logic.
- Adds RL-ready components:
  • true action consumption from agents
  • available-action mask generation
  • 10-dim observation/state vectors
  • minimal but meaningful reward shaping
"""

import simpy
import numpy as np
import gym
from gym import spaces

from utils.site import Sites
from utils.job import Jobs
from utils.task_generator import TaskGenerator
from utils.jobagent import JobAgent
from utils.operator import Operators
from MARL.common.terms import t


class ScheduleEnv(gym.Env):
    environment_name = "MASA-SimPy-Scheduler"

    def __init__(
        self,
        start_agents: int = 4,
        max_agents: int = 12,
        arrival_prob: float = 0.20,
        variable_ops: bool = True,
        seed: int = 123,
    ):
        # --- Core configuration ---
        self.start_agents = int(start_agents)
        self.max_agents = int(max_agents)
        self.arrival_prob = float(arrival_prob)
        self.variable_ops = bool(variable_ops)
        self.rng = np.random.default_rng(seed)

        # --- Simulation state ---
        self.workcenters = []
        self.jobs = []
        self.agents = []
        self.job_record_for_gant = []
        self.done = False
        self.step_count = 0
        self._completed_prev = 0

        self.sim_env = simpy.Environment()
        self.workcenter_resources = []
        self.task_gen = TaskGenerator()
        self.operators = None
        self.wait_time_dict = {}
        self._last_mean_wait = 0.0

        # RL interface
        self.action_space = None
        self.n_actions = 0
        self.obs_shape = 10
        self.state_shape = 10

        self.initialize()

    # ============================================================
    # === Initialization =========================================
    # ============================================================
    def initialize(self):
        sites_obj = Sites()
        jobs_obj = Jobs()

        self.workcenters = sites_obj.sites_object_list
        self.jobs = jobs_obj.jobs_object_list
        self.sim_env = simpy.Environment()
        self.workcenter_resources = [
            simpy.Resource(self.sim_env, capacity=1) for _ in range(len(self.workcenters))
        ]

        self.agents = []
        self.job_record_for_gant = []
        self.done = False
        self._completed_prev = 0
        self.step_count = 0
        self.wait_time_dict = {}
        self._last_mean_wait = 0.0

        # operators
        self.operators = Operators(sites_obj)
        self.operators.release_all()

        # RL spaces
        self.n_actions = len(self.workcenters)
        self.action_space = spaces.Discrete(self.n_actions)

        # initial JobAgents
        for aid in range(self.start_agents):
            task_objs = self.task_gen.generate_constrained_task(jobagent_id=aid)
            agent = JobAgent(agent_id=aid, job_object_list=task_objs, arrival_time=0)
            self.agents.append(agent)
            self.sim_env.process(self.jobagent_process(aid))

        print(
            f"\n=== Learning-Active Env Init ===\n"
            f"WorkCenters={len(self.workcenters)} | Operators={len(self.operators.operators_object_list)} "
            f"| StartAgents={self.start_agents}\n"
        )

    # ============================================================
    # === Candidate discovery / Mask =============================
    # ============================================================
    def get_avail_actions(self):
        """Mask over WorkCenters: 1 if free + operator + capability."""
        mask = np.zeros((len(self.workcenters),), dtype=np.float32)
        active_jobs = []
        for a in self.agents:
            if a.is_active and a.left_job:
                jid = a.left_job[0].index_id if hasattr(a.left_job[0], "index_id") else a.left_job[0]
                active_jobs.append(jid)
        if not active_jobs:
            return mask
        for wc_id, wc in enumerate(self.workcenters):
            if len(self.workcenter_resources[wc_id].users) != 0:
                continue
            has_op = any(
                (not op.is_busy) and (wc_id in op.qualified_workcenters)
                for op in self.operators.operators_object_list
            )
            if not has_op:
                continue
            can_any = any((jid in wc.resource_ids_list) for jid in active_jobs)
            if can_any:
                mask[wc_id] = 1.0
        return mask

    # ============================================================
    # === Observation / State ===================================
    # ============================================================
    def _build_obs(self):
        T = float(self.sim_env.now)
        n_agents = len(self.agents)
        n_active = len(self.get_active_agents())
        n_completed = len(self.job_record_for_gant)
        waits = list(self.wait_time_dict.values()) if self.wait_time_dict else []
        mean_wait = float(np.mean(waits)) if waits else 0.0

        obs = np.zeros(self.obs_shape, dtype=np.float32)
        obs[0] = min(1.0, T / 1000.0)
        obs[1] = n_agents / max(1.0, self.max_agents)
        obs[2] = n_active / max(1, n_agents)
        obs[3] = min(1.0, n_completed / 100.0)
        obs[4] = min(1.0, mean_wait / 100.0)
        return obs

    def _build_state(self):
        return self._build_obs().copy()

    # ============================================================
    # === Core SimPy job processes ===============================
    # ============================================================
    def jobagent_process(self, agent_id: int):
        agent = self.agents[agent_id]
        agent.total_wait_time = 0.0

        if agent.arrival_time > self.sim_env.now:
            yield self.sim_env.timeout(agent.arrival_time - self.sim_env.now)
        agent.is_active = True
        print(f"[t={self.sim_env.now}] {t('JobAgent')} {agent_id} entered system.")

        while agent.left_job:
            jid = agent.left_job[0].index_id if hasattr(agent.left_job[0], "index_id") else agent.left_job[0]
            yield self.sim_env.timeout(1)  # passive until dispatched
            agent.total_wait_time += 1.0
            self.wait_time_dict[agent_id] = agent.total_wait_time

        agent.is_active = False
        self.wait_time_dict[agent_id] = agent.total_wait_time

    def _dispatch_agent_to_wc(self, agent, wc_id):
        """Launch operation if possible; return True if started."""
        if not agent.is_active or not agent.left_job:
            return False
        if wc_id < 0 or wc_id >= len(self.workcenter_resources):
            return False
        if len(self.workcenter_resources[wc_id].users) != 0:
            return False
        j = agent.left_job[0]
        jid = j.index_id if hasattr(j, "index_id") else j
        if jid not in self.workcenters[wc_id].resource_ids_list:
            return False
        op = self.operators.find_free_operator(jid, wc_id)
        if op is None:
            return False

        wc_res = self.workcenter_resources[wc_id]

        def _op():
            with wc_res.request() as req:
                yield req
                op.assign_job(jid, wc_id)
                start = self.sim_env.now
                proc = self.jobs[jid].time_span
                yield self.sim_env.timeout(proc)
                end = self.sim_env.now
                op.release()
                self.save_env_info((start, end, jid, wc_id, agent.agent_id, op.operator_id))
                if agent.left_job and (
                    agent.left_job[0].index_id if hasattr(agent.left_job[0], "index_id") else agent.left_job[0]
                ) == jid:
                    agent.left_job.pop(0)

        self.sim_env.process(_op())
        return True

    # ============================================================
    # === Core step ==============================================
    # ============================================================
    def step(self, actions=None):
        self.step_count += 1
        # maybe spawn new jobs
        self.maybe_spawn()

        # --- apply actions ---
        if actions is not None:
            arr = np.asarray(actions).reshape(-1)
            for idx, agent in enumerate(self.agents):
                if idx < arr.shape[0]:
                    wc_choice = int(arr[idx])
                    self._dispatch_agent_to_wc(agent, wc_choice)

        # advance SimPy clock
        self.advance_clock()

        # --- reward shaping ---
        new_records = self.job_record_for_gant[self._completed_prev :]
        newly_completed_by = [rec[4] for rec in new_records]
        completed_now = len(new_records)
        self._completed_prev = len(self.job_record_for_gant)

        reward = -1.0 + 8.0 * completed_now
        waits = list(self.wait_time_dict.values()) if self.wait_time_dict else []
        cur_mean = float(np.mean(waits)) if waits else 0.0
        delta_wait = max(0.0, cur_mean - self._last_mean_wait)
        reward -= 0.2 * delta_wait
        self._last_mean_wait = cur_mean

        self.done = self.all_jobs_completed()
        if self.done:
            print(f"✅ All jobs finished @ t={self.sim_env.now}")

        # --- info pack ---
        current_waits = {a.agent_id: getattr(a, "total_wait_time", 0.0) for a in self.agents}
        current_waits.update(self.wait_time_dict)

        obs_vec = self._build_obs()
        state_vec = self._build_state()
        avail_mask = self.get_avail_actions()

        info = {
            "time": self.sim_env.now,
            "completed_jobs": len(self.job_record_for_gant),
            "episodes_situation": list(self.job_record_for_gant),
            f"n_{t('JobAgent')}s": len(self.agents),
            "active_agents": self.get_active_agents(),
            "new_records": list(new_records),
            "newly_completed_by": list(newly_completed_by),
            "wait_times": current_waits,
            "avail_actions": avail_mask,
            "obs": obs_vec,
            "state": state_vec,
        }

        print(
            f"[DEBUG] Step={self.step_count} | t={self.sim_env.now} | "
            f"completed={len(self.job_record_for_gant)} | active={len(self.get_active_agents())}"
        )
        return reward, self.done, info

    # ============================================================
    # === Utility helpers =======================================
    # ============================================================
    def save_env_info(self, record):
        self.job_record_for_gant.append(record)

    def all_jobs_completed(self):
        return all((not a.is_active) or (len(a.left_job) == 0) for a in self.agents)

    def get_active_agents(self):
        return [a.agent_id for a in self.agents if a.is_active]

    def advance_clock(self, max_time=None):
        if len(self.sim_env._queue) == 0:
            return
        next_time = self.sim_env._queue[0][0]
        if max_time and next_time > max_time:
            return
        self.sim_env.step()
        while len(self.sim_env._queue) > 0 and self.sim_env._queue[0][0] == self.sim_env.now:
            self.sim_env.step()

    def can_spawn_more(self) -> bool:
        return len(self.agents) < self.max_agents

    def add_new_agent(self, time_now: float):
        if not self.can_spawn_more():
            return False
        aid = len(self.agents)
        task_objs = self.task_gen.generate_constrained_task(jobagent_id=aid)
        agent = JobAgent(agent_id=aid, job_object_list=task_objs, arrival_time=time_now)
        self.agents.append(agent)
        self.sim_env.process(self.jobagent_process(aid))
        print(f"[ARRIVAL t={time_now}] New {t('JobAgent')} {aid}")
        return True

    def maybe_spawn(self):
        if self.can_spawn_more() and self.rng.random() < self.arrival_prob:
            self.add_new_agent(self.sim_env.now)

    def reset(self):
        self.initialize()
        return np.zeros(self.obs_shape, dtype=np.float32)

    def get_env_info(self):
        return {
            "n_agents": len(self.agents),
            "n_actions": self.n_actions,
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "episode_limit": 200,
        }

    # ============================================================
    # === Quick test ============================================
    # ============================================================
    def test_dynamic_arrivals(self, max_steps=50):
        self.reset()
        for _ in range(max_steps):
            acts = self.rng.integers(0, self.n_actions, size=(len(self.agents),))
            _, done, _ = self.step(acts)
            if done:
                break
        print("✅ Learning-Active Step test complete.")


if __name__ == "__main__":
    env = ScheduleEnv()
    env.test_dynamic_arrivals()
