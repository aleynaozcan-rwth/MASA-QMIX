"""
environment.py
Step 8A.6.2 – Reward Shaping Enhanced Environment
-------------------------------------------------
- Adds structured reward signal (wait penalty + completion bonus + utilization reward)
- Compatible with QMix-based rollout (8A.6)
- Returns (obs, state, avail_actions) tuples for learning
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
    environment_name = "MASA-QMIX-Scheduler"

    def __init__(self,
                 start_agents: int = 4,
                 max_agents: int = 12,
                 arrival_prob: float = 0.20,
                 variable_ops: bool = True,
                 seed: int = 123):
        # --- Core parameters ---
        self.start_agents = start_agents
        self.max_agents = max_agents
        self.arrival_prob = arrival_prob
        self.variable_ops = variable_ops
        self.rng = np.random.default_rng(seed)

        # --- Env structures ---
        self.sim_env = simpy.Environment()
        self.workcenters = []
        self.jobs = []
        self.agents = []
        self.job_record_for_gant = []
        self.workcenter_resources = []
        self.done = False
        self.step_count = 0
        self._completed_prev = 0
        self.wait_time_dict = {}

        # --- Operators & Task generator ---
        self.operators = None
        self.task_gen = TaskGenerator()

        # --- Gym/learning interface ---
        self.action_space = spaces.Discrete(21)

        # --- Reward tracking ---
        self.total_reward = 0.0
        self.operator_utilization = {}
        self.workcenter_utilization = {}

        self.initialize()

    # ============================================================
    # === Observation / State / Action Info ======================
    # ============================================================
    def observe(self):
        """
        Generate simplified observation/state/action space.
        For now: random low-dimensional representation (placeholder for feature extraction).
        """
        obs = np.zeros((len(self.agents), 10), dtype=np.float32)
        state = np.zeros((10,), dtype=np.float32)
        avail = np.ones((len(self.agents), self.action_space.n), dtype=np.float32)
        return obs, state, avail

    def get_env_info(self):
        return {
            "n_agents": len(self.agents),
            "n_actions": self.action_space.n,
            "state_shape": 10,
            "obs_shape": 10,
            "episode_limit": 200,
        }

    # ============================================================
    # === Initialization =========================================
    # ============================================================
    def initialize(self):
        sites_obj = Sites()
        jobs_obj = Jobs()

        self.workcenters = sites_obj.sites_object_list
        self.jobs = jobs_obj.jobs_object_list
        self.sim_env = simpy.Environment()
        self.workcenter_resources = [simpy.Resource(self.sim_env, capacity=1)
                                     for _ in range(len(self.workcenters))]

        self.agents = []
        self.job_record_for_gant = []
        self.done = False
        self._completed_prev = 0
        self.step_count = 0
        self.wait_time_dict = {}

        self.operators = Operators(sites_obj)
        self.operators.release_all()

        for aid in range(self.start_agents):
            task_objs = self.task_gen.generate_constrained_task(jobagent_id=aid)
            agent = JobAgent(agent_id=aid, job_object_list=task_objs, arrival_time=0)
            self.agents.append(agent)
            self.sim_env.process(self.jobagent_process(aid))

        print(f"\n=== Step 8A.6.2 Initialized ===")
        print(f"WorkCenters={len(self.workcenters)} | Operators={len(self.operators.operators_object_list)}")

    # ============================================================
    # === Candidate Discovery ====================================
    # ============================================================
    def _candidate_pairs_for_job(self, job_id):
        candidates = []
        for wc_id, wc in enumerate(self.workcenters):
            if job_id not in wc.resource_ids_list:
                continue
            if len(self.workcenter_resources[wc_id].users) > 0:
                continue
            op = self.operators.find_free_operator(job_id, wc_id)
            if op is not None:
                candidates.append((wc_id, op.operator_id))
        return candidates

    # ============================================================
    # === JobAgent Execution =====================================
    # ============================================================
    def jobagent_process(self, agent_id):
        agent = self.agents[agent_id]
        agent.total_wait_time = 0.0

        if agent.arrival_time > self.sim_env.now:
            yield self.sim_env.timeout(agent.arrival_time - self.sim_env.now)

        agent.is_active = True

        while agent.left_job:
            job_id = agent.left_job[0].index_id
            candidates = self._candidate_pairs_for_job(job_id)

            if not candidates:
                yield self.sim_env.timeout(1)
                agent.total_wait_time += 1.0
                self.wait_time_dict[agent_id] = agent.total_wait_time
                continue

            chosen_wc, chosen_op = self.rng.choice(candidates)
            wc_res = self.workcenter_resources[chosen_wc]
            operator = [op for op in self.operators.operators_object_list if op.operator_id == chosen_op][0]

            with wc_res.request() as req:
                yield req
                operator.assign_job(job_id, chosen_wc)

                start_t = self.sim_env.now
                p_time = self.jobs[job_id].time_span
                yield self.sim_env.timeout(p_time)
                end_t = self.sim_env.now

                operator.release()
                self.save_env_info((start_t, end_t, job_id, chosen_wc, agent_id, operator.operator_id))
                agent.left_job.pop(0)

        agent.is_active = False
        self.wait_time_dict[agent_id] = agent.total_wait_time

    # ============================================================
    # === Reward Computation =====================================
    # ============================================================
    def _compute_reward(self, completed_now):
        """
        Reward = completion bonus + utilization balance - waiting penalty
        """
        # --- Completion reward ---
        r_complete = completed_now * 10.0

        # --- Average waiting penalty ---
        mean_wait = np.mean(list(self.wait_time_dict.values())) if self.wait_time_dict else 0.0
        r_wait_penalty = -0.1 * mean_wait

        # --- Utilization metrics ---
        op_busy = np.mean([1 if op.is_busy else 0 for op in self.operators.operators_object_list])
        wc_busy = np.mean([len(wc.users) for wc in self.workcenter_resources])

        # encourage moderate load
        r_util = +2.0 * (1 - abs(op_busy - 0.5))  # peak at balanced utilization
        r_wc = +1.5 * (1 - abs(wc_busy - 0.5))

        total_r = r_complete + r_util + r_wc + r_wait_penalty
        return total_r

    # ============================================================
    # === Step function ==========================================
    # ============================================================
    def step(self, actions=None):
        self.step_count += 1
        self.maybe_spawn()
        prev_completed = len(self.job_record_for_gant)

        self.advance_clock()
        new_records = self.job_record_for_gant[prev_completed:]
        completed_now = len(new_records)

        reward = self._compute_reward(completed_now)

        self.done = self.all_jobs_completed() or self.step_count >= 200
        obs, state, avail = self.observe()

        info = {
            "time": self.sim_env.now,
            "completed_jobs": len(self.job_record_for_gant),
            "episodes_situation": list(self.job_record_for_gant),
            "active_agents": [a.agent_id for a in self.agents if a.is_active],
        }

        return (obs, state, avail), reward, self.done, info

    # ============================================================
    # === Other helpers ==========================================
    # ============================================================
    def advance_clock(self, max_time=None):
        if len(self.sim_env._queue) == 0:
            return
        next_time = self.sim_env._queue[0][0]
        if max_time and next_time > max_time:
            return
        self.sim_env.step()
        while len(self.sim_env._queue) > 0 and self.sim_env._queue[0][0] == self.sim_env.now:
            self.sim_env.step()

    def all_jobs_completed(self):
        return all((not a.is_active) or (len(a.left_job) == 0) for a in self.agents)

    def maybe_spawn(self):
        if len(self.agents) < self.max_agents and self.rng.random() < self.arrival_prob:
            aid = len(self.agents)
            task_objs = self.task_gen.generate_constrained_task(jobagent_id=aid)
            agent = JobAgent(agent_id=aid, job_object_list=task_objs, arrival_time=self.sim_env.now)
            self.agents.append(agent)
            self.sim_env.process(self.jobagent_process(aid))
            print(f"[ARRIVAL t={self.sim_env.now}] New {t('JobAgent')} {aid}")

    def save_env_info(self, record):
        self.job_record_for_gant.append(record)

    def reset(self):
        self.initialize()
        return self.observe()
