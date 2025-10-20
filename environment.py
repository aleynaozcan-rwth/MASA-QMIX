"""
environment.py
Step 8A.3 — Dynamic JobAgent Arrivals + WorkCenter–Operator Constraints + Explainable Logs
--------------------------------------------------------------------------------
7A:
  - Dynamic job-agent arrivals
  - WorkCenter (machine) and operator constraints
  - Explainable logs for (WorkCenter, operator) selection

8A.3:
  - Renamed Plane → JobAgent, Site → WorkCenter
  - Imports from utils.jobagent
  - Fully terminology-aligned version
  - NEW: per-JobAgent wait-time accounting exposed via info["wait_times"]
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
        # --- Config ---
        self.start_agents = int(start_agents)
        self.max_agents = int(max_agents)
        self.arrival_prob = float(arrival_prob)
        self.variable_ops = bool(variable_ops)
        self.rng = np.random.default_rng(seed)

        # --- Core structures ---
        self.workcenters = []
        self.jobs = []
        self.agents = []
        self.job_record_for_gant = []
        self.done = False
        self.step_count = 0

        # Step 7B additions
        self._completed_prev = 0

        # --- Simulation world ---
        self.sim_env = simpy.Environment()
        self.workcenter_resources = []

        # --- Generators & Operators ---
        self.task_gen = TaskGenerator()
        self.operators = None

        # --- NEW: per-JobAgent wait times (cumulative, SimPy time units)
        self.wait_time_dict = {}

        # --- Gym interface ---
        self.action_space = spaces.Discrete(21)

        # --- Initialize world ---
        self.initialize()

    # ============================================================
    # === Pretty Logs / Explainability ===========================
    # ============================================================

    def print_workcenter_job_map(self):
        print(f"\n📋 {t('WorkCenter').upper()}–{t('Job').upper()} MAPPING")
        for wc in self.workcenters:
            print(f"   {t('WorkCenter')} {wc.site_id:02d} → {t('Job')}s {wc.resource_ids_list}")

    def print_operator_workcenter_map(self):
        print(f"\n📋 {t('Operator').upper()}–{t('WorkCenter').upper()} MAPPING")
        for op in self.operators.operators_object_list:
            print(f"   {t('Operator')} {op.operator_id} → {t('WorkCenter')}s {op.qualified_workcenters}")

    # ============================================================
    # === Helpers ================================================
    # ============================================================

    def get_active_agents(self):
        """Return list of active JobAgent IDs."""
        return [a.agent_id for a in self.agents if a.is_active]

    def get_num_agents(self):
        return len(self.agents)

    # ============================================================
    # === Candidate discovery ====================================
    # ============================================================

    def _candidate_pairs_for_job(self, job_id):
        """Find feasible (WorkCenter, Operator) pairs for an operation."""
        candidates = []
        valid_wcs = []
        for wc_id, wc in enumerate(self.workcenters):
            if job_id in wc.resource_ids_list:
                valid_wcs.append(wc_id)
                # tezgâh boş mu?
                if len(self.workcenter_resources[wc_id].users) == 0:
                    op = self.operators.find_free_operator(job_id, wc_id)
                    if op is not None:
                        candidates.append((wc_id, op.operator_id))
        return valid_wcs, candidates

    # ============================================================
    # === JobAgent process =======================================
    # ============================================================

    def jobagent_process(self, agent_id: int):
        agent = self.agents[agent_id]
        # NEW: init wait counter
        agent.total_wait_time = 0.0

        # Respect dynamic arrival
        if agent.arrival_time > self.sim_env.now:
            yield self.sim_env.timeout(agent.arrival_time - self.sim_env.now)
        agent.is_active = True
        print(f"[t={self.sim_env.now}] {t('JobAgent')} {agent_id} entered the system.")

        while agent.left_job:
            current_job_id = (
                agent.left_job[0].index_id
                if hasattr(agent.left_job[0], "index_id")
                else agent.left_job[0]
            )

            valid_wcs, candidate_pairs = self._candidate_pairs_for_job(current_job_id)
            print(f"\n🛠️ [{t('JobAgent')} {agent_id}] Next {t('Job')}={current_job_id}")
            print(f"   Step 1: Valid {t('WorkCenter')}s (can perform {t('Job')} {current_job_id}): {valid_wcs}")

            if len(valid_wcs) > 0:
                print(f"   Step 2: {t('Operator')} availability per {t('WorkCenter')}:")
                for w_id in valid_wcs:
                    can_ops = []
                    for op in self.operators.operators_object_list:
                        if (not op.is_busy) and op.can_do_job(current_job_id, w_id):
                            can_ops.append(op.operator_id)
                    print(f"       {t('WorkCenter')} {w_id} → {t('Operator')}s {can_ops if can_ops else '[]'}")

            # hiçbir uygun ikili yoksa bekle
            if not candidate_pairs:
                print(f"[WAIT] No available {t('WorkCenter')}/{t('Operator')} for {t('Job')} {current_job_id} at t={self.sim_env.now}.")
                # 1 zaman birimi bekle, bekleme süresine ekle
                yield self.sim_env.timeout(1)
                agent.total_wait_time += 1.0
                # canlı iken anlık toplamını sakla
                self.wait_time_dict[agent_id] = agent.total_wait_time
                continue

            print(f"   Step 3: Feasible ({t('WorkCenter')}, {t('Operator')}) candidates → {candidate_pairs}")
            chosen_wc_id, chosen_op_id = self.rng.choice(candidate_pairs)
            print(f"✅ Decision: choose {t('WorkCenter')} {chosen_wc_id} with {t('Operator')} {chosen_op_id}")

            wc_resource = self.workcenter_resources[chosen_wc_id]
            operator = [op for op in self.operators.operators_object_list if op.operator_id == chosen_op_id][0]

            with wc_resource.request() as req:
                yield req
                operator.assign_job(current_job_id, chosen_wc_id)

                start_time = self.sim_env.now
                proc_time = self.jobs[current_job_id].time_span
                yield self.sim_env.timeout(proc_time)
                end_time = self.sim_env.now

                operator.release()
                self.save_env_info((start_time, end_time, current_job_id, chosen_wc_id, agent_id, operator.operator_id))
                agent.left_job.pop(0)
                print(
                    f"[t={self.sim_env.now}] {t('JobAgent')} {agent_id} finished {t('Job')} {current_job_id} "
                    f"at {t('WorkCenter')} {chosen_wc_id} by {t('Operator')} {operator.operator_id}"
                )

        print(f"[t={self.sim_env.now}] {t('JobAgent')} {agent_id} completed all {t('Job')}s.")
        agent.is_active = False
        # tamamlanan ajanın son bekleme süresini yaz
        self.wait_time_dict[agent_id] = agent.total_wait_time

    def save_env_info(self, record):
        """Append (start, end, operation, WorkCenter, JobAgent, operator)."""
        self.job_record_for_gant.append(record)

    # ============================================================
    # === Initialization =========================================
    # ============================================================

    def _create_initial_agents(self):
        for aid in range(self.start_agents):
            task_objs = self.task_gen.generate_constrained_task(jobagent_id=aid)
            agent = JobAgent(agent_id=aid, job_object_list=task_objs, arrival_time=0)
            self.agents.append(agent)
            self.sim_env.process(self.jobagent_process(aid))

    def initialize(self):
        sites_obj = Sites()
        jobs_obj = Jobs()

        self.workcenters = sites_obj.sites_object_list
        self.jobs = jobs_obj.jobs_object_list

        self.sim_env = simpy.Environment()
        self.workcenter_resources = [simpy.Resource(self.sim_env, capacity=1) for _ in range(len(self.workcenters))]

        self.agents = []
        self.job_record_for_gant = []
        self.done = False
        self._completed_prev = 0
        self.step_count = 0
        self.wait_time_dict = {}  # reset

        self.operators = Operators(sites_obj)
        self.operators.release_all()

        self._create_initial_agents()

        print(f"\n=== Step 8A.3: Dynamic Arrivals + Operator Constraints Test ===")
        self.print_workcenter_job_map()
        self.print_operator_workcenter_map()
        print(
            f"\n[INIT] Step 8A.3 env with {self.start_agents} initial {t('JobAgent')}s; "
            f"max_{t('JobAgent')}s={self.max_agents}."
        )

    # ============================================================
    # === Dynamic Arrivals =======================================
    # ============================================================

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
        print(f"[ARRIVAL t={time_now}] New {t('JobAgent')} {aid} spawned with {len(task_objs)} ops.")
        return True

    def maybe_spawn(self):
        if self.can_spawn_more() and self.rng.random() < self.arrival_prob:
            self.add_new_agent(self.sim_env.now)

    # ============================================================
    # === Clock control ==========================================
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

    # ============================================================
    # === Core step ==============================================
    # ============================================================

    def all_jobs_completed(self):
        return all((not a.is_active) or (len(a.left_job) == 0) for a in self.agents)

    def step(self, action=None):
        self.step_count += 1
        self.maybe_spawn()

        reward = -1
        self.advance_clock()

        new_records = self.job_record_for_gant[self._completed_prev:]
        newly_completed_by = [rec[4] for rec in new_records]
        completed_now = len(new_records)
        self._completed_prev = len(self.job_record_for_gant)

        reward += completed_now * 10

        self.done = self.all_jobs_completed()
        if self.done:
            print(f"✅ All {t('JobAgent')}s finished at SimPy time = {self.sim_env.now}")

        # expose current cumulative waits (for plotting)
        current_waits = {a.agent_id: getattr(a, "total_wait_time", 0.0) for a in self.agents}
        # also merge any finished agents from dict (safety)
        current_waits.update(self.wait_time_dict)

        info = {
            "time": self.sim_env.now,
            "completed_jobs": len(self.job_record_for_gant),
            "episodes_situation": list(self.job_record_for_gant),
            f"n_{t('JobAgent')}s": len(self.agents),
            "active_agents": self.get_active_agents(),
            "new_records": list(new_records),
            "newly_completed_by": list(newly_completed_by),
            "wait_times": current_waits,  # NEW
        }

        print(
            f"[DEBUG] Step={self.step_count} | t={self.sim_env.now} | "
            f"completed={len(self.job_record_for_gant)} | {t('JobAgent')}s={len(self.agents)}"
        )
        return reward, self.done, info

    def reset(self):
        self.initialize()
        return np.zeros(10)

    def get_env_info(self):
        return {
            "n_agents": len(self.agents),
            "n_actions": self.action_space.n if hasattr(self.action_space, "n") else 21,
            "state_shape": 10,
            "obs_shape": 10,
            "episode_limit": 200,
        }

    def test_dynamic_arrivals(self, max_steps=200):
        self.reset()
        for _ in range(max_steps):
            _, done, _ = self.step(None)
            if done:
                break
        print("✅ Step 8A.3 dynamic arrivals test completed.")


if __name__ == "__main__":
    env = ScheduleEnv()
    env.test_dynamic_arrivals()
