"""
environment.py
Step 7A/7B – Dynamic JobAgent Arrivals + WorkCenter–Operator Constraints + Explainable Logs
--------------------------------------------------------------------------------
7A:
  - Dynamic job-agent arrivals
  - WorkCenter (machine) and operator constraints
  - Explainable logs for (WorkCenter, operator) selection

7B (new):
  - Extra info() fields so rollout/replay can handle dynamic arrivals:
      * 'active_agents'      -> currently active JobAgent IDs
      * 'new_records'        -> newly appended Gantt rows since last step
      * 'newly_completed_by' -> list of JobAgent IDs that completed an operation at this step
"""

import simpy
import numpy as np
import gym
from gym import spaces

from utils.site import Sites
from utils.job import Jobs
from utils.task_generator import TaskGenerator
from utils.plane import Plane
from utils.operator import Operators
from MARL.common.terms import t


class ScheduleEnv(gym.Env):
    environment_name = "MASA-SimPy-Scheduler"

    def __init__(
        self,
        start_planes: int = 4,
        max_planes: int = 12,
        arrival_prob: float = 0.20,
        variable_ops: bool = True,
        seed: int = 123,
    ):
        # --- Config (Step 7A) ---
        self.start_planes = int(start_planes)
        self.max_planes = int(max_planes)
        self.arrival_prob = float(arrival_prob)
        self.variable_ops = bool(variable_ops)
        self.rng = np.random.default_rng(seed)

        # --- Basic structures (init placeholders) ---
        self.sites = []
        self.jobs = []
        self.planes = []
        self.job_record_for_gant = []
        self.done = False
        self.step_count = 0

        # For 7B
        self._completed_prev = 0

        # --- Simulation world ---
        self.sim_env = simpy.Environment()
        self.site_resources = []

        # --- Generators & Operators ---
        self.task_gen = TaskGenerator()
        self.operators = None

        # --- Gym interface ---
        self.action_space = spaces.Discrete(21)

        # --- Initialize world ---
        self.initialize()

    # ============================================================
    # === Pretty Logs / Explainability ===========================
    # ============================================================

    def print_site_job_map(self):
        print(f"\n📋 {t('site').upper()}–{t('job').upper()} MAPPING")
        for site in self.sites:
            print(f"   {t('site')} {site.site_id:02d} → {t('job')}s {site.resource_ids_list}")

    def print_operator_site_map(self):
        print(f"\n📋 {t('operator').upper()}–{t('site').upper()} MAPPING")
        for op in self.operators.operators_object_list:
            print(f"   {t('operator')} {op.operator_id} → {t('site')}s {op.qualified_sites}")

    # ============================================================
    # === Helpers for Step 7B ===================================
    # ============================================================

    def get_active_agents(self):
        """Return list of active JobAgent IDs."""
        return [p.plane_id for p in self.planes if p.is_active]

    def get_num_planes(self):
        return len(self.planes)

    # ============================================================
    # === Candidate discovery ====================================
    # ============================================================

    def _candidate_pairs_for_job(self, job_id):
        """
        Compute feasible (WorkCenter_id, operator_id) pairs for an operation:
        - WorkCenter must allow the operation
        - WorkCenter must be free
        - Operator must be qualified for that WorkCenter and not busy
        """
        candidates = []
        valid_sites = []
        for site_id, site in enumerate(self.sites):
            if job_id in site.resource_ids_list:
                valid_sites.append(site_id)
                if len(self.site_resources[site_id].users) == 0:
                    op = self.operators.find_free_operator(job_id, site_id)
                    if op is not None:
                        candidates.append((site_id, op.operator_id))
        return valid_sites, candidates

    # ============================================================
    # === JobAgent process =======================================
    # ============================================================

    def plane_process(self, plane_id: int):
        plane = self.planes[plane_id]

        # Respect dynamic arrival
        if plane.arrival_time > self.sim_env.now:
            yield self.sim_env.timeout(plane.arrival_time - self.sim_env.now)
        plane.is_active = True
        print(f"[t={self.sim_env.now}] {t('plane')} {plane_id} entered the system.")

        while plane.left_job:
            current_job_id = (
                plane.left_job[0].index_id
                if hasattr(plane.left_job[0], "index_id")
                else plane.left_job[0]
            )

            valid_sites, candidate_pairs = self._candidate_pairs_for_job(current_job_id)
            print(f"\n🛠️ [{t('plane')} {plane_id}] Next {t('job')}={current_job_id}")
            print(f"   Step 1: Valid {t('site')}s (can perform {t('job')} {current_job_id}): {valid_sites}")

            if len(valid_sites) > 0:
                print(f"   Step 2: {t('operator').capitalize()} availability per {t('site')}:")
                for s_id in valid_sites:
                    can_ops = []
                    for op in self.operators.operators_object_list:
                        if (not op.is_busy) and op.can_do_job(current_job_id, s_id):
                            can_ops.append(op.operator_id)
                    print(f"       {t('site')} {s_id} → {t('operator')}s {can_ops if can_ops else '[]'}")

            if not candidate_pairs:
                print(f"[WAIT] No available {t('site')}/{t('operator')} for {t('job')} {current_job_id} at t={self.sim_env.now}.")
                yield self.sim_env.timeout(1)
                continue

            print(f"   Step 3: Feasible ({t('site')}, {t('operator')}) candidates → {candidate_pairs}")
            chosen_site_id, chosen_op_id = self.rng.choice(candidate_pairs)
            print(f"✅ Decision: choose {t('site')} {chosen_site_id} with {t('operator')} {chosen_op_id}")

            site_resource = self.site_resources[chosen_site_id]
            operator = [op for op in self.operators.operators_object_list if op.operator_id == chosen_op_id][0]

            with site_resource.request() as req:
                yield req
                operator.assign_job(current_job_id, chosen_site_id)

                start_time = self.sim_env.now
                proc_time = self.jobs[current_job_id].time_span
                yield self.sim_env.timeout(proc_time)
                end_time = self.sim_env.now

                operator.release()
                self.save_env_info((start_time, end_time, current_job_id, chosen_site_id, plane_id, operator.operator_id))
                plane.left_job.pop(0)
                print(
                    f"[t={self.sim_env.now}] {t('plane')} {plane_id} finished {t('job')} {current_job_id} "
                    f"at {t('site')} {chosen_site_id} by {t('operator')} {operator.operator_id}"
                )

        print(f"[t={self.sim_env.now}] {t('plane')} {plane_id} completed all {t('job')}s.")
        plane.is_active = False

    def save_env_info(self, record):
        """Append (start, end, operation, WorkCenter, JobAgent, operator)."""
        self.job_record_for_gant.append(record)

    # ============================================================
    # === Initialization =========================================
    # ============================================================

    def _create_initial_planes(self):
        for pid in range(self.start_planes):
            task_objs = self.task_gen.generate_constrained_task(plane_id=pid)
            plane = Plane(plane_id=pid, job_object_list=task_objs, arrival_time=0)
            self.planes.append(plane)
            self.sim_env.process(self.plane_process(pid))

    def initialize(self):
        sites_obj = Sites()
        jobs_obj = Jobs()

        self.sites = sites_obj.sites_object_list
        self.jobs = jobs_obj.jobs_object_list

        self.sim_env = simpy.Environment()
        self.site_resources = [simpy.Resource(self.sim_env, capacity=1) for _ in range(len(self.sites))]

        self.planes = []
        self.job_record_for_gant = []
        self.done = False
        self._completed_prev = 0
        self.step_count = 0

        self.operators = Operators(sites_obj)
        self.operators.release_all()

        self._create_initial_planes()

        print(f"\n=== Step 7A: Dynamic Arrivals + {t('operator').capitalize()} Constraints Test ===")
        self.print_site_job_map()
        self.print_operator_site_map()
        print(
            f"\n[INIT] Step 7A env with {self.start_planes} initial {t('plane')}s; "
            f"max_{t('plane')}s={self.max_planes}."
        )

    # ============================================================
    # === Dynamic Arrivals =======================================
    # ============================================================

    def can_spawn_more(self) -> bool:
        return len(self.planes) < self.max_planes

    def add_new_plane(self, time_now: float):
        if not self.can_spawn_more():
            return False
        pid = len(self.planes)
        task_objs = self.task_gen.generate_constrained_task(plane_id=pid)
        plane = Plane(plane_id=pid, job_object_list=task_objs, arrival_time=time_now)
        self.planes.append(plane)
        self.sim_env.process(self.plane_process(pid))
        print(f"[ARRIVAL t={time_now}] New {t('plane')} {pid} spawned with {len(task_objs)} ops.")
        return True

    def maybe_spawn(self):
        if self.can_spawn_more() and self.rng.random() < self.arrival_prob:
            self.add_new_plane(self.sim_env.now)

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
        return all((not p.is_active) or (len(p.left_job) == 0) for p in self.planes)

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
            print(f"✅ All {t('plane')}s finished at SimPy time = {self.sim_env.now}")

        info = {
            "time": self.sim_env.now,
            "completed_jobs": len(self.job_record_for_gant),
            "episodes_situation": list(self.job_record_for_gant),
            f"n_{t('plane')}s": len(self.planes),
            "active_agents": self.get_active_agents(),
            "new_records": list(new_records),
            "newly_completed_by": list(newly_completed_by),
        }

        print(
            f"[DEBUG] Step={self.step_count} | t={self.sim_env.now} | "
            f"completed={len(self.job_record_for_gant)} | {t('plane')}s={len(self.planes)}"
        )
        return reward, self.done, info

    def reset(self):
        self.initialize()
        return np.zeros(10)

    def get_env_info(self):
        return {
            "n_agents": len(self.planes),
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
        print("✅ Step 7A/7B dynamic arrivals test completed.")


if __name__ == "__main__":
    env = ScheduleEnv()
    env.test_dynamic_arrivals()
