"""
environment.py
------------------------------------------------------------
Environment for MASA-QMIX Project
Step 6A — Add Operator Entity
------------------------------------------------------------
Each plane (agent) executes a sequence of jobs at available sites.
SimPy models the process timing; RL controls scheduling decisions.
Now, each job also requires an available human Operator qualified
for that job type.
------------------------------------------------------------
"""

import simpy
from utils.site import Sites
from utils.job import Jobs
from utils.task import Task
from utils.plane import Planes, Plane
from utils.operator import Operators
import numpy as np
import gym
from gym import spaces


class ScheduleEnv(gym.Env):
    environment_name = "MASA-SimPy-Scheduler"

    def __init__(self):
        # --- Core data containers ---
        self.sites = []
        self.jobs = []
        self.task = []
        self.planes_obj = Planes()
        self.planes = []
        self.job_record_for_gant = []
        self.done = False
        self.step_count = 0
        self._completed_prev = 0

        # --- Simulation ---
        self.sim_env = simpy.Environment()
        self.site_resources = []

        # --- NEW: Operators (Step 6A) ---
        self.operators_obj = Operators()
        self.operators = self.operators_obj.operators_object_list

        # --- Gym interface ---
        self.action_space = spaces.Discrete(21)

        # --- Initialize environment ---
        self.initialize()

    # ============================================================
    # === SimPy Process Definitions ===============================
    # ============================================================

    def find_site_for_job(self, job_id):
        """Return an available site that can process the given job."""
        for idx, site in enumerate(self.sites):
            if job_id in site.resource_ids_list and len(self.site_resources[idx].users) == 0:
                return idx
        return None

    def find_operator_for_job(self, job_id):
        """Return a free operator qualified for the given job, or None."""
        for op in self.operators:
            if (job_id in op.qualified_jobs) and not op.is_busy:
                return op
        return None

    def plane_process(self, plane_id):
        """Each plane sequentially processes its job list."""
        plane = self.planes[plane_id]

        while plane.left_job:
            current_job = plane.left_job[0]

            # --- find available site ---
            site_id = self.find_site_for_job(current_job)
            if site_id is None:
                yield self.sim_env.timeout(1)
                continue

            # --- find available operator (Step 6A) ---
            operator = self.find_operator_for_job(current_job)
            if operator is None:
                yield self.sim_env.timeout(1)
                continue

            operator.assign_job(current_job)

            # --- process execution ---
            site_resource = self.site_resources[site_id]
            with site_resource.request() as req:
                yield req
                start_time = self.sim_env.now
                job_obj = self.jobs[current_job]
                process_time = job_obj.time_span
                yield self.sim_env.timeout(process_time)
                end_time = self.sim_env.now

                # save record including operator id
                self.save_env_info((start_time, end_time, current_job, site_id, plane_id, operator.operator_id))

                plane.left_job.pop(0)
                operator.release()

                print(f"[t={self.sim_env.now}] Plane {plane_id} finished job {current_job} "
                      f"at site {site_id} (Operator {operator.operator_id})")

        print(f"[t={self.sim_env.now}] Plane {plane_id} completed all jobs.")

    def save_env_info(self, record):
        """Save a (start, end, job, site, plane, operator) tuple."""
        self.job_record_for_gant.append(record)

    # ============================================================
    # === Initialization =========================================
    # ============================================================

    def initialize(self):
        """Create sites, jobs, tasks, planes, operators, and SimPy resources."""
        sites_obj = Sites()
        jobs_obj = Jobs()
        task_obj = Task()

        self.sites = sites_obj.sites_object_list
        self.jobs = jobs_obj.jobs_object_list
        self.task = task_obj.simple_task_object
        self.planes = [Plane(pid, self.task[pid]) for pid in range(len(self.task))]

        # recreate SimPy environment
        self.sim_env = simpy.Environment()
        self.site_resources = [simpy.Resource(self.sim_env, capacity=1) for _ in range(len(self.sites))]

        # re-initialize operators
        self.operators_obj = Operators()
        self.operators = self.operators_obj.operators_object_list

        # launch plane processes
        for pid in range(len(self.planes)):
            self.sim_env.process(self.plane_process(pid))

        self.job_record_for_gant = []
        self.done = False
        self._completed_prev = 0
        self.step_count = 0
        print(f"[INIT] Environment initialized with {len(self.planes)} planes, "
              f"{len(self.sites)} sites, and {len(self.operators)} operators.")

    # ============================================================
    # === Clock Control (Step 5B) ================================
    # ============================================================

    def advance_clock(self, max_time=None):
        """Advance SimPy clock to the next event or until max_time."""
        if len(self.sim_env._queue) == 0:
            return
        next_time = self.sim_env._queue[0][0]
        if max_time and next_time > max_time:
            return
        self.sim_env.step()
        while len(self.sim_env._queue) > 0 and self.sim_env._queue[0][0] == self.sim_env.now:
            self.sim_env.step()

    # ============================================================
    # === Core Step Logic ========================================
    # ============================================================

    def all_jobs_completed(self):
        return all(len(p.left_job) == 0 for p in self.planes)

    def step(self, action=None):
        """Advance one SimPy tick and compute reward."""
        self.step_count += 1
        reward = -1  # base time penalty
        self.advance_clock()

        completed_now = len(self.job_record_for_gant) - self._completed_prev
        self._completed_prev = len(self.job_record_for_gant)
        reward += completed_now * 10

        self.done = self.all_jobs_completed()
        if self.done:
            print(f"✅ All planes finished at SimPy time = {self.sim_env.now}")

        info = {
            "time": self.sim_env.now,
            "completed_jobs": len(self.job_record_for_gant),
            "episodes_situation": list(self.job_record_for_gant),
        }

        print(f"[DEBUG] RL step={self.step_count} | SimPy time={self.sim_env.now} | "
              f"completed={len(self.job_record_for_gant)}")
        return reward, self.done, info

    def reset(self):
        self.initialize()
        return np.zeros(10)

    # ============================================================
    # === Environment Info (for MARL Framework) ==================
    # ============================================================

    def get_env_info(self):
        """Return core parameters expected by QMIX runner."""
        return {
            "n_agents": len(self.planes),
            "n_actions": self.action_space.n if hasattr(self.action_space, "n") else 21,
            "state_shape": 10,
            "obs_shape": 10,
            "episode_limit": 200,
        }

    # ============================================================
    # === Stand-alone Test =======================================
    # ============================================================

    def test_coexecution(self):
        print("=== Step 6A Test: SimPy–RL with Operators ===")
        self.reset()
        while not self.done:
            _, done, _ = self.step(None)
            if done:
                break
        print("✅ Step 6A simulation completed successfully.")


if __name__ == "__main__":
    env = ScheduleEnv()
    env.test_coexecution()
