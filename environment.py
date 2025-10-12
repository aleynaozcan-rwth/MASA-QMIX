"""
environment.py
------------------------------------------------------------
Environment for MASA-QMIX Project
Step 6B — Modify Action Space (machine + operator pairs)
------------------------------------------------------------
Each plane (agent) executes its own job sequence.
SimPy controls time and concurrency.
Now the RL agent’s action space is composed of all valid
(machine/site, operator) combinations.
------------------------------------------------------------
"""

import simpy
import numpy as np
import gym
from gym import spaces
from utils.site import Sites
from utils.job import Jobs
from utils.task import Task
from utils.plane import Planes, Plane
from utils.operator import Operators


class ScheduleEnv(gym.Env):
    environment_name = "MASA-SimPy-Scheduler"

    def __init__(self):
        # --- Core environment data ---
        self.sites = []
        self.jobs = []
        self.task = []
        self.planes_obj = Planes()
        self.planes = []
        self.job_record_for_gant = []
        self.done = False
        self.step_count = 0
        self._completed_prev = 0

        # --- Simulation objects ---
        self.sim_env = simpy.Environment()
        self.site_resources = []

        # --- Operators ---
        self.operators_obj = Operators()
        self.operators = self.operators_obj.operators_object_list

        # --- Initialize structures ---
        self.initialize()

        # --- NEW: machine–operator action pairs (Step 6B) ---
        self.valid_action_pairs = self.generate_action_pairs()
        self.action_space = spaces.Discrete(len(self.valid_action_pairs))
        print(f"[INIT] Action space size = {self.action_space.n}")

    # ============================================================
    # === Action-pair generation & utilities =====================
    # ============================================================

    def generate_action_pairs(self):
        """
        Build all valid (site_id, operator_id) pairs.
        A pair is valid if the operator can perform at least one
        of the site's available job types.
        """
        pairs = []
        for site in self.sites:
            for op in self.operators:
                if any(j in op.qualified_jobs for j in site.resource_ids_list):
                    pairs.append((site.site_id, op.operator_id))
        print(f"[INIT] Generated {len(pairs)} valid (site, operator) pairs.")
        return pairs

    def decode_action(self, action_idx):
        """Translate discrete index → (site_id, operator_id)."""
        if 0 <= action_idx < len(self.valid_action_pairs):
            return self.valid_action_pairs[action_idx]
        return None, None

    def get_avail_actions(self):
        """
        Return availability mask (1 = free, 0 = busy) for all action pairs.
        Used by RL to mask invalid actions.
        """
        avail = np.zeros(len(self.valid_action_pairs))
        for i, (site_id, op_id) in enumerate(self.valid_action_pairs):
            site_busy = len(self.site_resources[site_id].users) > 0
            op_busy = self.operators[op_id].is_busy
            if not site_busy and not op_busy:
                avail[i] = 1
        return avail

    # ============================================================
    # === SimPy Processes ========================================
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
        """Each plane sequentially executes its list of jobs."""
        plane = self.planes[plane_id]

        while plane.left_job:
            current_job = plane.left_job[0]

            # Find available site & operator
            site_id = self.find_site_for_job(current_job)
            if site_id is None:
                yield self.sim_env.timeout(1)
                continue

            operator = self.find_operator_for_job(current_job)
            if operator is None:
                yield self.sim_env.timeout(1)
                continue

            operator.assign_job(current_job)

            # Simulate job execution
            site_resource = self.site_resources[site_id]
            with site_resource.request() as req:
                yield req
                start_time = self.sim_env.now
                job_obj = self.jobs[current_job]
                yield self.sim_env.timeout(job_obj.time_span)
                end_time = self.sim_env.now

                self.save_env_info((start_time, end_time,
                                    current_job, site_id,
                                    plane_id, operator.operator_id))
                plane.left_job.pop(0)
                operator.release()

                print(f"[t={self.sim_env.now}] Plane {plane_id} finished job {current_job} "
                      f"at site {site_id} (Operator {operator.operator_id})")

        print(f"[t={self.sim_env.now}] Plane {plane_id} completed all jobs.")

    def save_env_info(self, record):
        """Store (start, end, job, site, plane, operator) for logging/Gantt."""
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

        # Reset simulation objects
        self.sim_env = simpy.Environment()
        self.site_resources = [simpy.Resource(self.sim_env, capacity=1)
                               for _ in range(len(self.sites))]

        # Recreate operator pool
        self.operators_obj = Operators()
        self.operators = self.operators_obj.operators_object_list

        # Launch plane processes
        for pid in range(len(self.planes)):
            self.sim_env.process(self.plane_process(pid))

        self.job_record_for_gant = []
        self.done = False
        self._completed_prev = 0
        self.step_count = 0
        print(f"[INIT] Environment initialized with {len(self.planes)} planes, "
              f"{len(self.sites)} sites, and {len(self.operators)} operators.")

    # ============================================================
    # === Clock Control (unchanged from 5B) ======================
    # ============================================================

    def advance_clock(self, max_time=None):
        """Advance SimPy clock to next event or until max_time."""
        if len(self.sim_env._queue) == 0:
            return
        next_time = self.sim_env._queue[0][0]
        if max_time and next_time > max_time:
            return
        self.sim_env.step()
        while len(self.sim_env._queue) > 0 and self.sim_env._queue[0][0] == self.sim_env.now:
            self.sim_env.step()

    # ============================================================
    # === Step / Reset / Info ====================================
    # ============================================================

    def all_jobs_completed(self):
        return all(len(p.left_job) == 0 for p in self.planes)

    def step(self, action=None):
        """
        RL step:
        - Advance SimPy clock by one event
        - Compute reward
        - Provide info (including available actions)
        """
        self.step_count += 1
        reward = -1
        self.advance_clock()

        completed_now = len(self.job_record_for_gant) - self._completed_prev
        self._completed_prev = len(self.job_record_for_gant)
        reward += completed_now * 10

        self.done = self.all_jobs_completed()
        avail_actions = self.get_avail_actions()

        info = {
            "time": self.sim_env.now,
            "completed_jobs": len(self.job_record_for_gant),
            "avail_actions": avail_actions,
            "episodes_situation": list(self.job_record_for_gant),
        }

        if self.done:
            print(f"✅ All planes finished at SimPy time = {self.sim_env.now}")

        print(f"[DEBUG] RL step={self.step_count} | SimPy time={self.sim_env.now} | "
              f"completed={len(self.job_record_for_gant)}")
        return reward, self.done, info

    def reset(self):
        self.initialize()
        return np.zeros(10)

    def get_env_info(self):
        """Provide core parameters for MARL framework."""
        return {
            "n_agents": len(self.planes),
            "n_actions": self.action_space.n,
            "state_shape": 10,
            "obs_shape": 10,
            "episode_limit": 200,
        }

    # ============================================================
    # === Stand-alone Test =======================================
    # ============================================================

    def test_coexecution(self):
        print("=== Step 6B Test: SimPy–RL with Machine+Operator Pairs ===")
        self.reset()
        while not self.done:
            _, done, _ = self.step(None)
            if done:
                break
        print("✅ Step 6B simulation completed successfully.")


if __name__ == "__main__":
    env = ScheduleEnv()
    env.test_coexecution()
