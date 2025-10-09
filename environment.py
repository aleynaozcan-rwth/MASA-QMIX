"""
In this code, "plane" stands for a job, and "site" stands for a station.
This environment schedules planes (jobs) to sites (stations) under resource constraints.
"""

import simpy
from utils.site import Sites
from utils.job import Jobs
from utils.task_generator import TaskGenerator        # ← NEW import
from utils.plane import Planes
from utils import util
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math


class ScheduleEnv(gym.Env):
    environment_name = "Boat Schedule"

    def __init__(self):
        # --- Basic initialization ---
        self.sites = []
        self.jobs = []
        self.task = []
        self.planes_obj = Planes()
        self.planes = []
        self.state = [[]]
        self.done = False
        self.state_left_time = []
        self.episode_time_slice = []
        self.plane_speed = 0
        self.initialize()

        # --- Gym settings ---
        self.action_space = spaces.Discrete(len(self.sites) + 3)
        self.id = "Boat Schedule"
        self.reward_threshold = -1000
        self.trials = 50

        self.job_record_for_gant = []
        self.sites_state_global = None
        self.state4marl = None
        self.obs4marl = None

        # --- SimPy setup ---
        self.sim_env = simpy.Environment()
        self.site_resources = []
        self._completed_prev = 0

    # === SimPy process definitions ===
    def find_site_for_job(self, job_id):
        """Find an idle site compatible with the given job."""
        if not self.site_resources or len(self.site_resources) != len(self.sites):
            return None
        for idx, site in enumerate(self.sites):
            if job_id in site.resource_ids_list and len(self.site_resources[idx].users) == 0:
                return idx
        return None

    def plane_process(self, plane_id):
        """Each plane executes its sequence of jobs as a SimPy process."""
        plane = self.planes[plane_id]
        while len(plane.left_job) > 0:
            current_job = plane.left_job[0]
            site_id = self.find_site_for_job(current_job.index_id)

            if site_id is None:
                yield self.sim_env.timeout(1)
                continue

            site_resource = self.site_resources[site_id]
            with site_resource.request() as req:
                yield req

                start_time = self.sim_env.now
                process_time = plane.execute_task(current_job, self.sites[site_id])
                yield self.sim_env.timeout(process_time)
                end_time = self.sim_env.now

                self.save_env_info((start_time, end_time, current_job.index_id, site_id, plane_id))
                print(f"[t={self.sim_env.now}] Plane {plane_id} finished job {current_job.index_id} at site {site_id}")

        print(f"[t={self.sim_env.now}] Plane {plane_id} completed all jobs.")

    def save_env_info(self, job_transition):
        """Record job transitions for analysis and Gantt visualization."""
        self.job_record_for_gant.append(job_transition)

    # === Initialization ===
    def initialize(self):
        sites_obj = Sites()
        self.sites_obj = sites_obj
        jobs_obj = Jobs()
        task_gen = TaskGenerator()                             # ← use TaskGenerator
        self.planes_obj = Planes()
        self.sites = sites_obj.sites_object_list
        self.jobs = jobs_obj.jobs_object_list
        self.task = task_gen.generate_tasks()                  # ← dynamic but identical task list
        self.planes = self.planes_obj.planes_object_list

        print(f"[INIT] Loaded {len(self.task)} tasks from TaskGenerator.")

        self.state = [
            [9, [1 if j in self.sites[i].resource_ids_list else 0 for j in range(9)]]
            for i in range(len(self.sites))
        ]
        self.sites_state_global = [-1 for _ in range(len(self.sites))]
        self.job_record_for_gant = []
        self.done = False
        self.state_left_time = np.array([0 for _ in range(len(self.sites))])
        self.episode_time_slice = []
        self.plane_speed = self.planes_obj.plane_speed
        self.obs4marl = [[] for _ in range(len(self.planes))]
        self.current_finishing_jobs = 0
        self.step_count = 0
        self._completed_prev = 0

    def reset(self):
        self.initialize()
        state = np.zeros(10)
        return state

    # === Step logic ===
    def all_jobs_completed(self):
        """Return True if all planes have finished all jobs."""
        return all(len(p.left_job) == 0 for p in self.planes)

    def step(self, action):
        """Perform one RL step and advance SimPy environment."""
        self.step_count += 1
        reward = -16  # baseline penalty for time passing
        self.done = False

        # --- Advance SimPy: drain all events that occur at the same timestamp ---
        if hasattr(self, "sim_env") and len(self.sim_env._queue) > 0:
            # SimPy 4.x stores events as tuples (time, priority, eid, event)
            next_time = self.sim_env._queue[0][0]
            if self.sim_env.now < next_time:
                self.sim_env.step()  # advance to next event
            while len(self.sim_env._queue) > 0 and self.sim_env._queue[0][0] == self.sim_env.now:
                self.sim_env.step()
            print(f"[DEBUG] Advanced SimPy → now={self.sim_env.now}")

        # --- Reward: +10 for each new completion since last step ---
        completed_now = len(self.job_record_for_gant) - self._completed_prev
        self._completed_prev = len(self.job_record_for_gant)
        reward += completed_now * 10

        # --- Check completion ---
        if self.all_jobs_completed():
            self.done = True
            print(f"✅ All jobs finished at SimPy time={self.sim_env.now}")

        print(
            f"[DEBUG] RL step={self.step_count} | t={self.sim_env.now} | "
            f"+completed={completed_now} | total={len(self.job_record_for_gant)}"
        )

        return reward, self.done, {}

    # === Co-execution test ===
    def test_coexecution(self, steps=5):
        print("Starting Step 3B Co-Execution Test")
        self.reset()

        # Initialize SimPy resources and start plane processes
        self.site_resources = [simpy.Resource(self.sim_env, capacity=1) for _ in range(len(self.sites))]
        for pid in range(len(self.planes)):
            self.sim_env.process(self.plane_process(pid))

        self.sim_env.run(until=0.01)

        for i in range(steps):
            dummy_action = [len(self.sites)] * len(self.planes)
            reward, done, info = self.step(dummy_action)
            print(
                f"Step {i} | Reward={reward:.2f} | SimPy time={self.sim_env.now} | "
                f"Completed jobs={len(self.job_record_for_gant)}"
            )

            if done:
                print("✅ Environment finished early.")
                break

        print("✅ Step 3B test completed successfully.")


# === Quick local test ===
if __name__ == "__main__":
    env = ScheduleEnv()
    env.test_coexecution()
