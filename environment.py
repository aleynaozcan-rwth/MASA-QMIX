'''
In this code, "plane" stands for a job, and "site" stands for a station.
This environment schedules planes (jobs) to sites (stations) under resource constraints.
'''

# === Added for Step 2A ===
import simpy
# =========================

from utils.site import Sites
from utils.job import Jobs
from utils.task import Task
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
        # Declare member variables
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

        # Action space: [0..len(sites)-1] for site selection + [wait, busy, finished]
        self.action_space = spaces.Discrete(len(self.sites) + 3)
        self.id = "Boat Schedule"
        self.reward_threshold = -1000
        self.trials = 50
        self.job_record_for_gant = []
        self.sites_state_global = None
        self.state4marl = None
        self.obs4marl = None

        # === Added for Step 2A: create SimPy environment ===
        self.sim_env = simpy.Environment()
        # ====================================================

        # === Added for Step 2B: SimPy site resources (capacity = 1 per site) ===
        self.site_resources = []
        # =======================================================================


    # === Added for Step 2A ===
    def clock(self, until_time):
        """A simple SimPy clock process that runs until the specified time."""
        while True:
            yield self.sim_env.timeout(1)
            if self.sim_env.now >= until_time:
                break

    def run_simpy(self, until_time=10):
        """Run the SimPy environment to test event scheduling."""
        self.sim_env.process(self.clock(until_time))
        self.sim_env.run()
        print(f"✅ SimPy environment ran successfully until t={until_time}")
    # =========================


    # === Added for Step 2B: Event-driven processes ===
    def find_site_for_job(self, job_id):
        """Find an idle site that can handle the given job."""
        if not self.site_resources or len(self.site_resources) != len(self.sites):
            return None
        for idx, site in enumerate(self.sites):
            if job_id in site.resource_ids_list and len(self.site_resources[idx].users) == 0:
                return idx
        return None

    def plane_process(self, plane_id):
        """SimPy process representing one plane executing its sequence of jobs."""
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

    def run_processes(self):
        """Start all plane processes and run the full simulation."""
        self.site_resources = [simpy.Resource(self.sim_env, capacity=1) for _ in range(len(self.sites))]
        for pid in range(len(self.planes)):
            self.sim_env.process(self.plane_process(pid))
        self.sim_env.run()
        print("✅ All plane processes completed successfully.")
    # ===================================================


    def initialize(self):
        sites_obj = Sites()
        self.sites_obj = sites_obj
        jobs_obj = Jobs()
        task_obj = Task()
        self.planes_obj = Planes()
        self.sites = sites_obj.sites_object_list
        self.jobs = jobs_obj.jobs_object_list
        self.task = task_obj.simple_task_object
        self.planes = self.planes_obj.planes_object_list

        self.state = [[9, [1 if j in self.sites[i].resource_ids_list else 0 for j in range(9)]]
                      for i in range(len(self.sites))]
        self.sites_state_global = [-1 for i in range(len(self.sites))]
        self.job_record_for_gant = []
        self.done = False
        self.state_left_time = np.array([0 for i in range(len(self.sites))])
        self.episode_time_slice = []
        self.plane_speed = self.planes_obj.plane_speed
        self.obs4marl = [[] for i in range(len(self.planes))]
        self.current_finishing_jobs = 0
        self.step_count = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.initialize()
        info = {"sites": [[[0, 0], self.state[i][0], self.state[i][1]] for i in range(len(self.sites))],
                "planes": [[self.planes[i].left_job[0].index_id,
                            self.jobs[self.planes[i].left_job[0].index_id].time_span,
                            len(self.planes[i].left_job)]
                           if len(self.planes[i].left_job) != 0 else [9, 0, len(self.planes[i].left_job)]
                           for i in range(len(self.planes))],
                "planes_obj": self.planes}
        state = self.conduct_state(info)
        return state


    def conduct_state(self, info):
        res = []
        for eve in info["sites"]:
            res += eve[2]
        self.state4marl = np.array(res)
        return np.array(res)


    def save_env_info(self, job_transition):
        """Keep track of job records for Gantt chart or logs."""
        self.job_record_for_gant.append(job_transition)


    def step(self, action):
        """Simplified RL step — includes SimPy sync check (Step 3A)."""
        self.step_count += 1

        # Dummy reward & info (placeholder)
        reward = -240
        self.episode_time_slice.append(1)

        # === Step 3A: SimPy pilot sync test (Fixed) ===
        if hasattr(self, "sim_env"):
            # Start the background clock only once
            if not hasattr(self, "_clock_started"):
                self.sim_env.process(self.clock(999))  # long-running background clock
                self._clock_started = True
            self.sim_env.step()
            print(f"[DEBUG] SimPy time after RL step: {self.sim_env.now}")
        # ==============================================

        done = False
        info = {"time": sum(self.episode_time_slice)}
        return reward, done, info


if __name__ == "__main__":
    env = ScheduleEnv()
    env.reset()
    print("Starting Pilot SimPy–RL Sync Test")
    for i in range(5):
        dummy_action = [len(env.sites)] * len(env.planes)
        reward, done, info = env.step(dummy_action)
        print(f"Step {i} | Reward: {reward:.2f} | SimPy time: {env.sim_env.now}")
    print("✅ Pilot integration test completed.")
