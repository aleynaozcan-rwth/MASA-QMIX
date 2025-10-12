"""
ScheduleEnv — Step 5A
Extend SimPy to all jobs and make environment RL-compatible.
"""

import simpy
from utils.site import Sites
from utils.job import Jobs
from utils.task import Task
from utils.plane import Planes, Plane
import numpy as np
import gym
from gym import spaces


class ScheduleEnv(gym.Env):
    environment_name = "Boat Schedule"

    def __init__(self):
        # --- Basic initialization ---
        self.sites = []
        self.jobs = []
        self.task = []
        self.planes_obj = Planes()
        self.planes = []
        self.done = False
        self.state_left_time = []
        self.step_count = 0
        self._completed_prev = 0
        self.initialize()

        # --- Gym settings ---
        self.action_space = spaces.Discrete(len(self.sites) + 3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # --- SimPy setup ---
        self.sim_env = simpy.Environment()
        self.site_resources = []

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
            site_id = self.find_site_for_job(current_job)

            if site_id is None:
                yield self.sim_env.timeout(1)
                continue

            site_resource = self.site_resources[site_id]
            with site_resource.request() as req:
                yield req
                start_time = self.sim_env.now
                job_obj = self.jobs[current_job]
                yield self.sim_env.timeout(job_obj.time_span)
                end_time = self.sim_env.now

                self.save_env_info((start_time, end_time, current_job, site_id, plane_id))
                plane.left_job.pop(0)
                print(f"[t={self.sim_env.now}] Plane {plane_id} finished job {current_job} at site {site_id}")

        print(f"[t={self.sim_env.now}] Plane {plane_id} completed all jobs.")

    def save_env_info(self, job_transition):
        """Record job transitions for Gantt analysis."""
        self.job_record_for_gant.append(job_transition)

    # === Initialization ===
    def initialize(self):
        sites_obj = Sites()
        self.sites_obj = sites_obj
        jobs_obj = Jobs()
        task_obj = Task()
        self.planes_obj = Planes()

        self.sites = sites_obj.sites_object_list
        self.jobs = jobs_obj.jobs_object_list
        self.task = task_obj.simple_task_object
        self.planes = [Plane(pid, self.task[pid]) for pid in range(len(self.task))]

        self.job_record_for_gant = []
        self.done = False
        self.step_count = 0
        self._completed_prev = 0
        print(f"[INIT] Environment initialized with {len(self.planes)} planes and per-plane task sequences.")

    def reset(self):
        self.initialize()
        return self.get_obs()

    # === RL information interface (for MARL integration) ===
    def get_env_info(self):
        return dict(
            n_agents=len(self.planes),
            n_actions=self.action_space.n,
            state_shape=10,
            obs_shape=10,
            episode_limit=200
        )

    # === Observation helpers ===
    def get_obs(self):
        """Per-agent observation: number of remaining jobs."""
        return [len(p.left_job) / 9.0 for p in self.planes]  # normalized 0-1

    def get_state(self):
        """Global state: remaining jobs per plane."""
        return np.array([len(p.left_job) for p in self.planes])

    # === Step logic (Step 5A RL-compatible) ===
    def step(self, actions):
        """
        Perform one RL step and advance SimPy environment for all planes.
        Returns: obs, reward, done, info
        """
        self.step_count += 1
        self.done = False

        # --- Start all planes on first step ---
        if self.step_count == 1:
            self.site_resources = [simpy.Resource(self.sim_env, capacity=1)
                                   for _ in range(len(self.sites))]
            for pid in range(len(self.planes)):
                self.sim_env.process(self.plane_process(pid))

        # --- Advance SimPy time ---
        if len(self.sim_env._queue) > 0:
            next_time = self.sim_env._queue[0][0]
            if self.sim_env.now < next_time:
                self.sim_env.step()
            while len(self.sim_env._queue) > 0 and self.sim_env._queue[0][0] == self.sim_env.now:
                self.sim_env.step()

        # --- Compute reward ---
        completed_now = len(self.job_record_for_gant) - self._completed_prev
        self._completed_prev = len(self.job_record_for_gant)
        reward = completed_now * 10 - 1  # baseline penalty

        # --- Observation + state ---
        obs = self.get_obs()

        # --- Check termination ---
        done = self.all_jobs_completed()
        if done:
            self.done = True
            print(f"✅ All planes finished at SimPy time={self.sim_env.now}")

        info = dict(time=self.sim_env.now,
                    step=self.step_count,
                    completed=len(self.job_record_for_gant))
        return obs, reward, done, info

    def all_jobs_completed(self):
        return all(len(p.left_job) == 0 for p in self.planes)
