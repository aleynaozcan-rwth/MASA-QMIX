"""
utils/jobagent.py
Step 8A.3 — Dynamic Arrivals + Replay-Safe JobAgent State
----------------------------------------------------------
Enhancements vs Step 7B:
- Renamed Plane → JobAgent, Site → WorkCenter
- Robust `reset()` for environment reuse
- `completed_at` timestamp for replay/log credit
- Safe `mark_completed()` helper
- Compatible with dynamic arrivals & SimPy-driven scheduling
"""

from typing import Optional
from utils.task import Task


# ======================================================================
# Container for all JobAgents
# ======================================================================

class JobAgents:
    """Manages all JobAgent objects and simple bookkeeping utilities."""

    def __init__(self, numbers=8, dynamic=False):
        self.dynamic = dynamic
        self.agents_object_list = []

        task = Task()
        for i in range(numbers):
            temp_object = JobAgent(i, task.simple_task_object, arrival_time=0.0)
            self.agents_object_list.append(temp_object)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_jobs(self):
        """Return (remaining_jobs, total_jobs) across all JobAgents."""
        left_jobs, all_jobs = 0, 0
        for a in self.agents_object_list:
            left_jobs += len(a.left_job)
            all_jobs += len(a.static_job_list)
        return left_jobs, all_jobs

    def add_new_agent(self, agent):
        """Add a new JobAgent object (for dynamic arrivals)."""
        self.agents_object_list.append(agent)
        print(f"[JobAgents] Added new JobAgent {agent.agent_id} at t={agent.arrival_time}, "
              f"{len(agent.left_job)} jobs.")

    def active_agent_ids(self):
        """Return IDs of currently active JobAgents."""
        return [a.agent_id for a in self.agents_object_list if a.is_active]

    def reset_all(self):
        """Reset all JobAgents (for environment re-init)."""
        for a in self.agents_object_list:
            a.reset()


# ======================================================================
# Individual JobAgent
# ======================================================================

class JobAgent:
    """
    Represents one JobAgent (former Plane) executing a predefined sequence of jobs.
    Step 8A.3 adds replay-safe state control and timestamps for explainable logs.
    """

    def __init__(self, agent_id, job_object_list, arrival_time: float = 0.0):
        # --- Core state ---
        self.agent_id = agent_id
        self.static_job_list = list(job_object_list)
        self.left_job = [j for j in job_object_list]
        self.finished_job = []
        self.time_spent = 0.0
        self.workcenter_history = []

        # --- Arrival / activity control ---
        self.arrival_time = float(arrival_time)
        self.is_active = False

        # --- Replay/log fields ---
        self.completed_at: Optional[float] = None  # SimPy time when completed

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------
    def execute_task(self, job_object, workcenter_object):
        """
        Execute next job in the sequence and update internal state.
        Returns processing time.
        """
        assert job_object.index_id == self.left_job[0].index_id, \
            f"JobAgent {self.agent_id} mismatch: expected job {self.left_job[0].index_id}, got {job_object.index_id}"

        t_proc = job_object.time_span
        self.time_spent += t_proc
        self.finished_job.append(job_object)
        self.left_job.pop(0)
        self.workcenter_history.append(workcenter_object.site_id)
        return t_proc

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def mark_completed(self, sim_time: Optional[float] = None):
        """Mark JobAgent as completed (for replay/log credit)."""
        self.is_active = False
        self.completed_at = sim_time
        print(f"[JobAgent] JobAgent {self.agent_id} completed all jobs at t={sim_time}")

    def reset(self):
        """Reset internal state (used on environment reset)."""
        self.left_job = list(self.static_job_list)
        self.finished_job.clear()
        self.time_spent = 0.0
        self.workcenter_history.clear()
        self.is_active = False
        self.completed_at = None

    # ------------------------------------------------------------------
    # Debug representation
    # ------------------------------------------------------------------
    def __repr__(self):
        status = "active" if self.is_active else "waiting"
        return (f"<JobAgent id={self.agent_id}, status={status}, "
                f"arrival={self.arrival_time}, left_jobs={len(self.left_job)}, "
                f"completed_at={self.completed_at}>")
