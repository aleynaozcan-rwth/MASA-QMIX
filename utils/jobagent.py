"""
utils/jobagent.py
Step 8A.5.3 — Machine-level Decision Integration
------------------------------------------------
Enhancements vs Step 8A.3:
- JobAgent can now choose and record actions at the Machine level.
- Each JobAgent tracks its last chosen Machine ID and speed factor.
- execute_task() accepts (machine_id, workcenter_id, speed_factor).
- Machine and WorkCenter histories stored separately for explainable logs.
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
    Represents one JobAgent executing a predefined sequence of jobs.
    Step 8A.5.3 adds machine-level decision tracking and explainable logs.
    """

    def __init__(self, agent_id, job_object_list, arrival_time: float = 0.0):
        # --- Core state ---
        self.agent_id = agent_id
        self.static_job_list = list(job_object_list)
        self.left_job = [j for j in job_object_list]
        self.finished_job = []
        self.time_spent = 0.0

        # --- Decision tracking ---
        self.last_chosen_machine: Optional[str] = None
        self.machine_history = []     # Stores chosen machine IDs
        self.workcenter_history = []  # Still stored for compatibility
        self.speed_factor_history = []  # For performance explainability

        # --- Arrival / activity control ---
        self.arrival_time = float(arrival_time)
        self.is_active = False

        # --- Replay/log fields ---
        self.completed_at: Optional[float] = None  # SimPy time when completed

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------
    def execute_task(self, job_object, machine_id, workcenter_id=None, speed_factor: float = 1.0):
        """
        Execute next job and update internal state.
        Machine-level integration (Step 8A.5.3):
          - Records selected machine and speed factor.
          - Returns actual processing time after speed adjustment.
        """
        assert job_object.index_id == self.left_job[0].index_id, \
            f"JobAgent {self.agent_id} mismatch: expected job {self.left_job[0].index_id}, got {job_object.index_id}"

        base_time = job_object.time_span
        adjusted_time = base_time * (1.0 / speed_factor)

        # Update internal metrics
        self.time_spent += adjusted_time
        self.finished_job.append(job_object)
        self.left_job.pop(0)
        self.last_chosen_machine = machine_id
        self.machine_history.append(machine_id)
        if workcenter_id is not None:
            self.workcenter_history.append(workcenter_id)
        self.speed_factor_history.append(speed_factor)

        print(f"[JobAgent {self.agent_id}] Executed {job_object.index_id} "
              f"on Machine {machine_id} (WC {workcenter_id}) "
              f"speed×{speed_factor} | duration={adjusted_time:.2f}")

        return adjusted_time

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
        self.machine_history.clear()
        self.workcenter_history.clear()
        self.speed_factor_history.clear()
        self.last_chosen_machine = None
        self.is_active = False
        self.completed_at = None

    # ------------------------------------------------------------------
    # Debug representation
    # ------------------------------------------------------------------
    def __repr__(self):
        status = "active" if self.is_active else "waiting"
        last_m = self.last_chosen_machine if self.last_chosen_machine else "-"
        return (f"<JobAgent id={self.agent_id}, status={status}, "
                f"last_machine={last_m}, left_jobs={len(self.left_job)}, "
                f"arrival={self.arrival_time}, completed_at={self.completed_at}>")
