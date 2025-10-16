"""
utils/plane.py
Step 7B — Dynamic Arrivals + Replay-Safe Plane State
-----------------------------------------------------
Enhancements vs Step 7A:
- Robust `reset()` method for env reuse.
- `completed_at` timestamp for replay/log credit.
- Safe `mark_completed()` helper.
- Compatible with dynamic arrivals & SimPy-driven scheduling.
"""

from typing import Optional
from utils.task import Task


# ======================================================================
# Container for all planes
# ======================================================================

class Planes:
    """Manages all Plane objects and simple bookkeeping utilities."""

    def __init__(self, numbers=8, dynamic=False):
        self.plane_speed = 20  # legacy, unused
        self.dynamic = dynamic
        self.planes_object_list = []

        task = Task()
        for i in range(numbers):
            temp_object = Plane(i, task.simple_task_object, arrival_time=0.0)
            self.planes_object_list.append(temp_object)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_jobs(self):
        """Return (remaining_jobs, total_jobs) across all planes."""
        left_jobs, all_jobs = 0, 0
        for p in self.planes_object_list:
            left_jobs += len(p.left_job)
            all_jobs += len(p.static_job_list)
        return left_jobs, all_jobs

    def add_new_plane(self, plane):
        """Add a new Plane object (for dynamic arrivals)."""
        self.planes_object_list.append(plane)
        print(f"[Planes] Added new plane {plane.plane_id} at t={plane.arrival_time}, "
              f"{len(plane.left_job)} jobs.")

    def active_plane_ids(self):
        """Return IDs of currently active planes."""
        return [p.plane_id for p in self.planes_object_list if p.is_active]

    def reset_all(self):
        """Reset all planes (for environment re-init)."""
        for p in self.planes_object_list:
            p.reset()


# ======================================================================
# Individual Plane
# ======================================================================

class Plane:
    """
    Represents one aircraft/agent that executes a predefined sequence of jobs.
    Step 7B adds replay-safe state control and timestamps for explainable logs.
    """

    def __init__(self, plane_id, job_object_list, arrival_time: float = 0.0):
        # --- Core state ---
        self.plane_id = plane_id
        self.static_job_list = list(job_object_list)
        self.left_job = [j for j in job_object_list]
        self.finished_job = []
        self.time_spent = 0.0
        self.site_history = []

        # --- Step 7A fields ---
        self.arrival_time = float(arrival_time)
        self.is_active = False

        # --- Step 7B additions ---
        self.completed_at: Optional[float] = None  # SimPy time when plane completed all jobs

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------
    def execute_task(self, job_object, site_object):
        """
        Execute next job in the sequence and update internal state.
        Returns processing time.
        """
        assert job_object.index_id == self.left_job[0].index_id, \
            f"Plane {self.plane_id} mismatch: expected job {self.left_job[0].index_id}, got {job_object.index_id}"

        t_proc = job_object.time_span
        self.time_spent += t_proc
        self.finished_job.append(job_object)
        self.left_job.pop(0)
        self.site_history.append(site_object.site_id)
        return t_proc

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def mark_completed(self, sim_time: Optional[float] = None):
        """Mark plane as completed (for replay/log credit)."""
        self.is_active = False
        self.completed_at = sim_time
        print(f"[Plane] Plane {self.plane_id} completed all jobs at t={sim_time}")

    def reset(self):
        """Reset internal state (used on environment reset)."""
        self.left_job = list(self.static_job_list)
        self.finished_job.clear()
        self.time_spent = 0.0
        self.site_history.clear()
        self.is_active = False
        self.completed_at = None

    # ------------------------------------------------------------------
    # Debug representation
    # ------------------------------------------------------------------
    def __repr__(self):
        status = "active" if self.is_active else "waiting"
        return (f"<Plane id={self.plane_id}, status={status}, "
                f"arrival={self.arrival_time}, left_jobs={len(self.left_job)}, "
                f"completed_at={self.completed_at}>")
