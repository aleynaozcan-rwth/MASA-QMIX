"""
Plane class
Contains: plane id, complete static task list, finished task list, remaining task list, current time spent.

Step 7A additions:
- Each Plane now has an `arrival_time` (float) → determines when the plane enters the simulation.
- Each Plane has an `is_active` flag → becomes True once its SimPy process starts (after its arrival time).
- Fully backward-compatible with previous plane/task logic.
"""
from utils.task import Task


class Planes:
    """
    Container for all Plane objects.
    Responsible for initialization and simple bookkeeping utilities
    such as counting remaining jobs across all planes.
    """

    def __init__(self, numbers=8, dynamic=False):
        """
        numbers : int
            Initial number of planes to create.
        dynamic : bool
            If True, allows additional planes to be added later (Step 7A dynamic arrivals).
        """
        # Plane speed retained only for backward compatibility (not used since Step 1B)
        self.plane_speed = 20
        self.dynamic = dynamic
        self.planes_object_list = []

        # Create initial planes
        task = Task()
        for i in range(numbers):
            temp_object = Plane(i, task.simple_task_object)
            self.planes_object_list.append(temp_object)

    # ----------------------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------------------
    def count_jobs(self):
        """Return (remaining_jobs, total_jobs) across all planes."""
        left_jobs = 0
        all_jobs = 0
        for eve in self.planes_object_list:
            left_jobs += len(eve.left_job)
            all_jobs += len(eve.static_job_list)
        return left_jobs, all_jobs

    def add_new_plane(self, plane):
        """Add a new Plane object (for dynamic arrivals)."""
        self.planes_object_list.append(plane)
        print(f"[Planes] Added new plane {plane.plane_id} at t={plane.arrival_time}, "
              f"{len(plane.left_job)} jobs.")


# ======================================================================
# Individual Plane definition
# ======================================================================

class Plane:
    """
    Represents one aircraft/agent that executes a predefined sequence of jobs.
    The plane keeps track of its completed and remaining jobs, cumulative
    time spent, and the sites it visited.

    Step 7A introduces:
    - `arrival_time` → simulation time when the plane appears.
    - `is_active` → True once its SimPy process starts.
    """

    def __init__(self, plane_id, job_object_list, arrival_time: float = 0.0):
        # -------- Agent STATE --------
        self.plane_id = plane_id                    # Unique agent ID
        self.static_job_list = list(job_object_list) # Immutable reference list
        self.left_job = [eve for eve in job_object_list]  # Remaining jobs
        self.finished_job = []                      # Completed jobs
        self.time_spent = 0                         # Accumulated processing time
        self.site_history = []                      # Ordered list of visited site IDs

        # -------- Step 7A additions --------
        self.arrival_time = float(arrival_time)     # Time of appearance in SimPy env
        self.is_active = False                      # False until its process starts
        # -----------------------------------

    # ------------------------------------------------------------------
    # Agent ACTION
    # ------------------------------------------------------------------
    def execute_task(self, job_object, site_object):
        """
        Execute the next job in the sequence.
        Parameters
        ----------
        job_object : Job
            The job to execute (must match the first in left_job list).
        site_object : Site
            The site (machine) where the job is executed.
        Returns
        -------
        time : float
            Processing time for this job.
        """
        # Sanity check
        assert job_object.index_id == self.left_job[0].index_id, \
            f"Plane {self.plane_id} mismatch: expected job {self.left_job[0].index_id}, got {job_object.index_id}"

        # --- Update plane state ---
        time = job_object.time_span
        self.time_spent += time
        self.finished_job.append(job_object)
        self.left_job.pop(0)                        # Remove from queue
        self.site_history.append(site_object.site_id)

        # (Spatial tracking removed since Step 1B)
        return time

    # ------------------------------------------------------------------
    # Helper methods (for debug/logging)
    # ------------------------------------------------------------------
    def __repr__(self):
        status = "active" if self.is_active else "waiting"
        return (f"<Plane id={self.plane_id}, status={status}, "
                f"arrival={self.arrival_time}, left_jobs={len(self.left_job)}>")
