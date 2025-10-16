"""
utils/task_generator.py
Step 7B.3 — Safe Dynamic Task Generation (No Repeated Operations)
-----------------------------------------------------------------
Enhancements:
- Guarantees non-empty operation sequences.
- Ensures no operation type repeats within one JobAgent.
- Auto-retries generation up to N times if a WorkCenter has no valid new jobs.
- Logs gracefully even if dynamic spawning occurs mid-simulation.
- Fully backward-compatible with Step 7A.
"""

import random
from utils.task import Task
from utils.job import Jobs
from utils.site import Sites


class TaskGenerator:
    """Creates random but valid job sequences (operations list) based on WorkCenter capabilities."""

    def __init__(self, max_retries: int = 5):
        self.jobs = Jobs()
        self.workcenters = Sites()
        self.max_retries = max_retries

        # Random machine speed scaling factors (for job duration variability)
        self.machine_speed = {
            wc.site_id: random.uniform(0.7, 1.4)
            for wc in self.workcenters.sites_object_list
        }

    # ------------------------------------------------------------------
    def generate_constrained_task(self, num_ops=None, jobagent_id=None):
        """
        Create a random but valid operation list for one JobAgent.
        Ensures:
        - Non-empty valid job sequence
        - No repeated job type within the same JobAgent
        """
        if num_ops is None:
            num_ops = random.randint(3, 7)

        ops_sequence = []
        used_job_ids = set()
        retries = 0

        while len(ops_sequence) < num_ops and retries < self.max_retries * num_ops:
            wc = random.choice(self.workcenters.sites_object_list)

            # Filter WorkCenter operations so none are already used
            valid_ops = [
                j for j in wc.resource_ids_list
                if j not in used_job_ids
            ]

            if not valid_ops:
                retries += 1
                continue

            op_id = random.choice(valid_ops)
            job_obj = self.jobs.jobs_object_list[op_id]

            # Apply machine speed multiplier
            speed_factor = self.machine_speed.get(wc.site_id, 1.0)
            duration = round(max(0.1, job_obj.time_span * speed_factor), 2)

            # Create a copy of the Job object with modified time_span
            new_job = type(job_obj)(
                index_id=job_obj.index_id,
                codes=job_obj.codes,
                name=job_obj.name,
                time_span=duration
            )
            ops_sequence.append(new_job)
            used_job_ids.add(op_id)

        # Fallback safeguard — avoid returning empty task lists
        if not ops_sequence:
            print(f"[WARN] TaskGenerator: Empty ops for JobAgent {jobagent_id}; generating fallback task.")
            default_job = self.jobs.jobs_object_list[0]
            fallback = type(default_job)(
                index_id=default_job.index_id,
                codes=default_job.codes,
                name=default_job.name,
                time_span=default_job.time_span
            )
            ops_sequence = [fallback]

        # -------- Logging (explainable trace) --------
        print(f"\n[TaskGen] JobAgent {jobagent_id if jobagent_id is not None else '?'} "
              f"generated → {len(ops_sequence)} unique operation(s):")
        for job in ops_sequence:
            possible_wcs = [
                wc.site_id for wc in self.workcenters.sites_object_list
                if job.index_id in wc.resource_ids_list
            ]
            print(f"   • {job.name:<12} (JobID={job.index_id}, "
                  f"duration={job.time_span}) → WorkCenters {possible_wcs}")
        print("------------------------------------------------------------")

        return ops_sequence
