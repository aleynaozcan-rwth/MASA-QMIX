"""
utils/task_generator.py
Step 7A+ – Dynamic Job Creation with Detailed Logging
-----------------------------------------------------
Generates task sequences constrained by Sites and Jobs definitions.
Prints each new job's operations and possible site mappings.
"""

import random
from utils.task import Task
from utils.job import Jobs
from utils.site import Sites


class TaskGenerator:
    """
    Creates random but valid job sequences (operations list) based on site capabilities.
    Step 7A+: Adds logging to show composition and mapping of each generated task.
    """

    def __init__(self):
        self.jobs = Jobs()     # All available operation types (0–8)
        self.sites = Sites()   # All 18 sites with resource_ids_list

        # Each site has its own random speed factor (0.7 – 1.4 range)
        self.machine_speed = {
            s.site_id: random.uniform(0.7, 1.4)
            for s in self.sites.sites_object_list
        }

    # -------------------------------------------------------------
    def generate_constrained_task(self, num_ops=None, plane_id=None):
        """
        Create a job sequence for one plane.
        Each operation is chosen from sites that can perform it.
        """
        if num_ops is None:
            num_ops = random.randint(3, 7)  # moderate length range (3–7 ops)

        ops_sequence = []
        for _ in range(num_ops):
            site = random.choice(self.sites.sites_object_list)
            valid_ops = site.resource_ids_list
            if not valid_ops:
                continue
            op_id = random.choice(valid_ops)
            job_obj = self.jobs.jobs_object_list[op_id]
            speed_factor = self.machine_speed[site.site_id]
            duration = round(job_obj.time_span * speed_factor, 1)
            new_job = type(job_obj)(
                index_id=job_obj.index_id,
                codes=job_obj.codes,
                name=job_obj.name,
                time_span=duration
            )
            ops_sequence.append(new_job)

        # --------- LOGGING (Step 7A Visualization Enhancement) ---------
        print(f"\n[TaskGen] New plane {plane_id if plane_id is not None else '?'} "
              f"generated → {len(ops_sequence)} operations:")
        for job in ops_sequence:
            possible_sites = [
                s.site_id for s in self.sites.sites_object_list
                if job.index_id in s.resource_ids_list
            ]
            print(f"   • {job.name:<12} (JobID={job.index_id}, "
                  f"duration={job.time_span}) → Sites {possible_sites}")
        print("------------------------------------------------------------")

        return ops_sequence
