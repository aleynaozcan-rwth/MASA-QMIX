"""
This file describes the content of a Task (a predefined operation sequence).
"""

from utils.job import Jobs


# -------------------------------------------------------------------------
# NOTE:
# This class defines a simple task sequence for a plane.
# - self.simple_task: a list of job IDs representing the order of operations.
#   Example: [5, 4, 2, 3, 8, 6, 7, 1, 0] means the plane must perform jobs
#   in that exact order.
# - The original comment states that, if executed optimally, this sequence
#   can be completed in 74 time units.
# - self.simple_task_object: the same sequence, but instead of raw IDs it stores
#   the full Job objects (with id, code, name, and time span).
#
# Reminder:
# In this project, each "job" type corresponds to a required service/resource.
# So a job ID also acts as a resource ID elsewhere in the code.
# -------------------------------------------------------------------------
class Task:
    def __init__(self):
        # Currently we only consider the simplest case: a serial (sequential) task plan
        self.simple_task = [5, 4, 2, 3, 8, 6, 7, 1, 0]  # Fastest completion time noted as 74

        # Build the same sequence but with full Job objects (not just IDs)
        jobs = Jobs()
        self.simple_task_object = [jobs.jobs_object_list[e] for e in self.simple_task]

    # Optional helper: get human-readable names of the sequence (useful for logging/debug)
    def simple_task_names(self):
        return [job.name for job in self.simple_task_object]
