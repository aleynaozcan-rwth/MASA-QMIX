"""
utils/operator.py
Step 6A – Introduce Operator entity
------------------------------------
Operators represent human resources who must also be available
for a job to start, in addition to a free site.
"""

class Operator:
    """Single operator with skill qualifications."""
    def __init__(self, operator_id, qualified_jobs):
        self.operator_id = operator_id
        self.qualified_jobs = qualified_jobs      # list of job IDs the operator can perform
        self.is_busy = False
        self.current_job = None

    def assign_job(self, job_id):
        """Mark operator as busy with a specific job."""
        self.is_busy = True
        self.current_job = job_id

    def release(self):
        """Free the operator after finishing the job."""
        self.is_busy = False
        self.current_job = None


class Operators:
    """Container that manages a group of Operator objects."""
    def __init__(self, num_operators=4, job_range=list(range(9))):
        # Example: 4 operators, all qualified for all 9 job types
        self.operators_object_list = [
            Operator(i, job_range) for i in range(num_operators)
        ]

    def find_free_operator(self, job_id):
        """Return the first free operator who can perform this job."""
        for op in self.operators_object_list:
            if (job_id in op.qualified_jobs) and not op.is_busy:
                return op
        return None
