"""
utils/operator.py
Step 7A — Refined Operator logic
------------------------------------
Operators now only define which sites they can work on.
Job capability is automatically derived from the sites' allowed jobs (site.py).
"""

class Operator:
    """Single operator who can work at specific sites."""
    def __init__(self, operator_id, qualified_sites, sites_ref):
        self.operator_id = operator_id
        self.qualified_sites = qualified_sites
        self.sites_ref = sites_ref  # Reference to Sites() class
        self.is_busy = False
        self.current_job = None

    def can_do_job(self, job_id, site_id):
        """Return True if operator can work at site_id and that site allows job_id."""
        if site_id not in self.qualified_sites:
            return False
        allowed_jobs = self.sites_ref.sites_object_list[site_id].resource_ids_list
        return job_id in allowed_jobs

    def assign_job(self, job_id, site_id):
        """Mark operator as busy and log assignment."""
        self.is_busy = True
        self.current_job = (job_id, site_id)
        print(f"[Operator] Operator {self.operator_id} assigned job {job_id} at site {site_id}")

    def release(self):
        """Free the operator after finishing the job."""
        if self.current_job:
            print(f"[Operator] Operator {self.operator_id} released from job {self.current_job}")
        self.is_busy = False
        self.current_job = None


class Operators:
    """Manages all Operator objects."""
    def __init__(self, sites_ref):
        self.operators_object_list = [
            Operator(0, [0,1,2,3,4,5], sites_ref),
            Operator(1, [6,7,8,9], sites_ref),
            Operator(2, [10,11,12,13], sites_ref),
            Operator(3, [14,15,16,17], sites_ref)
        ]

        print("\n[Init] Operators created:")
        for op in self.operators_object_list:
            print(f"   - Operator {op.operator_id} → sites {op.qualified_sites}")

    def find_free_operator(self, job_id, site_id):
        """Return the first free operator who can perform this job at this site."""
        for op in self.operators_object_list:
            if not op.is_busy and op.can_do_job(job_id, site_id):
                return op
        return None

    def release_all(self):
        """Free all operators (for environment resets)."""
        for op in self.operators_object_list:
            op.release()
