"""
utils/operator.py
Step 7B — Dynamic arrivals + Explainable operator state
--------------------------------------------------------
Operators now:
 - Track assignment history (for logging/analysis)
 - Have safe release() even if called redundantly
 - Support dynamic reallocation when planes arrive mid-episode
"""

class Operator:
    """Single operator who can work at specific sites."""

    def __init__(self, operator_id, qualified_sites, sites_ref):
        self.operator_id = operator_id
        self.qualified_sites = qualified_sites
        self.sites_ref = sites_ref  # Reference to Sites() class
        self.is_busy = False
        self.current_job = None
        self.current_site = None
        self.history = []  # (job_id, site_id, start_t, end_t)

    # ============================================================
    # === Capability check =======================================
    # ============================================================

    def can_do_job(self, job_id, site_id):
        """
        Return True if operator can work at site_id and that site allows job_id.
        """
        if site_id not in self.qualified_sites:
            return False
        allowed_jobs = self.sites_ref.sites_object_list[site_id].resource_ids_list
        return job_id in allowed_jobs

    # ============================================================
    # === Assignment / Release ===================================
    # ============================================================

    def assign_job(self, job_id, site_id, start_time=None):
        """Mark operator as busy and log assignment."""
        self.is_busy = True
        self.current_job = job_id
        self.current_site = site_id
        print(f"[Operator] Operator {self.operator_id} assigned job {job_id} at site {site_id}")
        # Store partial history (we’ll update end_time at release)
        self.history.append({
            "job_id": job_id,
            "site_id": site_id,
            "start_time": start_time,
            "end_time": None
        })

    def release(self, end_time=None):
        """
        Free the operator after finishing the job.
        Safe to call multiple times (idempotent).
        """
        if self.current_job is not None:
            print(f"[Operator] Operator {self.operator_id} released from job {self.current_job}")
            # Update last history record if it exists
            if self.history and self.history[-1]["end_time"] is None:
                self.history[-1]["end_time"] = end_time
        self.is_busy = False
        self.current_job = None
        self.current_site = None

    # ============================================================
    # === Diagnostics ============================================
    # ============================================================

    def __repr__(self):
        status = "BUSY" if self.is_busy else "FREE"
        return f"Operator(id={self.operator_id}, sites={self.qualified_sites}, status={status})"


class Operators:
    """Manages all Operator objects."""

    def __init__(self, sites_ref):
        self.operators_object_list = [
            Operator(0, [0, 1, 2, 3, 4, 5], sites_ref),
            Operator(1, [6, 7, 8, 9], sites_ref),
            Operator(2, [10, 11, 12, 13], sites_ref),
            Operator(3, [14, 15, 16, 17], sites_ref)
        ]

        print("\n[Init] Operators created:")
        for op in self.operators_object_list:
            print(f"   - Operator {op.operator_id} → sites {op.qualified_sites}")

    # ============================================================
    # === Lookup / Utility =======================================
    # ============================================================

    def find_free_operator(self, job_id, site_id):
        """Return the first free operator who can perform this job at this site."""
        for op in self.operators_object_list:
            if (not op.is_busy) and op.can_do_job(job_id, site_id):
                return op
        return None

    def release_all(self):
        """Free all operators (for environment resets)."""
        for op in self.operators_object_list:
            op.release()

    def get_busy_summary(self):
        """Return list of (op_id, job_id, site_id) for currently busy operators."""
        busy_ops = []
        for op in self.operators_object_list:
            if op.is_busy:
                busy_ops.append((op.operator_id, op.current_job, op.current_site))
        return busy_ops
