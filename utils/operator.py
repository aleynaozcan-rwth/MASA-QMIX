"""
utils/operator.py
Step 8A — WorkCenter–Operator System (Terminology Unified)
-----------------------------------------------------------
Operators now:
 - Track assignment history (for logging and analysis)
 - Have safe release() even if called redundantly
 - Support dynamic reallocation when JobAgents arrive mid-episode
 - Fully aligned with 8A terminology (WorkCenter / JobAgent)
"""

class Operator:
    """Single operator who can work at specific WorkCenters."""

    def __init__(self, operator_id, qualified_workcenters, workcenters_ref):
        self.operator_id = operator_id
        self.qualified_workcenters = qualified_workcenters
        self.workcenters_ref = workcenters_ref  # Reference to WorkCenters() environment object
        self.is_busy = False
        self.current_job = None
        self.current_workcenter = None
        self.history = []  # (job_id, workcenter_id, start_t, end_t)

    # ============================================================
    # === Capability check =======================================
    # ============================================================

    def can_do_job(self, job_id, workcenter_id):
        """
        Return True if operator can work at the given WorkCenter and that WorkCenter allows this job.
        """
        if workcenter_id not in self.qualified_workcenters:
            return False
        allowed_jobs = self.workcenters_ref.sites_object_list[workcenter_id].resource_ids_list
        return job_id in allowed_jobs

    # ============================================================
    # === Assignment / Release ===================================
    # ============================================================

    def assign_job(self, job_id, workcenter_id, start_time=None):
        """Mark operator as busy and log assignment."""
        self.is_busy = True
        self.current_job = job_id
        self.current_workcenter = workcenter_id
        print(f"[Operator] Operator {self.operator_id} assigned job {job_id} at WorkCenter {workcenter_id}")
        self.history.append({
            "job_id": job_id,
            "workcenter_id": workcenter_id,
            "start_time": start_time,
            "end_time": None
        })

    def release(self, end_time=None):
        """Free the operator after finishing the job. Safe to call multiple times."""
        if self.current_job is not None:
            print(f"[Operator] Operator {self.operator_id} released from job {self.current_job}")
            if self.history and self.history[-1]["end_time"] is None:
                self.history[-1]["end_time"] = end_time
        self.is_busy = False
        self.current_job = None
        self.current_workcenter = None

    # ============================================================
    # === Diagnostics ============================================
    # ============================================================

    def __repr__(self):
        status = "BUSY" if self.is_busy else "FREE"
        return f"Operator(id={self.operator_id}, workcenters={self.qualified_workcenters}, status={status})"


class Operators:
    """Manages all Operator objects."""

    def __init__(self, workcenters_ref):
        self.operators_object_list = [
            Operator(0, [0, 1, 2, 3, 4, 5], workcenters_ref),
            Operator(1, [6, 7, 8, 9], workcenters_ref),
            Operator(2, [10, 11, 12, 13], workcenters_ref),
            Operator(3, [14, 15, 16, 17], workcenters_ref)
        ]

        print("\n[Init] Operators created:")
        for op in self.operators_object_list:
            print(f"   - Operator {op.operator_id} → WorkCenters {op.qualified_workcenters}")

    # ============================================================
    # === Lookup / Utility =======================================
    # ============================================================

    def find_free_operator(self, job_id, workcenter_id):
        """Return the first free operator who can perform this job at this WorkCenter."""
        for op in self.operators_object_list:
            if (not op.is_busy) and op.can_do_job(job_id, workcenter_id):
                return op
        return None

    def release_all(self):
        """Free all operators (for environment resets)."""
        for op in self.operators_object_list:
            op.release()

    def get_busy_summary(self):
        """Return list of (op_id, job_id, workcenter_id) for currently busy operators."""
        busy_ops = []
        for op in self.operators_object_list:
            if op.is_busy:
                busy_ops.append((op.operator_id, op.current_job, op.current_workcenter))
        return busy_ops
