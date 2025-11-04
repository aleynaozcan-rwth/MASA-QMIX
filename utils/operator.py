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


# Default configuration mirror for Operators (Phase 3C.0)
# Mirrors YAML keys: 'operators'
# Example operator list with qualified machines and group mapping
# TODO(Phase3C.1): integrate with Operators.__init__() via merge_config(DEFAULT_OPERATORS, cfg)
DEFAULT_OPERATORS = [
    {"id": 1, "qualified_machines": ["M_0_0", "M_1_0"]},
    {"id": 2, "qualified_machines": ["M_0_1", "M_1_0"]},
]

class Operator:
    """Single operator who can work on specific machine names."""

    def __init__(self, operator_id, qualified_machines, workcenters_ref):
        self.operator_id = operator_id
        # qualified_machines: list of machine name strings (e.g., 'M_0_0')
        self.qualified_machines = list(qualified_machines)
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
        # Operator must be able to operate at least one machine inside the workcenter.
        # Prefer direct registry-based lookup for speed and clarity.
        try:
            registry = getattr(self.workcenters_ref, 'machine_registry', {}) or {}
            # collect machines in this workcenter
            machines_in_wc = [m for m, md in registry.items() if int(md.get('workcenter', -1)) == int(workcenter_id)]
            # intersect with operator-qualified machines
            candidate_machines = [m for m in machines_in_wc if m in self.qualified_machines]
            if not candidate_machines:
                return False
            # check capabilities via registry
            for m in candidate_machines:
                caps = registry.get(m, {}).get('capabilities', [])
                if int(job_id) in caps:
                    return True
            return False
        except Exception:
            # conservative fallback: try older WorkCenter object path
            try:
                wc_obj = self.workcenters_ref.workcenters_list[workcenter_id]
                candidate_machines = [m for m in wc_obj.machines.keys() if m in self.qualified_machines]
                if not candidate_machines:
                    return False
                for m in candidate_machines:
                    caps = wc_obj.machines.get(m, {}).get('capabilities', [])
                    if int(job_id) in caps:
                        return True
                return False
            except Exception:
                return False

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
        return f"Operator(id={self.operator_id}, machines={self.qualified_machines}, status={status})"


class Operators:
    """Manages all Operator objects."""

    def __init__(self, workcenters_ref):
        # Attempt to merge in-module DEFAULT_OPERATORS with any provided
        # operator configuration present on the WorkCenters object or its
        # attached config. This fills missing fields while remaining
        # non-destructive to inputs.
        try:
            from utils.config_loader import merge_config
            provided = None
            try:
                provided = getattr(workcenters_ref, 'operators', None)
            except Exception:
                provided = None
            if provided is None:
                try:
                    cfg = getattr(workcenters_ref, 'config', None) or {}
                    provided = cfg.get('operators') if isinstance(cfg, dict) else None
                except Exception:
                    provided = None
            # merge expects dicts; wrap list into {'operators': [...]}
            merged = merge_config({"operators": DEFAULT_OPERATORS}, {"operators": provided} if provided is not None else None)
            merged_operators = merged.get('operators', DEFAULT_OPERATORS)
        except Exception:
            merged_operators = DEFAULT_OPERATORS

        # Default operator qualification mapping (user-specified topology):
        # Operator 1 can work on machine numbers [1,4,5]
        # Operator 2 can work on machine numbers [2,3,5]
        # Map machine numbers to machine names using WorkCenters helper
        # Determine canonical ordered registry keys for machines (1..N) in a
        # way that respects YAML overrides. Preferred strategies in order:
        #  1) If machine_registry keys look like 'M1'..'M5', sort by integer
        #  2) Else, if workcenters_ref.machine_order contains keys present in
        #     machine_registry, use that order
        #  3) Else, build order by iterating workcenters_list and their machines
        registry = getattr(workcenters_ref, 'machine_registry', {}) or {}
        machine_order = []
        if registry:
            # attempt strategy 1: keys like 'M1','M2',... -> sort by digit
            import re
            keys = list(registry.keys())
            numeric_keys = []
            for k in keys:
                m = re.match(r'^M(\d+)$', k)
                if m:
                    numeric_keys.append((int(m.group(1)), k))
            if numeric_keys:
                numeric_keys.sort()
                machine_order = [k for (_, k) in numeric_keys]

            # strategy 2: use workcenters_ref.machine_order if it maps into registry
            if not machine_order:
                possible = getattr(workcenters_ref, 'machine_order', []) or []
                if all(p in registry for p in possible):
                    machine_order = list(possible)

            # strategy 3: flatten by iterating workcenters_list
            if not machine_order:
                mo = []
                for wc in getattr(workcenters_ref, 'workcenters_list', []):
                    for mname in wc.machines.keys():
                        if mname in registry:
                            mo.append(mname)
                # dedupe while preserving order
                seen = set()
                machine_order = [x for x in mo if not (x in seen or seen.add(x))]

        # Now pick machine names for numbers 1..5 safely
        def machine_for_num(n, fallback=None):
            try:
                return machine_order[n - 1]
            except Exception:
                return fallback

        # fallback default names if we couldn't build an order
        fallback_names = ["M_0_0", "M_0_1", "M_1_0", "M_1_1", "M_2_0"]
        m1 = machine_for_num(1, fallback_names[0])
        m2 = machine_for_num(2, fallback_names[1])
        m3 = machine_for_num(3, fallback_names[2])
        m4 = machine_for_num(4, fallback_names[3])
        m5 = machine_for_num(5, fallback_names[4])

        # operator-machine mapping (preserve machine-level constraints)
        op1_machines = [m1, m4, m5]
        op2_machines = [m2, m3, m5]

        # If an operators config was provided, prefer it to construct
        # Operator objects; otherwise, fall back to default mapping.
        try:
            if isinstance(merged_operators, (list, tuple)) and len(merged_operators) > 0:
                objs = []
                for entry in merged_operators:
                    try:
                        oid = entry.get('id', None) if isinstance(entry, dict) else None
                        q = entry.get('qualified_machines', []) if isinstance(entry, dict) else []
                        if oid is None:
                            # try to infer id from position
                            oid = len(objs) + 1
                        objs.append(Operator(oid, q, workcenters_ref))
                    except Exception:
                        continue
                self.operators_object_list = objs
            else:
                self.operators_object_list = [
                    Operator(1, op1_machines, workcenters_ref),
                    Operator(2, op2_machines, workcenters_ref),
                ]
        except Exception:
            # fallback to original explicit default
            self.operators_object_list = [
                Operator(1, op1_machines, workcenters_ref),
                Operator(2, op2_machines, workcenters_ref),
            ]

        # Compute and attach qualified_workcenters for convenience: map each
        # operator's qualified machine names to their WorkCenter indices using
        # the provided WorkCenters metadata object.
        try:
            # Build a helper to resolve various machine-name formats to the
            # canonical keys used in workcenters_ref.machine_registry.
            registry = getattr(workcenters_ref, 'machine_registry', {}) or {}
            order = getattr(workcenters_ref, 'machine_order', []) or []

            def resolve_registry_key(mname: str):
                # 1) direct key match
                if mname in registry:
                    return mname
                # 2) try using machine_order index -> registry key convention
                try:
                    idx = order.index(mname)
                    candidate = f'M{idx+1}'
                    if candidate in registry:
                        return candidate
                except ValueError:
                    pass
                # 3) try a loose normalization: drop underscores and attempt to
                # match keys like 'M1'.. by comparing digits
                import re
                digits = re.findall(r"\d+", mname)
                if digits:
                    # try the first number as ordinal
                    candidate = f'M{int(digits[0]) + 1}' if len(order) == len(registry) else f'M{digits[0]}'
                    if candidate in registry:
                        return candidate
                # nothing matched
                return None

            for op in self.operators_object_list:
                qualified_wcs = set()
                for mname in op.qualified_machines:
                    try:
                        key = resolve_registry_key(mname)
                        if not key:
                            continue
                        entry = registry.get(key)
                        if not entry:
                            continue
                        wc_idx = entry.get('workcenter')
                        if wc_idx is None:
                            continue
                        qualified_wcs.add(int(wc_idx))
                    except Exception:
                        # be resilient: skip any machine we can't resolve
                        continue
                # attach a stable sorted list for downstream tools
                op.qualified_workcenters = sorted(list(qualified_wcs))
        except Exception:
            # if anything fails, leave attribute absent for backward compatibility
            pass

        print("\n[Init] Operators created (derived from WorkCenters):")
        for op in self.operators_object_list:
            qwcs = getattr(op, 'qualified_workcenters', None)
            print(f"   - Operator {op.operator_id} → Machines {op.qualified_machines} qualified_workcenters={qwcs}")

    # ============================================================
    # === Lookup / Utility =======================================
    # ============================================================

    def find_free_operator(self, job_id, workcenter_id):
        """Return the first free operator who can perform this job at this WorkCenter."""
        for op in self.operators_object_list:
            if (not op.is_busy) and op.can_do_job(job_id, workcenter_id):
                return op
        return None

    def find_free_operator_for_machine(self, op_idx, machine_name):
        """
        Return the first free operator who is qualified for the given machine
        and who can perform operation `op_idx` on that machine. This is a
        machine-level lookup used when decisions are made at machine granularity.
        """
        try:
            registry = getattr(self.operators_object_list[0].workcenters_ref, 'machine_registry', {}) or {}
        except Exception:
            registry = {}
        for op in self.operators_object_list:
            try:
                if op.is_busy:
                    continue
                # must be qualified for the machine by name
                if machine_name not in op.qualified_machines:
                    continue
                # check machine capabilities
                caps = registry.get(machine_name, {}).get('capabilities', [])
                if int(op_idx) in caps:
                    return op
            except Exception:
                continue
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
