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
    {"id": "O1", "qualified_machines": ["M0", "M3", "M4"]},
    {"id": "O2", "qualified_machines": ["M1", "M2", "M4"]},
    {"id": "O3", "qualified_machines": ["M0", "M3", "M4"]},
    {"id": "O4", "qualified_machines": ["M1", "M2", "M4"]},
]
import logging
LOG = logging.getLogger(__name__)

class Operator:
    """Single operator who can work on specific machine names.

    Now owns a per-operator SimPy Resource so operator-level concurrency
    is enforced by the simulation (no more relying only on is_busy flags).
    """

    def __init__(self, operator_id, qualified_machines, workcenters_ref, env=None):
        # Normalize operator IDs to stable textual labels.
        # If callers pass an int (legacy code paths) convert to 'O{n}' to
        # ensure timeline/gantt output is always human-readable and
        # non-ambiguous (avoid naked numeric ids like '0').
        try:
            if isinstance(operator_id, int):
                self.operator_id = f"O{int(operator_id)}"
            elif operator_id is None:
                self.operator_id = "UNKNOWN"
            else:
                # keep stringy ids as-is but ensure type is str
                self.operator_id = str(operator_id)
        except Exception as e:
            self.operator_id = str(operator_id)
        # qualified_machines: list of machine name strings (e.g., 'M0')
        self.qualified_machines = list(qualified_machines)
        self.workcenters_ref = workcenters_ref  # Reference to WorkCenters() environment object
        # SimPy resource for this concrete operator. If env is None the
        # resource will be created lazily when an env is provided (fallback).
        self.resource = None
        if env is not None:
            try:
                import simpy
                self.resource = simpy.Resource(env, capacity=1)
            except Exception as e:
                self.resource = None
        self._manual_busy = False  # Fallback when resource unavailable
        self.current_job = None
        self.current_workcenter = None
        self.history = []  # (job_id, workcenter_id, start_t, end_t)
    
    @property
    def is_busy(self):
        """Check if operator is busy based on SimPy resource state.
        
        If resource is available, check if it has active users (busy).
        Otherwise fall back to manual flag for compatibility.
        """
        if self.resource is not None:
            # SimPy resource is busy if it has users (someone holds a request)
            return len(self.resource.users) > 0
        return self._manual_busy
    
    @is_busy.setter
    def is_busy(self, value):
        """Set manual busy flag (used when resource unavailable)."""
        self._manual_busy = bool(value)

    # ============================================================
    # === Capability check =======================================
    # ============================================================

    def can_do_job(self, op_idx, workcenter_id):
        """
        Return True if operator can perform operation `op_idx` at the given WorkCenter.

        This checks machine-level qualifications: operator must be qualified for
        at least one machine inside the workcenter and that machine's
        capabilities must include op_idx.
        """
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
                if int(op_idx) in caps:
                    return True
            return False
        except Exception as e:
            # conservative fallback: try older WorkCenter object path
            try:
                wc_obj = self.workcenters_ref.workcenters_list[workcenter_id]
                candidate_machines = [m for m in wc_obj.machines.keys() if m in self.qualified_machines]
                if not candidate_machines:
                    return False
                for m in candidate_machines:
                    caps = wc_obj.machines.get(m, {}).get('capabilities', [])
                    if int(op_idx) in caps:
                        return True
                return False
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                return False

    # ============================================================
    # === Assignment / Release ===================================
    # ============================================================

    def assign_job(self, job_id, workcenter_id, start_time=None):
        """Mark operator as busy and log assignment."""
        self.is_busy = True
        self.current_job = job_id
        self.current_workcenter = workcenter_id
        LOG.info("[Operator] Operator %s assigned job %s at WorkCenter %s", self.operator_id, job_id, workcenter_id)
        self.history.append({
            "job_id": job_id,
            "workcenter_id": workcenter_id,
            "start_time": start_time,
            "end_time": None
        })

    def release(self, end_time=None):
        """Free the operator after finishing the job. Safe to call multiple times."""
        if self.current_job is not None:
            LOG.info("[Operator] Operator %s released from job %s", self.operator_id, self.current_job)
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

    def __init__(self, workcenters_ref, env=None):
        # Attempt to merge in-module DEFAULT_OPERATORS with any provided
        # operator configuration present on the WorkCenters object or its
        # attached config. This fills missing fields while remaining
        # non-destructive to inputs.
        # Prefer explicit operator list provided on the WorkCenters object
        # (workcenters_ref.operators) or in its config; otherwise fall back
        # to the in-module DEFAULT_OPERATORS. This avoids depending on the
        # YAML merge helper and keeps behavior deterministic in config-free
        # mode.
        try:
            provided = getattr(workcenters_ref, 'operators', None)
        except Exception as e:
            provided = None
        if provided is None:
            try:
                cfg = getattr(workcenters_ref, 'config', None) or {}
                provided = cfg.get('operators') if isinstance(cfg, dict) else None
            except Exception as e:
                provided = None
        if isinstance(provided, (list, tuple)) and len(provided) > 0:
            merged_operators = list(provided)
        else:
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
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                return fallback

        # fallback default names if we couldn't build an order
        fallback_names = ["M0", "M1", "M2", "M3", "M4"]
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
                        objs.append(Operator(oid, q, workcenters_ref, env=env))
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                        continue
                self.operators_object_list = objs
            else:
                self.operators_object_list = [
                    Operator(1, op1_machines, workcenters_ref, env=env),
                    Operator(2, op2_machines, workcenters_ref, env=env),
                ]
        except Exception as e:
            # fallback to original explicit default
            self.operators_object_list = [
                Operator(1, op1_machines, workcenters_ref, env=env),
                Operator(2, op2_machines, workcenters_ref, env=env),
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
                    candidate = f'M{idx}'
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
                    candidate = f'M{int(digits[0])}' if len(order) == len(registry) else f'M{digits[0]}'
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
                    except Exception as e:
                        # be resilient: skip any machine we can't resolve
                        continue
                # attach a stable sorted list for downstream tools
                op.qualified_workcenters = sorted(list(qualified_wcs))
        except Exception as e:
            # if anything fails, leave attribute absent for backward compatibility
            pass

        LOG.info("\n[Init] Operators created (derived from WorkCenters):")
        for op in self.operators_object_list:
            qwcs = getattr(op, 'qualified_workcenters', None)
            LOG.info("   - Operator %s → Machines %s qualified_workcenters=%s", op.operator_id, op.qualified_machines, qwcs)

    # ============================================================
    # === Lookup / Utility =======================================
    # ============================================================

    def find_free_operator(self, op_idx, workcenter_id):
        """Return the first free operator who can perform operation `op_idx` at this WorkCenter."""
        for op in self.operators_object_list:
            try:
                if (not op.is_busy) and op.can_do_job(op_idx, workcenter_id):
                    return op
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                continue
        return None

    def find_free_operator_for_machine(self, op_idx, machine_name):
        """
        Return the first free operator who is qualified for the given machine
        and who can perform operation `op_idx` on that machine. This is a
        machine-level lookup used when decisions are made at machine granularity.
        """
        try:
            registry = getattr(self.operators_object_list[0].workcenters_ref, 'machine_registry', {}) or {}
        except Exception as e:
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
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                continue
        return None

    def find_free_operator_seeded_random(self, op_idx, workcenter_id, rng):
        """C7 FIX: Deterministic but balanced operator selection using seeded RNG.
        
        This replaces the nondeterministic find_free_operator() which was
        timing-dependent due to SimPy event ordering. Uses seeded random
        selection for reproducibility while maintaining load balance.
        
        Args:
            op_idx: Operation index
            workcenter_id: Workcenter ID
            rng: numpy.random.RandomState (seeded)
            
        Returns:
            Operator object if found, None if all busy or none qualified
        """
        # Get all qualified FREE operators
        qualified_free = []
        for op in self.operators_object_list:
            try:
                if (not op.is_busy) and op.can_do_job(op_idx, workcenter_id):
                    qualified_free.append(op)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                continue
        
        if not qualified_free:
            return None
        
        # C7 FIX: Sort by operator ID for deterministic ordering (critical for RNG consistency)
        # Handle both numeric and string operator IDs (e.g., 'O1', '1', 1)
        def _extract_numeric_id(op):
            try:
                opid = str(op.operator_id)
                # If ID is like 'O1', 'O2', extract the number part
                if opid.startswith('O') or opid.startswith('o'):
                    return int(opid[1:])
                # Otherwise try direct conversion
                return int(opid)
            except (ValueError, IndexError):
                # Fallback: use hash for consistent ordering
                return hash(str(op.operator_id))
        
        qualified_free.sort(key=_extract_numeric_id)
        
        # C7 FIX: Select randomly using seeded RNG
        # Handle single operator case (len=1) separately
        if len(qualified_free) == 1:
            return qualified_free[0]
        # Note: randint(low, high) is inclusive of both endpoints,
        # so we use high=len-1 to avoid IndexError
        selected_idx = rng.randint(0, len(qualified_free) - 1)
        return qualified_free[selected_idx]

    def find_free_operator_for_machine_seeded_random(self, op_idx, machine_name, rng):
        """C7 FIX: Machine-specific deterministic but balanced operator selection.
        
        Args:
            op_idx: Operation index
            machine_name: Machine name (e.g., 'M0', 'M1')
            rng: numpy.random.RandomState (seeded)
            
        Returns:
            Operator object if found, None otherwise
        """
        try:
            registry = getattr(self.operators_object_list[0].workcenters_ref, 'machine_registry', {}) or {}
        except Exception as e:
            registry = {}
        
        # Get all qualified FREE operators for this machine
        qualified_free = []
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
                    qualified_free.append(op)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                continue
        
        if not qualified_free:
            return None
        
        # C7 FIX: Sort by operator ID for deterministic ordering
        # Handle both numeric and string operator IDs (e.g., 'O1', '1', 1)
        def _extract_numeric_id(op):
            try:
                opid = str(op.operator_id)
                # If ID is like 'O1', 'O2', extract the number part
                if opid.startswith('O') or opid.startswith('o'):
                    return int(opid[1:])
                # Otherwise try direct conversion
                return int(opid)
            except (ValueError, IndexError):
                # Fallback: use hash for consistent ordering
                return hash(str(op.operator_id))
        
        qualified_free.sort(key=_extract_numeric_id)
        
        # C7 FIX: Select randomly using seeded RNG
        # Handle single operator case (len=1) separately
        if len(qualified_free) == 1:
            return qualified_free[0]
        # Note: randint(low, high) is inclusive of both endpoints,
        # so we use high=len-1 to avoid IndexError
        selected_idx = rng.randint(0, len(qualified_free) - 1)
        return qualified_free[selected_idx]

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
#--------------------------------------------------------------