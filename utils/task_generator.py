"""Task generator that delegates job metadata to `utils.job.Jobs`.

This file provides a single, consistent TaskGenerator implementation that
uses the canonical `utils.job.Jobs` registry.
"""

import random
import logging
from typing import Optional, Any
from utils.workcenter import WorkCenters
from utils.job import Jobs
import simpy
from typing import List


# Default configuration mirror for TaskGenerator (Phase 3C.0)
# Mirrors YAML keys: 'task_generator', 'processing_time_means', 'machines'
# Example: arrival_lambda and seq_length controls
# TODO(Phase3C.1): integrate with TaskGenerator via merge_config(DEFAULT_TASKGEN_PARAMS, cfg)
DEFAULT_TASKGEN_PARAMS = {
    "task_generator": {
        "arrival_lambda": 0.4,
        "seq_length": {"min": 1, "max": 5}
    }
}


class TaskGenerator:
    """Generates random valid operation sequences and injects them into a SimPy env.

    Accepts optional RNG instances so callers (MASAEnv) can inject seeded
    RNGs for deterministic behavior across components. If no RNGs are
    provided the TaskGenerator will create internal, non-deterministic ones.
    """

    def __init__(self, max_retries: int = 5, py_rng: Optional[random.Random] = None, np_rng: Optional[Any] = None):
        self.max_retries = max_retries
        # Python RNG: use injected one or create a local instance
        if py_rng is not None:
            self._py_rng = py_rng
        else:
            self._py_rng = random.Random()
        # NumPy RNG (optional): prefer injected np_rng for consistency
        self._np_rng = np_rng if np_rng is not None else None
        # No YAML/config-based initialization: Jobs use embedded defaults
        self.jobs = Jobs()
        self.workcenters = WorkCenters()

        # processing_time_means must be provided and is the sole source of durations
        # Behavior change: prefer in-code WorkCenter defaults unless YAML/config
        # is explicitly enabled via `enable_yaml=True` and provides the mapping.
        self.proc_time_means = {}
        # Load deterministic defaults from utils.workcenter.DEFAULT_PROCESSING_TIMES.
        if not self.proc_time_means:
            try:
                import utils.workcenter as _wc_mod  # type: ignore
                wc_defaults = getattr(_wc_mod, 'DEFAULT_PROCESSING_TIMES', None)
                if isinstance(wc_defaults, dict):
                    proc_by_op = {}
                    # wc_defaults: machine_name -> {OpName: mean}
                    for mname, ops_map in wc_defaults.items():
                        for opname, v in (ops_map or {}).items():
                            try:
                                proc_by_op.setdefault(opname, {})[mname] = float(v)
                            except Exception as e:
                                proc_by_op.setdefault(opname, {})[mname] = v
                    self.proc_time_means = proc_by_op
                    logging.getLogger(__name__).info("[TaskGenerator] Loaded processing_time_means from WorkCenter defaults")
            except Exception as e:
                # if this fails, keep proc_time_means empty and raise below
                pass

        if not self.proc_time_means:
            # strict mode: processing_time_means required when no fallback exists
            raise ValueError("processing_time_means is required for TaskGenerator to compute durations")
        # map machine order -> machine name from WorkCenters
        try:
            self.machine_name_by_wc = {idx: name for idx, name in enumerate(getattr(self.workcenters, 'machine_order', []))}
        except Exception as e:
            self.machine_name_by_wc = {}

    def generate_constrained_task(self, num_ops: Optional[int] = None, jobagent_id: Optional[int] = None):
        if num_ops is None:
            tg = DEFAULT_TASKGEN_PARAMS.get('task_generator', {})
            mn = int(tg.get('seq_length', {}).get('min', 1))
            mx = int(tg.get('seq_length', {}).get('max', 9))
            num_ops = int(self._py_rng.randint(max(1, mn), max(mn, mx)))

        ops_sequence = []
        used_job_ids = set()
        retries = 0
        # Precompute available operation indices from WorkCenters as a fallback
        try:
            available_ops_from_wc = list(getattr(self.workcenters, 'operations_map', {}).keys())
        except Exception as e:
            available_ops_from_wc = []

        while len(ops_sequence) < num_ops and retries < self.max_retries * num_ops:
            # Prefer selecting ops from Jobs registry if possible; fall back
            # to WorkCenters.operations_map when Jobs is empty or unavailable.
            op_id = None
            job_obj = None
            try:
                # If Jobs exposes a list-like access and contains items, try to use it
                if getattr(self, 'jobs', None) is not None and hasattr(self.jobs, '__len__') and len(self.jobs) > 0:
                    # Attempt legacy selection via workcenter resource lists if present
                    try:
                        wc = self._py_rng.choice(self.workcenters.workcenters_list)
                        valid_ops = [j for j in getattr(wc, 'resource_ids_list', []) if j not in used_job_ids]
                    except Exception as e:
                        valid_ops = []

                    if valid_ops:
                        op_id = self._py_rng.choice(valid_ops)
                        try:
                            job_obj = self.jobs[op_id]
                        except Exception as e:
                            job_obj = None
                    else:
                        # fallback: sample from jobs registry by integer indices if possible
                        try:
                            # try numeric keys or sequence indices
                            idx = int(self._py_rng.randrange(len(self.jobs)))
                            job_obj = self.jobs[idx]
                            op_id = getattr(job_obj, 'index_id', None)
                        except Exception as e:
                            job_obj = None

                # If job_obj still not found, sample from WorkCenters.operations_map
                if job_obj is None:
                    if available_ops_from_wc:
                        op_idx = int(self._py_rng.choice(available_ops_from_wc))
                        # create a minimal job-like object
                        try:
                            class _SimpleJobFallback:
                                def __init__(self, index_id):
                                    self.index_id = index_id
                                    self.codes = None
                                    self.name = f"Op{index_id+1}"
                            job_obj = _SimpleJobFallback(op_idx)
                            op_id = op_idx
                        except Exception as e:
                            job_obj = None
                    else:
                        job_obj = None
            except Exception as e:
                job_obj = None

            if job_obj is None:
                retries += 1
                continue

            try:
                new_job = type(job_obj)(job_obj.index_id, job_obj.codes, job_obj.name)
            except Exception as e:
                class _SimpleJob:
                    def __init__(self, index_id, codes, name):
                        self.index_id = index_id
                        self.codes = codes
                        self.name = name

                new_job = _SimpleJob(job_obj.index_id, getattr(job_obj, 'codes', None), getattr(job_obj, 'name', None))

            ops_sequence.append(new_job)
            used_job_ids.add(op_id)

        if not ops_sequence:
            # strict: no dynamic job could be generated
            raise ValueError("TaskGenerator failed to generate any operations; provide valid processing_time_means and machine topology")

        return ops_sequence

    def arrival_loop(self, env: simpy.Environment, arrival_lambda: float):
        if arrival_lambda is None or float(arrival_lambda) <= 0.0:
            return
        lam = float(arrival_lambda)
        
        # [STOCHASTIC_ARRIVAL] Lottery-based arrival is DEFAULT
        use_exponential = bool(getattr(self, 'use_exponential_arrival', True))
        use_lottery = not use_exponential  # Lottery unless explicitly disabled
        check_interval = float(getattr(self, 'arrival_check_interval', 1.0))
        lottery_choices = list(getattr(self, 'arrival_lottery_choices', [0, 2, 4, 6, 8, 10]))
        lottery_probs = list(getattr(self, 'arrival_lottery_probs', [0.10,0.25,0.35,0.2,0.1,0.05]))
        
        # Diagnostic: report proc_time_means size so we know whether
        # TaskGenerator has a durations mapping available at runtime.
        try:
            logging.getLogger(__name__).info("[Diag] proc_time_means length: %d", len(getattr(self, 'proc_time_means', {}) or {}))
            if use_lottery:
                logging.getLogger(__name__).info("[Diag] Lottery arrival (DEFAULT): check_interval=%.1f, choices=%s", check_interval, lottery_choices)
            else:
                logging.getLogger(__name__).info("[Diag] Legacy exponential arrival enabled (use --use_exponential_arrival)")
        except Exception as e:
            logging.getLogger(__name__).info("[Diag] proc_time_means length: (failed to compute)")
        # Allow env to be either a bare simpy.Environment or a MASAEnv wrapper.
        # If the provided `env` is a plain simpy.Environment it will not
        # expose attributes like `episode_limit` or `done`. Prefer using the
        # owner wrapper (self._owner_env) when available for stop-condition
        # checks so arrival loops do not become infinite when scheduled on
        # the raw simpy.Environment.
        sim_env = getattr(env, 'env', env)
        wrapper_env = getattr(self, '_owner_env', None)

        def _get_attr(attr, default=None):
            # Check the provided env first, then an owner wrapper, then the
            # underlying simpy.Environment.
            try:
                if env is not None and hasattr(env, attr):
                    return getattr(env, attr)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
            try:
                if wrapper_env is not None and hasattr(wrapper_env, attr):
                    return getattr(wrapper_env, attr)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
            try:
                if sim_env is not None and hasattr(sim_env, attr):
                    return getattr(sim_env, attr)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
            return default

        while float(_get_attr('now', 0.0)) < float(_get_attr('episode_limit', float('inf'))) and not bool(_get_attr('done', False)):
            if not use_exponential:
                # [LOTTERY_ARRIVAL - DEFAULT] Check at fixed intervals, draw from discrete distribution
                yield sim_env.timeout(check_interval)
                # Check if episode ended during check interval
                if bool(_get_attr('done', False)):
                    break
                # Draw next job delay from lottery
                try:
                    delay = float(self._py_rng.choices(lottery_choices, weights=lottery_probs, k=1)[0])
                except Exception as e:
                    delay = float(lottery_choices[0]) if lottery_choices else 0.0
                
                if delay > 0:
                    yield sim_env.timeout(delay)
                    # Check again after delay
                    if bool(_get_attr('done', False)):
                        break
                # If delay=0, job arrives immediately (no additional timeout)
            else:
                # Classic exponential inter-arrival
                try:
                    ia = float(self._py_rng.expovariate(lam))
                except Exception as e:
                    ia = float(1.0 / max(1e-12, lam))
                yield sim_env.timeout(ia)

            try:
                ops_objs = self.generate_constrained_task(jobagent_id=len(getattr(env, 'jobs', [])))
            except Exception as e:
                # Diagnostic: surface generation failures explicitly
                logging.getLogger(__name__).warning("[TaskGen] generation failed: %s", e, exc_info=True)
                try:
                    logging.getLogger(__name__).info("[Diag] generate_constrained_task() failed: %s", e)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                continue

            converted_ops = []
            capability_map = {op: [] for op in range(0, 32)}
            machine_to_wc = {}  # Track machine -> workcenter mapping
            try:
                for mid, mdata in getattr(getattr(env, 'workcenters_meta', {}), 'machine_registry', {}).items():
                    wc = int(mdata.get('workcenter', 0))
                    machine_to_wc[mid] = wc
                    caps = list(mdata.get('capabilities', []))
                    # Map operation -> MACHINE NAMES (not WorkCenter IDs)
                    for c in caps:
                        capability_map.setdefault(int(c), []).append(mid)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

            for jobobj in ops_objs:
                op_type = int(getattr(jobobj, 'index_id', 0))
                op_name = f"Op{op_type+1}"
                # capability_map now stores MACHINE NAMES that can do this operation
                allowed_machines = list(sorted(set(capability_map.get(op_type, []))))
                if not allowed_machines:
                    raise ValueError(f"No capable machines for operation {op_name}; check machine capabilities")

                # Build per_machine_indices and per_wc for each capable machine
                per_machine_indices = []
                per_wc = {}
                
                # Get machine_index mapping
                wc_meta = getattr(env, 'workcenters_meta', None)
                machine_list = []
                if wc_meta is not None:
                    try:
                        machine_list = list(getattr(wc_meta, 'machine_list', []))
                    except Exception:
                        machine_list = []
                
                # Build machine name -> index mapping
                machine_index_map = {}
                if machine_list:
                    machine_index_map = {mname: i for i, mname in enumerate(machine_list)}

                # Get processing times
                op_map = self.proc_time_means.get(op_name, {})

                for machine_name in allowed_machines:
                    # Check if duration exists for this machine
                    if machine_name not in op_map:
                        raise ValueError(
                            f"Missing duration for {op_name} on {machine_name}. "
                            f"Machine {machine_name} is capable (in capabilities list) but no processing time defined. "
                            f"Check DEFAULT_PROCESSING_TIMES consistency."
                        )

                    # Get machine index
                    mi = machine_index_map.get(machine_name, 0)
                    per_machine_indices.append(int(mi))
                    per_wc[int(mi)] = float(op_map.get(machine_name))

                converted_ops.append((op_type, per_machine_indices, per_wc))

            try:
                # Prefer owner_env callback (set by MASAEnv) so we can call
                # MASAEnv.add_job even when start() was invoked with the
                # bare simpy.Environment (tests assert that). Fall back to
                # calling add_job on the provided env if available.
                owner = getattr(self, '_owner_env', None)
                if owner is not None:
                    owner.add_job(converted_ops)
                else:
                    env.add_job(converted_ops)
            except Exception as e:
                # Diagnostic: make clear whether add_job raised when called
                logging.getLogger(__name__).exception("[Env] Failed to add dynamic job from TaskGenerator: %s", e)
                try:
                    logging.getLogger(__name__).info("[Diag] add_job() failed during TaskGenerator.arrival_loop at sim.now=%s: %s", getattr(getattr(env, 'env', env), 'now', None), e)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

    def start(self, env: simpy.Environment, arrival_lambda: float):
        try:
            # If a MASAEnv wrapper was provided, schedule on its internal simpy.Environment.
            sim_env = getattr(env, 'env', env)
            # Prefer to call arrival_loop with the MASAEnv wrapper so the
            # generator can resolve workcenters_meta and other attributes.
            # If this TaskGenerator was attached to an owner MASAEnv via
            # `_owner_env`, use that wrapper as the env argument for
            # arrival_loop; otherwise fall back to the provided env.
            wrapper_env = getattr(self, '_owner_env', None)
            call_env = wrapper_env if wrapper_env is not None else env
            sim_env.process(self.arrival_loop(call_env, arrival_lambda))
        except Exception as e:
            logging.getLogger(__name__).exception("[TaskGen] Failed to start arrival loop: %s", e)

    def create_job(self, num_ops: Optional[int] = None):
        """Create a single converted job (op tuples) suitable for MASAEnv.add_job().

        This helper mirrors the conversion logic used in `arrival_loop` but
        returns the converted operations list so callers (MASAEnv) can add
        it at the desired arrival time.
        """
        # Use owner env if available so we can consult workcenters_meta / config
        owner = getattr(self, '_owner_env', None)
        env = owner if owner is not None else None

        # Generate raw job objects (ops sequence)
        ops_objs = self.generate_constrained_task(num_ops=num_ops, jobagent_id=(len(getattr(env, 'jobs', [])) if env is not None else None))

        # Convert ops_objs into (op_type, allowed_machine_indices, per_machine) tuples
        converted_ops = []
        # Build capability_map: operation index -> list of MACHINE NAMES that support it
        # MACHINE-BASED (not WorkCenter-based) - WorkCenter is metadata only
        capability_map = {op: [] for op in range(0, 32)}
        machine_to_wc = {}  # Track machine -> workcenter mapping for metadata
        try:
            machine_registry = getattr(getattr(env, 'workcenters_meta', {}), 'machine_registry', {})
            for mname, mdata in machine_registry.items():
                wc = int(mdata.get('workcenter', 0))
                machine_to_wc[mname] = wc
                caps = list(mdata.get('capabilities', []))
                # Map operation -> machines that CAN DO this operation
                for c in caps:
                    capability_map.setdefault(int(c), []).append(mname)
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

        for jobobj in ops_objs:
            op_type = int(getattr(jobobj, 'index_id', 0))
            op_name = f"Op{op_type+1}"
            # capability_map now stores MACHINE NAMES that can do this operation
            allowed_machines = list(sorted(set(capability_map.get(op_type, []))))
            if not allowed_machines:
                raise ValueError(f"No capable machines for operation {op_name}; check machine capabilities")

            # For each capable machine, lookup duration from proc_time_means
            per_machine_indices = []
            per_wc = {}
            op_map = self.proc_time_means.get(op_name, {})

            wc_meta = getattr(env, 'workcenters_meta', None)
            machine_index_map = {}
            try:
                if wc_meta is not None:
                    machine_index_map = getattr(wc_meta, 'machine_index', {}) or {}
            except Exception as e:
                machine_index_map = {}

            # Extract WorkCenter IDs from capable machines (for backward compat)
            allowed_wcs = list(sorted(set(machine_to_wc.get(m, 0) for m in allowed_machines)))

            for machine_name in allowed_machines:
                # Check if duration exists for this machine
                if machine_name not in op_map:
                    raise ValueError(
                        f"Missing duration for {op_name} on {machine_name}. "
                        f"Machine {machine_name} is capable (in capabilities list) but no processing time defined. "
                        f"Check DEFAULT_PROCESSING_TIMES consistency."
                    )


                try:
                    mi = int(machine_index_map.get(machine_name))
                except Exception as e:
                    # Fallback: try to get machine index from WorkCenter
                    wc = machine_to_wc.get(machine_name, 0)
                    try:
                        mi = int(self.machine_name_by_wc.get(int(wc), 0))
                    except Exception as e:
                        mi = 0

                per_machine_indices.append(int(mi))
                per_wc[int(mi)] = float(op_map.get(machine_name))

            # IMPORTANT: keep the second tuple element as the list of allowed
            # workcenters (not machine indices) to remain compatible with
            # tests and legacy callers expecting workcenter ids here.
            converted_ops.append((op_type, allowed_wcs, per_wc))

        return converted_ops
