"""Task generator that delegates job metadata to `utils.job.Jobs`.

This file provides a single, consistent TaskGenerator implementation that
uses the canonical `utils.job.Jobs` registry.
"""

import random
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
        "arrival_lambda": 0.05,
        "seq_length": {"min": 1, "max": 5}
    }
}


class TaskGenerator:
    """Generates random valid operation sequences and injects them into a SimPy env.

    Accepts optional RNG instances so callers (MASAEnv) can inject seeded
    RNGs for deterministic behavior across components. If no RNGs are
    provided the TaskGenerator will create internal, non-deterministic ones.
    """

    def __init__(self, max_retries: int = 5, config_path: Optional[str] = None, py_rng: Optional[random.Random] = None, np_rng: Optional[Any] = None):
        self.max_retries = max_retries
        self.config = None
        # Python RNG: use injected one or create a local instance
        if py_rng is not None:
            self._py_rng = py_rng
        else:
            self._py_rng = random.Random()
        # NumPy RNG (optional): prefer injected np_rng for consistency
        self._np_rng = np_rng if np_rng is not None else None
        if config_path:
            try:
                from utils.config_loader import load_config

                self.config = load_config(config_path)
            except Exception:
                self.config = None

    # Merge in-module defaults with loaded config so missing sections are
    # filled by DEFAULT_TASKGEN_PARAMS and processing_time_means provided
    # via YAML/config (processing_time_means is authoritative and must
    # be supplied by callers in strict mode).
        try:
            from utils.config_loader import merge_config
            # Only merge task_generator defaults here. processing_time_means
            # MUST be provided by external YAML/config (strict mode).
            merged = merge_config({
                "task_generator": DEFAULT_TASKGEN_PARAMS.get("task_generator", DEFAULT_TASKGEN_PARAMS),
            }, self.config or {})
            # merged now contains 'task_generator' and 'processing_time_means' keys
            self.config = merged
        except Exception:
            # keep whatever self.config was if merge fails
            pass

        jobs_cfg = None
        if self.config is not None:
            jobs_cfg = self.config.get('jobs', None)

        self.jobs = Jobs(jobs_cfg)
        self.workcenters = WorkCenters()

        # processing_time_means must be provided and is the sole source of durations
        self.proc_time_means = {}
        if self.config is not None:
            self.proc_time_means = self.config.get('processing_time_means', {})
        if not self.proc_time_means:
            # Attempt to fall back to WorkCenters.DEFAULT_PROCESSING_TIMES when
            # TaskGenerator is used standalone (no env provided). The fallback
            # lives only in utils.workcenter and must be transposed to the
            # op->machine->mean shape expected by the rest of the code.
            try:
                # DEFAULT_PROCESSING_TIMES is defined at module-level in
                # utils.workcenter. Import the module and read the value.
                import utils.workcenter as _wc_mod  # type: ignore
                wc_defaults = getattr(_wc_mod, 'DEFAULT_PROCESSING_TIMES', None)
                if isinstance(wc_defaults, dict):
                    proc_by_op = {}
                    for mname, ops_map in wc_defaults.items():
                        for opname, v in (ops_map or {}).items():
                            try:
                                proc_by_op.setdefault(opname, {})[mname] = float(v)
                            except Exception:
                                proc_by_op.setdefault(opname, {})[mname] = v
                    self.proc_time_means = proc_by_op
            except Exception:
                pass

        if not self.proc_time_means:
            # strict mode: processing_time_means required when no fallback exists
            raise ValueError("processing_time_means is required for TaskGenerator to compute durations")
        machines_cfg = list(self.config.get('machines', {}).keys()) if isinstance(self.config, dict) else []
        self.machine_name_by_wc = {idx: name for idx, name in enumerate(machines_cfg)}

    def generate_constrained_task(self, num_ops: Optional[int] = None, jobagent_id: Optional[int] = None):
        if num_ops is None:
            if self.config and self.config.get('task_generator'):
                tg = self.config.get('task_generator', {})
                mn = tg.get('seq_length', {}).get('min', 1)
                mx = tg.get('seq_length', {}).get('max', 9)
                num_ops = int(self._py_rng.randint(max(1, mn), max(mn, mx)))
            else:
                num_ops = int(self._py_rng.randint(3, 7))

        ops_sequence = []
        used_job_ids = set()
        retries = 0

        while len(ops_sequence) < num_ops and retries < self.max_retries * num_ops:
            wc = self._py_rng.choice(self.workcenters.workcenters_list)
            valid_ops = [j for j in getattr(wc, 'resource_ids_list', []) if j not in used_job_ids]
            if not valid_ops:
                retries += 1
                continue

            op_id = self._py_rng.choice(valid_ops)
            # Build job-like object (metadata only). Durations are assigned
            # later when converting to per-machine durations using
            # processing_time_means. If a mapping is missing the code will
            # raise when converting.
            try:
                job_obj = self.jobs[op_id]
            except Exception:
                retries += 1
                continue

            try:
                new_job = type(job_obj)(job_obj.index_id, job_obj.codes, job_obj.name)
            except Exception:
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
            except Exception:
                pass
            try:
                if wrapper_env is not None and hasattr(wrapper_env, attr):
                    return getattr(wrapper_env, attr)
            except Exception:
                pass
            try:
                if sim_env is not None and hasattr(sim_env, attr):
                    return getattr(sim_env, attr)
            except Exception:
                pass
            return default

        while float(_get_attr('now', 0.0)) < float(_get_attr('episode_limit', float('inf'))) and not bool(_get_attr('done', False)):
            try:
                # Use the internal Python RNG for exponential inter-arrival
                ia = float(self._py_rng.expovariate(lam))
            except Exception:
                ia = float(1.0 / max(1e-12, lam))
            # yield on the simulation environment (simpy.Environment)
            yield sim_env.timeout(ia)

            try:
                ops_objs = self.generate_constrained_task(jobagent_id=len(getattr(env, 'jobs', [])))
            except Exception as e:
                print(f"[TaskGen] generation failed: {e}")
                continue

            converted_ops = []
            capability_map = {op: [] for op in range(0, 32)}
            try:
                for mid, mdata in getattr(getattr(env, 'workcenters_meta', {}), 'machine_registry', {}).items():
                    wc = int(mdata.get('workcenter', 0))
                    caps = list(mdata.get('capabilities', []))
                    for c in caps:
                        capability_map.setdefault(int(c), []).append(int(wc))
            except Exception:
                pass

            for jobobj in ops_objs:
                op_type = int(getattr(jobobj, 'index_id', 0))
                op_name = f"Op{op_type+1}"
                allowed_wcs = list(sorted(set(capability_map.get(op_type, []))))
                if not allowed_wcs:
                    raise ValueError(f"No allowed workcenters for operation {op_name}; check machine capabilities")
                per_wc = {}
                # For each allowed WC, resolve a machine name and lookup the
                # processing_time_means entry for that machine. Missing entries
                # are errors in strict mode.
                for wc in allowed_wcs:
                    # try to find a machine name that belongs to this workcenter
                    machine_name = None
                    try:
                        # find first machine in registry with matching workcenter
                        for mname, mdata in getattr(getattr(env, 'workcenters_meta', {}), 'machine_registry', {}).items():
                            if int(mdata.get('workcenter', -1)) == int(wc):
                                machine_name = mname
                                break
                    except Exception:
                        machine_name = None

                    if not machine_name:
                        raise ValueError(f"No machine found for workcenter {wc} when resolving durations for {op_name}")

                    # Prefer processing_time_means provided by the runtime env when
                    # available (MASAEnv will inject its merged config into
                    # env.config). Only fall back to the TaskGenerator's own
                    # proc_time_means (module-level DEFAULT_PROCESSING_TIMES
                    # transposed) when the env does not supply the mapping.
                    try:
                        env_proc = getattr(env, 'config', None) or {}
                        if isinstance(env_proc, dict):
                            op_map = env_proc.get(op_name, {}) or {}
                        else:
                            op_map = {}
                    except Exception:
                        op_map = {}
                    if not op_map:
                        op_map = self.proc_time_means.get(op_name, {})
                    # Strict mode: processing_time_means must contain an entry
                    # for the exact canonical machine name. We do not accept
                    # legacy or derived fallback keys here.
                    if machine_name in op_map:
                        per_wc[int(wc)] = float(op_map.get(machine_name))
                    else:
                        # Strict behaviour: only YAML-provided mappings allowed.
                        raise ValueError(f"Missing duration for {op_name} on {machine_name} (workcenter {wc})")

                converted_ops.append((op_type, allowed_wcs, per_wc))

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
                print(f"[Env] Failed to add dynamic job from TaskGenerator: {e}")

    def start(self, env: simpy.Environment, arrival_lambda: float):
        try:
            # If a MASAEnv wrapper was provided, schedule on its internal simpy.Environment.
            sim_env = getattr(env, 'env', env)
            # When MASAEnv constructs the TaskGenerator it sets `_owner_env`
            # to the MASAEnv instance so arrival_loop can consult wrapper
            # attributes (episode_limit / done). Keep behaviour backward
            # compatible by still scheduling on the provided simpy.Environment.
            sim_env.process(self.arrival_loop(env, arrival_lambda))
        except Exception as e:
            print(f"[TaskGen] Failed to start arrival loop: {e}")

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

        # Convert ops_objs into (op_type, allowed_wcs, per_wc) tuples
        converted_ops = []
        capability_map = {op: [] for op in range(0, 32)}
        try:
            for mid, mdata in getattr(getattr(env, 'workcenters_meta', {}), 'machine_registry', {}).items():
                wc = int(mdata.get('workcenter', 0))
                caps = list(mdata.get('capabilities', []))
                for c in caps:
                    capability_map.setdefault(int(c), []).append(int(wc))
        except Exception:
            pass

        for jobobj in ops_objs:
            op_type = int(getattr(jobobj, 'index_id', 0))
            op_name = f"Op{op_type+1}"
            allowed_wcs = list(sorted(set(capability_map.get(op_type, []))))
            if not allowed_wcs:
                raise ValueError(f"No allowed workcenters for operation {op_name}; check machine capabilities")
            per_wc = {}
            for wc in allowed_wcs:
                machine_name = None
                try:
                    for mname, mdata in getattr(getattr(env, 'workcenters_meta', {}), 'machine_registry', {}).items():
                        if int(mdata.get('workcenter', -1)) == int(wc):
                            machine_name = mname
                            break
                except Exception:
                    machine_name = None

                if not machine_name:
                    raise ValueError(f"No machine found for workcenter {wc} when resolving durations for {op_name}")

                try:
                    env_proc = getattr(env, 'config', None) or {}
                    if isinstance(env_proc, dict):
                        op_map = env_proc.get(op_name, {}) or {}
                    else:
                        op_map = {}
                except Exception:
                    op_map = {}
                if not op_map:
                    op_map = self.proc_time_means.get(op_name, {})

                if machine_name in op_map:
                    per_wc[int(wc)] = float(op_map.get(machine_name))
                else:
                    raise ValueError(f"Missing duration for {op_name} on {machine_name} (workcenter {wc})")

            converted_ops.append((op_type, allowed_wcs, per_wc))

        return converted_ops
