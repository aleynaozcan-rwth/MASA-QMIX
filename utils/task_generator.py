"""Task generator that delegates job metadata to `utils.job.Jobs`.

This file provides a single, consistent TaskGenerator implementation that
uses the canonical `utils.job.Jobs` registry.
"""

import random
from typing import Optional, Any
from utils.workcenter import WorkCenters
from utils.job import Jobs
import simpy


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

# Default processing time means mirror (Phase 3C.0)
# Mirrors YAML key: 'processing_time_means'
DEFAULT_PROCESSING_TIME_MEANS = {
    # synchronized with configs/env_config_enabled.yaml
    "Op1": {"M1": 1.225, "M3": 1.575, "M4": 1.4, "M5": 1.05},
    "Op2": {"M1": 1.05, "M3": 1.75},
    "Op3": {"M1": 1.575, "M4": 1.75, "M5": 1.925},
    "Op4": {"M2": 1.575, "M3": 1.68, "M5": 2.1},
    "Op5": {"M2": 1.75, "M1": 2.275},
    "Op6": {"M3": 2.1, "M5": 2.8},
    "Op7": {"M4": 1.82},
    "Op8": {"M5": 1.575, "M4": 2.625, "M2": 2.975},
    "Op9": {"M5": 1.925, "M1": 2.975, "M3": 3.15},
}


class TaskGenerator:
    """Generates random valid operation sequences and injects them into a SimPy env."""

    def __init__(self, max_retries: int = 5, config_path: Optional[str] = None):
        self.max_retries = max_retries
        self.config = None
        if config_path:
            try:
                from utils.config_loader import load_config

                self.config = load_config(config_path)
            except Exception:
                self.config = None

        # Merge in-module defaults with loaded config so missing sections are
        # filled by DEFAULT_TASKGEN_PARAMS and DEFAULT_PROCESSING_TIME_MEANS.
        try:
            from utils.config_loader import merge_config
            merged = merge_config({
                "task_generator": DEFAULT_TASKGEN_PARAMS.get("task_generator", DEFAULT_TASKGEN_PARAMS),
                "processing_time_means": DEFAULT_PROCESSING_TIME_MEANS,
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

        self.machine_speed = {getattr(wc, 'id', None): random.uniform(0.7, 1.4) for wc in self.workcenters.workcenters_list}

        self.proc_time_means = {}
        if self.config is not None:
            self.proc_time_means = self.config.get('processing_time_means', {})
            machines_cfg = list(self.config.get('machines', {}).keys())
            self.machine_name_by_wc = {idx: name for idx, name in enumerate(machines_cfg)}
        else:
            self.machine_name_by_wc = {}

    def generate_constrained_task(self, num_ops: Optional[int] = None, jobagent_id: Optional[int] = None):
        if num_ops is None:
            if self.config and self.config.get('task_generator'):
                tg = self.config.get('task_generator', {})
                mn = tg.get('seq_length', {}).get('min', 1)
                mx = tg.get('seq_length', {}).get('max', 9)
                num_ops = random.randint(max(1, mn), max(mn, mx))
            else:
                num_ops = random.randint(3, 7)

        ops_sequence = []
        used_job_ids = set()
        retries = 0

        while len(ops_sequence) < num_ops and retries < self.max_retries * num_ops:
            wc = random.choice(self.workcenters.workcenters_list)
            valid_ops = [j for j in getattr(wc, 'resource_ids_list', []) if j not in used_job_ids]
            if not valid_ops:
                retries += 1
                continue

            op_id = random.choice(valid_ops)
            try:
                job_obj = self.jobs[op_id]
            except Exception:
                retries += 1
                continue

            speed_factor = self.machine_speed.get(getattr(wc, 'id', None), 1.0)

            duration = None
            try:
                op_name = getattr(job_obj, 'name', None)
                if op_name and isinstance(self.proc_time_means, dict) and op_name in self.proc_time_means:
                    mmap = self.proc_time_means.get(op_name, {})
                    mname = self.machine_name_by_wc.get(getattr(wc, 'id', None))
                    candidates = []
                    if mname:
                        candidates.append(mname)
                    candidates.append(f"M{getattr(wc, 'id', None)}")
                    candidates.append(f"M_{getattr(wc, 'id', None)}_0")
                    for cand in candidates:
                        if cand in mmap:
                            try:
                                duration = float(mmap.get(cand))
                                break
                            except Exception:
                                continue
            except Exception:
                duration = None

            if duration is None:
                duration = round(max(0.1, getattr(job_obj, 'time_span', 1.0) * speed_factor), 2)

            try:
                new_job = type(job_obj)(job_obj.index_id, job_obj.codes, job_obj.name, duration)
            except Exception:
                class _SimpleJob:
                    def __init__(self, index_id, codes, name, time_span):
                        self.index_id = index_id
                        self.codes = codes
                        self.name = name
                        self.time_span = time_span

                new_job = _SimpleJob(job_obj.index_id, getattr(job_obj, 'codes', None), getattr(job_obj, 'name', None), duration)

            ops_sequence.append(new_job)
            used_job_ids.add(op_id)

        if not ops_sequence:
            try:
                default_job = self.jobs[0]
                try:
                    fallback = type(default_job)(default_job.index_id, default_job.codes, default_job.name, default_job.time_span)
                except Exception:
                    fallback = default_job
                ops_sequence = [fallback]
            except Exception:
                class _Minimal:
                    def __init__(self):
                        self.index_id = 0
                        self.codes = 'J0'
                        self.name = 'Job0'
                        self.time_span = 1.0

                ops_sequence = [_Minimal()]

        return ops_sequence

    def arrival_loop(self, env: simpy.Environment, arrival_lambda: float):
        if arrival_lambda is None or float(arrival_lambda) <= 0.0:
            return
        lam = float(arrival_lambda)
        # Allow env to be either a bare simpy.Environment or a MASAEnv wrapper.
        sim_env = getattr(env, 'env', env)
        while getattr(env, 'now', getattr(sim_env, 'now', 0.0)) < getattr(env, 'episode_limit', getattr(sim_env, 'episode_limit', float('inf'))) and not getattr(env, 'done', False):
            try:
                ia = float(random.expovariate(lam))
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
                allowed_wcs = list(sorted(set(capability_map.get(op_type, []))))
                if not allowed_wcs:
                    allowed_wcs = [int(random.randint(0, max(1, getattr(env, 'num_wcs', 1)) - 1))]
                per_wc = {}
                for wc in allowed_wcs:
                    try:
                        speed = env._speed_factor_for_wc(wc) if hasattr(env, '_speed_factor_for_wc') else 1.0
                    except Exception:
                        speed = 1.0
                    per_wc[int(wc)] = round(float(getattr(jobobj, 'time_span', 1.0)) / max(1e-6, speed), 6)
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
            sim_env.process(self.arrival_loop(env, arrival_lambda))
        except Exception as e:
            print(f"[TaskGen] Failed to start arrival loop: {e}")
