"""
utils/task_generator.py
Step 7B.3 — Safe Dynamic Task Generation (No Repeated Operations)
-----------------------------------------------------------------
Enhancements:
- Guarantees non-empty operation sequences.
- Ensures no operation type repeats within one JobAgent.
- Auto-retries generation up to N times if a WorkCenter has no valid new jobs.
- Logs gracefully even if dynamic spawning occurs mid-simulation.
- Fully backward-compatible with Step 7A.
"""

import random
from utils.task import Task
from utils.job import Jobs
from utils.workcenter import WorkCenters


class TaskGenerator:
    """Creates random but valid job sequences (operations list) based on WorkCenter capabilities.

    If a YAML config is provided (via config_path), the generator will use
    `task_generator` and `processing_time_means` sections to sample arrival
    rates and per-(op,machine) durations.
    """

    def __init__(self, max_retries: int = 5, config_path: str = None):
        self.max_retries = max_retries

        # load optional config
        self.config = None
        if config_path:
            try:
                from utils.config_loader import load_config
                self.config = load_config(config_path)
            except Exception:
                self.config = None

        # core structures
        self.jobs = Jobs()
        self.workcenters = WorkCenters()

        # Random machine speed scaling factors (for job duration variability)
        self.machine_speed = {
            getattr(wc, 'id', None): random.uniform(0.7, 1.4)
            for wc in self.workcenters.workcenters_list
        }

        # processing_time_means from config (optional)
        self.proc_time_means = {}
        if self.config is not None:
            self.proc_time_means = self.config.get('processing_time_means', {})
            # build optional mapping: workcenter index -> machine name used in config
            # (e.g., config machines keys like 'M1','M2' mapped to wc indices 0..)
            machines_cfg = list(self.config.get('machines', {}).keys())
            self.machine_name_by_wc = {idx: name for idx, name in enumerate(machines_cfg)}
        else:
            self.machine_name_by_wc = {}

        # If config provided, build a reduced workcenters list matching config machines
        # so TaskGenerator uses the same logical WCs as the YAML (avoids using default 18).
        if self.config is not None and self.config.get('machines'):
            class _WC:
                def __init__(self, wc_id, resource_ids_list):
                    self.id = wc_id
                    self.id = wc_id
                    # legacy attribute removed: do not set site_id; use workcenter id instead
                    self.resource_ids_list = resource_ids_list

            cfg_machines = self.config.get('machines', {})
            reduced_wcs = []
            for idx, (mname, mconf) in enumerate(cfg_machines.items()):
                caps = mconf.get('capable_ops', [])
                caps_idx = []
                for c in caps:
                    try:
                        if isinstance(c, str) and c.lower().startswith('op'):
                            caps_idx.append(int(c[2:]) - 1)
                        else:
                            caps_idx.append(int(c))
                    except Exception:
                        continue
                reduced_wcs.append(_WC(idx, caps_idx))
            # override workcenters.workcenters_list with reduced list
            try:
                self.workcenters.workcenters_list = reduced_wcs
            except Exception:
                # best-effort: leave original WorkCenters if override fails
                pass

    # ------------------------------------------------------------------
    def generate_constrained_task(self, num_ops=None, jobagent_id=None):
        """
        Create a random but valid operation list for one JobAgent.
        Ensures:
        - Non-empty valid job sequence
        - No repeated job type within the same JobAgent
        """
        if num_ops is None:
            # allow config-driven defaults via task_generator.seq_length
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

            # Filter WorkCenter operations so none are already used
            valid_ops = [
                j for j in wc.resource_ids_list
                if j not in used_job_ids
            ]

            if not valid_ops:
                retries += 1
                continue

            op_id = random.choice(valid_ops)
            job_obj = self.jobs.jobs_object_list[op_id]

            # Apply processing time logic: prefer config-driven per-(op,machine) means
            speed_factor = self.machine_speed.get(getattr(wc, 'id', None), 1.0)
            # job_obj.index_id or job_obj.codes can be used to map op types; fall back
            op_key = getattr(job_obj, 'index_id', None) or getattr(job_obj, 'codes', None)
            # try to read from self.proc_time_means using op name if available
            duration = None
            try:
                # job_obj.name might contain the op type like 'Op1'
                op_name = getattr(job_obj, 'name', None)
                if op_name and isinstance(self.proc_time_means, dict) and op_name in self.proc_time_means:
                    mmap = self.proc_time_means.get(op_name, {})
                    # prefer mapping provided by config (machine name per wc)
                    mname = self.machine_name_by_wc.get(getattr(wc, 'id', None))
                    # allow several common name formats: 'M1' or 'M_0_0' etc
                    candidates = []
                    if mname:
                        candidates.append(mname)
                    # legacy/alternate formats
                    candidates.append(f"M{getattr(wc, 'id', None)}")
                    candidates.append(f"M_{getattr(wc, 'id', None)}_0")
                    # check candidates in mmap
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
                # fallback to original multiplier on job_obj.time_span
                duration = round(max(0.1, job_obj.time_span * speed_factor), 2)

            # Create a copy of the Job object with modified time_span
            new_job = type(job_obj)(
                index_id=job_obj.index_id,
                codes=job_obj.codes,
                name=job_obj.name,
                time_span=duration
            )
            ops_sequence.append(new_job)
            used_job_ids.add(op_id)

        # Fallback safeguard — avoid returning empty task lists
        if not ops_sequence:
            print(f"[WARN] TaskGenerator: Empty ops for JobAgent {jobagent_id}; generating fallback task.")
            default_job = self.jobs.jobs_object_list[0]
            fallback = type(default_job)(
                index_id=default_job.index_id,
                codes=default_job.codes,
                name=default_job.name,
                time_span=default_job.time_span
            )
            ops_sequence = [fallback]

        # -------- Logging (explainable trace) --------
        print(f"\n[TaskGen] JobAgent {jobagent_id if jobagent_id is not None else '?'} "
              f"generated → {len(ops_sequence)} unique operation(s):")
        for job in ops_sequence:
            possible_wcs = [
                getattr(wc, 'id', None) for wc in self.workcenters.workcenters_list
                if job.index_id in getattr(wc, 'resource_ids_list', [])
            ]
            print(f"   • {job.name:<20} (OpTypeID={job.index_id}, duration={job.time_span}) "
                  f"→ WorkCenters {possible_wcs}")
        print("------------------------------------------------------------")

        return ops_sequence
