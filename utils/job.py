"""
This file defines the Job class, i.e., the abstract resource assurance class
"""
# -------------------------------------------------------------------------
# NOTE:
# In this project, "Jobs" represent different operation types (e.g. Refueling,
# Oxygen supply, Weapon mounting). Each job ID (0–8) has:
#   - a code (short identifier),
#   - a human-readable name,
#   - and a required time span.
#
# Important: Each job type corresponds directly to a service resource.
# Example:
#   - Job "Refueling" ↔ Fuel truck
#   - Job "Oxygen"   ↔ Oxygen supply unit
#   - Job "Power"    ↔ Generator
#   - Job "Weapon mounting" ↔ Weapon team
#
# Therefore, in other parts of the code (e.g. restrict_dict in WorkCenters),
# job IDs are reused as "resource IDs". In other words:
#   JOB TYPE == REQUIRED RESOURCE TYPE
# -------------------------------------------------------------------------


# Default configuration mirror for Jobs (Phase 3C.0)
# Mirrors YAML key: 'jobs'
# Example structure: list of job descriptors with code/name/time_span
# TODO(Phase3C.1): integrate with Jobs.__init__() via merge_config(DEFAULT_JOBS, cfg)
DEFAULT_JOBS = [
    {"index_id": 0, "codes": "J0", "name": "JobTypeA", "time_span": 1.0},
    {"index_id": 1, "codes": "J1", "name": "JobTypeB", "time_span": 2.0},
    {"index_id": 2, "codes": "J2", "name": "JobTypeC", "time_span": 3.0},
]


class Jobs:
    def __init__(self, jobs_cfg=None):
        """
        Jobs registry. By default this will use the legacy hard-coded lists, but
        callers can pass `jobs_cfg` to provide canonical job metadata from
        configuration.

        Supported `jobs_cfg` formats:
          - None: fall back to legacy embedded lists (backwards compatible)
          - dict with keys 'codes', 'names', 'times' each mapping to a list of equal length
          - list of dicts where each dict has keys: index_id (optional), codes, name, time_span
        """
        self.jobs_object_list = []

        # Merge in-module DEFAULT_JOBS with provided jobs_cfg so that
        # callers may omit fields. Use merge_config to keep inputs immutable.
        try:
            from utils.config_loader import merge_config  # local import avoid cycles
            # merge_config expects dicts; DEFAULT_JOBS is a list, so wrap it
            if isinstance(DEFAULT_JOBS, list):
                wrapped_defaults = {"jobs": DEFAULT_JOBS}
                wrapped_cfg = {"jobs": jobs_cfg} if jobs_cfg is not None else None
                merged = merge_config(wrapped_defaults, wrapped_cfg)
                jobs_cfg = merged.get("jobs")
            else:
                jobs_cfg = merge_config(DEFAULT_JOBS, jobs_cfg)
        except Exception:
            # keep original jobs_cfg if merge fails
            pass


        if jobs_cfg is None:
            # legacy defaults
            jobs_codes = ["ZCTF", "SBTF", "JY", "TYY", "TD", "YQ", "DQ", "GDDE", "GD"]
            jobs_names = ["Cockpit", "Equipment cabin", "Refueling", "Hydraulic", "Power supply", "Oxygen", "Nitrogen", "Inertial navigation", "Weapon mounting"]
            jobs_times = [10, 10, 15, 4, 6, 2, 2, 10, 15]
            for i in range(len(jobs_names)):
                temp_object = Job(i, jobs_codes[i], jobs_names[i], jobs_times[i])
                self.jobs_object_list.append(temp_object)
            return

        # If jobs_cfg is a dict with parallel lists
        try:
            if isinstance(jobs_cfg, dict) and all(k in jobs_cfg for k in ("codes", "names", "times")):
                codes = list(jobs_cfg.get("codes", []))
                names = list(jobs_cfg.get("names", []))
                times = list(jobs_cfg.get("times", []))
                ln = min(len(codes), len(names), len(times))
                for i in range(ln):
                    temp_object = Job(i, codes[i], names[i], times[i])
                    self.jobs_object_list.append(temp_object)
                return
        except Exception:
            pass

        # If jobs_cfg is a list of dicts with explicit fields
        try:
            if isinstance(jobs_cfg, (list, tuple)):
                for i, entry in enumerate(jobs_cfg):
                    if not isinstance(entry, dict):
                        continue
                    idx = entry.get("index_id", i)
                    codes = entry.get("codes") or entry.get("code") or f"J{idx}"
                    name = entry.get("name") or entry.get("title") or f"Job{idx}"
                    time_span = entry.get("time_span") or entry.get("time") or 1
                    temp_object = Job(int(idx), codes, name, float(time_span))
                    self.jobs_object_list.append(temp_object)
                return
        except Exception:
            pass

        # Fallback: keep empty list (caller should handle)
        self.jobs_object_list = []

    # Which jobs are reserved, because jobs differ at different assurance locations
    # reserved_job_id :[0,2,3,1,...]
    def reserved_jobs(self, reserved_job_id):
        # Keep only the jobs whose indices appear in reserved_job_id. If input
        # is invalid or indices missing, no change is made.
        try:
            if not reserved_job_id:
                return
            reserved_jobs = []
            for id in reserved_job_id:
                reserved_jobs.append(self.jobs_object_list[int(id)])
            self.jobs_object_list = reserved_jobs
        except Exception:
            # be robust: do not raise here; caller can validate beforehand
            return


class Job:
    def __init__(self, index_id, codes, name, time_span):
        self.index_id = index_id
        self.codes = codes
        self.name = name
        self.time_span = time_span
