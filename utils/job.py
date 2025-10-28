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


class Jobs:
    def __init__(self):
        # Create all job objects, and use the list index as the id index (similar to a dictionary)
        self.jobs_object_list = []
        # id: 0 1 2 3 4 5 6 7 8
        jobs_codes = ["ZCTF", "SBTF", "JY", "TYY", "TD", "YQ", "DQ", "GDDE", "GD"]
        jobs_names = ["Cockpit", "Equipment cabin", "Refueling", "Hydraulic", "Power supply", "Oxygen", "Nitrogen", "Inertial navigation", "Weapon mounting"]
        jobs_times = [10, 10, 15, 4, 6, 2, 2, 10, 15]
        for i in range(len(jobs_names)):
            # FIX: added jobs_names[i] as the 'name' argument
            temp_object = Job(i, jobs_codes[i], jobs_names[i], jobs_times[i])
            self.jobs_object_list.append(temp_object)

    # Which jobs are reserved, because jobs differ at different assurance locations
    # reserved_job_id :[0,2,3,1,...]
    def reserved_jobs(self, reserved_job_id):
        assert type(reserved_job_id[0]) == int
        reserved_jobs = []
        for id in reserved_job_id:
            reserved_jobs.append(self.jobs_object_list[id])
        self.jobs_object_list = reserved_jobs


class Job:
    def __init__(self, index_id, codes, name, time_span):
        self.index_id = index_id
        self.codes = codes
        self.name = name
        self.time_span = time_span
