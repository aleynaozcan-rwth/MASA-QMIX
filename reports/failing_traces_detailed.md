# Detailed failing traces (first failures)

Date: 2025-11-09T17:21:19.540419

## ERROR collecting tmp_smoke_test.py

```
tmp_smoke_test.py:14: in <module>
    print("Done", "Completed:", env.completed_jobs, "Active:", len(env.active_agents), "Pending:", len(env.pending_jobs))
                                ^^^^^^^^^^^^^^^^^^
E   AttributeError: 'MASAEnv' object has no attribute 'completed_jobs'
------------------------------- Captured stdout --------------------------------
[Lifecycle] New job added: Job_0 | total_jobs=1
[Lifecycle] New job added: Job_1 | total_jobs=2
[Lifecycle] New job added: Job_2 | total_jobs=3
[Lifecycle] New job added: Job_3 | total_jobs=4
[Lifecycle] New job added: Job_4 | total_jobs=5
[Lifecycle] New job added: Job_5 | total_jobs=6
[Lifecycle] New job added: Job_6 | total_jobs=7
[Lifecycle] New job added: Job_7 | total_jobs=8
[Lifecycle] New job added: Job_8 | total_jobs=9
```

**Analysis:** Trace contains references to legacy token(s): completed_jobs

