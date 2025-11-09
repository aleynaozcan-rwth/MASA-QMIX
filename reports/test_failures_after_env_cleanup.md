# Test Failures After environment.py Cleanup

Date: 2025-11-09T17:10:45.302390

- Total tests: 1
- Failures: 0
- Errors: 1
- Skipped: 0

## Failures grouped by file

###  — 1 failure(s)

- Test: `tmp_smoke_test`  
  - Type: error
  - Message: collection failure
  - Top traceback lines:
  ```
tmp_smoke_test.py:14: in <module>
    print("Done", "Completed:", env.completed_jobs, "Active:", len(env.active_agents), "Pending:", len(env.pending_jobs))
                                ^^^^^^^^^^^^^^^^^^
E   AttributeError: 'MASAEnv' object has no attribute 'completed_jobs'
  ```

## Most common missing/legacy tokens referenced in failures

- completed_jobs: referenced in 1 failure(s)
