# SYSTEM_STATUS
Generated: 2025-10-28T19:20:48.239152 UTC

## Summary

|Metric|Value|
|---|---|
|num_jobs_reset|10|
|mean_ops_per_job|None|
|n_machines|5|
|n_actions|5|
|obs_dim|(10, 11)|
|state_dim|64|
|episode_limit|200|
|arrival_lambda_config|0.05|
|reward_weights_runtime|{'alpha': 1.0, 'beta': 0.3, 'gamma': 0.1, 'delta': 0.1}|

## Static checks (sources & values)

- Config file: configs/env_config.yaml present: True

- Config snapshot saved to `artifacts/config_snapshot.json`

## Avail mask checks (sample)
- Job 0 mask_ones=[0, 2, 4]
  - machine_index=0 name=M1 capable=True machine_free=True operator_free=True
  - machine_index=2 name=M3 capable=True machine_free=True operator_free=True
- Job 1 mask_ones=[0, 2, 4]
  - machine_index=0 name=M1 capable=True machine_free=True operator_free=True
  - machine_index=2 name=M3 capable=True machine_free=True operator_free=True
- Job 2 mask_ones=[0, 2]
  - machine_index=0 name=M1 capable=True machine_free=True operator_free=True
  - machine_index=2 name=M3 capable=True machine_free=True operator_free=True

## Short run (no arrivals) key findings
- Gantt records (test_concurrency): 2 (see `artifacts/test_concurrency_out.json`)
- Concurrency examples within same WorkCenter: FOUND — sample entries written to `artifacts/test_concurrency_out.json` (`concurrency_same_wc_examples` non-empty).

Notes: The short scripted concurrency test drove `env.wait_for_decisions()` and resumed pending decisions with round-robin machine selections; this produced overlapping gantt records in the same WorkCenter (evidence that per-machine resources allow concurrent processing).

## Operator logs (recent)
- Sample (from concurrency test stdout):

  - "Operator 1 assigned job 0 at WorkCenter 0"
  - "Operator 2 assigned job 1 at WorkCenter 0"
  - "Operator 1 released from job 0"
  - "Operator 2 released from job 1"

Full operator/gantt details: `artifacts/test_concurrency_out.json` (gantt_records and concurrency_same_wc_examples)

## Arrival-open test
- dynamic adds count (test_arrivals): 9 — dynamic job addition ACTIVE (see `artifacts/test_arrivals_out.json`).
- sample dynamic add stdout recorded in `artifacts/test_arrivals_out.json` and console logs.

## Initial jobs comparison
The historical `my_data_and_graph/historydata/initial_jobs.txt` was in the older (per-WorkCenter) format and has been archived to `archive/initial_jobs_history_old.txt` to avoid accidental usage. The reset-produced (current) initial jobs are the authoritative source and were used during the recent checks (printed to `artifacts/initial_jobs_dump.txt`).

Action taken: archived old dump to `archive/initial_jobs_history_old.txt`.

## Notes & Warnings
- MASAEnv instantiated and exercised by multiple small tests (`tools/test_concurrency.py`, `tools/test_arrivals.py`, `tools/test_reward_envvars.py`).

## Automated checks summary

|Check|Result|Notes / artifacts|
|---|---:|---|
|Concurrency (machines in same WC can run concurrently)|✅|`artifacts/test_concurrency_out.json` (non-empty `concurrency_same_wc_examples`)|
|Reward weights set to best-found values (alpha=1.0, beta=0.3, gamma=0.1, delta=0.1)|✅|`artifacts/reward_envcheck.json`|
|Dynamic arrivals (task generator active)|✅|`artifacts/test_arrivals_out.json` (dynamic_adds_count=9)|
|Mask logic (machine-level avail mask)|✅|See `artifacts/runtime_snapshot.json` avail_checks sample|
|Operator assignment deterministic (basic check)|✅|Operator logs in concurrency test show deterministic assignments; further trace matching available on request|
|State/obs shape consistency|✅|state_dim=64, obs_dim=11 as reported in runtime snapshot|

## Conclusion: Ready for training?

- YES — automated checks PASSED. Recommended next step: start the 2000-episode training with the updated reward weights and EXP_SEED=42. See `scripts/run_train_qmix.py` and use the environment variables documented in the repo.