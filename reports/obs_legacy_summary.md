# Observation Legacy Scan — Summary

Date: 2025-11-09

This report lists discovered references to legacy observation attributes, helper functions, and 11-D naming across tracked source files. Artifacts, logs, and the `reports/` directory were excluded from the scan and are not modified.

## Severity classification
- HIGH: still used in active environment or observation building code paths (must be reviewed before removals).
- MEDIUM: used in tests, metrics, or runner code (needs coordination with tests and validators).
- LOW: docstrings, comments, tools, or analysis scripts (informational; safe to update after coordination).

---

## HIGH severity (active env / obs builder usage)

These occurrences are in `environment.py`, `utils/env_obs.py`, and `utils/jobagent.py` and indicate live state or helper functions that the environment and observation builder still reference or that the env uses in reward/metrics computation.

Examples (file:line)

- `environment.py:107` — `self.remaining_time = 0.0`
- `environment.py:117` — `def progress_ratio(self):`
- `environment.py:770` — `self.completed_jobs = 0`
- `environment.py:773` — `self._recent_rewards = deque(maxlen=20)`
- `environment.py:1346` — `self._recent_rewards.append(reward)`
- `environment.py:1361` — `self.completed_jobs += 1`
- `environment.py:1866` — `job.remaining_time = dur`
- `environment.py:2094` — `job.remaining_time = 0.0`
- `environment.py:2396` — `def _util_machines(self) -> float:`
- `environment.py:2415` — `def _util_ops(self) -> float:`
- `utils/env_obs.py:15` — doc/comment referencing `wait_time_norm -> job.wait_time / env.max_wait_time`
- `utils/env_obs.py:121` — `wait_time_norm = np.clip(wait_time_val / float(max_wait), 0.0, 1.0)`
- `utils/env_obs.py:142` — `legacy helpers such as _wip, _recent_rewards or completed_jobs.`
- `utils/jobagent.py:118` — `self.remaining_time = 0.0`
- `utils/jobagent.py:148` — `def progress_ratio(self):`

Notes and impact
- `completed_jobs`, `_recent_rewards`, `remaining_time`, and `progress_ratio` are still present and used by runtime bookkeeping and reward computation (`pop_decision_reward()`), and by the environment job lifecycle. Removing them requires a careful migration: replace usages in reward computations, metrics, and job lifecycle with canonical sources or remove reward shaping that depends on these fields.
- `_util_machines()` and `_util_ops()` are used by the `runner` and may be used in metrics and traces — they should be migrated if you plan to remove legacy util helpers.

Recommended next steps for HIGH items
1. Decide whether these KPI helpers should be removed or kept for logging — if removed, implement replacements for any code that reads them (reward, runner, metrics).
2. Triage `build_agent_obs()` and its tests to ensure they do not depend on these legacy features; the current builder is 6-D and does not use them, but environment bookkeeping still does.

---

## MEDIUM severity (tests, metrics, runner usage)

These occurrences appear in tests, the runner, and metrics utilities. They indicate places that either assert legacy behavior or compute metrics using legacy fields.

Selected hits

- `MARL/runner.py:1533` — `avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)`
- `MARL/runner.py:1534` — `util_m = self.env._util_machines()`
- `MARL/runner.py:1535` — `util_o = self.env._util_ops()`
- `my_data_and_graph/metrics.py:170` — `avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))`
- `tests/test_env_obs.py:11` — `aobs = env._build_agent_obs(job)` (test calls into builder)
- `tests/test_env_obs.py:12` — `aobs2 = env_obs.build_agent_obs(env, job)`
- `tests/test_observation_shapes.py:10` — `obs = env._build_agent_obs(env.jobs[0])`

Notes and impact
- Tests reference the builder and env bookkeeping; before removing or changing legacy helpers, update tests to use canonical attributes or to mock/mimic any removed data.
- Runner and metrics use `completed_jobs` and util helpers for logging and metrics aggregation; if you remove them, update the runner's metric calculations to use the canonical quantities (for example, compute avg wait using an alternative counter or event-based aggregation).

Recommended next steps for MEDIUM items
1. Update tests to rely on the canonical 6-D builder and explicit env attributes where appropriate.
2. Change runner metric computations to use canonical env attributes (or provide a compatibility shim while migrating).

---

## LOW severity (docstrings, comments, tools)

These are non-functional references (comments, docstrings, small utility scripts) that mention 11-D observations or legacy function names. These are safe to update anytime to avoid confusion.

Selected hits

- `MARL/runner.py:795` — comment: `- 11D observations (progress_ratio)`
- `tools/test_reward_envvars.py:26` — writes `reward_weights_runtime` artifact (tooling)

Notes
- The previous refactor intentionally kept report files and some docstrings for audit. Update them later for consistency.

---

## Machine-readable hits
All machine-readable entries (file, line, match line, and assigned severity) are written to `reports/obs_legacy_hits.json` in this repo.

---

## Next steps / options
- I can prepare a safe patch that:
  - (A) Converts runner and metrics calls to use canonical env attributes (low-risk API-preserving changes), and
  - (B) Updates tests accordingly (update expected shapes/values), then run tests.
- Or I can prepare a targeted patch to remove or migrate specific HIGH-risk fields (e.g., remove `_recent_rewards`, require callers to compute reward traces manually). That is more invasive and should be staged.

Which action should I perform next? (Options: `prepare-migration-pr`, `update-runner-metrics`, `update-tests-only`, `just-report`)


## Per-file legacy token inventory (enriched)

Below are per-file sections listing all matched legacy tokens found during the scan, a short description of what each token/identifier does, and why it likely remains in the codebase today.

### environment.py

- `completed_jobs`
  - What: integer counter of how many jobs have completed in the environment.
  - Why still present: used for runtime bookkeeping, metrics (avg wait time), and checkpoint/logging. It is actively read by the runner and metrics utilities. Marked HIGH because removing it would affect runtime metrics and reward computations.

- `_recent_rewards`
  - What: a deque used as a short rolling cache of recent reward values (maxlen=20).
  - Why still present: appended to in `pop_decision_reward()` for instrumentation or smoothing; useful for debug/monitoring; HIGH risk to remove without replacing telemetry.

- `remaining_time` (on job objects)
  - What: per-job remaining processing time (set when an operation starts, decremented when finished).
  - Why still present: used by job lifecycle logic and some legacy observation features. Removing/moving it requires careful replacement where job timing is needed.

- `progress_ratio` (job method)
  - What: convenience method returning how far a job progressed (used by older 11-D observation builder).
  - Why still present: legacy helper from older observation API; currently referenced in comments and possibly tooling. It is HIGH impact if removed while older analytics expect it.

- `_util_machines()` and `_util_ops()`
  - What: helper methods that compute utilization metrics for machines and operations (return floats in [0,1]).
  - Why still present: consumed by `runner.py` and metrics code to report utilization; MEDIUM/HIGH depending on whether you use those metrics in production traces.

- `_wip()`
  - What: simple count of jobs not finished (work-in-progress).
  - Why still present: used for quick active-job counts and compatibility with older code; can be replaced by `active_jobs_count()` but still referenced across code.

Notes: `environment.py` is the primary owner of legacy bookkeeping state; modifications here require coordination with runner/metrics/tests.

### utils/env_obs.py

- `wait_time_norm` (doc and code references)
  - What: normalized wait-time feature produced by the observation builder (job.wait_time / env.max_wait_time).
  - Why still present: the current 6-D builder still exposes a normalized wait_time feature; the docstring references the specific normalization formula. This is functional and not legacy per se, but the file also documents legacy helpers.

- references to `_recent_rewards`, `completed_jobs`, `_wip` in comments
  - What: notes about legacy helpers that older builders used.
  - Why still present: comments and historical rationale; the implementation is already canonical 6-D but the comments mention legacy helpers for traceability. Low risk but should be updated for clarity.

### utils/jobagent.py

- `remaining_time`
  - What: per-job remaining duration placeholder stored on JobAgent.
  - Why still present: job lifecycle and gantt/trace writers use it to compute durations and append gantt records; HIGH if removed without migrating consumers.

- `progress_ratio`
  - What: job-level helper that computes a fraction of work completed for a job.
  - Why still present: convenience helper used historically by the 11-D obs builder and by some debug traces. Safe to deprecate but must update any consumer code.

### MARL/runner.py

- references to `11D` and `progress_ratio` in comments
  - What: documentation note referring to the older 11-D observation layout.
  - Why still present: historical comment; LOW severity but should be updated.

- `avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)`
  - What: computes average wait using `total_wait_time` and `completed_jobs` counters.
  - Why still present: runner metrics use it for monitoring and logging. MEDIUM risk to migrate: tests and logging expectations may rely on this number.

- `util_m = self.env._util_machines()` and `util_o = self.env._util_ops()`
  - What: runner calls environment helpers to get utilization metrics.
  - Why still present: used for reporting and telemetry in the runner. MEDIUM risk — replaceable with alternative canonical metrics but requires coordinated changes.

### my_data_and_graph/metrics.py

- `completed_jobs` and `avg_wait` calculations
  - What: offline metric computations and ingestion expect `completed_jobs` and `total_wait_time` to compute average wait and utilization.
  - Why still present: metrics post-processing uses canonical runtime counters; MEDIUM risk to change.

### utils/gantt.py

- usage of `env.completed_jobs` and `env.total_wait_time`
  - What: Gantt plotting helpers compute average wait and other summary metrics using environment counters.
  - Why still present: used to generate visualizations/metrics; MEDIUM risk.

### tests/*

- calls into `env._build_agent_obs(job)` and `env_obs.build_agent_obs(env, job)`
  - What: tests exercise the public/compat observation builder APIs and assert shapes/values.
  - Why still present: test coverage for builder — these are MEDIUM because tests must be updated in lockstep with any removal of legacy helpers.

### tools/test_reward_envvars.py (tool)

- `reward_weights_runtime` artifact write
  - What: a small utility that checks environment picks up environment variables for reward shaping.
  - Why still present: developer tooling; LOW severity for the migration — it is not part of production runtime.


## Dependency Graph Summary

This graph maps modules that still depend on the legacy observation values / helpers discovered in the scan. Use this to plan an ordered migration: start by updating metrics/runner or provide compatibility shims in `environment.py`.

- environment.py
  - Provides: `completed_jobs`, `_recent_rewards`, `total_wait_time`, `remaining_time`, `progress_ratio`, `_util_machines()`, `_util_ops()`, `_wip()`
  - Consumed by: `MARL/runner.py`, `utils/gantt.py`, `my_data_and_graph/metrics.py`, parts of `environment`'s own reward computations, and some debug artifacts.

- utils/env_obs.py
  - Provides: canonical `build_agent_obs()` (6-D); documents legacy helpers (`_recent_rewards`, `_wip`, `completed_jobs`) in comments
  - Consumed by: `environment._build_agent_obs`, test harnesses, `smoke_test_obs_6d.py`.

- utils/jobagent.py
  - Provides: per-job fields `remaining_time`, `progress_ratio()`
  - Consumed by: `environment` job lifecycle, `utils/gantt.py`, and possibly older observation code.

- MARL/runner.py
  - Consumes: `env.completed_jobs`, `env.total_wait_time`, `env._util_machines()`, `env._util_ops()` for metrics and logging
  - Downstream impact: test expectations, CI smoke checks, dashboards that parse runner output.

- my_data_and_graph/metrics.py
  - Consumes: `completed_jobs` and `total_wait_time` for offline metrics

- utils/gantt.py
  - Consumes: same counters for average wait and plotting

- tests/*
  - Consume: `build_agent_obs` and environment APIs; tests will need updates if any of the above environment fields are removed or changed.


## Migration guidance (brief)

1. If you prefer minimal-risk: add compatibility shims in `environment.py` that keep the fields (e.g., `completed_jobs`) but mark them deprecated and update consumers over time.
2. If you prefer faster cleanup: update `runner.py`, `metrics.py`, and `gantt.py` to compute metrics from canonical sources (events or derived counters) and update tests, then remove fields from `environment.py` in a follow-up PR.
3. Update documentation and comments (e.g., runner comments mentioning 11-D) as the last step to avoid confusing future readers.

---

The machine-readable hits file `reports/obs_legacy_hits.json` was written alongside this summary and contains the raw match list with file/line/context and severity.

Next: I can either (A) prepare PR-ready patches to migrate the runner/metrics/tests to canonical attributes, or (B) create compatibility shims in `environment.py` and schedule follow-up removals. Which approach do you prefer?