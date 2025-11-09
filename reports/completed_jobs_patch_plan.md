# completed_jobs Patch Plan

Date: 2025-11-09T00:00:00Z

This document groups all occurrences of `completed_jobs` (from `reports/completed_jobs_hits.json`) and provides a prioritized, per-file patch plan to remove legacy `completed_jobs` usage by replacing it with computed finished counts or using a metrics API if available.

Summary
-------

- Files to patch: 7
- High-impact files: `MARL/runner.py`, `my_data_and_graph/metrics.py`, `environment.py`
- Estimated LOC changed: ~50-90 lines (conservative estimate; average 2–4 lines per match)

Why these replacements
----------------------
The codebase no longer maintains a runtime `completed_jobs` counter on `env`. Replacing reads and prints with the canonical computation:

  len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])

preserves correct semantics and is safe (fail-fast if `env.jobs` not present). When available, prefer `env.metrics.completed_jobs()` if a metrics API exists — that centralizes logic and keeps call sites concise.

Replacement contract (recommended)
---------------------------------
- Input: any site reading/printing/assigning `completed_jobs`.
- Output: integer count of finished jobs (0..N).
- Error modes: if `env.jobs` absent, use 0 as fallback or raise depending on caller context.

Edge cases considered
--------------------
- env.jobs may be None or missing: use getattr(env, 'jobs', []) and guard with or []
- jobs may not have `finished` attribute: use getattr(j, 'finished', False)
- Performance: computing len(...) is O(#jobs). For extremely large job lists, consider caching in a metrics API.

Per-file prioritized patches
---------------------------

NOTE: each file section includes exact replacement snippets you can apply with a one-liner patch or small edit. Tests and code that expect exact historical counts may need slight adjustments (e.g., variable names or order of logging). Prefer small, focused PRs.

1) MARL/runner.py — Impact: HIGH

Total matches: 7

Matches (by line)

- Line 1193 (read):
  Original:
  ```py
  before_completed = int(getattr(self.env, 'completed_jobs', 0))
  ```
  Suggested replacement:
  ```py
  before_completed = len([j for j in getattr(self.env, 'jobs', []) if getattr(j, 'finished', False)])
  ```

- Line 1246 (read):
  Original:
  ```py
  self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
  ```
  Suggested replacement (use computed count, guard by 1):
  ```py
  _completed_count = max(1, len([j for j in getattr(self.env, 'jobs', []) if getattr(j, 'finished', False)]))
  self.wait_time_records.append({0: (self.env.total_wait_time / _completed_count)})
  ```

- Line 1493 / 1533 / 1549 / 1574 (reads / reporting): similar replacements

Patch steps (recommended order)

1. Replace direct reads of `self.env.completed_jobs` or getattr(...'completed_jobs'...) with the computed expression shown above. Use a small helper local variable `_completed_count` when a value is reused within the same block to avoid recomputation.
2. Run unit tests and the runner integration smoke harness.
3. If performance is a concern (many jobs), add a TODO to replace with `env.metrics.completed_jobs()` and create a follow-up change to implement the metrics API.

Typical diff example
--------------------

Before

```py
before_completed = int(getattr(self.env, 'completed_jobs', 0))
avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
```

After

```py
before_completed = len([j for j in getattr(self.env, 'jobs', []) if getattr(j, 'finished', False)])
_completed_count = max(1, before_completed)
avg_wait = self.env.total_wait_time / _completed_count
```

2) my_data_and_graph/metrics.py — Impact: HIGH

Total matches: 4

Matches (by line)

- Line 108 (read in function signature / comment): function `_compute_from_gantt(..., completed_jobs: int = 0)` expects an int.
  Suggested change: change the function API to accept optional `env` or compute `completed_jobs` upstream as a count. Recommended two options:

Option A (internal compute): keep signature but compute inside caller rather than relying on passed integer. Replace call sites to pass computed integers from env when available.

Option B (API change, preferred): change `_compute_from_gantt` to not accept `completed_jobs` and instead compute avg_wait internally using len(...) if gantt data is available.

Example replacement at call site

Before

```py
completed_jobs = int(item.get('completed_jobs', 0))
util = _compute_from_gantt(gantt, ..., total_wait_time=total_wait_time, completed_jobs=completed_jobs)
```

After (compute from env.jobs when env is available)

```py
completed_jobs_count = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])
util = _compute_from_gantt(gantt, ..., total_wait_time=total_wait_time, completed_jobs=completed_jobs_count)
```

Or prefer to remove the parameter and let `_compute_from_gantt` compute from gantt records.

Patch steps

1. Search for call sites that pass `completed_jobs` integer; update to compute a count at the call site (or refactor `_compute_from_gantt` to compute inside).
2. Run metrics unit tests and any plotting utilities that rely on the function.

3) environment.py — Impact: MEDIUM

Total matches: 13 (mostly logging and assignment sites)

Notes

- Many occurrences are already present as computed lists named `completed_jobs = [j ...]`. We should canonicalize to a single variable name `completed_jobs_count` set to an int to avoid confusion and to reduce memory allocation.

Suggested canonicalization

Replace instances where a list of job objects named `completed_jobs` is created with an integer count variable:

Before

```py
completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
tfs.write(... Completed={len(completed_jobs)} ...)
```

After

```py
completed_jobs_count = len([j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)])
tfs.write(... Completed={completed_jobs_count} ...)
```

Patch steps

1. Replace the list-assignment with an integer-count variable named `completed_jobs_count` in logging blocks.
2. Update subsequent uses in the block to use the integer. If a full list of job objects is required by logic, keep the list but rename to `completed_jobs_list`.
3. Run lifecycle logging smoke tests and validate output formatting.

4) utils/gantt.py — Impact: MEDIUM

Total matches: 1

Match (line 644)

Original:

```py
avg_wait = float(getattr(env, 'total_wait_time', 0.0)) / max(1.0, float(getattr(env, 'completed_jobs', 0) or 1))
```

Suggested replacement:

```py
completed_count = max(1.0, float(len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])))
avg_wait = float(getattr(env, 'total_wait_time', 0.0)) / completed_count
```

Patch steps

1. Replace the getattr(... 'completed_jobs') expression with the computed count as shown.
2. Run gantt-related plotting and metrics tests.

5) utils/env_obs.py — Impact: LOW

Total matches: 1

Match (line 142): documentation string referencing legacy helpers.

Suggested change: update docstring to reflect new canonical approach and remove mention of `completed_jobs`.

Before

```py
legacy helpers such as _wip, _recent_rewards or completed_jobs.
```

After

```py
legacy helpers such as _wip or _recent_rewards. Completed-job counts are computed on-demand via env.jobs.
```

6) tmp_smoke_test.py — Impact: LOW

Total matches: 1

Match (line 14)

Original:

```py
print("Done", "Completed:", env.completed_jobs, "Active:", len(env.active_agents), "Pending:", len(env.pending_jobs))
```

Suggested replacement:

```py
completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])
print("Done", "Completed:", completed, "Active:", len(env.active_agents), "Pending:", len(env.pending_jobs))
```

7) tmp_test_gantt_decision_trace.py — Impact: LOW

Total matches: 1

Match (line 35): assignment `self.completed_jobs = 0` in a test helper. Tests should avoid mutating env state that no longer exists.

Suggested replacement: remove the assignment or replace `self.completed_jobs` uses with an explicit counter in the test fixture (e.g., `self._completed_jobs_count = 0`) and adjust assertions to compute finished jobs from `env.jobs`.

Machine-readable plan
---------------------

See `reports/completed_jobs_patch_plan.json` for a per-file machine-readable plan including replacement snippets.

Final notes and follow-ups
-------------------------

- Run tests after each file patch (small PRs recommended).
- If several call sites compute the count repeatedly inside tight loops, consider adding `env.metrics.completed_jobs()` to centralize and optimize counting.
- If you want, I can produce per-file git-style patch diffs (unified diffs) for the top N files (recommended: start with `MARL/runner.py` and `my_data_and_graph/metrics.py`).

End of plan.
