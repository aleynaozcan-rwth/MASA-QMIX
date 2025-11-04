# CODE_HYGIENE_REPORT_V2

Generated: 2025-11-03

Summary
-------
This report documents each Python file in the repository, its inferred purpose, key dependencies, and a short recommendation in line with the "Option B" hygiene policy:
- Single source of defaults: `MARL/common/arguments.py`.
- No ad-hoc `args.<field> = ...` assignments outside that file (tests exempted).
- Use `self.run_args` (Runner-run copy) for runtime-derived values and `self.args` for CLI-level immutable settings.

This is intentionally concise so it is easy to review and act on.

Legend
------
- Purpose: quick one-line description.
- Depends: obvious first-order runtime imports / modules used (not exhaustive).
- Recommendation: concrete action (Keep / Review / Refactor / Remove / Move defaults to `MARL/common/arguments.py`).

Files
-----

# Top-level scripts & env

- `environment.py`
  - Purpose: SimPy-based MASA environment implementation (wait_for_decisions/pop_decision_reward etc).
  - Depends: simpy, utils/*, workcenter/job classes
  - Recommendation: Keep. Ensure it exposes `get_env_info()` and does not mutate global args; prefer reading flags from runner-provided args.

- `main.py`
  - Purpose: Orchestration entrypoint (parses args and launches Runner/training).
  - Depends: `MARL/common/arguments.py`, `MARL/runner.py`
  - Recommendation: Keep as thin orchestration (does not mutate args). Confirm it uses `get_common_args()` only and does not assign defaults.

- `run_masa_qmix.sh` (not Python; mentioned in repo)
  - Recommendation: Keep; verify it passes CLI flags instead of editing files.

# MARL package

- `MARL/__init__.py`
  - Purpose: package initializer
  - Recommendation: Keep.

- `MARL/runner.py`
  - Purpose: Orchestration, training loop, evaluation, KPI and gantt writes.
  - Depends: `MARL/common/rollout.py`, `MARL/agent/agent.py`, `MARL/common/replay_buffer.py`, `utils/gantt.py`
  - Recommendation: Keep. (Done) Normalize `self.args` vs `self.run_args` and avoid mutating global args. You applied a minimal normalization patch — expand the rule consistently across the file. Remove fallback GUI/plot code where not needed or wrap under configurable flags.

- `MARL/common/arguments.py`
  - Purpose: canonical CLI/default argument definitions (Option-B source of truth)
  - Depends: none
  - Recommendation: Keep as canonical. Move any remaining default assignments from other files here.

- `MARL/common/rollout.py`
  - Purpose: RolloutWorker and CommRolloutWorker; runs event-driven SimPy episodes, builds transitions, interacts with env.
  - Depends: Runner (calls), Agents, ReplayBuffer
  - Recommendation: Keep. Continue migrating shaping/avail/mask logic here so Runner remains coordinator only.

- `MARL/common/replay_buffer.py`
  - Purpose: Episode storage & sampling for replay-based updates.
  - Depends: numpy, random
  - Recommendation: Keep. Ensure its constructor reads seed/run_args properly (use run_args.seed supplied by Runner).

- `MARL/common/utils.py`
  - Purpose: Training utilities (returns, n-step, masks, tensor helpers).
  - Recommendation: Keep.

- `MARL/common/mask_utils.py`
  - Purpose: helper for availability masks
  - Recommendation: Keep.

- `MARL/common/analyse.py`
  - Purpose: offline analysis helpers used by plotting and reports
  - Recommendation: Keep/Review (non-critical to core runtime).

- `MARL/common/terms.py`
  - Purpose: small helper for canonical term strings (labels)
  - Recommendation: Keep.

# MARL/agent

- `MARL/agent/agent.py`
  - Purpose: Agents facade that instantiates policy classes and provides selection/training APIs.
  - Depends: MARL/policy/*
  - Recommendation: Keep. Ensure it consumes args (read-only) and never writes defaults.

# MARL/policy (policy implementations)
- `MARL/policy/qmix.py`
- `MARL/policy/maven.py`
- `MARL/policy/coma.py`
- `MARL/policy/central_v.py`
- `MARL/policy/vdn.py`
- `MARL/policy/reinforce.py`
- `MARL/policy/qtran_alt.py`
  - Purpose: policy classes and training logic for various algorithms.
  - Depends: MARL/network/*, MARL/common/utils, torch
  - Recommendation: Keep. They should read from `args` for static hyperparams and not perform ad-hoc assignment to `args`.

# MARL/network (neural nets)
- `MARL/network/base_net.py`
- `MARL/network/qtran_net.py`
- `MARL/network/qmix_net.py`
- `MARL/network/maven_net.py`
- `MARL/network/g2anet.py`
- `MARL/network/commnet.py`
- `MARL/network/coma_critic.py`
- `MARL/network/vdn_net.py`
  - Purpose: neural network architectures used by policies
  - Depends: torch, MARL/common/arguments (read-only)
  - Recommendation: Keep.

# utils (supporting environment & IO)
- `utils/gantt.py`
  - Purpose: Centralized gantt CSV formatting and selection-log helpers (migrated functions).
  - Recommendation: Keep; use everywhere and remove legacy writers across code after migration.

- `utils/util.py`
  - Purpose: misc helpers used widely
  - Recommendation: Keep; audit for any `args` writes.

- `utils/job.py`, `utils/jobagent.py`, `utils/workcenter.py`, `utils/operator.py`, `utils/machine_registry.py`
  - Purpose: domain model classes (jobs, tasks, machines, operators, registries)
  - Recommendation: Keep. Ensure they do not assign to global args; if they need config, accept it through env/runner parameters.

- `utils/task_generator.py`
  - Purpose: generate tasks/jobs for sim environment
  - Recommendation: Keep; tests use it — ensure no ad-hoc args writes.

- `utils/env_obs.py`
  - Purpose: observation shaping functions for environments
  - Recommendation: Keep; may be moved under MARL/common if heavily used by policies.

- `utils/config_loader.py`
  - Purpose: parse/prepare environment config files
  - Recommendation: Keep; ensure no side-effectful assignments to args.

- `utils/PDRs/shortestDistence.py`
  - Purpose: helper algorithm used by tools or planners
  - Recommendation: Keep or move into utils/path if you prefer consistent module layout.

# scripts/
- `scripts/run_train_qmix.py`
  - Purpose: convenience runnable script to start training
  - Recommendation: Keep; ensure it only passes CLI flags and does not mutate `MARL/common/arguments.py`.

# tools/
- `tools/*` (many small scripts: `test_arrivals.py`, `check_simpy_integration.py`, `run_initial_trace.py`, `smoke_generate_gantt.py`, `generate_pretty_gantt.py`, `run_smoke_gantt.py`, `test_concurrency.py`, `test_reward_envvars.py`)
  - Purpose: experiments, smoke tests, helper utilities
  - Recommendation: These are allowed to create runtime overrides but should not write defaults to `MARL/common/arguments.py`. Mark them as allowed (whitelist) if they truly need to set defaults; otherwise remove ad-hoc assignments.

# tests/
- `tests/*` (20+ test files)
  - Purpose: unit & integration tests (including `test_args_no_ad_hoc_assignments.py` which enforces Option-B)
  - Recommendation: Keep. Tests are allowed to construct local `args` fixtures and may write attributes for isolation.

# Top-level duplicates & housekeeping
- Duplicate file references (some files appear in multiple places in the repo listing) — ensure there are no accidental duplicate modules across paths. The file search found 124 Python files; confirm there are no near-duplicate implementations (e.g., repeated helpers under different folders). If duplicates exist, consolidate into `utils/` or `MARL/common/`.

Actionable next steps (prioritized)
-------------------------------
1. Finish Runner normalization: do a quick pass across `MARL/runner.py` replacing any accidental uses of the constructor-local `args` variable with `self.args` or `self.run_args` according to rule (CLI vs runtime). Aim for consistency rather than changing semantics. (Status: partially applied)
2. Migrate remaining legacy gantt writers: replace per-file writers with `utils/gantt.write_scheduling_trace` and remove fallbacks after tests exercise the CSV path.
3. Audit non-test `tools/` scripts for ad-hoc `args` assignments. If they intentionally set runtime overrides, add them to the test whitelist in `tests/test_args_no_ad_hoc_assignments.py` or refactor to pass flags via CLI.
4. Create a small linter test that warns about usage of bare `args` variable inside function scope (optional but helpful).
5. Consolidate repeated helpers into `MARL/common/` or `utils/` where appropriate.

Notes / Assumptions
-------------------
- Many files read `args` for static configuration — this is fine. The policy here only forbids writing into the shared args object outside `MARL/common/arguments.py`.
- I inferred each file's purpose from common naming and known project structure. If you want, I can produce a second, more detailed report that includes import graphs and usage counts per symbol.

If you'd like, I can now:
- A) perform the remaining Runner normalization sweep (update any lingering `args.` usages to `self.args`/`self.run_args`) and run tests again, or
- B) produce an extended report with import/usage graphs for every file.


---
End of CODE_HYGIENE_REPORT_V2.md
