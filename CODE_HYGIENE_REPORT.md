# CODE HYGIENE REPORT — MASA-QMIX

Date: 2025-11-03
Scope: Phase 1 — static repository analysis & dependency/topology mapping for enforcing Option B (arguments.py is the only default writer; no other module mutates args).

Summary
-------
This report documents the dependency topology and per-file hygiene checks for the core files in this repository. I focused on the modules central to training and environment operation: entrypoints, `MARL/` (runner, rollout, agent/policy), `environment.py`, and the `utils/` helpers introduced during the earlier refactor. The report identifies locations that read `args`, any arg mutations outside `MARL/common/arguments.py`, import / layer crossings, IO-in-core-logic, and other code smells relevant to Option B.

This is Phase 1 (analysis only). No code was changed while generating this report.

Top-level topology summary
--------------------------
- Entry points
  - `main.py` (CLI) — canonical entrypoint for MARL training. It calls `get_common_args()` then constructs a `Runner`.
  - `scripts/run_train_qmix.py` — alternative entrypoint used by some workflows; currently left as-is per instructions.
  - tools: `tools/run_smoke_gantt.py`, `tools/run_initial_trace.py`, etc. — convenience scripts which should use `get_smoke_args()` rather than mutating args.

- Training loop path
  - main.py -> Runner (MARL/runner.py) -> RolloutWorker (MARL/common/rollout.py) -> ReplayBuffer (MARL/common/replay_buffer.py) -> Agents (MARL/agent/agent.py) -> Policies (MARL/policy/*.py) and Networks (MARL/network/*.py)

- Environment path
  - `environment.MASAEnv` provides SimPy integration and runtime shapes via `get_env_info()`, `reset()`, and event-driven `wait_for_decisions()`.
  - Runner queries `env.get_env_info()` and then sets run-local shapes (via `self.run_args`) so policies/agents/readers get consistent dims.

- Where agents/policies constructed
  - Runner constructs `Agents` / `CommAgents` instances and passes `run_args` to them. Policies and networks read `.args` for hyper-parameters and shapes (n_agents, n_actions, obs_shape, state_shape).

- Output files & artifacts
  - Runner writes many artifacts to `./my_data_and_graph/historydata/` (CSV/PNG/JSON logs, learning metrics, gantt snapshots, run_summary.json).

Dependency map (file → imports and known importers)
--------------------------------------------------
Notes:
- I list the primary static imports in each file (not every stdlib import).
- "Importers" lists modules I found that import the file (best-effort from static analysis and run-time observations).

## File: main.py
Purpose: CLI entrypoint and orchestrator for MARL training (calls Runner).
Inputs: args (from `MARL/common/arguments.get_common_args()`); user selection of mode.
Outputs/Side-effects: instantiates Runner, may trigger training/evaluation, prints summary to stdout.
Findings:
- [ ] args mutation? (yes/no): no — `main.py` does not assign `args.<field> = ...` (it calls `get_*_args()` to augment args).
- [ ] misplaced config? (yes/no): no
- [ ] cross-layer import smell? (yes/no): minor — it imports `environment.MASAEnv` and then `MARL.runner.Runner` which is normal for an orchestrator.
- [ ] long function / god-object? (yes/no): no
- [ ] IO in core logic? (yes/no): prints to stdout only
Notes: `main.py` adheres to Option B: it reads args and forwards them; it has a comment explicitly saying not to mutate `args` here.
Imported modules: `environment`, `MARL.runner`, `MARL.common.arguments`
Imported by: CLI / user scripts, tests

## File: MARL/common/arguments.py
Purpose: Centralized CLI defaults and algorithm groups (single source of truth for defaults).
Inputs: none (reads nothing at import time); exposes `get_common_args()` and `get_smoke_args()`.
Outputs/Side-effects: constructs an `args` namespace and returns it. NOTE: this is the only file allowed to set defaults.
Findings:
- [ ] args mutation? (yes/no): yes — intentionally (this file defines defaults). This is the approved single mutator.
- [ ] misplaced config? (yes/no): no
- [ ] cross-layer import smell? (yes/no): no
- [ ] long function / god-object? (yes/no): no (but many grouped helpers)
- [ ] IO in core logic? (yes/no): no
Notes: Several helper functions (get_mixer_args, get_smoke_args) mutate an args namespace (intentionally). This file is the authorised writer for defaults.
Imported by: `main.py`, `MARL/common/rollout.py` (fallback), many scripts/tests call `get_smoke_args()`.

## File: MARL/runner.py
Purpose: Orchestrates training/evaluation loop; queries env for runtime shapes; instantiates Agents, Buffer and RolloutWorker; writes logs and artifacts.
Inputs: `env` (MASAEnv), `args` (read-only shared CLI namespace). It creates a run-local copy (`self.run_args`) and uses it for runtime-derived values.
Outputs/Side-effects: extensive IO (history files, PNG/CSV, run_summary.json, logs); may call `env.reset()` and `env.wait_for_decisions()`; writes metrics via `_append_learning_metrics()` and many file writes at end-of-run.
Findings:
- [ ] args mutation? (yes/no): PARTIAL — Runner creates `self.run_args = deepcopy(args)` and writes runtime-derived values into `self.run_args` (per Option B), not the shared `args`. It does, however, still call `setattr(self.env, "quiet_env", ...)` — that writes to `env`, not `args`.
- [ ] misplaced config? (yes/no): some IO and artifact writing is heavy but expected for Runner; no training hyperparams written here other than using `self.run_args`.
- [ ] cross-layer import smell? (yes/no): Runner imports plotting utilities and includes large plotting logic in-file (could be split), but this is acceptable for a top-level orchestrator.
- [ ] long function / god-object? (yes/no): yes — `Runner` is a large class with many responsibilities (logging, plotting, kpi writing, snapshotting, training loop). Candidate to split into smaller helper classes in Phase 2 but no functional change now.
- [ ] IO in core logic? (yes/no): yes — Runner intentionally writes many artifacts. This is acceptable but should be documented.
Notes: Runner already follows Option B by using `self.run_args`. However many `self.args.*` reads persist throughout (`self.args.n_epoch`, `self.args.evaluate_cycle`, etc.) — these are reads only and are allowed. Important: Runner still occasionally uses `self.args` in prints and file paths; these are reads, not writes.
Imported modules: MARL.common.rollout, MARL.agent.agent, MARL.common.replay_buffer, matplotlib
Imported by: `main.py`, scripts

## File: MARL/common/rollout.py
Purpose: Encapsulates rollout logic for both legacy step-based envs and event-driven SimPy envs; provides helpers that Runner uses to turn decision-batches into replay transitions and to apply actions.
Inputs: `env`, `agents`, `buffer`, `args` (read-only or run-local passed by Runner).
Outputs/Side-effects: may write small scheduling log lines (to scheduling_trace.csv) in `process_and_apply_actions()`, and places brief print/log messages.
Findings:
- [ ] args mutation? (yes/no): no — it reads args and fallbacks to `MARL.common.arguments.get_common_args()` when needed downstream. Does not write back to global args.
- [ ] misplaced config? (yes/no): no
- [ ] cross-layer import smell? (yes/no): moderate — `RolloutWorker` references `MARL.common.mask_utils` and environment internals; this is expected.
- [ ] long function / god-object? (yes/no): the class is quite long and contains many helper sections; it's central for decision→transition conversion.
- [ ] IO in core logic? (yes/no): small IO (scheduling_trace.csv append) — acceptable but should be noted.
Notes: The API already accepts `args` as constructor param and `runner_args` as override in helpers — that's good for Option B.
Imported modules: MARL.common.mask_utils (optional), numpy, torch (optional)
Imported by: `MARL/runner.py`

## File: MARL/common/replay_buffer.py
Purpose: Episode-based replay buffer used by Runner and Agents; stores transitions and samples minibatches.
Inputs: constructor args (episode_capacity, seed), `store_episode()` input transitions.
Outputs/Side-effects: in-memory buffer; minimal file IO (none in current code). Accepts `n_actions` when sampling.
Findings:
- [ ] args mutation? (yes/no): no
- [ ] misplaced config? (yes/no): no
- [ ] cross-layer import smell? (yes/no): no
- [ ] long function / god-object? (yes/no): no
- [ ] IO in core logic? (yes/no): no
Notes: Buffer.sample uses `n_actions` parameter (Runner passes `self.run_args.n_actions`) — good.
Imported by: `MARL/runner.py`, `MARL/common/rollout.py`

## File: MARL/agent/agent.py
Purpose: High-level agent wrappers (`Agents`, `CommAgents`) which construct policies and delegate training calls.
Inputs: `args` (namespace) in constructor; it expects shapes in args (n_agents, n_actions, rnn_hidden_dim, etc.).
Outputs/Side-effects: constructs policy objects, calls `.train()` and `.select_actions()` during runtime.
Findings:
- [ ] args mutation? (yes/no): no — constructors read args but don't mutate them. They expect the caller (Runner) to provide run-local shapes.
- [ ] misplaced config? (yes/no): no
- [ ] cross-layer import smell? (yes/no): some coupling to policy modules (expected)
- [ ] long function / god-object? (yes/no): contains several helpers; medium-sized.
- [ ] IO in core logic? (yes/no): no
Notes: Agents accept `args` in constructor. For strict Option B we must ensure Runner passes `self.run_args` (it already does in many places) — verify all call-sites in Runner use `run_args` (some prints may still reference `self.args`).
Imported by: `MARL/runner.py`

## File: MARL/policy/qmix.py and other policy files (qtran_base.py, maven.py, reinforce.py, central_v.py, etc.)
Purpose: Policy implementations with training / loss calculation for different algorithms.
Inputs: `args` (namespace) in constructors; networks read args.* for hyperparameters and shapes.
Outputs/Side-effects: training internal state; save/load model depending on args
Findings:
- [ ] args mutation? (yes/no): no — they read `args` for hyperparams and training settings (gamma, lr, grad_clip, target_update_cycle).
- [ ] misplaced config? (yes/no): sometimes they perform load_model behavior (IO) when `args.load_model` is set; this is part of model lifecycle.
- [ ] cross-layer import smell? (yes/no): heavy coupling with `MARL/network/*` modules (expected)
- [ ] long function / god-object? (yes/no): large training routines exist, but they belong here.
- [ ] IO in core logic? (yes/no): model save/load may perform IO — expected.
Notes: Policies are expected to read hyperparameters from the provided args. Ensure Runner passes `self.run_args` or explicit shapes to Agents so these modules never attempt to read mutated global `args`.
Imported by: `MARL/agent/agent.py`

## File: MARL/network/*.py
Purpose: Neural network building blocks for policies (qmix_net, qtran_net, commnet, g2anet, vdn_net, maven_net, coma_critic).
Inputs: `args` used for dimensions (n_agents, n_actions, rnn_hidden_dim, attention_dim, etc.)
Outputs/Side-effects: PyTorch module objects; no file IO.
Findings:
- [ ] args mutation? (yes/no): no — they read `args` fields extensively (shapes). This is expected; must be given a run-local args with correct shapes.
- [ ] cross-layer import smell? (yes/no): network modules read args heavily — acceptable as they need dims.
- [ ] long function / god-object? (yes/no): modules are medium-sized but acceptable.
Notes: Many network modules expect `self.args.n_actions`, `self.args.n_agents`, etc. Ensure agents/policies receive run-local args before instantiation.
Imported by: `MARL/policy/*`, `MARL/agent/agent.py`

## File: environment.py (MASAEnv)
Purpose: SimPy-based manufacturing environment with decision batching API, dynamic arrivals, gantt recordings and configuration via YAML.
Inputs: optional `config_path` (YAML), constructor args for counts/dimensions, environment variables for reward overrides.
Outputs/Side-effects: maintains the SimPy world; exposes `reset()`, `get_env_info()`, `wait_for_decisions()`, `pop_decision_reward()`, and `add_job()`. Writes `print()` messages and may call `self.logger.warning()` or `print` on config load failures.
Findings:
- [ ] args mutation? (yes/no): no — environment does not write `args`.
- [ ] misplaced config? (yes/no): it loads YAML config (expected). The environment also supports env-var overrides for reward coefficients (acceptable but should be documented).
- [ ] cross-layer import smell? (yes/no): environment imports utils modules (`utils.workcenter`, `utils.jobagent`, `utils.task_generator`, `utils.config_loader`) — this is expected; directionality is env -> utils (OK).
- [ ] long function / god-object? (yes/no): the class is long and complex; it implements a full SimPy orchestration. This is expected for an environment.
- [ ] IO in core logic? (yes/no): prints and logger calls for warnings; no file writes except `print()` and logging. It does not mutate `args`.
Notes: Environment delegates machine registry creation to `utils.workcenter` or `utils.machine_registry` when config is present; `get_env_info()` returns canonical runtime shapes (n_actions, n_agents, state/obs shapes, episode_limit).
Imported modules: utils.workcenter, utils.task_generator, utils.config_loader
Imported by: `main.py`, Runner, tools/tests

## File: utils/env_obs.py
Purpose: Pure helper functions to build agent-level observations and global state vector.
Inputs: `env`, `job` objects
Outputs/Side-effects: none (pure functions)
Findings:
- [ ] args mutation? (yes/no): no
- [ ] side-effects at import time? (yes/no): no
Notes: Good single-responsibility helper.
Imported by: `environment.py`, tests

## File: utils/task_generator.py
Purpose: Centralized TaskGenerator with arrival loop and generate_constrained_task.
Inputs: config_path optional; relies on `utils.workcenter` and `utils.job` data.
Outputs/Side-effects: prints and starts SimPy processes when `start()` called.
Findings:
- [ ] args mutation? (yes/no): no
- [ ] side-effects at import time? (yes/no): no
Notes: Provides `start()` / `arrival_loop()` compatible with MASAEnv.
Imported by: `environment.py`

## File: utils/machine_registry.py
Purpose: Extracted registry builder for converting YAML config into a machine/workcenter metadata object.
Inputs: parsed config dictionary
Outputs/Side-effects: returns `(wc_meta, num_wcs, num_ops)`; pure function.
Findings:
- [ ] args mutation? (yes/no): no
Notes: Good helper for centralizing config parsing. `MASAEnv` delegates to this logic when config present.
Imported by: `environment.py` (indirect), `utils.workcenter` in places

## File: utils/gantt.py
Purpose: Formatting and CSV write helpers for gantt records.
Inputs: gantt record lists
Outputs/Side-effects: `write_gantt_csv()` writes files; other utilities are pure.
Findings:
- [ ] args mutation? (yes/no): no
- [ ] IO in core logic? (yes/no): `write_gantt_csv()` writes disk — belongs to utils/IO helpers and should be called explicitly by Runner/tools.
Notes: Acceptable — keep as utility.
Imported by: tools and possibly Runner in the future

## File: utils/config_loader.py
Purpose: YAML config loader (uses PyYAML) returning a dict.
Inputs: path to YAML file
Outputs/Side-effects: raises helpful error when PyYAML missing; not otherwise side-effecting.
Findings:
- [ ] args mutation? (yes/no): no
Notes: Good small helper.
Imported by: `environment.py`, `utils/task_generator.py`

## File: utils/jobagent.py
Purpose: Data container / helper for job agent objects used by env-based JobAgent and related helpers. Contains JobAgents collection and JobAgent class.
Inputs: constructor params; may be used by env and TaskGenerator
Outputs/Side-effects: prints during `execute_task()` and `mark_completed()` (informational)
Findings:
- [ ] args mutation? (yes/no): no
- [ ] IO in core logic? (yes/no): prints for debug; no file writes
Notes: Good compatibility shim for environment.
Imported by: `environment.py`, `utils/task_generator.py`

## File: tools/run_smoke_gantt.py and other `tools/` scripts
Purpose: small smoke/demo helpers that emulate quick runs and plot gantt.
Inputs: should call `MARL.common.arguments.get_smoke_args()` instead of mutating args inline.
Outputs/Side-effects: call Runner or Runner-like flows; write CSV/PNG artifacts.
Findings:
- [ ] args mutation? (yes/no): some script files previously mutated `args` but were updated (per earlier session) to call `get_smoke_args()` instead. Confirmed: `tools/run_smoke_gantt.py` no longer writes `args.<field>`.
- [ ] misplaced config? (yes/no): no
- [ ] IO in core logic? (yes/no): yes — intended behavior.
Notes: Ensure all tools use `get_smoke_args()` and do not locally assign defaults.
Imported by: user/test workflows

Other files
-----------
- `scripts/run_train_qmix.py` — script entrypoint; should be audited to ensure it does not assign to `args` at module-level; treat similar to `main.py`.
- `tests/` — many tests rely on `get_smoke_args()` for small deterministic runs; tests also assert absence of ad-hoc `args` writes (lint test already present).

Findings: cross-repo issues & code smell checklist
-----------------------------------------------
- Args mutation outside `arguments.py`:
  - I scanned the main runtime files; there were historical ad-hoc assignments but your prior work migrated most into `MARL/common/arguments.py` and introduced `self.run_args` in Runner. Current analysis shows the only authorised writer is `MARL/common/arguments.py`.
  - Confirmed: `tools/run_smoke_gantt.py` was updated (no ad-hoc args writes). The `tests/test_args_no_ad_hoc_assignments.py` provides a lint gate.

- God files / long classes:
  - `MARL/runner.py` is large and mixes orchestration, plotting, KPI writing and training loop. It's acceptable for now, but I recommend splitting plotting/logging and the training loop into helper modules in Phase 3.

- Cross-layer imports:
  - The environment imports `utils.*` (good). `MARL` imports environment only indirectly via `main.py` — acceptable.
  - Some helper modules write to scheduling CSVs directly (RolloutWorker writes scheduling_trace.csv). This is an IO-in-library smell: prefer Runner to control artifact writes. We should centralize all artifact writes in Runner or dedicated IO helpers.

- Side effects at import time:
  - I did not find heavy side-effects during import in the inspected files; helpers are import-safe and only read files when explicitly asked (e.g., `utils.config_loader.load_config` reads YAML when called). Good.

- Hidden globals and module-level state:
  - A few modules (Runner, Env) hold `history_dir`, `gantt_records`, etc. This is expected runtime state — just document where the state lives.

- Hardcoded paths:
  - Runner uses `./my_data_and_graph/historydata/` as the default history dir while `arguments.py` also defines `history_dir` default. Recommend unifying usage so Runner reads `args.history_dir` (read-only) or `self.run_args.history_dir`. Currently Runner sets `self.history_dir = './my_data_and_graph/historydata/'` — that's an explicit path in Runner which duplicates the default set in `arguments.py`. This is a duplication to fix in Phase 2: prefer reading `args.history_dir` (read-only) and defaulting in `arguments.py`.

- Duplicated logic:
  - Several places build CSV lines for gantt snapshots (Runner code and RolloutWorker both construct CSV lines). Consider centralizing CSV serialization via `utils/gantt.py`.

- Random seed & determinism:
  - RNGs are set in env (np RandomState), RolloutWorker, TaskGenerator. That's OK, but document central seed policy (args.seed should be respected by Runner and passed to created components).

Per-file checklist (selected high-value files)
----------------------------------------------
Below are required template entries for the highest-impact files. For brevity I include the most relevant ~20 files; remaining files follow the same patterns (they predominantly read args).

(Template per-file: Purpose / Inputs / Outputs / Findings checklist)

## File: /home/aleynaozcan/projects/MASA-QMIX/main.py
Purpose: CLI orchestrator — constructs `env` and `Runner`, selects algorithm-specific arg augmenters.
Inputs: `args` from `MARL/common/arguments.get_common_args()`.
Outputs/Side-effects: constructs and runs Runner; prints summary; writes pickles for random baseline.
Findings:
- [ ] args mutation? (yes/no): no
- [ ] misplaced config? (yes/no): no
- [ ] cross-layer import smell? (yes/no): no
- [ ] long function / god-object? (yes/no): no
- [ ] IO in core logic? (yes/no): only printing and small pickling in random baseline
Notes: OK.

## File: /home/aleynaozcan/projects/MASA-QMIX/MARL/common/arguments.py
Purpose: central arguments and algorithm-groups definitions
Inputs: none at import time (parser built at call-time)
Outputs/Side-effects: returns `args` namespace with defaults; intentionally mutates `args` inside helper functions.
Findings:
- [ ] args mutation? (yes/no): YES (only file allowed to do so)
- [ ] misplaced config? (yes/no): no
- [ ] cross-layer import smell? (yes/no): no
- [ ] long function / god-object? (yes/no): no
- [ ] IO in core logic? (yes/no): no
Notes: This file is the single source-of-truth and will remain authorised writer.

## File: /home/aleynaozcan/projects/MASA-QMIX/MARL/runner.py
Purpose: main training orchestrator, logging, checkpointing, plotting
Inputs: env, args
Outputs/Side-effects: writing many history files, plots, run_summary.json, scheduling_trace.csv, etc.
Findings:
- [ ] args mutation? (yes/no): NO (it uses `self.run_args` for writeable runtime content); does write to `env` attributes (quiet_env) and writes files.
- [ ] misplaced config? (yes/no): minor duplication: `self.history_dir` default duplicated vs `arguments.py`
- [ ] cross-layer import smell? (yes/no): moderate — large class, does plotting & logging that could be delegated
- [ ] long function / god-object? (yes/no): YES (Runner is large)
- [ ] IO in core logic? (yes/no): YES
Notes: Good run-local adoption of args; recommend next-phase cleanup: move plotting/IO into `utils/gantt` and centralize `history_dir` read from args.

## File: /home/aleynaozcan/projects/MASA-QMIX/environment.py
Purpose: SimPy environment implementing MASA scheduling with event-driven API
Inputs: optional YAML config; constructor args
Outputs/Side-effects: sim world; print/log warnings; returns `get_env_info()`
Findings:
- [ ] args mutation? (yes/no): NO
- [ ] misplaced config? (yes/no): NO (loading config in env is acceptable)
- [ ] cross-layer import smell? (yes/no): NO
- [ ] long function / god-object? (yes/no): YES (large class but acceptable)
- [ ] IO in core logic? (yes/no): minor prints
Notes: `get_env_info()` is authoritative for shapes; Runner should rely on it exclusively.

## File: /home/aleynaozcan/projects/MASA-QMIX/MARL/common/rollout.py
Purpose: RolloutWorker that turns decision-batches into transitions and applies actions.
Inputs: env, agents, buffer, args (prefer read-only `run_args`)
Outputs/Side-effects: may append to scheduling CSV and log messages
Findings:
- [ ] args mutation? (yes/no): NO
- [ ] misplaced config? (yes/no): NO
- [ ] cross-layer import smell? (yes/no): moderate
- [ ] long function / god-object? (yes/no): moderate
- [ ] IO in core logic? (yes/no): small writes to scheduling_trace.csv
Notes: Move IO into Runner or util writer in Phase 2.

## File: /home/aleynaozcan/projects/MASA-QMIX/utils/env_obs.py
Purpose: pure helpers to build obs/state
Findings: no issues

## File: /home/aleynaozcan/projects/MASA-QMIX/utils/task_generator.py
Purpose: dynamic task generator / arrival loop
Findings: no args writes; uses config loader; contains print statements for trace

## File: /home/aleynaozcan/projects/MASA-QMIX/utils/machine_registry.py
Purpose: builds WorkCenter metadata from YAML config
Findings: pure, returns structured metadata

Remaining files and patterns
----------------------------
- The majority of `MARL/policy/*.py` and `MARL/network/*.py` read `.args` fields extensively for network sizes and hyperparameters. This is expected and acceptable if Runner constructs them with a run-local args namespace (currently Runner passes `run_args` to `Agents` at creation). Ensure that all callers that make network/policy objects pass `run_args` rather than the shared `args`.

- `utils/gantt.py` centralizes CSV formatting; Runner contains duplicated CSV writing. Consolidate in Phase 2.

Immediate issues to address in Phase 2 (recommendations)
-------------------------------------------------------
1. Unify `history_dir` default: Runner currently hardcodes `self.history_dir = './my_data_and_graph/historydata/'`. Prefer using `args.history_dir` (read only) as the canonical path. This is a non-functional change (read-only use) and safe.

2. Move Runner's plotting/CSV duplication to `utils/gantt.py` functions (e.g., `write_gantt_csv`, `format_gantt_records`) and replace ad-hoc CSV writers in `Runner.run` and `RolloutWorker.process_and_apply_actions()`.

3. Confirm all places that construct `Agents`, `CommAgents`, policy and network constructors receive `run_args` (Runner already does this in several places). Audit `scripts/` and `tools/` that may construct Agent/Policy objects directly.

4. Remove `RolloutWorker` -level scheduling CSV writes. Instead, expose structured hooks and let Runner write scheduling_trace.csv once at the end of epoch. This allows single-authority for all output artifacts.

5. Replace any remaining ad-hoc `args.<field> =` occurrences outside `MARL/common/arguments.py` (search & fix). The repository already has a lint test to detect regressions; run it after changes.

6. Break down `Runner` responsibilities: create `Runner.logger` / `Runner.io` helper to handle logs, gantt snapshots and metric writes.

Validation checklist (Phase 4 plan)
----------------------------------
- Run `python -m py_compile $(git ls-files '*.py')` to catch syntax errors.
- Run smoke scripts (e.g., `PYTHONPATH=. python tools/run_smoke_gantt.py`) and verify identical outputs.
- Run `pytest -q` (with appropriate PYTHONPATH or editable install) and ensure tests pass.
- Ensure `tests/test_args_no_ad_hoc_assignments.py` still passes after any edits.

Files changed in analysis: none (Phase 1 only).

Planned Phase 2 actions (per-file structural remediation)
--------------------------------------------------------
- For each file flagged in this report I will produce a small diff that:
  - removes `args.<field> = ...` assignments (if present) and moves defaults to `MARL/common/arguments.py` or uses `get_smoke_args()` in tools.
  - ensure constructors that need runtime shapes accept either explicit shapes or `runner_args` (Runner will pass `self.run_args`).
  - centralize IO into `utils/gantt.py` and a small `MARL/common/io.py` if necessary.

Deliverables for Phase 2 (what I'll produce):
- Small, focused patches (one file at a time) with tests run after each change.
- A commit message pattern: `Refactor: enforce Option B — remove args mutations from <file>` or `Chore: pass shapes explicitly to <module>`.

Appendix: Quick scan for other ad-hoc args assignments
-----------------------------------------------------
I performed a code-scan during analysis. The repo contains a lint test that enforces no ad-hoc `args.<field> =` writes except in `MARL/common/arguments.py`. At the time of this analysis the lint test passes. If you want, I can now (Phase 2) perform the automated per-file remediation steps described above.

---

If you'd like me to proceed to Phase 2 now, tell me whether you prefer a single large PR with a set of patches, or incremental small PRs (one file / one focused test change at a time). I recommend the incremental approach (safer, easier to review): I will take one file, apply the Option B fix, run the local tests and smoke run, then move to the next file.

