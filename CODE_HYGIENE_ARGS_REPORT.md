# Args Hygiene Check — Quick Report

Date: 2025-11-03
Scope: repository-wide scan for
- non-whitelisted `args.<field> = ...` assignments (only allowed in `MARL/common/arguments.py`), and
- mixed uses of `args.` vs `self.args` (possible sources of subtle bugs / style drift).

Findings
--------
1) Non-whitelisted assignments
- Result: CLEAN. The only files that contain `args.<field> =` assignments are:
  - `MARL/common/arguments.py` (canonical Option-B defaults — allowed)
  - test files under `tests/` (test fixtures set up `args` for tests — allowed)

  No other Python file in the repo contains `args.<field> =` assignments.

2) Mixed uses of `args.` vs `self.args`
- Several files contain both plain `args.` references and `self.args` references. That pattern is not automatically a bug (common legitimate patterns include: the `__init__(self, args)` method using the local `args` parameter before assigning `self.args = args`), but it can indicate places to standardize.

- Files observed to contain both `args.` and `self.args` (exclude `MARL/common/arguments.py`, tests):
  - `MARL/runner.py` — previously fixed in this session (now consistently uses `self.args` for CLI-level reads).
  - `MARL/policy/qmix.py`
  - `MARL/policy/central_v.py`
  - `MARL/policy/coma.py`
  - `MARL/policy/reinforce.py`
  - `MARL/policy/maven.py`
  - `MARL/policy/vdn.py`
  - `MARL/policy/qtran_base.py`
  - `MARL/policy/qtran_alt.py`
  - `MARL/policy/qtran_alt.py`
  - `MARL/policy/qtran_alt.py` (multiple occurrences across policy modules)
  - `MARL/agent/agent.py`
  - `MARL/network/coma_critic.py` — example of a minor mixed expression: `self.fc3 = nn.Linear(args.critic_dim, self.args.n_actions)` (recommend unify)
  - `MARL/network/commnet.py`
  - `MARL/network/g2anet.py`
  - `MARL/network/maven_net.py`
  - `MARL/network/qmix_net.py`
  - `MARL/common/utils.py`
  - `MARL/common/rollout.py` (rollout intentionally supports both `self.args` and an optional `runner_args` fallback)
  - `main.py` (orchestration; reads `args` provided by the entrypoint)

Notes on these mixed occurrences
- Many of the flagged files are classes whose constructors accept an `args` parameter, set `self.args = args`, and then later use `self.args` — this is OK and expected.
- A small number of locations do mix the *local* `args` parameter and `self.args` inside the same file/class (for example `coma_critic.py` line where `args.critic_dim` is used together with `self.args.n_actions`). These are low-risk but inconsistent and worth standardizing.
- `MARL/common/rollout.py` intentionally allows either (it uses `runner_args` fallback semantics). Leave as-is unless you want stricter uniformity.

Recommendations (next steps)
---------------------------
1) Non-urgent but high value: standardize policy / network modules to one of two patterns:
   - Constructor receives `args` and immediately assigns `self.args = args`, then *everywhere else in the class use `self.args`*.
   - Or, if the module is a lightweight factory or top-level util that never stores `self.args`, keep using `args` as a parameter consistently (do not mix).

   Small automated fix: in classes that already do `self.args = args`, replace remaining bare `args.` uses in that file with `self.args.` (restrict replacements to occurrences after the `self.args = args` line).

2) Add a small unit test or linter rule (flake8 plugin or a lightweight grep-based pytest) to detect `args.<field> =` outside `MARL/common/arguments.py` and to warn when both `args.` and `self.args` occur in the same file (optionally limit to method bodies, not `__init__`). You already have `tests/test_args_no_ad_hoc_assignments.py` — consider augmenting it to also warn on mixed uses.

3) If you'd like, I can automatically apply safe fixes for the low-risk files (e.g., `coma_critic.py`) and run the test suite afterwards. I will:
   - create a small script that replaces `args.` with `self.args.` only in file regions after `self.args = args` in that file (simple heuristic), or
   - produce a PR-style patch with the suggested manual changes so you can review them.

Conclusion
----------
- No ad-hoc `args.<field> =` assignments found outside the allowed locations.
- Several files show mixed `args.` and `self.args` uses; most are legitimate patterns, a few are inconsistent and safe to normalize.

If you want, I will proceed to automatically normalize the small, low-risk mixed occurrences (one file at a time and run tests after each change). Which approach do you prefer? 
