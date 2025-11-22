# Episode Termination Plan
## Episode Termination Refactor Plan (MASA-QMIX – SimPy Environment)
# IMPORTANT: Before applying ANY changes, read and obey **CoPilotRules.md**
Copilot MUST follow all rules defined in CoPilotRules.md FIRST, and apply this plan
ONLY within those constraints. Do NOT violate or override any rule from CoPilotRules.md.
This document describes a **strict, minimal-change plan** for cleaning up the
episode termination logic in the MASA-QMIX SimPy-based environment.

Copilot MUST follow this plan EXACTLY and MUST NOT introduce any behavior
outside the scope described here.

---

## 0. Global Rules for This Refactor

- DO NOT introduce any new parameters, flags, classes, functions, or files.
- DO NOT change function signatures unless absolutely required by this plan.
- DO NOT add any default values, fallbacks, or "if None then X" logic.
- DO NOT invent new config sources (no new YAML, JSON, CLI args, etc.).
- DO NOT refactor or "clean up" unrelated code.
- DO NOT touch epsilon decay, exploration logic, reward functions, replay buffer,
  QMIX, or observation/state definitions in this step.
- DO NOT move code between files (no big structural refactors).

Only modify:

- `environment.py`
- `MARL/common/arguments.py`
- `MARL/common/rollout.py`
- (Optionally) `runner.py` **if and only if** it directly duplicates
  episode termination logic.

Make the smallest possible changes to:

- unify episode termination logic,
- remove dual sources of truth,
- remove invalid termination conditions (`all jobs finished`),
- keep the system behavior deterministic and consistent.

---

## 1. Single Source of Truth for `episode_limit`

### Goal

There must be EXACTLY ONE source of truth for `episode_limit`.

### 1.1. Keep the CLI/arguments definition as canonical

In `MARL/common/arguments.py`, KEEP the existing argument for `--episode_limit`:

```python
parser.add_argument(
    '--episode_limit',
    type=int,
    default=500,
    help='Max SimPy time per episode'
)
# Episode Termination Plan
## Episode Termination Refactor Plan (MASA-QMIX – SimPy Environment)

This document describes a strict, minimal-change plan for cleaning up the
episode termination logic in the MASA-QMIX SimPy-based environment.

Copilot MUST follow this plan EXACTLY and MUST NOT introduce any behavior
outside the scope described here.

---

## 0. Global Rules for This Refactor

- DO NOT introduce new parameters, flags, classes, functions, or files.
- DO NOT change function signatures unless explicitly required by this plan.
- DO NOT add default values, fallbacks, or “if None then X” logic.
- DO NOT invent new config sources (no new YAML, JSON, CLI args, etc.).
- DO NOT refactor or clean up unrelated code.
- DO NOT touch epsilon decay, exploration, reward logic, replay buffer,
  QMIX, or observation/state definitions.
- DO NOT move logic between files or reorganize modules.

Allowed files to modify:
- `environment.py`
- `MARL/common/arguments.py`
- `MARL/common/rollout.py`
- Optionally `runner.py` ONLY if it duplicates episode termination logic.

Changes MUST be the smallest possible that achieve:
- unified episode termination,
- removal of dual truth paths,
- removal of invalid termination (`all jobs finished`),
- deterministic behavior.

---

## 1. Single Source of Truth for episode_limit

### 1.1 Canonical definition: arguments.py
Keep the existing argument definition exactly as the source of truth:

parser.add_argument(
    '--episode_limit',
    type=int,
    default=500,
    help='Max SimPy time per episode'
)

Do not rename this parameter or create an alternative.

### 1.2 Remove internal defaults in environment.py
Find any logic that resolves episode_limit internally, such as:

self.episode_limit = _resolve(('episode_limit',), int, default=600)

Replace it with the explicit authoritative value:

self.episode_limit = int(self.args.episode_limit)

NO fallback logic allowed:
- No getattr(...)
- No default=...
- No or 600
- No silently replacing missing values

If the argument is missing, the program must fail.

---

## 2. Episode Termination Logic – Environment as the Single Authority

The environment MUST be the only place that decides when an episode ends.

Termination is based ONLY on:
1. SimPy time limit (env.now >= episode_limit)
2. Emergency/safety conditions (rare; see Section 3)

Every other termination condition MUST be removed.

### 2.1 Remove “all jobs finished”
Locate every usage of:

all(j.finished for j in self.jobs)

including inside:
- step()
- _check_pending_activation()
- is_episode_done()
- any helper

Remove it from termination logic.
This condition is invalid in dynamic arrival settings.

It may remain ONLY for logging/metrics, but must not affect self.done.

### 2.2 Define a single, unified predicate

Define or update the termination predicate:

def is_episode_done(self) -> bool:
    time_limit_reached = float(self.env.now) >= float(self.episode_limit)
    env_done_flag = getattr(self, 'done', False)
    return time_limit_reached or env_done_flag

Rules:
- time_limit_reached = normal termination
- env_done_flag = emergency termination ONLY
- no second definitions elsewhere

### 2.3 Ensure self.done is set ONCE
When env.now >= episode_limit:
- set self.done = True in one central place
- avoid multiple scattered writes

---

## 3. Emergency / Safety Termination (Rare Only)

Emergency termination is NOT normal termination.

It exists ONLY for:
- infinite loops,
- deadlocks,
- zero-progress states.

### 3.1 Keep the existing safety guard
Inside wait_for_decisions():

iterations += 1
if iterations >= max_iterations:
    log warning
    self.done = True
    return empty batch

This is allowed.
Do NOT create additional safety guards.
Do NOT use all jobs finished as a safety condition.

### 3.2 Meaning of self.done after refactor
self.done MUST mean ONLY:

1. Time limit reached, or
2. Emergency guard triggered

Nothing else may set this flag.

---

## 4. Rollout Logic – Must Not Duplicate Termination Logic

### 4.1 Allowed termination conditions in rollout.py

while True:
    batch, sim_time = self.env.wait_for_decisions()
    if not batch:
        break
    if self.env.done:
        break

This is allowed.

NOT allowed:
- manual rechecking of time limit
- fallback defaults (getattr(..., 1e9))
- new conditions like num_decisions == 0

### 4.2 Remove duplicate checks
Any code resembling:

if getattr(self.env, "t", 0) >= getattr(self.env, "episode_limit", 1e9):

MUST be removed.

Rollout trusts ONLY:
- empty batch
- env.done

---

## 5. Runner – Minimal Adjustments Only

Runner MUST NOT implement its own episode termination logic.

Remove any termination logic such as:
- time limit checks
- job-finished checks
- fallback-based conditions

Runner should rely ONLY on:
- environment (env.done)
- rollout loop termination

Do NOT introduce new parameters or logic.

---

## 6. No New Definitions, No Side Effects

Copilot MUST NOT:
- introduce new parameters (max_episode_steps etc.)
- add helpers/wrappers
- relocate code
- add default values
- adjust unrelated logic (epsilon, reward, buffer, QMIX)

The ONLY allowed changes are those required to:

1. Remove dual episode_limit definitions  
2. Remove all jobs finished as a termination condition  
3. Make environment the sole termination authority  
4. Preserve emergency guard  
5. Leave everything else untouched  

Any change beyond this plan MUST NOT be done.

---
