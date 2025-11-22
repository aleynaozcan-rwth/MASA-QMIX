# You MUST strictly follow all prohibitions defined in Co-Pilot Rules.md.
Do NOT improvise. Do NOT optimize. Do NOT insert defaults.  
Apply EXACTLY the edits below and NOTHING ELSE.

---

# Epsilon Decay Migration Plan (FINAL — EXACT PATCH VERSION)

## Objective

Replace all previous epsilon decay systems with a single, deterministic, time-normalized decay based ONLY on:

- epsilon_start  
- epsilon_end  
- episode_limit (SimPy time)  
- env.now  

Formula:

epsilon(t) = epsilon_start − (t / episode_limit) × (epsilon_start − epsilon_end)

This is the ONLY allowed epsilon decay after migration.

---

# STEP 1 — Update arguments.py (EXACT PATCH)

**File:** MARL/common/arguments.py

## REMOVE COMPLETELY:

Search and delete all of the following entries (delete the entire argument block + any associated help text + any warnings referencing them):

- epsilon_anneal_steps  
- epsilon_anneal_episodes  
- Any warnings referencing epsilon_anneal_steps  
- Any fallback usage referencing epsilon decay  

## KEEP ONLY:

The following arguments must remain and must be the ONLY epsilon-related arguments:

- epsilon_start (float)  
- epsilon_end (float)  
- episode_limit (int)  

## UPDATE HELP STRINGS:

The help string for both epsilon_start and epsilon_end must explain:

"Epsilon decays linearly with simulation time based on: epsilon(t) = epsilon_start − (t / episode_limit) × (epsilon_start − epsilon_end)"

No other decay modes must be mentioned.

---

# STEP 2 — Implement decay ONLY in rollout.py (EXACT INSERT LOCATION)

**File:** MARL/common/rollout.py

## REMOVE from RolloutWorker.__init__:

Delete the ENTIRE initialization related to:

- epsilon_anneal_steps  
- _eps_decay  
- Any getattr(..., default=...) fallback for epsilon_start or epsilon_end  

## REMOVE fallback epsilon inside _select_actions():

Locate inside _select_actions():

The fallback:

epsilon = float(getattr(self, 'epsilon', 1.0))

DELETE this fallback completely.

## INSERT DECAY LOGIC (EXACT LOCATION):

You MUST insert the decay calculation inside:

RolloutWorker.run_event_driven_episode()

AFTER the following line:

actions, _ = self._select_actions(obs_list, avail, evaluate=evaluate, epsilon=self.epsilon)

and BEFORE any transition logging, reward logging, or buffer append.

Insert EXACTLY this code:

current_t = float(self.env.now)  
limit_t = float(self.env.episode_limit)  
fraction = min(1.0, current_t / limit_t)  
self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)

This becomes the ONLY epsilon update mechanism.

Rollout is the SOLE owner of epsilon evolution.  
No other file may modify epsilon.

---

# STEP 3 — Remove epsilon decay from runner.py (EXACT DELETE BLOCK)

**File:** MARL/runner.py

DELETE THE ENTIRE BLOCK in run() that starts with:

rollout_worker_epsilon_start =

and ends with:

self.rolloutWorker.epsilon =

Remove the entire block including all fallback calculations, all references to deprecated parameters, and all references to epsilon decay.

DELETE ANY fallback reads such as:

getattr(self.rolloutWorker, 'epsilon', ...)

Runner is strictly prohibited from modifying epsilon.  
Runner may ONLY log:

self.rolloutWorker.epsilon

Nothing else.

---

# STEP 4 — Clean qmix.py and agent.py (EXACT BEHAVIOR RULE)

**Files:**

- MARL/policy/qmix.py  
- MARL/agent/agent.py  

## DELETE COMPLETELY:

Any fallback logic like:

epsilon = float(getattr(self.args, 'epsilon', 0.0))

## AFTER REMOVAL:

If epsilon is None → raise an error:

raise ValueError("epsilon must be provided explicitly")

No silent fallback, no defaulting, no implicit creation.

---

# STEP 5 — Update YAML configs (EXACT DELETE)

**Files:** configs/*.yaml

## REMOVE COMPLETELY:

- epsilon_decay_steps  
- epsilon_anneal_steps  
- epsilon_anneal_episodes  

## KEEP:

epsilon_start: 1.0  
epsilon_end: 0.05  
episode_limit: NNN (user-defined)

No other epsilon-related keys allowed.

---

# STEP 6 — Fix tests (EXACT REPLACEMENTS)

Search and DELETE any test references to:

- epsilon_anneal_steps  
- epsilon_anneal_episodes  
- epsilon_decay_steps  
- args.epsilon  

Replace all with:

- epsilon_start  
- epsilon_end  
- episode_limit  

Tests must compute epsilon using the new time-normalized formula.

---

# VALIDATION REQUIREMENTS (MUST PASS)

- Epsilon(t) is perfectly linear decreasing over time.  
- Epsilon(t) is identical across identical seeds.  
- Epsilon does NOT depend on:
  - decision count  
  - job arrivals  
  - number of agents  

Search MUST return 0 hits after migration:

- epsilon_anneal_steps  
- epsilon_anneal_episodes  
- epsilon_decay_steps  
- _eps_decay  
- args.epsilon  

No fallback patterns allowed anywhere.

---

# END OF PLAN
