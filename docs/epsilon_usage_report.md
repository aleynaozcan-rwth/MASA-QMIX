# Epsilon Usage Report - Complete Workspace Analysis

**Generated:** 2025-11-22  
**Branch:** refactor-modules-hybridreward-v12  
**Status:** COMPREHENSIVE SCAN - ALL EPSILON REFERENCES

---

## Executive Summary

**Total Epsilon References Found:** 200+ across 30+ files  
**Critical Files:** 5 (arguments.py, rollout.py, runner.py, qmix.py, agent.py)  
**Deprecated Parameters:** 1 (`epsilon_anneal_steps`)  
**Active Parameters:** 3 (`epsilon_start`, `epsilon_end`, `epsilon_anneal_episodes`)  
**Decay Mechanisms:** 2 (step-based DEPRECATED, episode-based ACTIVE)  
**Conflicts/Issues:** 3 major architectural problems identified

---

## 1. Parameter Definitions

### 1.1 Canonical Source: `MARL/common/arguments.py`

#### Active Parameters (Lines 160-167)

```python
# Line 160
parser.add_argument('--epsilon_start', type=float, default=1.0)

# Line 161  
parser.add_argument('--epsilon_end', type=float, default=0.05)

# Line 163-164
parser.add_argument('--epsilon_anneal_steps', type=int, default=1600,
                    help='[DEPRECATED] Use --epsilon_anneal_episodes instead. This parameter is ignored.')

# Line 166-167
parser.add_argument('--epsilon_anneal_episodes', type=int, default=None,
                    help='Number of episodes to linearly anneal epsilon. If None, uses n_epoch * n_episodes')
```

**Status:**
- ✅ `epsilon_start`: Active, default=1.0
- ✅ `epsilon_end`: Active, default=0.05  
- ⚠️ `epsilon_anneal_steps`: **DEPRECATED** but still defined (default=1600)
- ❌ `epsilon_anneal_episodes`: Active but **default=None** (causes runtime error!)

#### Deprecation Warning (Lines 224-229)

```python
if hasattr(args, 'epsilon_anneal_steps') and args.epsilon_anneal_steps != 30000:
    logging.getLogger(__name__).warning(
        f"[PHASE9] --epsilon_anneal_steps is DEPRECATED and ignored. "
        f"Use --epsilon_anneal_episodes instead. Current value: {args.epsilon_anneal_steps}"
    )
```

**Problem:** Warning checks for `!= 30000` but default is `1600` - inconsistent!

---

### 1.2 Config Files (YAML)

#### `configs/env_config_enabled.yaml` (Lines 104-106)
```yaml
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_steps: 100000  # ← OLD NAME, not used
```

#### `configs/env_no_arrival.yaml` (Lines 104-106)
```yaml
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay_steps: 100000  # ← OLD NAME, not used
```

#### `configs/env_config.yaml` (Lines 99-101)
```yaml
# COMMENTED OUT:
# epsilon_start: 1.0
# epsilon_end: 0.05
# epsilon_decay_steps: 100000
```

**Problem:** Config files use `epsilon_decay_steps` (old name), but code uses `epsilon_anneal_episodes` (new name). Mismatch!

---

## 2. Runtime Storage & Initialization

### 2.1 RolloutWorker (`MARL/common/rollout.py`)

#### Constructor Parameters (Lines 38-40)
```python
def __init__(
    self,
    ...
    epsilon_start: Optional[float] = None,
    epsilon_end: Optional[float] = None,
    epsilon_anneal_steps: Optional[int] = None,  # ← DEPRECATED but still accepted!
    ...
):
```

#### Initialization Logic (Lines 66-89)

**epsilon_start:**
```python
# Line 67-72
if hasattr(self.args, 'epsilon_start') and getattr(self.args, 'epsilon_start') is not None:
    self.epsilon_start = float(getattr(self.args, 'epsilon_start'))
elif epsilon_start is not None:
    self.epsilon_start = float(epsilon_start)
else:
    self.epsilon_start = float(getattr(self.args, 'epsilon_start', 1.0))  # ← FALLBACK
```

**epsilon_end:**
```python
# Line 74-79
if hasattr(self.args, 'epsilon_end') and getattr(self.args, 'epsilon_end') is not None:
    self.epsilon_end = float(getattr(self.args, 'epsilon_end'))
elif epsilon_end is not None:
    self.epsilon_end = float(epsilon_end)
else:
    self.epsilon_end = float(getattr(self.args, 'epsilon_end', 0.05))  # ← FALLBACK
```

**epsilon_anneal_steps (DEPRECATED):**
```python
# Line 81-86
if hasattr(self.args, 'epsilon_anneal_steps') and getattr(self.args, 'epsilon_anneal_steps') is not None:
    self.epsilon_anneal_steps = int(getattr(self.args, 'epsilon_anneal_steps'))
elif epsilon_anneal_steps is not None:
    self.epsilon_anneal_steps = int(epsilon_anneal_steps)
else:
    self.epsilon_anneal_steps = int(getattr(self.args, 'epsilon_anneal_steps', 50000))  # ← FALLBACK
```

**epsilon initialization:**
```python
# Line 88
self.epsilon = float(self.epsilon_start)
```

**Decay rate calculation (UNUSED - decay moved to runner.py):**
```python
# Line 89
self._eps_decay = (self.epsilon_start - self.epsilon_end) / max(1, self.epsilon_anneal_steps)
```

**Diagnostic logging interval:**
```python
# Line 101
self.epsilon_log_every = int(getattr(self.args, 'epsilon_diagnostics_every', 50) or 50)
```

**Storage Locations:**
- `self.epsilon_start` (float)
- `self.epsilon_end` (float)
- `self.epsilon_anneal_steps` (int) - **DEPRECATED, stored but not used**
- `self.epsilon` (float) - **CURRENT VALUE**
- `self._eps_decay` (float) - **CALCULATED BUT UNUSED**
- `self.epsilon_log_every` (int) - diagnostic interval

---

## 3. Epsilon Decay Logic

### 3.1 OLD Location (REMOVED): `rollout.py` per-decision decay

**Previously at Line ~880-911 (NOW COMMENTED OUT):**
```python
# [PHASE1-FIX] Epsilon decay moved to per-episode level in runner.py
# No longer decay per decision step - this caused circular dependency
# where epsilon schedule depended on number of decisions made
```

**Old formula (NO LONGER EXECUTED):**
```python
self.epsilon = max(float(self.epsilon_end), float(self.epsilon) - float(self._eps_decay))
```

**Status:** ❌ DEAD CODE - `self._eps_decay` is calculated (line 89) but never used

---

### 3.2 NEW Location (ACTIVE): `runner.py` per-episode decay

#### Training Loop Decay (Lines 1436-1471)

```python
# Line 1436
# Training episodes use evaluate=False for epsilon-greedy exploration
evaluate = False

# Line 1438
episode, ep_r, win_tag, gantt_data = self._run_event_driven_episode(global_ep_idx, evaluate=evaluate)

# Lines 1453-1462
if not evaluate:
    rollout_worker_epsilon_start = float(getattr(self.rolloutWorker, 'epsilon_start', 1.0))
    rollout_worker_epsilon_end = float(getattr(self.rolloutWorker, 'epsilon_end', 0.05))
    rollout_worker_epsilon_anneal_episodes = int(getattr(self.args, 'epsilon_anneal_episodes', 
                                                         self.args.n_epoch * self.args.n_episodes))
    eps_decay_per_episode = (rollout_worker_epsilon_start - rollout_worker_epsilon_end) / max(1, rollout_worker_epsilon_anneal_episodes)
    self.rolloutWorker.epsilon = max(rollout_worker_epsilon_end, 
                                      float(self.rolloutWorker.epsilon) - eps_decay_per_episode)
```

**CRITICAL BUG (Line 1458-1459):**
```python
rollout_worker_epsilon_anneal_episodes = int(getattr(self.args, 'epsilon_anneal_episodes', 
                                                     self.args.n_epoch * self.args.n_episodes))
```
- `self.args.epsilon_anneal_episodes` is `None` (from arguments.py line 166)
- `getattr(..., fallback)` returns `None` because attribute exists (not missing)
- `int(None)` → **TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'**

**Decay Formula:**
```python
eps_decay_per_episode = (epsilon_start - epsilon_end) / max(1, epsilon_anneal_episodes)
new_epsilon = max(epsilon_end, current_epsilon - eps_decay_per_episode)
```

**Example:**
- `epsilon_start=1.0`, `epsilon_end=0.05`, `n_epoch=1`, `n_episodes=2`
- `epsilon_anneal_episodes = None` → tries `1 * 2 = 2` episodes
- `eps_decay_per_episode = (1.0 - 0.05) / 2 = 0.475`
- Episode 1: `epsilon = 1.0 - 0.475 = 0.525`
- Episode 2: `epsilon = 0.525 - 0.475 = 0.05`

---

## 4. Epsilon Consumption (Action Selection)

### 4.1 RolloutWorker (`rollout.py`)

#### Selection Method (Lines 118-134)
```python
def _select_actions(self, obs_batch: List[Any], avail_batch: Optional[List[Any]], 
                   evaluate: bool = False, epsilon: float = None) -> Tuple[List[Any], Any]:
    # Use current epsilon if not provided
    if epsilon is None:
        epsilon = float(getattr(self, 'epsilon', 1.0))  # ← FALLBACK
    
    try:
        # Try to pass epsilon if the method accepts it
        return self.agents.select_actions(obs_batch, avail_batch, evaluate=evaluate, epsilon=epsilon), None
    except TypeError:
        # Fallback if method doesn't accept epsilon
        return self.agents.select_actions(obs_batch, avail_batch, evaluate=evaluate), None
```

#### Episode Loop Usage (Line 863)
```python
actions, _ = self._select_actions(obs_list, avail, evaluate=evaluate, epsilon=self.epsilon)
```

**Status:** ✅ Correctly passes `self.epsilon` to action selection

---

### 4.2 Agent Wrapper (`MARL/agent/agent.py`)

#### Selection Method (Lines 113-127)
```python
def select_actions(self, obs_batch, avail_batch=None, evaluate=False, epsilon=None):
    try:
        # Try to pass epsilon to policy if it accepts it
        return self.policy.select_actions(obs_batch, avail_batch, evaluate=evaluate, epsilon=epsilon)
    except TypeError:
        # Fallback if policy doesn't accept epsilon
        return self.policy.select_actions(obs_batch, avail_batch, evaluate=evaluate)
```

**Status:** ✅ Propagates epsilon to policy layer

---

### 4.3 QMIX Policy (`MARL/policy/qmix.py`)

#### Selection Method (Lines 274-344)
```python
def select_actions(self, obs_batch, avail_batch=None, evaluate: bool = False, epsilon: float = None):
    # [PHASE8-FIX] Task 8.4 & 8.5: Epsilon fallback with warning (required for training)
    if epsilon is None:
        epsilon = float(getattr(self.args, 'epsilon', 0.0))  # ← FALLBACK
        if not evaluate:
            logging.getLogger(__name__).warning(
                "[PHASE8] select_actions called without epsilon during training. "
                f"Falling back to args.epsilon={epsilon}. Please provide epsilon explicitly."
            )
    
    # ... Q-value computation ...
    
    # epsilon-greedy (Line 343-344)
    if (not evaluate) and (float(_np.random.rand()) < float(epsilon)):
        # random action from available
```

**Status:** ✅ Uses epsilon for exploration, ⚠️ has fallback to `args.epsilon`

---

### 4.4 Other Policies

**VDN** (`MARL/policy/vdn.py` Line 61):
```python
def learn(self, batch, max_episode_len, train_step, epsilon=None):
```
- Accepts epsilon but doesn't use it in action selection (uses Q-values directly)

**COMA** (`MARL/policy/coma.py` Lines 88, 103, 231, 250):
```python
def learn(self, batch, max_episode_len, train_step, epsilon):  # ← NOT Optional!
def _get_action_prob(self, batch, max_episode_len, epsilon):
# Line 250
action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
```
- **REQUIRED parameter** (not Optional)
- Uses epsilon for probability smoothing

**REINFORCE** (`MARL/policy/reinforce.py` Lines 59, 79, 131, 151):
- Same pattern as COMA

**CENTRAL_V** (`MARL/policy/central_v.py` Lines 73, 92, 147, 166):
- Same pattern as COMA

**QTRAN_BASE** (`MARL/policy/qtran_base.py` Line 71):
```python
def learn(self, batch, max_episode_len, train_step, epsilon=None):
```

**QTRAN_ALT** (`MARL/policy/qtran_alt.py` Line 69):
```python
def learn(self, batch, max_episode_len, train_step, epsilon=None):
```

**MAVEN** (`MARL/policy/maven.py` Line 75):
```python
def learn(self, batch, max_episode_len, train_step, epsilon=None):
```

---

## 5. Epsilon Diagnostics & Logging

### 5.1 Runner Logging (runner.py)

#### Epsilon Log Path (Line 1016)
```python
self._epsilon_log_path = os.path.join(self.history_dir, f'epsilon_diagnostics_{worker_id}.txt')
```

#### Active Args Logging (Lines 1365-1367)
```python
"[ACTIVE_ARGS] epsilon_start=%g" % float(getattr(a, "epsilon_start", -1)),
"[ACTIVE_ARGS] epsilon_end=%g" % float(getattr(a, "epsilon_end", -1)),
"[ACTIVE_ARGS] epsilon_anneal_steps=%d" % int(getattr(a, "epsilon_anneal_steps", -1)),
```

#### Epoch Start Diagnostics (Lines 1387-1393)
```python
eps = float(getattr(self.rolloutWorker, 'epsilon', getattr(self.args, 'epsilon', None)))
print(f"[Diagnostics] Epoch {epoch} start | epsilon={eps} | buffer_len={buf_len}")
df.write(f"{time.time()},{epoch},epoch_start,epsilon={eps},buffer_len={buf_len}\n")
```

#### Per-Episode Logging (Lines 1467-1469)
```python
if global_ep_idx % 20 == 0:
    with open(self._epsilon_log_path, 'a') as ef:
        ef.write(f"{time.time()},{global_ep_idx},{self.rolloutWorker.epsilon:.6f}\n")
    print(f"[PHASE1-EPSILON] Episode {global_ep_idx}: epsilon={self.rolloutWorker.epsilon:.4f}")
```

#### Epoch End Diagnostics (Lines 1793-1798)
```python
eps_end = float(getattr(self.rolloutWorker, 'epsilon', getattr(self.args, 'epsilon', None)))
df.write(f"{time.time()},{epoch},epoch_end,epsilon={eps_end},buffer_len={buf_len_end}\n")
```

---

### 5.2 Rollout Logging (rollout.py Lines 882-911)

```python
# [PHASE1-FIX] Epsilon decay moved to per-episode level in runner.py
# No longer decay per decision step - this caused circular dependency

# --- Epsilon diagnostics (periodic, non-fatal) ---
if getattr(self, 'epsilon_log_every', 0) > 0 and (self.step_counter % int(self.epsilon_log_every) == 0):
    try:
        msg = f"[EPSILON_DECAY] step={self.step_counter}, epsilon={self.epsilon:.4f}"
    except Exception as e:
        try:
            msg = f"[EPSILON_DECAY] step={self.step_counter}, epsilon={float(self.epsilon)}"
        except Exception as e2:
            msg = f"[EPSILON_DECAY] step={self.step_counter}, epsilon={getattr(self, 'epsilon', 'NA')}"
```

**Status:** ⚠️ Still logs per-step but no longer decays per-step

---

## 6. Fallback Patterns (VIOLATIONS OF CO-PILOT RULES)

### 6.1 Arguments Fallbacks

| Location | Code | Default | Status |
|----------|------|---------|--------|
| `rollout.py:72` | `getattr(self.args, 'epsilon_start', 1.0)` | 1.0 | ❌ Violates rules |
| `rollout.py:79` | `getattr(self.args, 'epsilon_end', 0.05)` | 0.05 | ❌ Violates rules |
| `rollout.py:86` | `getattr(self.args, 'epsilon_anneal_steps', 50000)` | 50000 | ❌ Violates rules |
| `rollout.py:125` | `getattr(self, 'epsilon', 1.0)` | 1.0 | ❌ Violates rules |
| `qmix.py:287` | `getattr(self.args, 'epsilon', 0.0)` | 0.0 | ❌ Violates rules |
| `runner.py:1456` | `getattr(self.rolloutWorker, 'epsilon_start', 1.0)` | 1.0 | ❌ Violates rules |
| `runner.py:1457` | `getattr(self.rolloutWorker, 'epsilon_end', 0.05)` | 0.05 | ❌ Violates rules |
| `runner.py:1387` | `getattr(self.rolloutWorker, 'epsilon', getattr(self.args, 'epsilon', None))` | None | ❌ Violates rules |
| `runner.py:1795` | `getattr(self.rolloutWorker, 'epsilon', getattr(self.args, 'epsilon', None))` | None | ❌ Violates rules |

**Total Fallback Violations:** 9

---

## 7. Unused / Dead Code

### 7.1 Dead Calculations

**`rollout.py` Line 89:**
```python
self._eps_decay = (self.epsilon_start - self.epsilon_end) / max(1, self.epsilon_anneal_steps)
```
- **Status:** ❌ CALCULATED BUT NEVER USED
- Decay moved to runner.py, this calculation is orphaned

---

### 7.2 Deprecated Parameters Still Stored

**`rollout.py` Lines 81-86:**
```python
self.epsilon_anneal_steps = int(getattr(self.args, 'epsilon_anneal_steps'))
```
- **Status:** ⚠️ STORED BUT MARKED DEPRECATED
- Used ONLY in dead `_eps_decay` calculation
- Not used in runner.py (uses `epsilon_anneal_episodes` instead)

---

### 7.3 Commented Decision-Step Decay

**`rollout.py` Lines 878-880:**
```python
# [PHASE1-FIX] Epsilon decay moved to per-episode level in runner.py
# No longer decay per decision step - this caused circular dependency
# where epsilon schedule depended on number of decisions made
```
- **Status:** ✅ Properly commented, no execution

---

## 8. Architectural Problems

### 8.1 CRITICAL: `epsilon_anneal_episodes = None` Bug

**Location:** `runner.py` Lines 1458-1459  
**Problem:** 
```python
rollout_worker_epsilon_anneal_episodes = int(getattr(self.args, 'epsilon_anneal_episodes', 
                                                     self.args.n_epoch * self.args.n_episodes))
```
- `self.args.epsilon_anneal_episodes` exists but is `None`
- `getattr` returns `None` (attribute exists, doesn't use fallback)
- `int(None)` raises **TypeError**

**Solution Required:**
```python
if self.args.epsilon_anneal_episodes is None:
    rollout_worker_epsilon_anneal_episodes = self.args.n_epoch * self.args.n_episodes
else:
    rollout_worker_epsilon_anneal_episodes = int(self.args.epsilon_anneal_episodes)
```

---

### 8.2 CONFLICT: Deprecated vs Active Parameters

**Problem:** Both `epsilon_anneal_steps` (deprecated) and `epsilon_anneal_episodes` (active) are defined

| Parameter | Status | Where Stored | Where Used |
|-----------|--------|--------------|------------|
| `epsilon_anneal_steps` | DEPRECATED | `rollout.py:82-86` | NOWHERE (dead) |
| `epsilon_anneal_episodes` | ACTIVE | `arguments.py:166` | `runner.py:1458` |

**Issue:**
- `rollout.py` still initializes `epsilon_anneal_steps`
- `rollout.py` calculates `_eps_decay` using `epsilon_anneal_steps` (dead code)
- `runner.py` uses `epsilon_anneal_episodes` (but crashes on None)
- Config files use `epsilon_decay_steps` (neither parameter name!)

---

### 8.3 DUPLICATION: Epsilon Stored in Multiple Places

**Storage Locations:**
1. `self.rolloutWorker.epsilon` (primary, updated by runner.py)
2. `self.rolloutWorker.epsilon_start` (initial value, read-only)
3. `self.rolloutWorker.epsilon_end` (final value, read-only)
4. `self.args.epsilon` (fallback in qmix.py, shouldn't exist)
5. `self.args.epsilon_start` (definition)
6. `self.args.epsilon_end` (definition)
7. `self.args.epsilon_anneal_steps` (deprecated)
8. `self.args.epsilon_anneal_episodes` (active but None)

**Problem:** 8 different epsilon-related attributes across 2 objects!

---

## 9. Tests

### 9.1 Epsilon Decay Tests (`tests/test_epsilon_decay_episode_based.py`)

```python
def test_epsilon_decay_formula():
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_anneal_steps = 5000  # ← USES OLD NAME!
    
    def compute_epsilon(episode_count):
        if episode_count >= epsilon_anneal_steps:
            return epsilon_end
        decay_fraction = float(episode_count) / float(epsilon_anneal_steps)
        return epsilon_start - decay_fraction * (epsilon_start - epsilon_end)
```

**Tests:**
- Episode 0: epsilon=1.0 ✅
- Episode 10: epsilon≈0.9981 ✅
- Episode 100: epsilon≈0.981 ✅
- Episode 1000: epsilon≈0.81 ✅
- Episode 5000: epsilon=0.05 ✅
- Episode 6000+: epsilon=0.05 (clamped) ✅

---

### 9.2 Phase1 Tests

**`tests/test_phase1_standalone.py`:**
```python
args.epsilon_start = 1.0
args.epsilon_end = 0.05
args.epsilon_anneal_steps = 50000  # ← OLD NAME
args.epsilon = 0.0  # ← FALLBACK VALUE
```

**`tests/test_phase1_failfast.py`:**
```python
args.epsilon_start = 1.0
args.epsilon_end = 0.05
args.epsilon_anneal_steps = 50000  # ← OLD NAME
args.epsilon = 0.0  # ← FALLBACK VALUE
common_args.epsilon_start = 1.0
common_args.epsilon_end = 0.05
common_args.epsilon_anneal_steps = 50000  # ← OLD NAME
mixer_args.epsilon = 0.0  # ← FALLBACK VALUE
```

**Problem:** Tests use deprecated `epsilon_anneal_steps` name!

---

## 10. Summary of Issues

### 10.1 Critical Bugs

1. **TypeError in runner.py (Line 1458):**  
   `int(None)` when `epsilon_anneal_episodes=None`

2. **Dead Code in rollout.py (Line 89):**  
   `self._eps_decay` calculated but never used

3. **Deprecated Parameter Still Stored (rollout.py Lines 81-86):**  
   `epsilon_anneal_steps` initialized despite being deprecated

---

### 10.2 Architectural Problems

1. **Dual Decay Mechanisms:**  
   - OLD: rollout.py per-decision (removed, but calculation remains)
   - NEW: runner.py per-episode (active but buggy)

2. **Parameter Name Inconsistency:**  
   - Arguments: `epsilon_anneal_episodes`
   - Rollout: `epsilon_anneal_steps`
   - Configs: `epsilon_decay_steps`

3. **Excessive Fallbacks:**  
   - 9 getattr() calls with defaults (violates Co-Pilot Rules)

4. **Storage Duplication:**  
   - 8 epsilon-related attributes across 2 objects

---

### 10.3 Deprecation Issues

1. **Incomplete Deprecation:**  
   - `epsilon_anneal_steps` marked deprecated in arguments.py
   - But still initialized in rollout.py
   - Warning checks wrong value (30000 vs 1600)

2. **Config File Lag:**  
   - YAML files use `epsilon_decay_steps` (old name)
   - Code expects `epsilon_anneal_episodes` (new name)

---

## 11. Recommendations

**DO NOT IMPLEMENT - ANALYSIS ONLY**

### Priority 1: Fix Critical Bug
- Change `runner.py:1458-1459` to handle `None` properly

### Priority 2: Remove Dead Code
- Delete `rollout.py:89` (`_eps_decay` calculation)
- Delete `rollout.py:81-86` (`epsilon_anneal_steps` initialization)

### Priority 3: Unify Naming
- Rename config `epsilon_decay_steps` → `epsilon_anneal_episodes`
- Update all tests to use `epsilon_anneal_episodes`

### Priority 4: Eliminate Fallbacks
- Remove all 9 `getattr(..., default=X)` patterns
- Fail immediately if epsilon parameters missing

### Priority 5: Consolidate Storage
- Keep ONLY `rolloutWorker.epsilon` (current value)
- Keep ONLY `rolloutWorker.epsilon_start` (initial)
- Keep ONLY `rolloutWorker.epsilon_end` (final)
- Remove `args.epsilon` (shouldn't exist)

---

**END OF REPORT**
