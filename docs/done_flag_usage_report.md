# Done Flag Usage Report - MASA-QMIX Environment

## Overview

This document provides a comprehensive analysis of `self.done` flag usage across the MASA-QMIX codebase after the Episode Termination Refactor.

**Status:** ✅ Clean and Centralized  
**Last Updated:** 2025-11-22  
**Branch:** refactor-modules-hybridreward-v12

---

## 1. Initialization (2 Locations)

### 1.1 `environment.py` Line 350 - `__init__()`
```python
self.done = False
```
**Purpose:** Initial flag state when environment is created  
**Scope:** Constructor initialization

### 1.2 `environment.py` Line 537 - `reset()`
```python
self.done = False
```
**Purpose:** Reset flag to False at the start of each new episode  
**Scope:** Episode reset

---

## 2. Done Flag Checks (Read Operations - 5 Locations)

### 2.1 `environment.py` Line 619 - `wait_for_decisions()`
```python
if self.done:
    return [], float(self.t)
```
**Purpose:** Early exit if episode already terminated  
**Returns:** Empty batch to signal episode end  
**Type:** Guard clause

### 2.2 `environment.py` Line 644 - `wait_for_decisions()` loop condition
```python
while not self.pending_decisions and not bool(_get_attr_from_env('done', False)):
```
**Purpose:** Loop continuation condition  
**Behavior:** Stops advancing simulation if done=True  
**Type:** Loop guard

### 2.3 `environment.py` Line 1554 - `_is_episode_done()`
```python
env_done = getattr(self, 'done', False)
return time_limit_reached or env_done
```
**Purpose:** Unified episode termination predicate  
**Returns:** True if episode has ended (time limit OR done flag)  
**Type:** Status query

### 2.4 `MARL/common/rollout.py` Line 696 - `build_transitions_from_decision_batch()`
```python
tr['done'] = getattr(self.env, 'done', False)
```
**Purpose:** Copy done flag into transition dict for replay buffer  
**Type:** Data capture

### 2.5 `MARL/common/rollout.py` Line 963 - `run_event_driven_episode()`
```python
if getattr(self.env, "done", False):
    break
```
**Purpose:** Episode loop termination check  
**Type:** Loop exit condition

---

## 3. Done Flag Set Operations (4 Locations)

### A) Normal Termination (Time Limit) - 2 Locations

#### 3.1 `environment.py` Line 611 - `step()`
```python
done = (self.t >= self.episode_limit)
self.done = self.done or done
```
**Trigger:** SimPy time reaches or exceeds `episode_limit`  
**Type:** Normal termination  
**Context:** Called after `env.run(until=...)`

#### 3.2 `environment.py` Line 1388 - `_check_pending_activation()`
```python
if self.env.now >= self.episode_limit:
    self.done = True
    if not getattr(self.decisions_ready, 'triggered', False):
        self.decisions_ready.succeed()
```
**Trigger:** Time limit during pending job activation check  
**Type:** Normal termination  
**Context:** Called during job lifecycle management

---

### B) Emergency Termination (Safety Guards) - 2 Locations

#### 3.3 `environment.py` Line 647 - `wait_for_decisions()` time limit guard
```python
if float(self.env.now) >= float(self.episode_limit):
    self.done = True
    logging.getLogger(__name__).debug("wait_for_decisions: episode limit reached (t=%.2f >= %.2f)", ...)
    return [], float(self.env.now)
```
**Trigger:** Time limit reached during wait loop  
**Type:** Emergency termination  
**Purpose:** Ensure episode ends even if normal path fails

#### 3.4 `environment.py` Line 655 - `wait_for_decisions()` max iterations guard
```python
if iterations >= max_iterations:
    logging.getLogger(__name__).warning("wait_for_decisions: max iterations (%d) reached...", ...)
    self.done = True
    return [], float(self.env.now)
```
**Trigger:** Loop iterations exceed `max_iterations=1000`  
**Type:** Emergency termination (deadlock prevention)  
**Purpose:** Safety guard against infinite loops

---

## 4. External/Tool Usage (3 Locations)

### 4.1 `utils/task_generator.py` Line 212
```python
while float(_get_attr('now', 0.0)) < float(_get_attr('episode_limit', float('inf'))) and not bool(_get_attr('done', False)):
```
**Purpose:** Task generator respects episode termination  
**Type:** External component coordination

### 4.2 `tools/test_concurrency.py` Line 60
```python
while not env.done and env.env.now < env.episode_limit:
```
**Purpose:** Test tool loop control  
**Type:** Testing utility

### 4.3 `tools/smoke_generate_gantt.py` Line 96
```python
while not getattr(env, 'done', False) and env.env.now < env.episode_limit and steps < 1000:
```
**Purpose:** Gantt generation tool loop control  
**Type:** Visualization utility

---

## 5. Summary Table

| Usage Type | File | Line(s) | Operation | Count |
|------------|------|---------|-----------|-------|
| **Initialize** | `environment.py` | 350, 537 | `self.done = False` | 2 |
| **Read (Check)** | `environment.py` | 619, 644, 1554 | Check done status | 3 |
| **Read (Copy)** | `rollout.py` | 696, 963 | Read for transitions/loop | 2 |
| **Set (Normal)** | `environment.py` | 611, 1388 | Time limit → `done=True` | 2 |
| **Set (Emergency)** | `environment.py` | 647, 655 | Safety → `done=True` | 2 |
| **External** | `task_generator.py`, `tools/` | 212, 60, 96 | External checks | 3 |
| **Total** | - | - | - | **14** |

---

## 6. Control Flow Diagram

```
Episode Start
    ↓
self.done = False (init/reset)
    ↓
┌─────────────────────────────────────┐
│  Episode Execution Loop             │
│  (rollout.py)                       │
│                                     │
│  while True:                        │
│    batch = env.wait_for_decisions() │
│        ↓                            │
│    Check: self.done? → return []    │ (Line 619)
│        ↓                            │
│    Loop: while not done             │ (Line 644)
│        ↓                            │
│    Check: time_limit? → done=True   │ (Line 647)
│    Check: max_iter? → done=True     │ (Line 655)
│        ↓                            │
│    if not batch → break             │
│    if env.done → break              │ (Line 963)
│        ↓                            │
│    step()                           │
│        ↓                            │
│    Check: t >= limit? → done=True   │ (Line 611)
└─────────────────────────────────────┘
    ↓
Episode End
```

---

## 7. Semantics After Refactor

### `self.done = True` Means ONLY:

1. **Time Limit Reached**: `env.now >= episode_limit` (normal termination)
2. **Emergency Guard Triggered**: Infinite loop prevention (rare)

### `self.done = True` Does NOT Mean:

- ❌ All jobs finished (removed in refactor)
- ❌ No more decisions pending (temporary state, not termination)
- ❌ Manual intervention/stop

---

## 8. Validation Checklist

✅ **Single Source of Truth**: `episode_limit` from `args.episode_limit`  
✅ **No Dual Paths**: Removed fallback `default=600`  
✅ **No "All Jobs Finished"**: Removed from termination logic  
✅ **Centralized Authority**: Environment controls done flag  
✅ **Emergency Guards Preserved**: `max_iterations=1000` safety guard intact  
✅ **Rollout Trust**: Rollout only checks `env.done`, no duplicate logic  

---

## 9. Key Differences from Previous Implementation

| Aspect | Before Refactor | After Refactor |
|--------|----------------|----------------|
| **episode_limit source** | `_resolve(..., default=600)` | `int(args.episode_limit)` |
| **Termination conditions** | Time limit OR all jobs finished | Time limit ONLY |
| **Done flag semantics** | Mixed (completion + time) | Pure (time + emergency) |
| **Duplicate checks** | Yes (rollout had time checks) | No (rollout trusts env) |
| **Fallback logic** | Multiple defaults | No fallbacks |

---

## 10. Related Documentation

- **Episode Termination Mechanism**: `docs/episode_termination_mechanism.md`
- **Episode Termination Plan**: `.copilot/EPISODE_TERMINATION_PLAN.md`
- **Co-Pilot Rules**: `.copilot/Co-Pilot Rules.md`

---

## 11. Maintenance Notes

### When to Modify `self.done`:

- ✅ Adding new emergency guards (e.g., memory limit, wall-clock timeout)
- ✅ Changing time limit calculation logic (rare)

### When NOT to Modify `self.done`:

- ❌ Adding new episode end conditions (use time limit instead)
- ❌ Creating completion metrics (use `all(j.finished)` for metrics ONLY, not termination)
- ❌ Implementing early stopping (use time limit override, not done flag)

### Testing Done Flag:

When testing episode termination, verify:
1. `done=False` at episode start
2. `done=True` when `env.now >= episode_limit`
3. `done=True` triggers episode loop exit
4. Emergency guards set `done=True` appropriately
5. `done` flag reset on `env.reset()`

---

**End of Report**
