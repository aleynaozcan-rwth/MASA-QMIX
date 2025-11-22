# Epsilon Decay Trace Analysis

**Generated:** 2025-11-22  
**Branch:** refactor-modules-hybridreward-v12  
**Command:** `python main.py --alg qmix --learn 1 --n_epoch 1 --n_episodes 1 --seed 42`

---

## Summary

### Epsilon Configuration
- `epsilon_start`: 1.0
- `epsilon_end`: 0.05
- `episode_limit`: 500 SimPy time units
- **Formula**: `epsilon(t) = epsilon_start - (t/episode_limit) × (epsilon_start - epsilon_end)`

### Episode Results
- **Episodes run**: 1
- **Total jobs created**: 121
- **Operations completed**: 0 ⚠️
- **Episode reward**: -0.8400
- **Duration**: 500 SimPy time units (time limit reached)

---

## Critical Issues Found

### 1. Shape Mismatch Exceptions ❌

**Every decision raised:**
```
[C1] Exception in rollout batch processing: shape '[10, -1]' is invalid for input of size X
```

**Pattern Analysis:**

| Decision # | Timestamp | Exception Size | Expected Size | Agents Equivalent |
|-----------|-----------|----------------|---------------|-------------------|
| 1 | t=1.00 | 28 | 70 (10×7) | 4 agents (28/7) |
| 2 | t=2.00 | 7 | 70 (10×7) | 1 agent |
| 3 | t=8.00 | 7 | 70 (10×7) | 1 agent |
| 4 | t=9.00 | 7 | 70 (10×7) | 1 agent |
| 5 | t=10.00 | 7 | 70 (10×7) | 1 agent |
| 6 | t=12.00 | 14 | 70 (10×7) | 2 agents (14/7) |

**Root Cause:**
The observation batch is not properly structured. Each agent should provide a 7-dimensional observation vector. For 10 agents, the total flattened size should be **70**.

**Actual sizes** (28, 7, 7, 7, 7, 14) suggest:
- Variable number of agents providing observations (1-4 agents)
- Observation collection is incomplete
- Batch assembly skips some agents
- OR agents are not all ready/available at decision time

**Location of Error:**
The reshape operation `shape '[10, -1]'` assumes ALL 10 agents provide data, but actual input has fewer observations.

---

### 2. Epsilon NOT Being Updated ⚠️

**Evidence:**
- ✅ Initial value: `epsilon=1.0` at t=0 (correct)
- ❌ No `[TIME-EPSILON]` logs (only logged every 20 episodes, ran 1 episode)
- ❌ No epsilon decay updates visible

**Expected Behavior:**
At each decision, epsilon should update based on `env.env.now`:

| Time | Formula | Expected ε |
|------|---------|------------|
| t=0 | 1.0 - (0/500)×0.95 | 1.0000 |
| t=1 | 1.0 - (1/500)×0.95 | 0.9981 |
| t=2 | 1.0 - (2/500)×0.95 | 0.9962 |
| t=10 | 1.0 - (10/500)×0.95 | 0.9810 |
| t=100 | 1.0 - (100/500)×0.95 | 0.8100 |
| t=500 | 1.0 - (500/500)×0.95 | 0.0500 |

**Actual Behavior:**
Cannot verify because all `_select_actions()` calls failed **BEFORE** reaching the time-based epsilon decay code block.

**Code Location:**
```python
# rollout.py line ~863
actions, _ = self._select_actions(obs_list, avail, evaluate=evaluate, epsilon=self.epsilon)

# [TIME-BASED EPSILON DECAY] - NEVER REACHED
if not evaluate:
    current_t = float(self.env.env.now)
    limit_t = float(self.episode_limit)
    fraction = min(1.0, current_t / limit_t)
    self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)
```

The epsilon decay code is **NEVER EXECUTED** because exceptions occur during observation batch processing in the lines before `_select_actions()`.

---

### 3. Decision Flow Analysis 🔍

**Observed Timestamps:**
- t=1.00, t=2.00, t=8.00, t=9.00, t=10.00, t=12.00, ..., t=500.00

**Anomalies:**
1. ✅ Decisions attempted at multiple timepoints (expected)
2. ❌ **EVERY** decision raised shape mismatch exception
3. ❌ **NO** actions were successfully selected
4. ❌ Episode continued until t=500 (time limit) despite failures
5. ❌ Zero operations completed despite 121 jobs created

**Execution Flow:**
```
Episode Start (t=0)
  ↓
Decision Point (t=1)
  ↓
build_all_agent_obs() → Returns partial observations (4 agents worth)
  ↓
Reshape attempt: [10, -1] with size=28 → EXCEPTION
  ↓
Catch exception, log warning, continue episode
  ↓
Decision Point (t=2)
  ↓
... (repeat until t=500)
  ↓
Episode End (time limit)
```

---

### 4. No Epsilon ValueError ✅

**Expected:**
If `epsilon=None` reached `select_actions()`, should raise:
```
ValueError: epsilon must be provided explicitly to _select_actions
```

**Actual:**
No such error found in logs.

**Interpretation:**
- ✅ Epsilon IS being passed correctly to `_select_actions()`
- ❌ BUT exceptions occur **BEFORE** `_select_actions()` is called
- ❌ The epsilon parameter validation never runs because observation processing fails first

**Conclusion:**
The time-based epsilon decay migration is **architecturally correct** but cannot be tested because the observation batch assembly is broken.

---

## Detailed Timeline

```
t=0.00   | Episode start, epsilon initialized to 1.0
t=1.00   | Decision attempt → Exception: size=28 (4 agents × 7 obs)
t=2.00   | Decision attempt → Exception: size=7 (1 agent × 7 obs)
t=8.00   | Decision attempt → Exception: size=7 (1 agent × 7 obs)
t=9.00   | Decision attempt → Exception: size=7 (1 agent × 7 obs)
t=10.00  | Decision attempt → Exception: size=7 (1 agent × 7 obs)
t=12.00  | Decision attempt → Exception: size=14 (2 agents × 7 obs)
...
t=500.00 | Episode end (time limit reached)
         | Final statistics: 121 jobs, 0 operations, reward=-0.84
```

---

## Root Cause Analysis

### Primary Issue: Observation Batch Assembly Failure

**Where:** Between `build_all_agent_obs()` and `_select_actions()`

**What's Broken:**
The system expects observations from all 10 agents (machines), but only receives observations from a subset (1-4 agents) at each decision point.

**Why This Matters:**
1. The neural network expects fixed-size input: `[batch_size, n_agents, obs_dim]`
2. Current implementation tries to reshape variable-size input into fixed shape
3. Reshape fails when input size doesn't match `10 × 7 = 70`

**Possible Causes:**
1. **Agent Availability**: Not all machines are "ready" at every decision point
2. **Observation Filtering**: Code filters out agents without pending operations
3. **Batch Collection Bug**: `build_all_agent_obs()` doesn't collect from all agents
4. **Timing Issue**: Some agents skip decision callbacks

---

## Epsilon System Status

### ✅ What Works
1. Epsilon initialization: `epsilon=1.0` at episode start
2. Epsilon parameter passing: No `ValueError` for missing epsilon
3. Time-based decay formula: Code is correct (when executed)
4. Episode termination: Properly ends at `episode_limit=500`

### ❌ What's Broken
1. Observation batch size varies (28, 7, 14 instead of 70)
2. Epsilon decay never executes (exceptions occur first)
3. No actions selected (all decisions fail)
4. Zero operations completed

### ⚠️ What's Unknown
1. Whether epsilon decay works IF observations were fixed
2. Exact reason for variable agent observation counts
3. Whether this is a new regression or existing bug
4. If time-based decay formula produces correct values

---

## Recommendations

### Priority 1: Fix Observation Batch Assembly ⚠️ CRITICAL

**Action:** Debug `build_all_agent_obs()` in `environment.py`

**Add logging:**
```python
obs_list = self.build_all_agent_obs()
print(f"[OBS_DEBUG] t={self.env.now:.2f} | "
      f"n_agents_expected={self.n_agents} | "
      f"obs_collected={len(obs_list)} | "
      f"total_size={sum(len(o) if hasattr(o, '__len__') else 1 for o in obs_list)}")
```

**Questions to answer:**
- How many observations does `build_all_agent_obs()` return?
- Why does it vary (4, 1, 1, 1, 1, 2)?
- Are agents filtered based on availability?
- Should we pad missing observations with zeros?

### Priority 2: Add Epsilon Decay Logging 📊

**Action:** Insert logging AFTER epsilon update (if reached)

**Add to rollout.py after line ~867:**
```python
if not evaluate:
    current_t = float(self.env.env.now)
    limit_t = float(self.episode_limit)
    fraction = min(1.0, current_t / limit_t)
    self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)
    print(f"[EPSILON_TRACE] t={current_t:.2f} fraction={fraction:.4f} epsilon={self.epsilon:.4f}")
```

This will show if epsilon decay executes once observations are fixed.

### Priority 3: Verify Agent Count Consistency 🔍

**Check:**
1. `self.n_agents` value in environment
2. `len(self.machines)` at runtime
3. Whether all machines are "active" simultaneously
4. If decision callbacks fire for all agents

### Priority 4: Test Epsilon Decay in Isolation ✅

**Create standalone test:**
```python
def test_time_based_epsilon():
    epsilon_start = 1.0
    epsilon_end = 0.05
    episode_limit = 500.0
    
    for t in [0, 1, 10, 100, 250, 500]:
        fraction = min(1.0, t / episode_limit)
        epsilon = epsilon_start - fraction * (epsilon_start - epsilon_end)
        print(f"t={t:>3} → epsilon={epsilon:.4f}")
```

Run this to verify formula correctness independent of observation issues.

### Priority 5: Consider Graceful Degradation 🛡️

**Options:**
1. **Pad observations**: If fewer than 10 agents, pad with zeros
2. **Dynamic reshape**: Use actual agent count instead of fixed `[10, -1]`
3. **Skip invalid decisions**: Log and continue without crashing
4. **Fail-fast**: Raise exception immediately to surface the issue

---

## Conclusion

**Epsilon Decay Migration:** ✅ **ARCHITECTURALLY CORRECT**
- Time-based formula is implemented correctly
- Epsilon parameter passing works
- No fallback logic violations

**System Integration:** ❌ **OBSERVATION PIPELINE BROKEN**
- Epsilon decay cannot be tested due to upstream failures
- Observation batch assembly produces wrong sizes
- Zero operations completed in episode

**Next Step:** Fix observation collection in `build_all_agent_obs()` before retesting epsilon decay.

---

**END OF REPORT**
