# Episode Termination Mechanism - MASA-QMIX Implementation

## Summary

Episode termination in MASA-QMIX is **DYNAMIC** and controlled by a **combination of three conditions** checked at multiple locations in the codebase.

**Answer:** E) A combination of mechanisms

---

## Primary Termination Conditions (Logical OR)

An episode ends when **ANY** of these three conditions becomes true:

1. **SimPy Time Limit Reached**: `self.env.now >= self.episode_limit`
2. **All Jobs Finished**: `all(j.finished for j in self.jobs)`
3. **Done Flag Set**: `self.done = True`

---

## Exact Code Locations

### 1. Environment Termination Checks (`environment.py`)

#### Line 610 - `step()` method
```python
done = (self.t >= self.episode_limit) or all(j.finished for j in self.jobs)
self.done = self.done or done
return obs, float(reward), bool(done), info
```

#### Lines 646-648 - `wait_for_decisions()` loop
```python
if float(self.env.now) >= float(self.episode_limit):
    self.done = True
    logging.getLogger(__name__).debug("wait_for_decisions: episode limit reached (t=%.2f >= %.2f)", float(self.env.now), float(self.episode_limit))
    return [], float(self.env.now)
```

#### Lines 651-655 - Safety limit in `wait_for_decisions()`
```python
iterations += 1
if iterations >= max_iterations:
    logging.getLogger(__name__).warning("wait_for_decisions: max iterations (%d) reached at t=%.2f, returning empty batch", max_iterations, float(self.env.now))
    self.done = True
    return [], float(self.env.now)
```

#### Lines 1387-1390 - `_check_pending_activation()`
```python
if all(j.finished for j in self.jobs) or self.env.now >= self.episode_limit:
    self.done = True
    if not getattr(self.decisions_ready, 'triggered', False):
        self.decisions_ready.succeed()
```

#### Lines 1555-1559 - `is_episode_done()` method
```python
all_finished = all(getattr(j, 'finished', False) for j in self.jobs)
time_limit_reached = float(self.env.now) >= float(self.episode_limit)
env_done = getattr(self, 'done', False)

return all_finished or time_limit_reached or env_done
```

---

### 2. Rollout Worker Termination Checks (`MARL/common/rollout.py`)

#### Lines 839-843 - Main episode loop
```python
while True:
    batch, sim_time = self.env.wait_for_decisions()
    # empty batch may mean done
    if not batch:
        break
```

#### Lines 966-970 - Episode end conditions
```python
# stop if env signals done
if getattr(self.env, "done", False):
    break
# safety: break if time limit reached
if getattr(self.env, "t", 0.0) >= getattr(self.env, "episode_limit", getattr(args, 'n_steps', 1e9)):
    break
```

---

## Episode Limit Configuration

### Definition Locations

**Primary:** `environment.py` line 180
```python
self.episode_limit = _resolve(('episode_limit',), int, default=600)
```

**Arguments:** `MARL/common/arguments.py` line 122
```python
parser.add_argument('--episode_limit', type=int, default=500)
```

### Current Value
- Default: `500` (from arguments.py)
- Can be overridden via command line: `--episode_limit <value>`
- Represents **SimPy simulation time** (not RL steps, not real time)

---

## Episode Length Characteristics

### Deterministic vs Dynamic

**DYNAMIC** - Episode length varies based on:

1. **Job Completion Time**
   - How long it takes for all jobs to finish
   - Depends on: operation durations, machine availability, operator availability
   - Influenced by: job arrival rate, number of operations per job

2. **Time Limit**
   - Whether SimPy time reaches `episode_limit` before all jobs finish
   - Current default: 500 SimPy time units

3. **Safety Limits**
   - `max_iterations=1000` in `wait_for_decisions()` prevents infinite loops
   - Triggers when environment has no progress

### NOT Fixed By

- ❌ **NOT** a fixed number of RL decision steps
- ❌ **NOT** a fixed number of actions taken
- ❌ **NOT** real-world time

### Variable Factors

Decision count per episode varies based on:
- `arrival_lambda=0.2` (job arrival rate)
- `job_min_ops=2`, `job_max_ops=4` (operations per job)
- Operation execution speed
- Machine and operator availability

---

## Episode Control Flow

```
Runner (runner.py)
    ↓
RolloutWorker.run_event_driven_episode() (rollout.py)
    ↓
    while True:
        ↓
        env.wait_for_decisions() (environment.py)
            ↓
            Check: self.env.now >= self.episode_limit? → done=True
            Check: all jobs finished? → done=True
            Check: max_iterations reached? → done=True
            ↓
            Return batch or empty list
        ↓
        if not batch → break
        if env.done → break
        if env.t >= episode_limit → break
    ↓
    Return episode data
```

---

## Termination Summary Table

| Mechanism | File | Line(s) | Condition | Type |
|-----------|------|---------|-----------|------|
| Time Limit | `environment.py` | 610, 646, 1387, 1555 | `self.env.now >= self.episode_limit` | Primary |
| Jobs Finished | `environment.py` | 610, 1387, 1555 | `all(j.finished for j in self.jobs)` | Primary |
| Done Flag | `environment.py` | Multiple | `self.done = True` | State Variable |
| Rollout Check | `rollout.py` | 966-970 | Reads `env.done` and `env.t >= episode_limit` | Secondary |
| Safety Limit | `environment.py` | 651-655 | `iterations >= max_iterations (1000)` | Safety Guard |
| Empty Batch | `rollout.py` | 841-843 | `if not batch: break` | Loop Control |

---

## Key Takeaways

1. **Episode Controller**: `rollout.py` drives episode execution by calling `env.wait_for_decisions()` in a loop
2. **Episode Length**: Dynamic, bounded by `episode_limit=500` SimPy seconds OR all jobs finishing
3. **Termination Logic**: Distributed across `environment.py` (primary checks) and `rollout.py` (secondary checks)
4. **Decision Count**: Variable per episode, NOT a fixed number
5. **Time Type**: SimPy simulation time (discrete-event time), not real-world time or RL step count

---

## Related Parameters

- `episode_limit`: Max SimPy time per episode (default: 500)
- `arrival_lambda`: Job arrival rate (default: 0.2)
- `n_episodes`: Number of episodes per epoch (default: 4)
- `n_epoch`: Number of training epochs (default: 400)
- `max_iterations`: Safety limit in `wait_for_decisions()` (hardcoded: 1000)

---

**Last Updated:** 2025-11-22
**Branch:** refactor-modules-hybridreward-v12
