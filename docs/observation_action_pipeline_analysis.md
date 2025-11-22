# Observation/Action Pipeline Architectural Analysis

**Date:** 2025-01-27  
**Status:** ANALYSIS ONLY - NO CODE MODIFICATIONS YET  
**Purpose:** Diagnose shape mismatch preventing epsilon decay from executing

---

## Executive Summary

The epsilon decay implementation is **architecturally correct** but never executes due to upstream observation shape mismatches. Root cause: **event-driven decision batch system** creates variable-size batches (1-4 items) while neural network expects fixed `[n_agents=10, obs_dim=7]` shapes.

**Key Finding:** MASA-QMIX uses an **asynchronous event-driven scheduling system** where only jobs requiring decisions at a given simulation time are included in the batch. This is fundamentally incompatible with QMIX's centralized training architecture which expects observations for ALL agents simultaneously.

---

## 1. Architecture Overview

### 1.1 System Design Philosophy

**MASA-QMIX Hybrid Architecture:**
- **Simulation Layer:** Event-driven discrete-event simulation (SimPy)
- **RL Layer:** Multi-agent reinforcement learning (QMIX)
- **Observation System:** Per-job observations built on-demand
- **Decision System:** Asynchronous - only jobs needing decisions are queried

**Design Mismatch:**
```
SimPy Event-Driven Model          QMIX Centralized Training
------------------------          -------------------------
• Jobs request decisions          • Requires ALL agent obs
  WHEN they need them            • Fixed batch shape
• Variable batch size (1-N)      • [n_agents, obs_dim]
• Asynchronous timing            • Synchronous batch processing
```

---

## 2. Data Flow Architecture

### 2.1 Observation Construction Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: JOB REQUESTS MACHINE (event-driven)                    │
│ File: environment.py::_request_machine()                       │
│ Trigger: Job needs to process next operation                   │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: BUILD SINGLE DECISION ITEM                             │
│ File: environment.py lines 935-952                             │
│                                                                 │
│ decision_item = {                                               │
│     'job_id': job.id,                                           │
│     'obs': self._build_agent_obs(job),  # 7-element vector     │
│     'avail_row': self._avail_row_for_job(job),                 │
│     'allowed_machine_indices': [...],                           │
│     'per_machine_durations': {...},                             │
│     'resume_evt': simpy.Event(self.env)                         │
│ }                                                               │
│                                                                 │
│ self.pending_decisions.append(decision_item)                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: WAIT FOR DECISIONS                                     │
│ File: environment.py::wait_for_decisions() lines 614-675       │
│ Returns: (batch, sim_time)                                     │
│                                                                 │
│ batch = list(self.pending_decisions)  # Variable size!         │
│ self.pending_decisions = []                                     │
│                                                                 │
│ Batch size: 1-N jobs (wherever N ≤ max_active_agents=10)      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: EXTRACT OBSERVATIONS                                   │
│ File: rollout.py::run_episode() line 844                       │
│                                                                 │
│ batch, sim_time = self.env.wait_for_decisions()                │
│ obs_list = [item.get('obs') for item in batch]                 │
│                                                                 │
│ Result: List of K observations, each shape (7,)                │
│ where K = len(batch) = variable (1-4 in traces)                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: SELECT ACTIONS                                         │
│ File: rollout.py line 851                                      │
│                                                                 │
│ actions, _ = self._select_actions(                             │
│     obs_list,         # List of K×(7,) arrays                  │
│     avail,            # List of K×(n_machines,) arrays         │
│     evaluate=evaluate,                                          │
│     epsilon=self.epsilon                                        │
│ )                                                               │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: QMIX POLICY EXPECTS FIXED SHAPE                        │
│ File: qmix.py::select_actions() lines 289-311                  │
│                                                                 │
│ # Comment: "Normalize obs_batch to (1, n_agents, obs_dim)"     │
│ if obs_batch.ndim == 2:                                         │
│     obs_batch = obs_batch.reshape(1, self.n_agents, -1)        │
│                                                                 │
│ ASSUMPTION: obs_batch has exactly n_agents=10 observations     │
│ ACTUAL INPUT: K observations where K < n_agents                │
│                                                                 │
│ ❌ FAILURE: ValueError("cannot reshape array of size K*7       │
│             into shape '(1, 10, -1)'")                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Shape Expectations vs Reality

### 3.1 Canonical Dimensions

**From `environment.py::get_env_info()` line 1675:**
```python
"n_agents": self.max_jobs,           # 10 (from args)
"obs_dim": self.obs_dim_agent,       # 7 (canonical)
```

**From `utils/env_obs.py::build_agent_obs()`:**
```python
# Returns: np.array([...], dtype=np.float32) shape (7,)
# Elements:
[
    current_op_type,              # float: 0-K (operation type index)
    total_ops,                    # float: total operation count
    remaining_ops,                # float: ops left to complete
    wait_time,                    # float: normalized wait time
    theoretical_machine_count,    # float: machines that can run this op
    free_machine_count,           # float: available machines
    n_jobs_active                 # float: total active jobs
]
```

### 3.2 Expected vs Actual Shapes

| Layer | Expected Shape | Actual Shape | Result |
|-------|---------------|--------------|--------|
| **Decision Batch** | N/A (variable by design) | `List[Dict]` len=1-4 | ✅ Correct |
| **Observation Extraction** | `List[n_agents × (7,)]` | `List[K × (7,)]` where K<10 | ❌ Mismatch |
| **QMIX Input (2D)** | `(n_agents, obs_dim)` = `(10, 7)` | `(K, 7)` where K∈{1,2,4} | ❌ Mismatch |
| **QMIX Input (3D)** | `(1, n_agents, obs_dim)` = `(1, 10, 7)` | `(1, K, 7)` | ❌ Mismatch |
| **Network Forward** | Total 70 elements | Total K×7 elements (7, 14, 28) | ❌ Mismatch |
| **Q-Values Output** | `(1, n_agents, n_actions)` = `(1, 10, 5)` | Cannot compute due to input failure | ❌ Blocked |

**Trace Evidence (from `docs/epsilon_trace_analysis.md`):**
```
[Decision 1] Batch size=4 → 4 obs → 4×7=28 elements
  ValueError: cannot reshape array of size 28 into shape '(1, 10, -1)'
  
[Decision 2] Batch size=1 → 1 obs → 1×7=7 elements
  ValueError: cannot reshape array of size 7 into shape '(1, 10, -1)'
  
[Decision 3] Batch size=2 → 2 obs → 2×7=14 elements
  ValueError: cannot reshape array of size 14 into shape '(1, 10, -1)'
```

---

## 4. Critical Code Locations

### 4.1 Observation Builder (Correct Implementation)

**File:** `utils/env_obs.py`  
**Function:** `build_agent_obs(env, job, job_index)` lines 56-79

```python
def build_agent_obs(env, job, job_index: int):
    """Build 7-element observation for a single agent/job."""
    # ... constructs 7-element vector ...
    return np.array([...], dtype=np.float32)  # Shape: (7,)
```

**Assessment:** ✅ Correct - produces canonical 7-element observation

---

### 4.2 Decision Batch Construction (Event-Driven)

**File:** `environment.py`  
**Function:** `_request_machine()` lines 935-952

```python
# Build decision item for THIS job only
decision_item = {
    'job_id': job.id,
    'obs': self._build_agent_obs(job),  # Single 7-element vector
    'avail_row': self._avail_row_for_job(job),
    'allowed_machine_indices': allowed_machine_indices,
    'per_machine_durations': {},
    'resume_evt': simpy.Event(self.env)
}

self.pending_decisions.append(decision_item)  # Add to batch
```

**Assessment:** ✅ Correct for event-driven design - only builds observations for jobs requesting decisions

---

### 4.3 Batch Retrieval (Returns Variable Size)

**File:** `environment.py`  
**Function:** `wait_for_decisions()` lines 614-675

```python
def wait_for_decisions(self):
    """Run sim until at least one decision is pending or episode ends."""
    # ... advance simulation ...
    
    batch = list(self.pending_decisions)  # Variable size: 1-N items
    self.pending_decisions = []
    return batch, float(self.t)
```

**Assessment:** ✅ Correct for event-driven system, but incompatible with QMIX's fixed-shape expectation

---

### 4.4 Observation Extraction (No Padding)

**File:** `MARL/common/rollout.py`  
**Function:** `run_episode()` line 844

```python
batch, sim_time = self.env.wait_for_decisions()
obs_list = [item.get('obs') for item in batch]  # Extract K observations
```

**Assessment:** ❌ Missing padding - should pad to n_agents with zeros/masks

---

### 4.5 Action Selection (Fixed Shape Assumption)

**File:** `MARL/policy/qmix.py`  
**Function:** `select_actions()` lines 289-311

```python
# Comment: "Normalize obs_batch to tensor shape (1, n_agents, obs_dim)"
if obs_batch.ndim == 2:
    # ASSUMPTION: obs_batch has shape (n_agents, obs_dim)
    obs_batch = obs_batch.reshape(1, self.n_agents, -1)
    # ❌ FAILS when obs_batch has K < n_agents rows
```

**Assessment:** ❌ Rigid reshape - assumes exactly n_agents observations without validation

---

### 4.6 Network Forward Pass

**File:** `MARL/network/base_net.py`  
**Class:** `RNNAgent`  
**Function:** `forward()` lines 31-56

```python
def forward(self, obs, hidden_state=None):
    # obs expected: (batch, seq, input_shape)
    # For QMIX: (1, n_agents, obs_dim)
    x = F.relu(self.fc1(obs))
    # ... RNN processing ...
    return q, h
```

**Assessment:** ⚠️ No explicit validation - relies on upstream providing correct shapes

---

## 5. Reshape Operations Audit

### 5.1 All Reshape Locations

| File | Line | Operation | Assumption | Status |
|------|------|-----------|------------|--------|
| `qmix.py` | 228 | `reshape(episode_num * self.n_agents, -1)` | Exactly n_agents per episode | ❌ Fails |
| `qmix.py` | 292-298 | `reshape(1, self.n_agents, -1)` | Input has n_agents rows | ❌ Fails |
| `qmix.py` | 311 | `view(1, self.n_agents, -1)` | Q-values for n_agents | ❌ Blocked |
| `rollout.py` | 526 | `n_agents = getattr(args, 'n_agents', len(u_list))` | Fallback to batch size | ⚠️ Violates Co-Pilot Rules |
| `base_net.py` | 42 | `x.view(b, t, -1)` | Expects valid batch dimension | ⚠️ Downstream of failure |

---

## 6. Agent Count Semantics

### 6.1 Conceptual Model

**MASA-QMIX uses three distinct agent-related concepts:**

1. **`max_jobs` (args.n_agents):** Maximum concurrent jobs in system (capacity limit) = 10
2. **`active_agents`:** Currently active jobs (list in `environment.py` line 286)
3. **`decision batch`:** Subset of active agents requiring decisions at current sim time

```
Capacity Hierarchy:
max_jobs (10) ≥ len(active_agents) ≥ len(decision_batch)
     ↓                ↓                      ↓
  Static            Dynamic              Highly Dynamic
  Config            (jobs arrive/        (only jobs needing
  Parameter         complete)            machine selection)
```

### 6.2 Active Agent Management

**File:** `environment.py` lines 1378-1381

```python
max_active = int(getattr(self, 'max_active_agents', 0))
if max_active <= 0 or len(self.active_agents) < max_active:
    # Add next pending job to active agents
    self.active_agents.append(next_job)
```

**Agent Lifecycle:**
1. Job arrives → added to `pending_jobs` queue
2. Capacity available → moved to `active_agents` list
3. Job requests machine → added to `pending_decisions` batch
4. Agent selects action → removed from `pending_decisions`
5. Job completes → removed from `active_agents`

**Key Insight:** At any given simulation time:
- `len(active_agents)` ≤ `max_jobs` (10)
- `len(pending_decisions)` ≤ `len(active_agents)`
- Only jobs needing machine selection appear in `pending_decisions`

---

## 7. Root Cause Analysis

### 7.1 Fundamental Architecture Conflict

**Event-Driven Simulation:**
- Jobs request decisions **asynchronously** (when ready for next operation)
- Batch size determined by **simulation events**, not RL configuration
- Efficient for discrete-event systems (only process what's needed)

**Centralized Multi-Agent RL (QMIX):**
- Requires observations for **all agents simultaneously**
- Mixer network combines Q-values across **fixed number of agents**
- Fixed batch shapes for neural network efficiency

**Conflict:** MASA-QMIX tries to combine both paradigms without proper adapter layer

---

### 7.2 Missing Component: Padding/Masking Layer

**Required Functionality:**
1. **Padding:** Expand decision batch to fixed size (n_agents=10)
2. **Masking:** Track which agents are real vs padded
3. **Q-Value Filtering:** Only use Q-values for real agents
4. **Mixer Handling:** Mask out padded agents in centralized value function

**Current State:** No such layer exists - observations go directly from variable-size batch to fixed-shape neural network

---

### 7.3 Why Epsilon Decay Never Executes

**Code Flow:**
```python
# rollout.py lines 844-860
obs_list = [item.get('obs') for item in batch]  # Variable size
actions, _ = self._select_actions(obs_list, avail, evaluate, epsilon=self.epsilon)
# ❌ EXCEPTION RAISED - never reaches next lines

# Time-based epsilon decay (NEVER EXECUTED)
if self.epsilon is not None and hasattr(self, 'args'):
    self.epsilon = compute_time_based_epsilon(...)
```

**Trace Evidence:**
```
Decision 1: Shape mismatch exception
Decision 2: Shape mismatch exception (epsilon unchanged)
Decision 3: Shape mismatch exception (epsilon unchanged)
...
Final epsilon: 1.0 (never decayed)
Total operations completed: 0 (no actions ever selected)
```

---

## 8. Where Padding Should Occur

### 8.1 Option 1: Environment Layer (Recommended)

**Location:** `environment.py::wait_for_decisions()`  
**Approach:** Pad batch before returning

**Pros:**
- Centralized solution - single modification point
- Preserves event-driven efficiency until batch construction
- Downstream code (rollout, policy) works without changes

**Cons:**
- Environment becomes aware of RL architecture details
- Requires mask metadata in batch items

**Implementation Sketch:**
```python
def wait_for_decisions(self):
    batch = list(self.pending_decisions)
    self.pending_decisions = []
    
    # Pad to n_agents
    n_agents = self.max_jobs
    real_count = len(batch)
    
    # Create padding items with zero observations and invalid actions
    dummy_obs = np.zeros(self.obs_dim_agent, dtype=np.float32)
    for i in range(real_count, n_agents):
        batch.append({
            'job_id': -1,  # Invalid job ID
            'obs': dummy_obs,
            'avail_row': np.zeros(self.n_machines, dtype=np.int32),
            'mask': 0  # Padded agent
        })
    
    # Add mask to real items
    for i in range(real_count):
        batch[i]['mask'] = 1  # Real agent
    
    return batch, float(self.t)
```

---

### 8.2 Option 2: Rollout Layer

**Location:** `rollout.py::run_episode()` line 844  
**Approach:** Pad observations after extraction

**Pros:**
- Keeps environment layer clean (no RL-specific logic)
- RL-specific padding in RL-specific file

**Cons:**
- Must also pad avail_actions, masks, and track real agent count
- More complex due to multiple parallel lists (obs, avail, masks)

**Implementation Sketch:**
```python
batch, sim_time = self.env.wait_for_decisions()
obs_list = [item.get('obs') for item in batch]
avail_list = [item.get('avail_row') for item in batch]

# Pad to n_agents
n_agents = self.args.n_agents
real_count = len(obs_list)
dummy_obs = np.zeros(self.env.obs_dim_agent, dtype=np.float32)
dummy_avail = np.zeros(self.env.n_machines, dtype=np.int32)

obs_padded = obs_list + [dummy_obs] * (n_agents - real_count)
avail_padded = avail_list + [dummy_avail] * (n_agents - real_count)
mask = [1] * real_count + [0] * (n_agents - real_count)

actions, _ = self._select_actions(obs_padded, avail_padded, evaluate, epsilon, mask=mask)
```

---

### 8.3 Option 3: Policy Layer

**Location:** `qmix.py::select_actions()`  
**Approach:** Dynamic reshaping based on actual input size

**Pros:**
- Most flexible - handles any batch size
- No upstream changes needed

**Cons:**
- **Breaks QMIX architecture** - mixer expects fixed agent count
- Requires rewriting centralized value function logic
- Violates QMIX paper's design (fixed multi-agent system)

**Assessment:** ❌ Not recommended - fundamentally incompatible with QMIX

---

### 8.4 Recommendation

**Use Option 1 (Environment Layer)** because:
1. ✅ Single modification point
2. ✅ Preserves QMIX's fixed-agent architecture
3. ✅ Explicit mask metadata travels with batch
4. ✅ Minimal downstream changes
5. ✅ Aligns with Co-Pilot Rules (no silent fallbacks - padding is explicit and visible)

---

## 9. Modification Checklist

### 9.1 Required Code Changes (If Approved)

**Phase 1: Add Padding Logic**
- [ ] `environment.py::wait_for_decisions()` - Add padding to n_agents, create mask field
- [ ] `rollout.py::run_episode()` - Extract mask from batch items
- [ ] `qmix.py::select_actions()` - Accept mask parameter, validate input size

**Phase 2: Propagate Masks**
- [ ] `rollout.py::_select_actions()` - Pass mask to policy
- [ ] `qmix.py::forward()` - Mask Q-values for padded agents
- [ ] `qmix_mixer.py` - Handle masked agents in centralized value function

**Phase 3: Validation**
- [ ] Add shape validation with explicit error messages (Co-Pilot Rule A5)
- [ ] Remove fallback logic in `rollout.py` line 526 (violates Rule A6)
- [ ] Add unit tests for padding with various batch sizes (1, 5, 10)
- [ ] Add integration test for masked Q-value computation

**Phase 4: Testing**
- [ ] Test with batch size = 1 (single agent)
- [ ] Test with batch size = n_agents (no padding needed)
- [ ] Test with batch size between 1 and n_agents
- [ ] Verify epsilon decay now executes after padding fix
- [ ] Verify operations complete successfully

---

## 10. Shape Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CURRENT (BROKEN) FLOW                            │
└─────────────────────────────────────────────────────────────────────┘

Environment                   Rollout                  Policy/Network
-----------                   -------                  --------------
Batch (K items)    ──────>   Extract K obs   ──────>  Expects 10 obs
K ∈ {1,2,3,4}                List[(7,)] ×K            Reshape to (10,7)
                                                      ❌ SHAPE MISMATCH


┌─────────────────────────────────────────────────────────────────────┐
│                    PROPOSED (FIXED) FLOW                            │
└─────────────────────────────────────────────────────────────────────┘

Environment                   Rollout                  Policy/Network
-----------                   -------                  --------------
Batch (K items)               Extract K obs            Receives 10 obs
K ∈ {1,2,3,4}                List[(7,)] ×K            Array (10,7)
    ↓                             ↓                         ↓
Pad to 10 items   ──────>   Extract 10 obs  ──────>  Reshape (10,7)
Add mask field               + mask [1,1,0,0,...]    ✅ SHAPE MATCH
Dummy obs for 10-K                                    Mask Q-values
```

---

## 11. Compliance with Co-Pilot Rules

**Rule Analysis:**

✅ **A1 (No Silent Fallbacks):** Current failure is GOOD - exception raised rather than silent default  
❌ **A6 (No Defensive Fallbacks):** `rollout.py` line 526 has fallback `n_agents = getattr(args, 'n_agents', len(u_list))`  
✅ **A5 (Fail-Fast):** Shape mismatch raises ValueError immediately  
⚠️ **A2 (No Ad-Hoc Fixes):** Current padding proposal must be systematic, not ad-hoc  

**Proposed Padding Design Compliance:**
- ✅ Explicit masking (not implicit)
- ✅ Validates real agent count
- ✅ Raises ValueError if mask is missing or inconsistent
- ✅ No silent size mismatches

**Required Removal:**
- Line 526 in `rollout.py`: Remove `getattr(args, 'n_agents', len(u_list))` fallback
- Must fail if `args.n_agents` is missing (Co-Pilot Rule A6)

---

## 12. Epsilon Decay Verification Plan

**After Padding Fix Applied:**

1. **Smoke Test:**
   ```bash
   python main.py --alg qmix --env MASAEnv --epsilon_start 1.0 --epsilon_end 0.05 --episode_limit 500
   ```
   **Expected:** No shape exceptions, epsilon decays from 1.0 → 0.05

2. **Trace Verification:**
   Enable detailed logging in `rollout.py` lines 855-860:
   ```python
   LOG.info(f"[EPSILON_DECAY] t={sim_time:.2f}, epsilon={self.epsilon:.4f}")
   ```
   **Expected:** Log entries showing epsilon decreasing over time

3. **Completion Verification:**
   Check environment metrics at episode end:
   ```python
   total_ops = sum(env trace completed operations)
   ```
   **Expected:** `total_ops > 0` (actions successfully selected)

4. **Shape Verification:**
   Add assertion in `qmix.py` line 292:
   ```python
   assert obs_batch.shape[0] == self.n_agents, \
       f"Shape mismatch: got {obs_batch.shape[0]}, expected {self.n_agents}"
   ```
   **Expected:** No assertion failures

---

## 13. Open Questions for User

1. **Padding Location Preference:** Approve Option 1 (Environment) or prefer Option 2 (Rollout)?

2. **Mask Propagation:** Should mask travel through entire pipeline or just to Q-value computation?

3. **Mixer Handling:** How should QMIX mixer handle padded agents? Options:
   - Mask them out (Q-values = 0)
   - Exclude from mixing computation
   - Use special "inactive" Q-value

4. **Validation Strictness:** Should we fail if batch size ever exceeds n_agents, or just log warning?

5. **Episode Limit Handling:** Current code advances sim in small steps. Should we optimize batch collection timing?

---

## 14. Summary

**Problem:** Event-driven decision batching (variable size) incompatible with QMIX's fixed-shape neural network architecture.

**Root Cause:** No padding/masking layer between simulation and RL components.

**Impact:** Epsilon decay never executes because action selection crashes before reaching decay code.

**Solution:** Add padding layer in `wait_for_decisions()` to normalize batch size to n_agents with explicit masking.

**Estimated Effort:** 4-6 hours (implementation + testing + validation)

**Risk:** Low - padding is well-understood pattern, changes localized to 3-4 files.

---

## 15. Next Steps

**USER ACTION REQUIRED:**

1. **Review this analysis** - Confirm architectural understanding is correct
2. **Choose padding location** - Approve Option 1, 2, or propose alternative
3. **Approve modifications** - Give go-ahead to implement code changes
4. **Clarify mask propagation** - Specify how masks should be used in mixer

**Once approved, I will:**

1. Implement padding logic with explicit masking
2. Update all downstream reshape operations
3. Remove Co-Pilot Rule violations (fallback logic)
4. Add comprehensive tests for variable batch sizes
5. Verify epsilon decay executes correctly
6. Run full episode with operation completion validation

---

**END OF ANALYSIS**
