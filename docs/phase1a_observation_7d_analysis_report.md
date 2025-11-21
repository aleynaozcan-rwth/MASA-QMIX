# PHASE 1A — COMPREHENSIVE PRE-IMPLEMENTATION ANALYSIS
## 7-Element Observation Migration Impact Report

**Report Date:** 2025-11-20  
**Status:** ANALYSIS-ONLY — NO CODE CHANGES  
**Scope:** Complete dependency mapping before migrating from 6D → 7D observation

---

## =================================================================================
## === QUALIFICATION LOGIC REPORT (PHASE 1A.1) ===
## =================================================================================

### A. ALL QUALIFICATION DETERMINATION LOCATIONS

This section documents every location where "which machines can process operation type X" logic exists.

---

#### **LOCATION 1: `environment.py` → `_avail_row_for_job()` (lines 1426-1462)**

**File + Line Range:**  
`/home/cc253232/MASA-QMIX/environment.py:1426-1462`

**Logic Description:**
- **PRIMARY SOURCE OF TRUTH** for machine qualification at operation level
- Resolves operation index (`op_idx_local`) from job's current operation
- Uses `self.workcenters_meta.machine_registry` as authoritative source
- For each machine in `machine_list`, checks if `op_idx_local` is in machine's `capabilities` list
- Returns numpy array (length = num_machines) where `1` = qualified, `0` = not qualified

**Key Code Pattern:**
```python
registry = getattr(self.workcenters_meta, 'machine_registry', {}) or {}
mlist_local = list(getattr(self.workcenters_meta, 'machine_list', []) or [])
for i, mname in enumerate(mlist_local):
    caps = registry.get(mname, {}).get('capabilities', [])
    if int(op_idx_local) in caps:
        row[i] = 1
```

**Canonical Status:**  
✅ **THIS IS THE CANONICAL SOURCE**

**Dependencies:**
- `workcenters_meta.machine_registry` (dict: machine_name → {capabilities: [op_indices], workcenter: int})
- `workcenters_meta.machine_list` (list of machine names)
- Fallback: If registry doesn't mark any machines, falls back to legacy `allowed_machine_indices` from operation tuple

**Potential Divergence:**  
- If `machine_registry` is not properly initialized, falls back to operation tuple's `allowed_machine_indices`
- This fallback could diverge if different parts of code modify operation tuples independently

---

#### **LOCATION 2: `environment.py` → `_build_avail_actions()` (lines 1346-1425)**

**File + Line Range:**  
`/home/cc253232/MASA-QMIX/environment.py:1346-1425`

**Logic Description:**
- Batch version of qualification logic for ALL jobs
- Calls `_avail_row_for_job(job)` for each job (delegates to LOCATION 1)
- **Additional filtering:** Also checks machine resource availability and operator group availability
- Returns 2D array: `(num_jobs, num_machines)` where `1` = both qualified AND available

**Key Code Pattern:**
```python
for idx, j in enumerate(self.jobs):
    if j.finished:
        continue
    row = self._avail_row_for_job(j)  # ← DELEGATES TO LOCATION 1
    if row is None:
        continue
    
    # Additional checks: machine_free[m] and operator_free
    for m in range(n_m):
        if int(row[m]) != 1:
            continue
        if not machine_free[m]:
            continue
        # operator group checks...
        avail[idx, m] = 1
```

**Canonical Status:**  
✅ **DERIVATIVE (delegates to canonical source)**

**Dependencies:**
- Calls `_avail_row_for_job()` → LOCATION 1
- Additional dependencies: `machine_resources`, `operator_groups`, `eligible_operator_groups_by_wc`

**Potential Divergence:**  
- None with respect to qualification logic (delegates to canonical source)
- Could diverge on **availability** checks if resource states are inconsistent

---

#### **LOCATION 3: `MARL/common/mask_utils.py` → `build_machine_major_mask()` (lines 76-150)**

**File + Line Range:**  
`/home/cc253232/MASA-QMIX/MARL/common/mask_utils.py:76-150`

**Logic Description:**
- RL-facing wrapper for availability mask generation
- **Preferred path:** Calls `env._build_avail_actions()` and extracts row for specific job
- **Fallback path:** Calls `env._avail_row_for_job(job)` directly
- Converts result to list of ints (0 or 1)

**Key Code Pattern:**
```python
# Preferred: batch matrix lookup
if hasattr(env, '_build_avail_actions'):
    matrix = env._build_avail_actions()
    row = matrix[int(job_idx)]
    return [int(bool(x)) for x in list(row)]

# Fallback: direct call
row = env._avail_row_for_job(job_or_job_id)
return [int(bool(x)) for x in list(row)]
```

**Canonical Status:**  
✅ **WRAPPER (delegates to canonical source)**

**Dependencies:**
- Preferred: `env._build_avail_actions()` → LOCATION 2 → LOCATION 1
- Fallback: `env._avail_row_for_job()` → LOCATION 1

**Potential Divergence:**  
- None (pure delegation, no independent qualification logic)
- Risk: If job indexing is inconsistent between `jobs` list and matrix rows

---

#### **LOCATION 4: `utils/workcenter.py` → `WorkCenters.create_decision_item()` (lines 226-350)**

**File + Line Range:**  
`/home/cc253232/MASA-QMIX/utils/workcenter.py:226-350`

**Logic Description:**
- Helper to build decision metadata for a job's operation
- **Re-derives** allowed machines by scanning `machine_registry` for machines with matching capabilities
- Returns `allowed_machines`, `allowed_machine_indices`, `per_machine_durations`
- Also calls `env._build_agent_obs(job)` and `env._avail_row_for_job(job)`

**Key Code Pattern:**
```python
op_idx_local = int(op_type) if (op_type is not None) else int(getattr(job, 'current_op_idx', 0))
mlist = list(getattr(self, 'machine_list', []))
for mname in mlist:
    mreg = self.machine_registry.get(mname, {})
    caps = mreg.get('capabilities', [])
    if op_idx_local in caps:
        allowed_machines.append(mname)
        allowed_machine_indices.append(int(mindex.get(mname, len(allowed_machine_indices))))
```

**Canonical Status:**  
⚠️ **DUPLICATE LOGIC** (re-scans `machine_registry` instead of calling `_avail_row_for_job`)

**Dependencies:**
- `self.machine_registry` (same source as LOCATION 1)
- Calls `env._avail_row_for_job(job)` for decision item, but **also independently derives allowed machines**

**Potential Divergence:**  
🔴 **HIGH RISK OF DIVERGENCE**
- This function independently scans `machine_registry` to build `allowed_machines`
- If `machine_registry` is modified after initialization, this could diverge from `_avail_row_for_job()`
- **RECOMMENDATION:** Replace independent scan with call to `_avail_row_for_job()` and convert binary row to machine name list

---

#### **LOCATION 5: `environment.py` → `_job_process()` operator qualification check (lines 1000-1043)**

**File + Line Range:**  
`/home/cc253232/MASA-QMIX/environment.py:1000-1043`

**Logic Description:**
- During job execution, checks if operators are qualified for chosen machine
- Uses `op_obj.can_do_job(op_idx_local, wc_idx)` method on Operator objects
- Fallback: `mname in op_obj.qualified_machines`
- **DIFFERENT CONCERN:** This is operator→machine qualification, not operation→machine qualification

**Key Code Pattern:**
```python
is_qualified = False
if wc_idx_for_m is not None:
    is_qualified = bool(op_obj.can_do_job(op_idx_local, wc_idx_for_m))
else:
    is_qualified = mname in op_obj.qualified_machines
```

**Canonical Status:**  
⚠️ **ORTHOGONAL CONCERN** (operator qualification, not operation→machine qualification)

**Dependencies:**
- `Operators` object and its `can_do_job()` method
- `qualified_machines` attribute on operator objects

**Potential Divergence:**  
🟡 **MEDIUM RISK**
- This checks **operator** qualification (can operator X run machine Y?)
- Could conflict with operation→machine qualification if operator cannot run a machine that operation theoretically requires
- **RECOMMENDATION:** Ensure operator qualifications are subset of machine capabilities

---

#### **LOCATION 6: `utils/workcenter.py` → `WorkCenter.__init__()` capability assignment (lines 95-145)**

**File + Line Range:**  
`/home/cc253232/MASA-QMIX/utils/workcenter.py:95-145`

**Logic Description:**
- During `WorkCenter` initialization, assigns capabilities to machines
- Reads from module-level `DEFAULT_WORKCENTERS` dictionary
- Maps operation names like `"Op1"` to 0-based indices
- Populates `self.machines[mid] = {'workcenter': wc_id, 'capabilities': caps_idx}`

**Key Code Pattern:**
```python
default_machines = DEFAULT_WORKCENTERS.get('machines', {})
meta = default_machines.get(mid)
if isinstance(meta, dict):
    raw_caps = meta.get('capable_ops', None) or meta.get('capabilities', None)
    if isinstance(raw_caps, (list, tuple)):
        caps_idx = []
        for c in raw_caps:
            if isinstance(c, str) and c.lower().startswith('op'):
                caps_idx.append(int(c[2:]) - 1)
```

**Canonical Status:**  
✅ **INITIALIZATION SOURCE** (populates canonical `machine_registry`)

**Dependencies:**
- Module-level `DEFAULT_WORKCENTERS` dict
- Must match `configs/env_config_enabled.yaml` processing times

**Potential Divergence:**  
🟡 **MEDIUM RISK**
- If `DEFAULT_WORKCENTERS` is manually edited without updating config YAML, capabilities could diverge
- **RECOMMENDATION:** Validate `DEFAULT_WORKCENTERS` matches config YAML in tests

---

### B. CANONICAL SOURCE DETERMINATION

**Single Source of Truth:**  
✅ `environment.py:_avail_row_for_job()` (LOCATION 1)

**Reason:**
- Used by all mask generation paths
- Directly queries authoritative `machine_registry`
- No independent duplication of qualification logic

**Duplicate/Derivative Locations:**
- LOCATION 2 (`_build_avail_actions`) → ✅ Delegates correctly
- LOCATION 3 (`mask_utils.build_machine_major_mask`) → ✅ Delegates correctly
- LOCATION 4 (`WorkCenters.create_decision_item`) → 🔴 **DUPLICATES LOGIC** (re-scans registry)
- LOCATION 5 (operator qualification in `_job_process`) → ⚠️ Orthogonal concern (operator qualification)
- LOCATION 6 (`WorkCenter.__init__`) → ✅ Initialization source (feeds canonical registry)

---

### C. DIVERGENCE RISK SUMMARY

| Location | Risk Level | Issue | Recommendation |
|----------|-----------|-------|----------------|
| LOCATION 1 | ✅ None | Canonical source | Keep as-is |
| LOCATION 2 | ✅ None | Delegates to LOCATION 1 | Keep as-is |
| LOCATION 3 | ✅ None | Pure wrapper | Keep as-is |
| LOCATION 4 | 🔴 HIGH | Duplicates registry scan | Replace with call to `_avail_row_for_job()` |
| LOCATION 5 | 🟡 MEDIUM | Operator qualification orthogonal | Validate operator quals ⊆ machine caps |
| LOCATION 6 | 🟡 MEDIUM | Initialization from dict | Validate against config YAML in tests |

---

## =================================================================================
## === SHAPE & CAPACITY SCAN (PHASE 1A.2) ===
## =================================================================================

### A. HARDCODED OBSERVATION SHAPE REFERENCES (obs_shape = 6, obs_dim_agent = 6)

---

#### **Reference 1: `environment.py:139` — Canonical shape definition**

**File + Line:** `/home/cc253232/MASA-QMIX/environment.py:139`

**Exact Text:**
```python
self.obs_dim_agent = 6  # fixed by canonical obs builder (see utils/env_obs.py)
```

**Assessment:**  
✅ **SAFE (Source of Truth)**  
This is the authoritative definition. Must be updated to `7` during migration.

**Impact:**  
- **Phase 1A Core Change #1:** Update to `self.obs_dim_agent = 7`

---

#### **Reference 2: `MARL/common/arguments.py:128` — Default fallback**

**File + Line:** `/home/cc253232/MASA-QMIX/MARL/common/arguments.py:128`

**Exact Text:**
```python
parser.add_argument('--obs_shape', type=int, default=6)
```

**Assessment:**  
🟡 **POTENTIALLY HARMFUL (Fallback conflicts with environment)**  
This is a CLI default. If code bypasses Runner and reads `args.obs_shape` directly, will get stale value.

**Impact:**  
- **Phase 1B Change:** Update to `default=7`
- Risk if scripts construct env without Runner (e.g., standalone tests)

---

#### **Reference 3: Test files with hardcoded assertions**

**Locations:**
- `tests/test_observation_shapes.py:14` → `assert obs.shape == (6,)`
- `tests/test_env_obs.py:14` → `assert aobs.shape == (6,)`
- `tests/smoke_test_obs_6d.py:196` → `assert len(obs) == env.obs_dim_agent == 6`

**Assessment:**  
✅ **SAFE (Test fixtures)**  
These are validation tests that explicitly check 6-element observation. Will require updates during Phase 1C.

**Impact:**  
- **Phase 1C:** Update all assertions to `assert obs.shape == (7,)` or `== 7`
- **Note:** `smoke_test_obs_6d.py` filename indicates it's specifically for 6D validation — may need renaming to `smoke_test_obs_7d.py`

---

#### **Reference 4: Test setup fixtures (`test_phase1_failfast.py`, `test_phase1_standalone.py`)**

**Locations:**
- `tests/test_phase1_failfast.py` → 8 occurrences: lines 31, 61, 97, 129, 161, 196, 229, 260, 302, 318
- `tests/test_phase1_standalone.py` → 8 occurrences: lines 62, 101, 140, 181, 223, 262, 304, 344

**Exact Text:**
```python
args.obs_shape = 6
```

**Assessment:**  
✅ **SAFE (Test fixtures)**  
These manually set `args.obs_shape` for isolated unit tests. Will need batch update.

**Impact:**  
- **Phase 1C:** Batch replace `args.obs_shape = 6` → `args.obs_shape = 7` in all test files
- Use grep/sed: `sed -i 's/args\.obs_shape = 6/args.obs_shape = 7/g' tests/*.py`

---

#### **Reference 5: `utils/env_obs.py:132` — Shape validation in observation builder**

**File + Line:** `/home/cc253232/MASA-QMIX/utils/env_obs.py:132`

**Exact Text:**
```python
if obs.shape[0] != 6:
    raise RuntimeError('Canonical observation must be length 6')
```

**Assessment:**  
🔴 **MUST BE UPDATED (Hard validation)**  
This is a runtime assertion that will **fail immediately** if observation builder returns 7 elements.

**Impact:**  
- **Phase 1A Core Change #2:** Update to `if obs.shape[0] != 7:`
- **Phase 1A Core Change #3:** Update error message to `'Canonical observation must be length 7'`

---

#### **Reference 6: `tools/run_initial_trace.py:11` — Hardcoded in standalone script**

**File + Line:** `/home/cc253232/MASA-QMIX/tools/run_initial_trace.py:11`

**Exact Text:**
```python
env = MASAEnv(num_jobs=4, num_operators=2, num_wcs=3, seed=123, obs_dim_agent=6, config_path='configs/env_no_arrival.yaml')
```

**Assessment:**  
🟡 **POTENTIALLY HARMFUL (Script bypasses environment default)**  
This script explicitly passes `obs_dim_agent=6`, overriding environment's canonical value.

**Impact:**  
- **Phase 1C:** Remove `obs_dim_agent=6` argument (let environment default to 7)
- Or update to `obs_dim_agent=7`

---

### B. MAXIMUM JOBS / CAPACITY REFERENCES (max_jobs, max_agents, n_agents)

These references relate to **capacity management** for dynamic job arrival systems.

---

#### **Reference 7: `environment.py:133` — num_jobs resolution**

**File + Line:** `/home/cc253232/MASA-QMIX/environment.py:133`

**Exact Text:**
```python
self.num_jobs = _resolve(('n_agents', 'num_jobs'), int, default=0)
```

**Assessment:**  
✅ **SAFE (Deterministic resolution)**  
Resolves `num_jobs` from either `args.n_agents` or `args.num_jobs` via canonical helper.

**Context:**  
- This is for **initial job count**, not dynamic capacity
- Used to derive `self.max_jobs` at line 163

---

#### **Reference 8: `environment.py:163` — max_jobs derived from num_jobs**

**File + Line:** `/home/cc253232/MASA-QMIX/environment.py:163`

**Exact Text:**
```python
self.max_jobs = int(self.num_jobs)
```

**Assessment:**  
✅ **SAFE (Capacity derived from num_jobs)**  
This sets maximum job capacity based on initial job count.

**Context:**  
- Currently, `max_jobs` equals `num_jobs` (no dynamic arrival support)
- For **dynamic arrivals**, this should be replaced with explicit capacity parameter

**Future Consideration (Phase 3):**  
- Add `args.max_capacity` to support dynamic arrivals where `max_jobs > num_jobs`

---

#### **Reference 9: `environment.py:289-291` — max_active_agents capacity validation**

**File + Line:** `/home/cc253232/MASA-QMIX/environment.py:289-291`

**Exact Text:**
```python
if not hasattr(args, 'n_agents') or args.n_agents is None:
    raise ValueError("args.n_agents is required to set max_active_agents capacity")
self.max_active_agents = int(args.n_agents)
```

**Assessment:**  
🟡 **POTENTIALLY HARMFUL (Uses args.n_agents for capacity)**  
This is one of 4 remaining `args.n_agents` dependencies identified in previous analysis.

**Context:**  
- Sets **maximum concurrent active jobs** based on `args.n_agents`
- For dynamic arrivals, should derive from `self.max_jobs` instead of args

**Future Consideration (Phase 3):**  
- Replace with `self.max_active_agents = int(self.max_jobs)`
- Remove dependency on `args.n_agents` for capacity management

---

#### **Reference 10: `environment.py:1552` — Capacity fallback chain**

**File + Line:** `/home/cc253232/MASA-QMIX/environment.py:1552`

**Exact Text:**
```python
capacity = self.max_active_agents or self.num_jobs or self.args.n_agents
```

**Assessment:**  
🟡 **POTENTIALLY HARMFUL (Fallback to args.n_agents)**  
This is a fallback chain in `add_job()` for capacity enforcement.

**Context:**  
- Fallback logic: prefer `max_active_agents`, then `num_jobs`, then `args.n_agents`
- Last fallback to `args.n_agents` could cause stale capacity values

**Future Consideration (Phase 3):**  
- Remove `self.args.n_agents` from fallback chain
- Enforce that `max_active_agents` is always set during `__init__`

---

#### **Reference 11: `environment.py:1616` — get_env_info() returns n_agents**

**File + Line:** `/home/cc253232/MASA-QMIX/environment.py:1616`

**Exact Text:**
```python
"n_agents": self.max_jobs
```

**Assessment:**  
✅ **SAFE (Correct propagation)**  
Returns `max_jobs` as `n_agents` for RL system. This is correct: RL agent count = job capacity.

**Context:**  
- Runner reads this and injects into `args.n_agents`
- RL agent/network/buffer all use `args.n_agents` from this injection

---

#### **Reference 12: `utils/env_obs.py:13, 49, 117, 152, 167` — Observation normalization uses max_jobs**

**Files + Lines:** `/home/cc253232/MASA-QMIX/utils/env_obs.py:13, 49, 117, 152, 167`

**Exact Text (sample):**
```python
# Line 13 (docstring)
# 3. n_jobs_active_norm -> env.active_jobs_count() / env.max_jobs

# Line 49
max_jobs = float(_require_positive(env, 'max_jobs'))

# Line 117
n_jobs_active_norm = np.clip(active_jobs_val / float(max_jobs), 0.0, 1.0)
```

**Assessment:**  
✅ **SAFE (Uses max_jobs for normalization)**  
Observation normalization correctly uses `env.max_jobs` as capacity reference.

**Context:**  
- Element 3 of 6-element observation (will become element 6 of 7-element)
- This ensures observations are normalized to job capacity, not agent count

---

#### **Reference 13: `environment.py:641, 646, 663, 668, 677` — Reward normalization uses max_jobs**

**Files + Lines:** `/home/cc253232/MASA-QMIX/environment.py:641, 646, 663, 668, 677`

**Exact Text (sample):**
```python
# Line 641
CompletedNorm = float(completed_count) / float(max(1, self.max_jobs))

# Line 663
WIPNorm = float(wip_count) / float(max(1, self.max_jobs))

# Line 677
throughput_delta = float(self._throughput_history[-1] - self._throughput_history[-2]) / float(max(1, self.max_jobs))
```

**Assessment:**  
✅ **SAFE (Correct reward normalization)**  
Reward components are normalized by `max_jobs` (job capacity).

**Context:**  
- Used for reward computation in hybrid reward system
- Ensures rewards scale with capacity

---

### C. SUMMARY OF SHAPE & CAPACITY FINDINGS

#### **Hardcoded Shape=6 References (11 total)**

| Location | Type | Phase 1 Action |
|----------|------|----------------|
| `environment.py:139` | Source of truth | Update to `7` |
| `arguments.py:128` | CLI default | Update to `7` |
| `utils/env_obs.py:132` | Runtime validation | Update to `7` |
| `tools/run_initial_trace.py:11` | Script hardcode | Remove or update |
| `tests/test_observation_shapes.py:14` | Test assertion | Update to `7` |
| `tests/test_env_obs.py:14` | Test assertion | Update to `7` |
| `tests/smoke_test_obs_6d.py:196` | Test assertion | Update to `7` |
| `tests/test_phase1_failfast.py` (8 locations) | Test fixtures | Batch replace |
| `tests/test_phase1_standalone.py` (8 locations) | Test fixtures | Batch replace |

---

#### **Capacity/n_agents References (Assessment)**

| Location | Type | Dynamic Arrival Safe? | Phase 3 Action |
|----------|------|----------------------|----------------|
| `environment.py:133` (num_jobs resolve) | Initialization | ✅ Safe | None needed |
| `environment.py:163` (max_jobs = num_jobs) | Capacity init | 🟡 Static only | Add `max_capacity` arg |
| `environment.py:289-291` (max_active_agents) | Capacity enforce | 🔴 Uses args.n_agents | Derive from max_jobs |
| `environment.py:1552` (capacity fallback) | Fallback chain | 🔴 Falls back to args | Remove args fallback |
| `environment.py:1616` (get_env_info) | RL injection | ✅ Safe | None needed |
| `utils/env_obs.py` (normalization) | Observation norm | ✅ Safe | None needed |
| `environment.py` (reward norm) | Reward norm | ✅ Safe | None needed |

**Conclusion:**  
- Current system assumes **static job count** (`max_jobs == num_jobs`)
- For **dynamic arrivals**, Phase 3 must decouple `max_jobs` (capacity) from `num_jobs` (initial count)
- Observation/reward normalization already correctly uses `max_jobs` ✅

---

## =================================================================================
## === MASK CONSISTENCY REPORT (PHASE 1A.3) ===
## =================================================================================

### A. ALL MASK-BUILDING FUNCTIONS

---

#### **MASK FUNCTION 1: `environment.py:_avail_row_for_job()` (lines 1426-1462)**

**Purpose:**  
Returns per-machine qualification row for a single job's current operation.

**Qualification Logic Source:**  
✅ Uses `workcenters_meta.machine_registry` → machine `capabilities` list

**Availability Check:**  
❌ **QUALIFICATION ONLY** (does not check current availability)

**Operator Constraints:**  
❌ Not considered in this function

**Consistency with `theoretical_machine_count`:**  
✅ **PERFECT MATCH**  
`theoretical_machine_count` should count machines where `row[m] == 1`

**Output Format:**  
Numpy array (shape: `(n_machines,)`, dtype: `int32`, values: `0` or `1`)

---

#### **MASK FUNCTION 2: `environment.py:_build_avail_actions()` (lines 1346-1425)**

**Purpose:**  
Returns availability matrix for ALL jobs, considering both qualification and current resource availability.

**Qualification Logic Source:**  
✅ Delegates to `_avail_row_for_job()` for each job

**Availability Check:**  
✅ **CURRENT AVAILABILITY ENFORCED**
- Checks `machine_free[m]` (machine resource not in use)
- Checks operator group availability via `eligible_operator_groups_by_wc`

**Operator Constraints:**  
✅ **OPERATOR GROUP AVAILABILITY CHECKED**
```python
eligible_groups = eligible_map.get(int(wc_i), [])
found_free_op = False
for g in eligible_groups:
    gi = int(g)
    if 0 <= gi < len(operator_free) and operator_free[gi]:
        found_free_op = True
        break
if found_free_op:
    avail[idx, m] = 1
```

**Consistency with `theoretical_machine_count` and `free_machine_count`:**  
- `theoretical_machine_count` → matches `_avail_row_for_job()` ✅
- `free_machine_count` → matches `_build_avail_actions()` row ✅

**Output Format:**  
Numpy array (shape: `(n_jobs, n_machines)`, dtype: `int32`, values: `0` or `1`)

---

#### **MASK FUNCTION 3: `MARL/common/mask_utils.py:build_machine_major_mask()` (lines 76-150)**

**Purpose:**  
RL-facing wrapper that returns per-machine availability for a job (used by rollout/policy).

**Qualification Logic Source:**  
✅ Delegates to `env._build_avail_actions()` or `env._avail_row_for_job()`

**Availability Check:**  
🟡 **DEPENDS ON CODE PATH**
- If calls `_build_avail_actions()` → includes availability ✅
- If calls `_avail_row_for_job()` → qualification only ❌

**Operator Constraints:**  
🟡 **DEPENDS ON CODE PATH**
- Matrix path: includes operator checks ✅
- Direct path: no operator checks ❌

**Consistency Check:**  
🟡 **CONDITIONAL MATCH**
- Matrix path: matches `free_machine_count` ✅
- Direct path: matches `theoretical_machine_count` ✅

**Code Logic:**
```python
# Preferred path: matrix
if hasattr(env, '_build_avail_actions'):
    matrix = env._build_avail_actions()  # includes availability
    row = matrix[int(job_idx)]
    return [int(bool(x)) for x in list(row)]

# Fallback path: direct row
row = env._avail_row_for_job(job_or_job_id)  # qualification only
return [int(bool(x)) for x in list(row)]
```

---

#### **MASK FUNCTION 4: `environment.py:_job_process()` decision-time mask (line 892)**

**Purpose:**  
During decision item creation, includes `avail_row` for the job.

**Qualification Logic Source:**  
✅ Calls `self._avail_row_for_job(job)`

**Availability Check:**  
❌ **QUALIFICATION ONLY** (decision item mask is static qualification row)

**Operator Constraints:**  
❌ Not included in decision item's `avail_row`

**Consistency Check:**  
✅ **MATCHES `theoretical_machine_count`**

**Code:**
```python
decision_item = {
    'job_id': job.id,
    'obs': self._build_agent_obs(job),
    'avail_row': self._avail_row_for_job(job),  # ← qualification only
    'allowed_machine_indices': allowed_machine_indices,
    'per_machine_durations': {},
}
```

---

#### **MASK FUNCTION 5: `utils/workcenter.py:create_decision_item()` (lines 226-350)**

**Purpose:**  
Creates decision metadata including `avail_row`.

**Qualification Logic Source:**  
✅ Calls `env._avail_row_for_job(job)`

**Availability Check:**  
❌ **QUALIFICATION ONLY**

**Operator Constraints:**  
❌ Not included

**Consistency Check:**  
✅ **MATCHES `theoretical_machine_count`**

**Code:**
```python
decision_item = {
    "job_id": getattr(job, 'id', getattr(job, 'agent_id', None)),
    "obs": env._build_agent_obs(job),
    "avail_row": env._avail_row_for_job(job),  # ← qualification only
    ...
}
```

---

### B. MASK USAGE IN ROLLOUT/DECISION FLOW

---

#### **Usage Point 1: `MARL/common/rollout.py` → Action selection mask (lines 180-188)**

**Context:**  
During episode rollout, computes availability mask for action selection.

**Mask Source:**  
Calls `build_machine_major_mask(self.env, jid)`

**Consistency:**  
🟡 **CONDITIONAL**
- If `build_machine_major_mask` uses matrix path → matches `free_machine_count` ✅
- If uses direct path → matches `theoretical_machine_count` ✅

**Code:**
```python
if item.get('avail_row') is None and build_machine_major_mask is not None:
    try:
        jid = item.get('job_id')
        row = build_machine_major_mask(self.env, jid)
        item['avail_row'] = row
```

---

#### **Usage Point 2: `MARL/runner.py` → Batch decision mask (lines 2558-2610)**

**Context:**  
Runner processes batch of decisions and builds availability masks.

**Mask Source:**  
Calls `build_machine_major_mask(self.env, jid)`

**Consistency:**  
🟡 **CONDITIONAL** (same as Usage Point 1)

---

### C. MASK vs. OBSERVATION CONSISTENCY ANALYSIS

---

#### **Observation Element: `theoretical_machine_count` (Element 5)**

**Definition:**  
Count of machines qualified for job's current operation (regardless of current availability).

**Must Match:**  
✅ `_avail_row_for_job()` output (count of `1`s)

**Mask Functions That Match:**
- ✅ `_avail_row_for_job()` → EXACT MATCH
- ✅ `_job_process()` decision item `avail_row` → EXACT MATCH
- ✅ `create_decision_item()` `avail_row` → EXACT MATCH
- ✅ `build_machine_major_mask()` (direct path) → EXACT MATCH

**Mask Functions That DON'T Match:**
- ❌ `_build_avail_actions()` → includes availability filter (would be ≤ theoretical count)
- ❌ `build_machine_major_mask()` (matrix path) → includes availability filter

**RECOMMENDATION:**  
When computing `theoretical_machine_count` for observation, **MUST call `_avail_row_for_job()` directly**, not `_build_avail_actions()`.

---

#### **Observation Element: `free_machine_count` (Element 6)**

**Definition:**  
Count of qualified machines that are **currently available** (machine resource free, operator available).

**Must Match:**  
✅ `_build_avail_actions()` output (count of `1`s in row)

**Mask Functions That Match:**
- ✅ `_build_avail_actions()` → EXACT MATCH
- ✅ `build_machine_major_mask()` (matrix path) → EXACT MATCH

**Mask Functions That DON'T Match:**
- ❌ `_avail_row_for_job()` → qualification only (no availability)
- ❌ `build_machine_major_mask()` (direct path) → qualification only

**RECOMMENDATION:**  
When computing `free_machine_count` for observation, **MUST call `_build_avail_actions()` and extract job's row**, not `_avail_row_for_job()`.

---

### D. INCONSISTENCIES DETECTED

---

#### **INCONSISTENCY 1: Mask granularity depends on code path**

**Location:** `MARL/common/mask_utils.py:build_machine_major_mask()`

**Issue:**  
- Matrix path (preferred): Returns availability mask (qualification + availability)
- Direct path (fallback): Returns qualification-only mask

**Impact on 7D Observation:**  
- If rollout uses matrix path → observation `free_machine_count` matches mask ✅
- If rollout uses direct path → observation `free_machine_count` will exceed mask count ❌

**Severity:** 🔴 **HIGH**

**Mitigation:**  
- **OPTION A:** Force `build_machine_major_mask()` to always use matrix path
- **OPTION B:** Add explicit flag: `include_availability=True/False`
- **OPTION C:** Split into two functions: `build_qualification_mask()` and `build_availability_mask()`

**RECOMMENDATION:** Implement OPTION C during Phase 1A

---

#### **INCONSISTENCY 2: Decision item `avail_row` is qualification-only, but agent needs availability**

**Location:** `environment.py:_job_process()` line 892, `utils/workcenter.py:create_decision_item()` line 348

**Issue:**  
- Decision item stores `avail_row` from `_avail_row_for_job()` (qualification only)
- Agent/policy needs current availability mask for action selection
- Rollout/runner must re-compute mask via `build_machine_major_mask()`

**Impact on 7D Observation:**  
- Observation `free_machine_count` requires availability check
- But decision item `avail_row` only provides qualification
- Creates inconsistency between decision item mask and observation

**Severity:** 🟡 **MEDIUM**

**Mitigation:**  
- **OPTION A:** Store both `qualification_row` and `availability_row` in decision item
- **OPTION B:** Add method `get_availability_row(job)` to distinguish from qualification
- **OPTION C:** Always recompute availability mask at decision time (current behavior)

**RECOMMENDATION:** Implement OPTION A during Phase 1A (add both rows to decision item)

---

#### **INCONSISTENCY 3: `WorkCenters.create_decision_item()` duplicates qualification logic**

**Location:** `utils/workcenter.py:create_decision_item()` lines 270-285

**Issue:**  
- Independently scans `machine_registry` to build `allowed_machines` list
- Duplicates qualification logic from `_avail_row_for_job()`
- If `machine_registry` is modified post-init, could diverge

**Impact on 7D Observation:**  
- `theoretical_machine_count` derived from `_avail_row_for_job()` might not match `allowed_machines` list
- Decision item metadata could be inconsistent with observation

**Severity:** 🔴 **HIGH**

**Mitigation:**  
Replace independent scan with:
```python
avail_row = env._avail_row_for_job(job)
allowed_machines = [mlist[i] for i in range(len(mlist)) if avail_row[i] == 1]
allowed_machine_indices = [i for i in range(len(mlist)) if avail_row[i] == 1]
```

**RECOMMENDATION:** Implement during Phase 1A

---

### E. MASK CONSISTENCY VERIFICATION CHECKLIST

Before implementing 7D observation, verify:

- ✅ `theoretical_machine_count` helper calls `_avail_row_for_job()` directly
- ✅ `free_machine_count` helper calls `_build_avail_actions()` and extracts row
- ✅ Decision items store both `qualification_row` and `availability_row`
- ✅ `build_machine_major_mask()` documents which type of mask it returns
- ✅ `WorkCenters.create_decision_item()` delegates qualification to `_avail_row_for_job()`
- ✅ Test that `theoretical_machine_count ≥ free_machine_count` always (availability is subset of qualification)

---

## =================================================================================
## === END OF REPORT ===
## =================================================================================

---

## SUMMARY & NEXT STEPS

### Key Findings

1. **Qualification Logic:** Single canonical source (`_avail_row_for_job`) but LOCATION 4 duplicates logic 🔴
2. **Shape References:** 11 hardcoded `obs_shape=6` references requiring updates ✅
3. **Capacity Management:** Current system assumes static job count; Phase 3 needs dynamic arrival support 🟡
4. **Mask Consistency:** 3 inconsistencies detected (granularity, decision item, duplication) 🔴

### Implementation Blockers

Must resolve before Phase 1A implementation:

1. 🔴 **BLOCKER 1:** Fix `WorkCenters.create_decision_item()` to delegate qualification (don't duplicate)
2. 🔴 **BLOCKER 2:** Decide mask granularity strategy (split functions or add flag)
3. 🟡 **ADVISORY:** Add both qualification + availability rows to decision items

### Ready for Implementation

Once blockers resolved:

- ✅ Add `get_theoretical_machine_count(job)` → calls `_avail_row_for_job()`, counts `1`s
- ✅ Add `get_free_machine_count(job)` → calls `_build_avail_actions()`, extracts row, counts `1`s
- ✅ Extend `build_agent_obs()` to 7 elements
- ✅ Update `self.obs_dim_agent = 7`
- ✅ Update all test assertions
- ✅ Verify `theoretical_machine_count ≥ free_machine_count` in tests

**Status:** READY FOR APPROVAL TO PROCEED TO PHASE 1B (after blocker resolution)
