# Action Selection Reasoning Audit – 3-Step Logging Analysis

**Date**: November 23, 2025  
**Branch**: make-it-work-and-converge-v2  
**Question**: Can we support clean 3-step reasoning logging: MachinesCanDo → MachinesFree → MachineOperatorFeasible?

---

## STEP 0 – Action Selection Pipeline Map

### Key Functions Identified:

**1. Decision Item Creation** (utils/workcenter.py)
- `WorkCenters.create_decision_item(env, job, op)` (line 230)
  - Builds initial decision metadata: allowed_machines, allowed_machine_indices, per_machine_durations
  - Uses `machine_registry['capabilities']` to determine MachinesCanDo
  - Calls `env._avail_row_for_job(job)` to populate `avail_row`

**2. Availability Row Construction** (environment.py)
- `_avail_row_for_job(job)` (line 1598-1658)
  - Returns per-job, per-machine availability vector (shape: n_machines)
  - **This is where the 3-step logic SHOULD occur**
  - Checks: capability → operator qualified & free
  - Returns `np.array([0 or 1, ...])` for each machine

**3. Full Availability Matrix** (environment.py)
- `_build_avail_actions()` (line 1518-1598)
  - Builds (n_jobs × n_machines) availability matrix
  - For each job: calls `_avail_row_for_job(job)`
  - Applies additional filters: machine_free, operator_free (WorkCenter-level)
  - Returns `np.array([[0/1, ...], ...])`

**4. Action Selection** (MARL/common/rollout.py → MARL/policy/qmix.py)
- `RolloutWorker.decide_batch(batch, evaluate)` (line 186)
  - Extracts `obs_list` and `avail` from decision batch
  - Calls `_select_actions(obs_list, avail, evaluate)`
- `RolloutWorker._select_actions(obs_batch, avail_batch, evaluate, epsilon)` (line 122)
  - Routes to `self.agents.select_actions(...)`
- `QMIX.select_actions(obs_batch, avail_batch, evaluate, epsilon)` (line 274)
  - Computes Q-values from neural network
  - Applies availability mask to Q-values
  - Performs epsilon-greedy or greedy action selection
  - Returns list of integer actions (machine indices)

**5. Action Processing & Application** (MARL/common/rollout.py)
- `RolloutWorker.process_and_apply_actions(batch, raw_actions, sim_time)` (line 218)
  - Validates chosen action against `avail_row`
  - Converts action index to machine name
  - Resumes SimPy event with chosen action
  - Logs selection to scheduling_trace.csv

### Pipeline Flow Summary:

```
Job needs decision
    ↓
WorkCenters.create_decision_item()
    ├─ Determines allowed_machines (from capabilities)
    └─ Calls env._avail_row_for_job(job) → avail_row
        ↓
Decision batch sent to RolloutWorker.decide_batch()
    ↓
RolloutWorker._select_actions(obs, avail)
    ↓
QMIX.select_actions(obs, avail, epsilon)
    ├─ Forward pass → Q-values
    ├─ Mask Q-values by avail
    └─ Epsilon-greedy selection → action (machine index)
        ↓
RolloutWorker.process_and_apply_actions()
    ├─ Validate action against avail_row
    ├─ Convert index to machine name
    └─ Resume SimPy event
```

---

## STEP 1 – Does 3-Step Reasoning Exist in Current Code?

### Desired 3-Step Flow:
1. **MachinesCanDo**: Static capability-level (operation → machines)
2. **MachinesFree**: Subset of MachinesCanDo that are not busy
3. **MachineOperatorFeasible**: Subset of MachinesFree with free qualified operator

### Current Implementation Analysis:

#### In `_avail_row_for_job()` (environment.py, line 1598-1658):

**CRITICAL FINDING**: The 3 steps are **COLLAPSED into a single loop**:

```python
for i, mname in enumerate(mlist_local):
    caps = registry.get(mname, {}).get('capabilities', [])
    if int(op_idx_local) in caps:  # ← STEP 1: MachinesCanDo
        # Machine can do this operation - now check if any FREE qualified operator exists
        wc_idx = registry.get(mname, {}).get('workcenter', None)
        has_free_qualified_operator = False
        
        if self.operators is not None and wc_idx is not None:
            # Check all operators: must be qualified for THIS MACHINE and FREE
            for op_obj in self.operators.operators_object_list:
                if mname in op_obj.qualified_machines:  # ← Operator qualified for machine
                    if op_obj.can_do_job(op_idx_local, wc_idx):  # ← Operator can do operation
                        if not op_obj.is_busy:  # ← Operator is FREE
                            has_free_qualified_operator = True
                            break
            
            # Only mark machine as available if free qualified operator exists
            if has_free_qualified_operator:
                row[i] = 1  # ← FINAL RESULT: All 3 conditions met
```

**PROBLEM**: 
- ❌ **MachinesFree** (STEP 2) is **NOT CHECKED** in `_avail_row_for_job()`
- The function only checks: capability (STEP 1) AND operator feasibility (STEP 3)
- Machine busy status is checked LATER in `_build_avail_actions()`

#### In `_build_avail_actions()` (environment.py, line 1518-1598):

**SECOND LAYER OF FILTERING**:

```python
# Compute per-machine free flags
machine_free = [self._resource_free(self.machine_resources[m]) for m in range(n_m)]

for idx, j in enumerate(self.jobs):
    row = self._avail_row_for_job(j)  # ← Returns capability + operator
    
    for m in range(n_m):
        if int(row[m]) != 1:  # ← Skip if not capable or no operator
            continue
        
        if not machine_free[m]:  # ← STEP 2: Check if machine is FREE
            continue
        
        # Additional WorkCenter-level operator check (redundant/buggy)
        wc_i = registry.get(mname, {}).get('workcenter')
        eligible_groups = eligible_map.get(int(wc_i), [])
        # ... more checks ...
        
        if found_free_op:
            avail[idx, m] = 1  # ← FINAL mask bit
```

**ARCHITECTURE ISSUE**: The 3 steps are **split across two functions**:
- `_avail_row_for_job()`: capability + operator feasibility (STEP 1 + STEP 3)
- `_build_avail_actions()`: machine free (STEP 2) + redundant operator checks

### Answer to STEP 1 Question:

**NO**, there is **NO clear separation** of the 3 steps. Current implementation has:

1. **MachinesCanDo** (STEP 1): Checked in `_avail_row_for_job()` via `if op_idx_local in caps`
   - ✅ Clean, explicit check
   - Variable: `caps` (list of operation indices machine can perform)

2. **MachinesFree** (STEP 2): Checked in `_build_avail_actions()` via `machine_free[m]`
   - ⚠️ Happens in DIFFERENT function
   - ⚠️ Applied AFTER operator checks already ran
   - Variable: `machine_free` (list of booleans per machine)

3. **MachineOperatorFeasible** (STEP 3): Checked in `_avail_row_for_job()` via operator loop
   - ⚠️ Happens BEFORE machine free check
   - ⚠️ Can waste computation checking operators for busy machines
   - Variable: `has_free_qualified_operator` (boolean, per machine)

**Logical Order in Code**: STEP 1 (capability) → STEP 3 (operator) → STEP 2 (machine free)  
**Desired Order**: STEP 1 (capability) → STEP 2 (machine free) → STEP 3 (operator)

**Information Loss**: 
- We cannot distinguish "capable but busy" from "capable but no operator" because both result in `avail[job, machine] = 0`
- Intermediate states are NOT stored in separate variables that persist beyond the loops

---

## STEP 2 – Information Available at Decision Time

### What Information Exists:

At the moment a decision is created (in `create_decision_item()`), the following data is available:

**1. Static Capability Information:**
- `machine_registry[machine_name]['capabilities']`: List of op_idx each machine can perform
- `allowed_machines`: List of machine names that can do current operation (derived from capabilities)
- `allowed_machine_indices`: Corresponding machine indices
- **Storage**: YES, stored in `decision_item['allowed_machines']` and `decision_item['allowed_machine_indices']`

**2. Machine Free Status:**
- `machine_resources[m]`: SimPy Resource for each machine
- `_resource_free(res)`: Boolean check `len(res.users) < res.capacity`
- Computed in `_build_avail_actions()` as `machine_free = [...]`
- **Storage**: NO, only computed transiently in `_build_avail_actions()`, not stored in decision_item

**3. Operator Availability:**
- `operators.operators_object_list`: List of Operator objects
- For each operator: `qualified_machines`, `is_busy`, `can_do_job(op_idx, wc_idx)`
- Checked in `_avail_row_for_job()` via loop over operators
- **Storage**: NO, only result (`has_free_qualified_operator`) determines row[i] bit, details lost

**4. Final Availability Mask:**
- `avail_row`: Per-job, per-machine availability vector (0/1 array)
- **Storage**: YES, stored in `decision_item['avail_row']`

### What Action Selection Function Sees:

When `QMIX.select_actions(obs_batch, avail_batch)` is called:

**Available**:
- `obs_batch`: Per-agent observations (feature vectors)
- `avail_batch`: Final availability masks (0/1 arrays) = `avail_row` from decision items
- Q-values from neural network forward pass

**NOT Available**:
- ❌ Which machines were capable but busy
- ❌ Which machines were capable and free but had no operator
- ❌ Which specific operators were considered
- ❌ Per-machine capability list (MachinesCanDo)
- ❌ Per-machine free status at decision time

### Summary:

**At decision time, the code knows:**
- ✅ Final mask (avail_row): which machines pass ALL checks
- ✅ Allowed machines (from capabilities, stored in decision_item)
- ❌ Machine free status (computed but not stored)
- ❌ Operator feasibility details (computed but not stored)

**The action selection function sees:**
- ✅ Only the final 0/1 mask (avail_row)
- ❌ NO intermediate capability or free information

**Information exists but is NOT kept separate:**
- Capability info: Computed in `_avail_row_for_job()`, used to filter, then discarded
- Machine free: Computed in `_build_avail_actions()`, used to filter, then discarded
- Operator checks: Computed in `_avail_row_for_job()`, boolean result used, details discarded

---

## STEP 3 – Can We Log 3-Step Reasoning As-Is?

### Question: Can we produce MachinesCanDo / MachinesFree / MachineOperatorFeasible WITHOUT changing behavior?

### Analysis:

**1. MachinesCanDo (STEP 1):**

✅ **YES, can be reconstructed easily**

**Where**: Already computed in `_avail_row_for_job()` and `create_decision_item()`

**How to extract**:
```python
# In create_decision_item (already done):
allowed_machines = []  # Machines that can do this operation
for mname in mlist:
    caps = machine_registry.get(mname, {}).get('capabilities', [])
    if op_idx_local in caps:
        allowed_machines.append(mname)

# Already stored in decision_item:
decision_item['allowed_machines'] = allowed_machines  # ← This is MachinesCanDo!
```

**Logging**: Can log directly from `decision_item['allowed_machines']` or reconstruct from `machine_registry`.

---

**2. MachinesFree (STEP 2):**

⚠️ **PARTIALLY POSSIBLE** - requires passing data through functions

**Where**: Computed in `_build_avail_actions()` but NOT stored

**Problem**: 
- `create_decision_item()` calls `env._avail_row_for_job(job)` which does NOT check machine free status
- Machine free status is only checked later in `_build_avail_actions()`
- `decision_item['avail_row']` already includes operator filtering, so we can't separate machine_free from operator_free

**Two options**:

**Option A: Recompute at logging time** (no code changes to action selection):
```python
# At logging/timeline generation time:
machine_free_now = [env._resource_free(env.machine_resources[m]) for m in range(n_machines)]
machines_can_do = decision_item['allowed_machines']
machines_free = [m for m in machines_can_do if machine_free_now[machine_index[m]]]
```

**Problem with Option A**: Timing mismatch! 
- Decision was made at time T
- Logging happens at time T + delta
- Machine free status may have changed between decision and logging
- ❌ **Cannot accurately reconstruct decision-time state**

**Option B: Store machine_free status in decision_item** (requires minor code change):
```python
# In create_decision_item or _avail_row_for_job:
machine_free_at_decision = [env._resource_free(env.machine_resources[m]) for m in range(n_machines)]
decision_item['machine_free_at_decision'] = machine_free_at_decision  # ADD THIS
```

**Impact**: Non-invasive, doesn't change behavior, only adds metadata

---

**3. MachineOperatorFeasible (STEP 3):**

❌ **NO, CANNOT be reconstructed accurately from current code**

**Where**: Computed in `_avail_row_for_job()` but details are lost

**Problem**: The code checks operators in a loop and sets a boolean flag:

```python
has_free_qualified_operator = False
for op_obj in self.operators.operators_object_list:
    if mname in op_obj.qualified_machines:
        if op_obj.can_do_job(op_idx_local, wc_idx):
            if not op_obj.is_busy:
                has_free_qualified_operator = True
                break  # ← Found one, stop searching

if has_free_qualified_operator:
    row[i] = 1
else:
    row[i] = 0  # ← This bit tells us NOTHING about why it failed
```

**Information lost**:
- ❌ Which operators were qualified but busy?
- ❌ Which operators were free but not qualified?
- ❌ Were there NO qualified operators at all, or were they all busy?

**To reconstruct**, we would need:

**Option A: Store operator eligibility details per machine** (requires code change):
```python
# In _avail_row_for_job, for each machine:
operator_details = {
    'qualified_operators': [],  # Operators qualified for this machine+operation
    'qualified_and_free': [],   # Subset that are also free
    'qualified_but_busy': [],   # Subset that are busy
}

for op_obj in self.operators.operators_object_list:
    if mname in op_obj.qualified_machines:
        if op_obj.can_do_job(op_idx_local, wc_idx):
            operator_details['qualified_operators'].append(op_obj.operator_id)
            if not op_obj.is_busy:
                operator_details['qualified_and_free'].append(op_obj.operator_id)
            else:
                operator_details['qualified_but_busy'].append(op_obj.operator_id)

# Store per-machine operator details:
decision_item['operator_details_per_machine'][machine_index] = operator_details
```

**Option B: Recompute at logging time** (same timing problem as machine_free):
- Operator busy status may have changed between decision and logging
- ❌ **Inaccurate**

---

### Can We Log Without Changing Behavior?

**Summary**:

| Step | Can Reconstruct? | Method | Accurate? |
|------|-----------------|--------|-----------|
| MachinesCanDo | ✅ YES | Use `decision_item['allowed_machines']` | ✅ YES |
| MachinesFree | ⚠️ PARTIAL | Recompute machine_free OR store in decision_item | ⚠️ NO (timing) / ✅ YES (stored) |
| MachineOperatorFeasible | ❌ NO | Would need to store operator details per machine | N/A |

**Conclusion**: 
- **MachinesCanDo**: ✅ Can log cleanly, already available
- **MachinesFree**: ⚠️ Requires storing `machine_free` status in decision_item (minor change)
- **MachineOperatorFeasible**: ❌ Requires substantial refactor to store per-machine operator details

**Current `avail_row` conflates**:
- Machine capability ✅
- Operator availability ✅
- But MISSING machine free status (checked later in different function)

---

## STEP 4 – Why Current Implementation Cannot Support Clean Logging

### Precise Information Loss Points:

**1. Operator Feasibility Loop** (environment.py, line 1620-1638):

```python
has_free_qualified_operator = False  # ← Single boolean flag
for op_obj in self.operators.operators_object_list:
    if mname in op_obj.qualified_machines:
        if op_obj.can_do_job(op_idx_local, wc_idx):
            if not op_obj.is_busy:
                has_free_qualified_operator = True
                break  # ← Early exit loses details
```

**Lost information**:
- Cannot tell if M1 was excluded because:
  - No operators are qualified for M1 at all, OR
  - All qualified operators for M1 are busy, OR
  - M1 cannot do this operation type (different issue)

**Result**: `row[i] = 0` for three different reasons that cannot be distinguished.

---

**2. Machine Free Check Happens After Operator Check** (environment.py, line 1556-1561):

```python
# In _build_avail_actions:
for m in range(n_m):
    if int(row[m]) != 1:  # ← Already filtered by capability + operator
        continue
    
    if not machine_free[m]:  # ← Now check machine free
        continue  # ← Set avail[idx, m] = 0 (stays default)
```

**Lost information**:
- Cannot tell if machine M2 was excluded because:
  - M2 is busy (checked here), OR
  - M2 has no free operator (checked earlier in `_avail_row_for_job`)

**Result**: Both result in `avail[idx, m] = 0`, indistinguishable.

---

**3. Timing Mismatch Between Decision Creation and Action Application**:

```
T0: create_decision_item() → builds decision_item with avail_row
    (Machine M1 is FREE at T0)

T1: Decision batch sent to policy

T2: Policy selects action

T3: process_and_apply_actions() resumes SimPy event
    (Machine M1 may now be BUSY at T3)

T4: Logging/timeline generation
    (If we recompute machine_free here, M1 status may have changed again)
```

**Lost information**: Decision-time state vs logging-time state mismatch.

---

### Summary of Why Clean Logging is Not Possible:

**1. Collapsed Checks**: Multiple failure reasons → single 0 bit
   - "No operator" vs "operator busy" vs "machine busy" all become `avail = 0`

**2. No Intermediate Storage**: Transient variables not saved
   - `has_free_qualified_operator` boolean, but not operator IDs or busy states
   - `machine_free` list computed but not stored in decision_item

**3. Split Logic**: 3-step reasoning split across 2 functions
   - `_avail_row_for_job()`: capability + operator (STEP 1 + 3)
   - `_build_avail_actions()`: machine free (STEP 2)
   - Makes it impossible to trace full reasoning for single job

**4. Timing Issues**: State changes between decision and logging
   - Cannot accurately reconstruct decision-time state from current-time state

**Example scenario we CANNOT log accurately**:

> "Job_0.Op7 at t=5.0: MachinesCanDo=[M1, M3], MachinesFree=[M3], MachineOperatorFeasible=[M3].  
> M1 was capable and had qualified operator O2, but O2 was busy at t=5.0."

Why? Because by the time we log (maybe t=5.5), O2 might be free again, and we have no record of O2's state at t=5.0.

---

## STEP 5 – How to Add Clean 3-Step Logging (Refactor Proposal)

### Minimal Changes to Enable Logging:

**Goal**: Store intermediate reasoning steps in `decision_item` without changing environment behavior.

### Proposed Changes:

#### **Change 1: Refactor `_avail_row_for_job()` to Return Detailed Information**

**Current signature**:
```python
def _avail_row_for_job(self, job: JobAgent) -> np.ndarray:
    # Returns: row of 0/1 values (shape: n_machines)
```

**New signature** (backward compatible):
```python
def _avail_row_for_job(self, job: JobAgent, return_details: bool = False):
    # Returns: 
    #   If return_details=False: row (np.ndarray) - LEGACY
    #   If return_details=True: (row, details_dict)
```

**New return value when `return_details=True`**:
```python
details = {
    'machines_can_do': [],  # List of machine indices that can do operation
    'machines_can_do_names': [],  # Corresponding machine names
    'operator_details': {},  # Dict: machine_idx → operator info
}

for i, mname in enumerate(mlist_local):
    caps = registry.get(mname, {}).get('capabilities', [])
    if int(op_idx_local) in caps:
        # STEP 1: Capable
        details['machines_can_do'].append(i)
        details['machines_can_do_names'].append(mname)
        
        # STEP 3: Check operators
        operator_info = {
            'qualified_operators': [],  # All operators that can work here
            'free_operators': [],       # Qualified AND free
            'busy_operators': [],       # Qualified BUT busy
        }
        
        for op_obj in self.operators.operators_object_list:
            if mname in op_obj.qualified_machines:
                if op_obj.can_do_job(op_idx_local, wc_idx):
                    operator_info['qualified_operators'].append(op_obj.operator_id)
                    if not op_obj.is_busy:
                        operator_info['free_operators'].append(op_obj.operator_id)
                    else:
                        operator_info['busy_operators'].append(op_obj.operator_id)
        
        details['operator_details'][i] = operator_info
        
        # Set row[i] based on availability
        if operator_info['free_operators']:
            row[i] = 1
        else:
            row[i] = 0

return (row, details) if return_details else row
```

**Impact**: 
- ✅ Backward compatible (default `return_details=False` preserves existing behavior)
- ✅ Provides detailed reasoning when needed

---

#### **Change 2: Store Machine Free Status in Decision Item**

**Location**: `create_decision_item()` in utils/workcenter.py

**Add**:
```python
def create_decision_item(self, env: Any, job: Any, op: Any) -> Dict:
    # ... existing code ...
    
    # Compute machine free status at decision time
    machine_free_at_decision = {}
    n_machines = len(self.machine_list)
    for i, mname in enumerate(self.machine_list):
        try:
            machine_res = env.machine_resources[i]
            machine_free_at_decision[mname] = env._resource_free(machine_res)
        except Exception as e:
            machine_free_at_decision[mname] = None  # Unknown
    
    # Get detailed availability info
    avail_row, avail_details = env._avail_row_for_job(job, return_details=True)
    
    decision_item = {
        "job_id": getattr(job, 'id', getattr(job, 'agent_id', None)),
        "obs": env._build_agent_obs(job),
        "avail_row": avail_row,  # Final mask (for policy)
        "allowed_machines": list(allowed_machines),
        "allowed_machine_indices": list(dict.fromkeys(allowed_machine_indices)),
        
        # NEW: Store reasoning details
        "machines_can_do": avail_details['machines_can_do_names'],  # STEP 1
        "machine_free_at_decision": machine_free_at_decision,       # STEP 2
        "operator_details_per_machine": avail_details['operator_details'],  # STEP 3
        
        # ... rest of existing fields ...
    }
    
    return decision_item
```

**Impact**:
- ✅ No behavior change (decision_item has extra fields that are ignored by policy)
- ✅ All information available for logging

---

#### **Change 3: Update `_build_avail_actions()` to Store Machine Free Info**

**Current**: `_build_avail_actions()` computes `machine_free` transiently

**Option A**: Don't change (use stored info from decision_item instead)

**Option B**: Add method to query machine free status:
```python
def get_machine_free_status(self):
    """Return current machine free status for all machines."""
    n_m = len(self.workcenters_meta.machine_list)
    machine_free = {}
    for i, mname in enumerate(self.workcenters_meta.machine_list):
        machine_free[mname] = self._resource_free(self.machine_resources[i])
    return machine_free
```

**Recommendation**: Option A - use decision_item stored info for accurate decision-time state.

---

### Where to Add Logging:

**Location 1**: Timeline generation (utils/gantt.py)

When building timeline text, use decision_item fields:

```python
def format_decision_reasoning(decision_item, chosen_machine_name):
    """Generate human-readable 3-step reasoning explanation."""
    
    # STEP 1: MachinesCanDo
    machines_can_do = decision_item.get('machines_can_do', [])
    
    # STEP 2: MachinesFree (filter step 1 by free status)
    machine_free_map = decision_item.get('machine_free_at_decision', {})
    machines_free = [m for m in machines_can_do if machine_free_map.get(m, False)]
    
    # STEP 3: MachineOperatorFeasible (filter step 2 by operator availability)
    operator_details = decision_item.get('operator_details_per_machine', {})
    machine_index = {m: i for i, m in enumerate(decision_item['allowed_machines'])}
    
    machines_with_operator = []
    for m in machines_free:
        m_idx = machine_index.get(m)
        if m_idx is not None:
            op_info = operator_details.get(m_idx, {})
            if op_info.get('free_operators'):
                machines_with_operator.append(m)
    
    # Build explanation
    explanation = []
    explanation.append(f"MachinesCanDo: {machines_can_do}")
    explanation.append(f"MachinesFree: {machines_free}")
    explanation.append(f"MachineOperatorFeasible: {machines_with_operator}")
    
    # Per-machine reasoning
    for m in machines_can_do:
        if m == chosen_machine_name:
            explanation.append(f"  → {m}: SELECTED")
        elif m not in machines_free:
            explanation.append(f"  → {m}: capable but BUSY")
        else:
            m_idx = machine_index.get(m)
            op_info = operator_details.get(m_idx, {})
            if not op_info.get('free_operators'):
                if op_info.get('qualified_operators'):
                    busy_ops = ', '.join(op_info['busy_operators'])
                    explanation.append(f"  → {m}: free but no available operator (qualified: {busy_ops} - all busy)")
                else:
                    explanation.append(f"  → {m}: free but NO qualified operators")
            else:
                explanation.append(f"  → {m}: feasible but not chosen")
    
    return '\n'.join(explanation)
```

**Location 2**: Scheduling trace CSV (MARL/common/rollout.py)

Add columns to scheduling_trace.csv:

```csv
time,job_id,machines_can_do,machines_free,machines_operator_feasible,chosen_machine,reason
5.0,Job_0,"['M1','M3']","['M3']","['M3']",M3,"M1:busy; M3:selected"
```

---

### Implementation Steps:

1. **Modify `_avail_row_for_job()`**: Add `return_details` parameter and collect operator info
2. **Modify `create_decision_item()`**: Call `_avail_row_for_job(return_details=True)` and store machine_free status
3. **Add logging helper**: `format_decision_reasoning()` in utils/gantt.py
4. **Update timeline generation**: Call logging helper and include in timeline output
5. **Update scheduling_trace.csv**: Add new columns with 3-step reasoning

**Behavior preserved**: 
- `avail_row` computation unchanged (same 0/1 values)
- Policy sees same masks
- Only difference: extra metadata stored and logged

---

## STEP 6 – Final Answers

### 1. "Does my current action selection implementation naturally support a 3-step reasoning flow?"

**Answer: NO**

**Explanation**:
The current implementation does NOT naturally support the 3-step flow because:

1. **Logic is split across two functions**:
   - `_avail_row_for_job()`: Checks capability (STEP 1) and operator availability (STEP 3)
   - `_build_avail_actions()`: Checks machine free status (STEP 2)

2. **Steps execute in wrong order**:
   - Current: Capability → Operator → Machine Free
   - Desired: Capability → Machine Free → Operator
   - This wastes computation checking operators for machines that are busy

3. **No intermediate storage**:
   - Each check produces a boolean result that either sets a mask bit or not
   - Details (which operators were busy, which machines were free but no operator, etc.) are lost
   - Cannot reconstruct reasoning after the fact

4. **Single collapsed output**:
   - All checks result in a single 0/1 bit in the mask
   - `avail[job, machine] = 0` could mean 4+ different things:
     - Not capable
     - Capable but busy
     - Capable and free but no qualified operators exist
     - Capable and free but all qualified operators are busy

---

### 2. "If yes, can I log it cleanly without changing behavior? If no, what minimal refactor is needed?"

**Answer: NO (cannot log cleanly without changes), but minimal refactor is straightforward**

**What's needed**:

#### **Minimal Refactor (Recommended)**:

1. **Extend `_avail_row_for_job()` signature** to optionally return details:
   ```python
   def _avail_row_for_job(self, job, return_details=False):
       # Returns (row, details) when return_details=True
       # Returns row when return_details=False (backward compatible)
   ```

2. **Store machine free status** in decision_item:
   ```python
   decision_item['machine_free_at_decision'] = {
       'M0': True, 'M1': False, 'M2': True, ...
   }
   ```

3. **Store operator details** per machine in decision_item:
   ```python
   decision_item['operator_details_per_machine'] = {
       0: {'qualified': ['O1'], 'free': ['O1'], 'busy': []},
       1: {'qualified': ['O2'], 'free': [], 'busy': ['O2']},
       ...
   }
   ```

4. **Add logging helper** that reads these fields and formats 3-step reasoning

**Behavior preserved**: 
- ✅ Mask values unchanged
- ✅ Action selection unchanged
- ✅ Only adds metadata (ignored by policy)

**Cost**: 
- ~50 lines of code in `_avail_row_for_job()`
- ~20 lines in `create_decision_item()`
- ~100 lines for logging helper
- Negligible performance impact (same checks already done, just storing results)

---

### 3. "Does action selection have access to intermediate information I care about, or is it lost?"

**Answer: INFORMATION IS LOST**

**What exists at decision time**:
- ✅ Machine capabilities (from machine_registry)
- ✅ Machine free status (from machine_resources)
- ✅ Operator qualifications (from operators.operators_object_list)
- ✅ Operator busy status (from operator.is_busy)

**What reaches the policy**:
- ✅ Final mask only (`avail_row`: 0/1 array)
- ❌ NOT: MachinesCanDo list
- ❌ NOT: MachinesFree list
- ❌ NOT: Operator details (which were qualified, which were busy)

**What is lost in between**:
- **Operator loop details** (line 1623-1636 in environment.py):
  - Loop finds first free qualified operator and breaks
  - If found: `row[i] = 1`, else: `row[i] = 0`
  - ❌ **LOST**: Which operators were considered, which were qualified, which were busy

- **Machine free filtering** (line 1556-1561 in environment.py):
  - Computed in `_build_avail_actions()` but not stored
  - ❌ **LOST**: Decision-time machine busy status (only current-time status available later)

**Consequence**:
- Cannot distinguish failure modes without refactor
- Timeline logging shows "eligible=['M1']" but cannot explain why M3 was chosen
- "Reason: M1 no qualified operator" might be wrong (maybe M1 was busy, not operator issue)

---

## Summary Table

| Question | Answer | Details |
|----------|--------|---------|
| **Does code support 3-step reasoning?** | ❌ NO | Logic split across functions, wrong order, no intermediate storage |
| **Can we log it without changes?** | ❌ NO | Information lost before logging, timing mismatches |
| **Is intermediate info available?** | ⚠️ PARTIALLY | Exists during computation but NOT stored or passed to policy |
| **What's needed for clean logging?** | Refactor | Store details in decision_item: machines_can_do, machine_free, operator_details |
| **Behavior change required?** | ❌ NO | Only add metadata, mask values unchanged |
| **Estimated effort** | ~200 lines | Extend _avail_row_for_job, modify create_decision_item, add logging helper |

---

## Recommendation

**Implement the minimal refactor proposed in STEP 5**:

1. Extend `_avail_row_for_job()` to return detailed operator information
2. Store machine free status in decision_item at creation time
3. Add logging helper to format 3-step reasoning from stored details
4. Update timeline generation to use new helper

This will enable clean, accurate logging of:
```
[t=5.0] Job_0.Op7 decision:
  MachinesCanDo: [M1, M3]
  MachinesFree: [M3]  (M1 busy)
  MachineOperatorFeasible: [M3]  (M1: no free operator - O2 qualified but busy)
  → Selected: M3 with O1
```

**Benefits**:
- ✅ Complete reasoning transparency
- ✅ No behavior changes (same masks, same actions)
- ✅ Accurate decision-time state (not reconstructed from stale data)
- ✅ Enables debugging of eligibility issues
- ✅ Confirms fixes from ANALYSIS_MACHINE_OPERATOR_ELIGIBILITY.md

---

**End of Action Selection Reasoning Audit**
