# Workspace-Wide Consistency Audit – Machine / Operation / Operator Eligibility

**Date**: January 2025  
**Branch**: make-it-work-and-converge-v2  
**Analysis Focus**: Machine capabilities, operator qualifications, action masking, and timeline logging consistency

---

## STEP 0 – Key Modules and Entry Points

### Core Modules Located:

**1. utils/workcenter.py**
- `DEFAULT_WORKCENTERS`: Defines machines, capabilities, workcenters
- `DEFAULT_PROCESSING_TIMES`: Processing time definitions per machine-operation pair
- `WorkCenters.__init__()`: Builds machine_registry with capabilities
- `WorkCenters.create_decision_item()`: Creates decision items with allowed_machines and per_machine_durations
- `WorkCenter.__init__()`: Constructs individual workcenter with machine metadata

**2. utils/operator.py**
- `DEFAULT_OPERATORS`: Defines operator qualifications (which machines each can work on)
- `Operator.can_do_job(op_idx, workcenter_id)`: Checks if operator can perform operation at workcenter
- `Operators.__init__()`: Creates operator objects with qualified_machines and qualified_workcenters
- `Operators.find_free_operator(op_idx, workcenter_id)`: WorkCenter-based lookup
- `Operators.find_free_operator_for_machine(op_idx, machine_name)`: Machine-based lookup
- `Operators.find_free_operator_seeded_random()`: Deterministic operator selection
- `Operators.find_free_operator_for_machine_seeded_random()`: Deterministic machine-specific selection

**3. environment.py**
- `_build_avail_actions()` (line 1518): Builds availability matrix for all jobs × machines
- `_avail_row_for_job(job)` (line 1598): Builds availability row for single job
- Uses `machine_registry['capabilities']` to check machine-operation compatibility
- Checks operator availability via `Operators.can_do_job()` and `qualified_machines`
- Fallback logic at line 1648-1654 when no machines marked

**4. utils/gantt.py**
- `generate_scheduling_timeline()` (line 180): Produces timeline output
- Line 326-342: Builds `jobs_allowed` mapping from job operations
- Line 390-394: Extracts `eligible_idxs` and `eligible_names` for logging
- Line 406: Formats timeline start text with eligible machines
- Line 410-470: Generates "Reason: selected..." explanations
- Uses `decision_trace` if available, otherwise reconstructs from records

**5. MARL/common/rollout.py**
- Extracts `avail_row` from decision items
- Validates action selection against availability masks
- Line 356: Calls `gantt_utils.append_selection_log()` with allowed_machine_indices and avail_actions

---

## STEP 1 – Canonical Sources of Truth

### 1.1 Machine Capabilities

**Location**: `utils/workcenter.py` → `DEFAULT_WORKCENTERS["machines"]`

```python
"machines": {
    "M0": {"wc": "WC1", "capable_ops": ["Op1", "Op2", "Op3", "Op5", "Op9"]},
    "M1": {"wc": "WC1", "capable_ops": ["Op4", "Op5", "Op8"]},
    "M2": {"wc": "WC2", "capable_ops": ["Op1", "Op2", "Op4", "Op6", "Op9"]},
    "M3": {"wc": "WC2", "capable_ops": ["Op1", "Op3", "Op7", "Op8"]},
    "M4": {"wc": "WC3", "capable_ops": ["Op1", "Op3", "Op4", "Op6", "Op8", "Op9"]},
}
```

**Internal representation**: Op names converted to 0-based indices:
- Op1 → 0, Op2 → 1, Op3 → 2, Op4 → 3, Op5 → 4, Op6 → 5, Op7 → 6, Op8 → 7, Op9 → 8

**Machine-Operation Matrix** (derived):

| Machine | WorkCenter | Capable Operations (0-indexed) |
|---------|------------|-------------------------------|
| M0      | WC1 (0)    | [0, 1, 2, 4, 8] (Op1, Op2, Op3, Op5, Op9) |
| M1      | WC1 (0)    | [3, 4, 7] (Op4, Op5, Op8) |
| M2      | WC2 (1)    | [0, 1, 3, 5, 8] (Op1, Op2, Op4, Op6, Op9) |
| M3      | WC2 (1)    | [0, 2, 6, 7] (Op1, Op3, Op7, Op8) |
| M4      | WC3 (2)    | [0, 2, 3, 5, 7, 8] (Op1, Op3, Op4, Op6, Op8, Op9) |

### 1.2 Processing Times

**Location**: `utils/workcenter.py` → `DEFAULT_PROCESSING_TIMES`

Operations defined per machine:

| Machine | Operations Defined |
|---------|-------------------|
| M0      | Op1, Op2, Op3, Op4, Op5, Op8, Op9 (7 ops) |
| M1      | Op4, Op5, Op8 (3 ops) |
| M2      | Op1, Op2, Op3, Op4, Op6, Op7, Op8, Op9 (8 ops) |
| M3      | Op1, Op3, Op7, Op8 (4 ops) |
| M4      | Op1, Op3, Op4, Op6, Op8, Op9 (6 ops) |

### 1.3 Operator Qualifications

**Location**: `utils/operator.py` → `DEFAULT_OPERATORS`

```python
DEFAULT_OPERATORS = [
    {"id": "O1", "qualified_machines": ["M0", "M3", "M4"]},
    {"id": "O2", "qualified_machines": ["M1", "M2", "M4"]},
]
```

**Operator-Machine Matrix**:

| Operator | Qualified Machines | Derived qualified_workcenters |
|----------|-------------------|------------------------------|
| O1       | M0, M3, M4        | [0, 1, 2] (WC1, WC2, WC3)    |
| O2       | M1, M2, M4        | [0, 1, 2] (WC1, WC2, WC3)    |

### 1.4 WorkCenter Configuration

**Location**: `utils/workcenter.py` → `WorkCenters.__init__()`

```python
default_map = {
    0: ["M0", "M1"],  # WorkCenter 0 (WC1)
    1: ["M2", "M3"],  # WorkCenter 1 (WC2)
    2: ["M4"],        # WorkCenter 2 (WC3)
}

eligible_operator_groups_by_wc = {
    0: [0],  # WC1 → operator group 0
    1: [1],  # WC2 → operator group 1
    2: [2],  # WC3 → operator group 2
}
```

### 1.5 Intended Single Source of Truth

**Question**: "Can machine M do operation op_idx?"
- **Answer**: `machine_registry[machine_name]['capabilities']` must contain `op_idx`
- **Source**: `DEFAULT_WORKCENTERS["machines"][machine_name]["capable_ops"]`

**Question**: "Can operator O work on machine M for operation op_idx?"
- **Answer**: 
  1. `machine_name in operator.qualified_machines` (operator qualified for machine)
  2. `op_idx in machine_registry[machine_name]['capabilities']` (machine can do operation)
- **Source**: `DEFAULT_OPERATORS[operator]["qualified_machines"]` + machine capabilities

**Question**: "Which WorkCenter does machine M belong to?"
- **Answer**: `machine_registry[machine_name]['workcenter']`
- **Source**: `DEFAULT_WORKCENTERS["machines"][machine_name]["wc"]`

---

## STEP 2 – Hard Inconsistencies in Static Data

### ⚠️ **CRITICAL MISMATCH FOUND: Op4 on M0**

**Issue**: `DEFAULT_PROCESSING_TIMES["M0"]` defines **Op4: 1.575**, but `DEFAULT_WORKCENTERS["machines"]["M0"]["capable_ops"]` does **NOT** include Op4.

**Impact**: 
- If job generation uses `DEFAULT_PROCESSING_TIMES`, it may include Op4 as executable on M0
- But `_avail_row_for_job()` will mark M0 as unavailable (capabilities check fails)
- Result: Invalid action space or missing valid actions

**Fix Options**:
1. **Add Op4 to M0 capabilities**: `capable_ops": ["Op1", "Op2", "Op3", "Op4", "Op5", "Op9"]`
2. **Remove Op4 from M0 processing times**: Delete `"Op4": 1.575` entry

**Recommendation**: **Option 2** – Remove Op4 from `DEFAULT_PROCESSING_TIMES["M0"]` to match current capability definition. This preserves the existing capability matrix design.

### ⚠️ **MINOR MISMATCH: Op3 on M2**

**Issue**: `DEFAULT_PROCESSING_TIMES["M2"]` defines **Op3: 1.575**, but `DEFAULT_WORKCENTERS["machines"]["M2"]["capable_ops"]` does **NOT** include Op3.

**Impact**: Same as above.

**Fix**: **Remove Op3 from `DEFAULT_PROCESSING_TIMES["M2"]`**

### ⚠️ **MINOR MISMATCH: Op7 on M2**

**Issue**: `DEFAULT_PROCESSING_TIMES["M2"]` defines **Op7: 2.10**, but `DEFAULT_WORKCENTERS["machines"]["M2"]["capable_ops"]` does **NOT** include Op7.

**Impact**: Same as above.

**Fix**: **Remove Op7 from `DEFAULT_PROCESSING_TIMES["M2"]`**

### ✅ **Consistent Entries** (Selected Examples):

- M3 + Op7: ✅ In both capable_ops and processing times
- M1 + Op4, Op5, Op8: ✅ All consistent
- M4 + Op1, Op3, Op4, Op6, Op8, Op9: ✅ All consistent

### Full Consistency Matrix:

| Machine | In capable_ops but NOT in processing_times | In processing_times but NOT in capable_ops |
|---------|-------------------------------------------|-------------------------------------------|
| M0      | Op9 | **Op4** ❌ |
| M1      | (none) | (none) ✅ |
| M2      | Op9 | **Op3** ❌, **Op7** ❌ |
| M3      | (none) | (none) ✅ |
| M4      | Op9 | (none) ✅ |

**Note**: Op9 missing from processing times on M0, M2, M4 is acceptable if operations are generated differently, but Op4 on M0, Op3 and Op7 on M2 create direct contradictions.

---

## STEP 3 – Eligibility Logic End-to-End Trace

### 3.1 Allowed Machines Construction

**Function**: `WorkCenters.create_decision_item()` (workcenter.py, line 243-344)

**Logic**:
1. Extract `op_idx_local` from operation tuple
2. Iterate through `machine_list`
3. For each machine, check: `if op_idx_local in machine_registry[mname]['capabilities']`
4. If true, add to `allowed_machines` and `allowed_machine_indices`
5. Lookup duration from `DEFAULT_PROCESSING_TIMES[machine_name][op_name]`

**✅ Correct**: Uses `machine_registry['capabilities']` as source of truth.

### 3.2 Action Mask Construction

**Function**: `environment._avail_row_for_job(job)` (environment.py, line 1598-1658)

**Logic**:
1. Extract `op_idx_local` from job's current operation
2. Iterate through `machine_list`
3. For each machine:
   - Check: `if op_idx_local in machine_registry[mname]['capabilities']` ✅
   - Check: machine has free qualified operator
     - Operator must be in `qualified_machines` for this machine ✅
     - Operator must pass `can_do_job(op_idx_local, wc_idx)` ✅
     - Operator must be free (not busy) ✅
   - If all pass, set `row[i] = 1`

4. **⚠️ FALLBACK** (line 1648-1654): If no machines marked:
   ```python
   if not row.any():
       # Use operation tuple's allowed_machine_indices
       for idx in allowed_machine_indices:
           row[int(idx)] = 1
   ```

**Problem with fallback**:
- If `machine_registry` is empty or malformed, fallback uses `allowed_machine_indices` from operation tuple
- This bypasses operator availability checks!
- Result: Action mask includes machines without qualified operators → invalid actions selected

**Function**: `environment._build_avail_actions()` (environment.py, line 1518-1598)

**Logic**:
1. For each job, call `_avail_row_for_job(job)` ✅
2. For each machine in row:
   - Check machine is free ✅
   - Check operator availability (via `eligible_operator_groups_by_wc`) ⚠️

**⚠️ Issue**: Line 1553-1574 uses **WorkCenter-based** operator eligibility (`eligible_operator_groups_by_wc`), but should use **machine-based** eligibility (`qualified_machines`).

**Current code**:
```python
eligible_groups = eligible_map.get(int(wc_i), []) if wc_i is not None else []
# Then checks if any operator in eligible_groups is free
```

**Problem**: 
- `eligible_operator_groups_by_wc = {0: [0], 1: [1], 2: [2]}` maps WC → operator groups
- But both O1 and O2 can work in all workcenters (qualified_workcenters = [0,1,2])
- This creates mismatch between WorkCenter-level and machine-level eligibility

**Correct logic should be** (already in `_avail_row_for_job`):
```python
if mname in operator.qualified_machines and operator.can_do_job(op_idx, wc_idx) and not operator.is_busy:
    available = True
```

### 3.3 Operator Selection Functions

**Machine-based** (✅ Correct):
- `find_free_operator_for_machine(op_idx, machine_name)`: 
  - Checks `machine_name in operator.qualified_machines`
  - Checks `op_idx in machine_registry[machine_name]['capabilities']`
  - Uses machine-level eligibility

**WorkCenter-based** (⚠️ Potentially incorrect):
- `find_free_operator(op_idx, workcenter_id)`:
  - Uses `operator.can_do_job(op_idx, workcenter_id)`
  - Internally checks machines in workcenter
  - Can return operator qualified for ANY machine in workcenter, not necessarily the chosen machine

**Recommendation**: Environment should use **machine-based** operator selection (`find_free_operator_for_machine`) rather than WorkCenter-based to ensure operator is qualified for the specific chosen machine.

---

## STEP 4 – Logging / Timeline Logic Analysis

### 4.1 Eligible Machines List Construction

**Location**: `utils/gantt.py`, line 326-342 and 390-391

**Process**:
1. Build `jobs_allowed` dictionary from job definitions:
   ```python
   for idx, op in enumerate(ops):
       allowed_machine_indices = list(op[1])  # From operation tuple
       opname = f"Op{...}"
       amap[opname] = list(allowed_machine_indices)
   jobs_allowed[jid] = amap
   ```

2. During timeline generation:
   ```python
   eligible_idxs = jobs_allowed.get(int(job_id), {}).get(op_name, [])
   eligible_names = [mlist[i] for i in eligible_idxs]
   ```

**✅ Correct**: Uses operation tuple's `allowed_machine_indices` from job definition.

**⚠️ Limitation**: This is **static capability** from job generation, not **dynamic availability** at decision time.

### 4.2 "Reason: selected..." Generation

**Location**: `utils/gantt.py`, line 410-490

**Logic**:
1. **If `decision_trace` exists** (line 410-470):
   - Extract `chosen_machine`, `chosen_operator`, `eligibilities` from trace
   - Build "Reason: selected M3 & O1 (selected)" ✅
   - For each eligibility entry:
     - Check `machine_busy`, `operator_available`, `qualified`
     - Generate: "M0 no qualified operator", "M1 free / O2 free but not chosen", etc.

2. **If no `decision_trace`** (line 471-550):
   - Reconstructs from timeline records
   - Checks if machines/operators were busy at start time
   - **⚠️ May not reflect actual decision-time eligibility checks**

### 4.3 Known Logging Bugs

**Bug #1**: "eligible=['M1']" when chosen="M3"
- **Root cause**: `eligible_names` shows static job definition (Op7 defined for M1)
- **Chosen machine**: M3 (from actual execution)
- **Why mismatch**: M3's capability may have been added after job generation, or fallback logic used

**Bug #2**: "no qualified operator" when operator exists
- **Example**: Timeline shows "M1 no qualified operator" but O2 is qualified for M1
- **Root cause**: `decision_trace['eligibilities']` may check WorkCenter-level eligibility instead of machine-level
- Or: Operator was busy at decision time but logging checks post-execution state

**Bug #3**: Timeline shows machine started but was not in eligible list
- **Example**: Op7 eligible=[M1] but started on M3
- **Root cause**: Fallback in `_avail_row_for_job` bypassed capability check

---

## STEP 5 – Fallback / Default Behaviors Review

### 5.1 `_avail_row_for_job()` Fallback (environment.py, line 1648-1654)

**Code**:
```python
if not row.any():
    # Fallback to operation tuple's allowed_machine_indices
    for idx in allowed_machine_indices:
        if 0 <= int(idx) < row.shape[0]:
            row[int(idx)] = 1
```

**When triggered**: 
- `machine_registry` is empty
- No machines have required capability
- Operator checks failed for all machines

**Problem**: 
- Bypasses operator availability checks
- May mark machines as available when no qualified operator exists
- Causes invalid action selection

**Recommendation**: 
- **Remove fallback** and let `row` remain all zeros if no valid machines
- Add explicit error/warning when this occurs
- Or: Keep fallback but log warning to detect misconfiguration

### 5.2 `Operator.can_do_job()` Fallback (operator.py, line 72-109)

**Code**:
```python
try:
    # Primary: use machine_registry
    ...
except Exception as e:
    # Fallback: use workcenters_list
    try:
        wc_obj = self.workcenters_ref.workcenters_list[workcenter_id]
        ...
    except Exception as e:
        logging.getLogger(__name__).warning(...)
        return False
```

**When triggered**: `machine_registry` access fails

**Problem**: 
- Fallback to old `workcenters_list` structure may use stale data
- Silent failure returns `False`, hiding configuration errors

**Recommendation**:
- **Remove fallback** and require `machine_registry` to be valid
- Or: Add strict mode that raises exception instead of returning False

### 5.3 `_build_avail_actions()` Default Behaviors (environment.py, line 1525-1541)

**Code**:
```python
machine_free = [True] * n_m  # Default to all free
if getattr(self, 'machine_resources', None):
    machine_free = [actual check]
else:
    # Fallback: assume True
    machine_free = [True] * n_m
```

**Problem**: If resource system not initialized, assumes all machines free

**Recommendation**: 
- Fail fast if `machine_resources` is None
- Don't default to permissive behavior

### 5.4 `eligible_operator_groups_by_wc` Usage

**Location**: Multiple places (environment.py line 1553, 1574)

**Purpose**: Maps WorkCenter → eligible operator groups

**Current values**: `{0: [0], 1: [1], 2: [2]}`

**Problem**: 
- Both O1 and O2 have `qualified_workcenters = [0, 1, 2]` (can work in all WCs)
- But `eligible_operator_groups_by_wc` restricts to one group per WC
- Creates artificial limitation

**Recommendation**: 
- **Deprecate `eligible_operator_groups_by_wc`**
- Use direct machine-level checks: `machine_name in operator.qualified_machines`
- Remove WorkCenter-level operator filtering

---

## STEP 6 – Automated Consistency Check Script

```python
#!/usr/bin/env python3
"""
consistency_check.py - Validate machine/operator/operation configurations
"""
import sys
from utils.workcenter import DEFAULT_WORKCENTERS, DEFAULT_PROCESSING_TIMES, WorkCenters
from utils.operator import DEFAULT_OPERATORS, Operators

def check_capability_processing_time_consistency():
    """Check that capable_ops matches DEFAULT_PROCESSING_TIMES keys."""
    print("=" * 60)
    print("CONSISTENCY CHECK: Capabilities vs Processing Times")
    print("=" * 60)
    
    errors = []
    machines_config = DEFAULT_WORKCENTERS.get("machines", {})
    
    for machine_name, machine_meta in machines_config.items():
        capable_ops_names = machine_meta.get("capable_ops", [])
        processing_times = DEFAULT_PROCESSING_TIMES.get(machine_name, {})
        
        # Convert capable_ops to set
        capable_set = set(capable_ops_names)
        processing_set = set(processing_times.keys())
        
        # Find mismatches
        in_capable_not_processing = capable_set - processing_set
        in_processing_not_capable = processing_set - capable_set
        
        if in_processing_not_capable:
            errors.append({
                "machine": machine_name,
                "type": "processing_times_extra",
                "operations": list(in_processing_not_capable)
            })
            print(f"❌ {machine_name}: Operations in processing_times but NOT in capable_ops:")
            print(f"   {list(in_processing_not_capable)}")
        
        if in_capable_not_processing:
            print(f"⚠️  {machine_name}: Operations in capable_ops but NOT in processing_times:")
            print(f"   {list(in_capable_not_processing)}")
    
    if not errors:
        print("✅ All machines have consistent capabilities and processing times")
    else:
        print(f"\n❌ Found {len(errors)} critical mismatches")
    
    return errors

def check_operator_machine_validity():
    """Verify operators reference valid machines."""
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECK: Operator Qualifications")
    print("=" * 60)
    
    workcenters = WorkCenters()
    machine_names = set(workcenters.machine_registry.keys())
    
    errors = []
    for op_config in DEFAULT_OPERATORS:
        op_id = op_config["id"]
        qualified = op_config["qualified_machines"]
        
        invalid_machines = [m for m in qualified if m not in machine_names]
        if invalid_machines:
            errors.append({
                "operator": op_id,
                "invalid_machines": invalid_machines
            })
            print(f"❌ {op_id}: References non-existent machines: {invalid_machines}")
        else:
            print(f"✅ {op_id}: All qualified machines valid {qualified}")
    
    return errors

def check_operation_executability():
    """For each operation type, verify at least one machine can execute it."""
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECK: Operation Executability")
    print("=" * 60)
    
    workcenters = WorkCenters()
    operations_map = workcenters.operations_map
    
    # Check operations 0-8 (Op1-Op9)
    for op_idx in range(9):
        op_name = f"Op{op_idx + 1}"
        machines = operations_map.get(op_idx, [])
        
        if not machines:
            print(f"❌ {op_name} (index {op_idx}): NO machines can execute this operation!")
        else:
            print(f"✅ {op_name} (index {op_idx}): Executable on {machines}")
    
    return operations_map

def check_machine_operator_coverage():
    """Verify each machine has at least one qualified operator."""
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECK: Machine-Operator Coverage")
    print("=" * 60)
    
    workcenters = WorkCenters()
    operators = Operators(workcenters)
    
    errors = []
    for machine_name in workcenters.machine_list:
        qualified_ops = [
            op.operator_id 
            for op in operators.operators_object_list 
            if machine_name in op.qualified_machines
        ]
        
        if not qualified_ops:
            errors.append({"machine": machine_name})
            print(f"❌ {machine_name}: NO operators qualified!")
        else:
            print(f"✅ {machine_name}: Qualified operators: {qualified_ops}")
    
    return errors

def main():
    print("\n" + "=" * 60)
    print("WORKSPACE CONSISTENCY VALIDATION")
    print("=" * 60 + "\n")
    
    errors_cap = check_capability_processing_time_consistency()
    errors_op = check_operator_machine_validity()
    ops_map = check_operation_executability()
    errors_cov = check_machine_operator_coverage()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_errors = len(errors_cap) + len(errors_op) + len(errors_cov)
    
    if total_errors == 0:
        print("✅ ALL CHECKS PASSED")
        return 0
    else:
        print(f"❌ {total_errors} ERRORS FOUND")
        print("\nRequired fixes:")
        for err in errors_cap:
            print(f"  - Remove {err['operations']} from DEFAULT_PROCESSING_TIMES['{err['machine']}']")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

**Usage**:
```bash
python consistency_check.py
```

---

## STEP 7 – Executive Summary and Recommendations

### Critical Issues Found:

1. **❌ Data Mismatch**: Op4 defined in `DEFAULT_PROCESSING_TIMES["M0"]` but NOT in capabilities
2. **❌ Data Mismatch**: Op3 defined in `DEFAULT_PROCESSING_TIMES["M2"]` but NOT in capabilities
3. **❌ Data Mismatch**: Op7 defined in `DEFAULT_PROCESSING_TIMES["M2"]` but NOT in capabilities
4. **⚠️ Logic Bug**: `_avail_row_for_job()` fallback bypasses operator checks
5. **⚠️ Architecture Issue**: Mixed use of WorkCenter-level and machine-level operator eligibility
6. **⚠️ Logging Bug**: Timeline eligible list shows job definition, not actual decision-time availability
7. **⚠️ Logging Bug**: "no qualified operator" messages may be incorrect due to WorkCenter checks

### Recommended Fixes (Priority Order):

#### **PRIORITY 1: Fix Data Mismatches**

**File**: `utils/workcenter.py`

**Change 1**: Remove Op4 from M0 processing times (line ~53-60)
```python
"M0": {
    "Op1": 1.225,
    "Op2": 1.05,
    "Op3": 1.575,
    # REMOVE: "Op4": 1.575,  # ❌ M0 cannot do Op4
    "Op5": 2.275,
    "Op8": 1.575,
    "Op9": 2.975,
},
```

**Change 2**: Remove Op3 and Op7 from M2 processing times (line ~68-77)
```python
"M2": {
    "Op1": 1.575,
    "Op2": 1.75,
    # REMOVE: "Op3": 1.575,  # ❌ M2 cannot do Op3
    "Op4": 1.68,
    "Op6": 2.1,
    # REMOVE: "Op7": 2.10,  # ❌ M2 cannot do Op7
    "Op8": 2.625,
    "Op9": 3.15,
},
```

#### **PRIORITY 2: Remove Unsafe Fallback**

**File**: `environment.py`, line 1648-1654

**Current code**:
```python
if not row.any():
    for idx in allowed_machine_indices:
        row[int(idx)] = 1
```

**Recommended fix**:
```python
if not row.any():
    # Log warning instead of silently using fallback
    logging.getLogger(__name__).warning(
        f"[ELIGIBILITY] No machines available for job {job.id} op_idx {op_idx_local}. "
        f"Check machine_registry capabilities and operator availability."
    )
    # Keep row as all zeros - agent must handle invalid state
    # Or: raise exception in strict mode
```

#### **PRIORITY 3: Unify Operator Eligibility Logic**

**File**: `environment.py`, line 1553-1590

**Current code** (uses WorkCenter-level `eligible_operator_groups_by_wc`):
```python
eligible_groups = eligible_map.get(int(wc_i), [])
if operator_free is None or not eligible_groups:
    avail[idx, m] = 1
```

**Recommended fix** (use machine-level `qualified_machines` already in `_avail_row_for_job`):
```python
# _avail_row_for_job already does the right checks:
# - machine_name in operator.qualified_machines
# - operator.can_do_job(op_idx, wc_idx)
# - not operator.is_busy

# In _build_avail_actions, simply use the row from _avail_row_for_job
# and only apply machine_free check:
for m in range(n_m):
    if int(row[m]) != 1:
        continue
    if not machine_free[m]:
        continue
    avail[idx, m] = 1  # Row already validated operators
```

**Simplification**: Remove redundant operator checks from `_build_avail_actions` since `_avail_row_for_job` already validates operators.

#### **PRIORITY 4: Fix Timeline Logging**

**File**: `utils/gantt.py`, line 390-391

**Current** (shows static job definition):
```python
eligible_idxs = jobs_allowed.get(int(job_id), {}).get(op_name, [])
eligible_names = [mlist[i] for i in eligible_idxs]
```

**Recommended** (use decision-time data):
```python
# If decision_trace exists, use its allowed_machines
if decision_trace and 'allowed_machines' in decision_trace:
    eligible_names = decision_trace['allowed_machines']
else:
    # Fallback to job definition
    eligible_idxs = jobs_allowed.get(int(job_id), {}).get(op_name, [])
    eligible_names = [mlist[i] for i in eligible_idxs]
```

**Requires**: Store `allowed_machines` in `decision_trace` dict at decision time.

**File**: `environment.py` or `rollout.py` (where decision_trace is created)

**Add**:
```python
decision_trace = {
    "chosen_machine": machine_name,
    "chosen_operator": operator_id,
    "allowed_machines": [mlist[i] for i in allowed_machine_indices],  # ADD THIS
    "eligibilities": [...]
}
```

#### **PRIORITY 5: Deprecate eligible_operator_groups_by_wc**

**Rationale**: Creates mismatch between WorkCenter-level and machine-level logic

**Files affected**:
- `utils/workcenter.py`: Remove or document as deprecated
- `environment.py`: Remove usage in `_build_avail_actions`
- All operator selection: Use `find_free_operator_for_machine` (machine-level)

---

### Testing Checklist:

- [ ] Run `consistency_check.py` script
- [ ] Verify Op4 removed from M0 processing times
- [ ] Verify Op3 and Op7 removed from M2 processing times
- [ ] Create test job with Op7 → verify only M3 available (not M1)
- [ ] Create test job with operation requiring operator → verify unavailable if no free operator
- [ ] Check timeline output: eligible list matches action mask
- [ ] Check timeline output: "no qualified operator" only when truly no operator
- [ ] Verify no fallback warnings in normal operation
- [ ] Run full training: verify no invalid actions selected
- [ ] Verify action masks all have at least one valid action (or episode handled gracefully)

---

**End of Analysis Report**
