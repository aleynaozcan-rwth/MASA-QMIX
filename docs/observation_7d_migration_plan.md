# PHASE 1 — FULL IMPACT PLAN FOR 7-ELEMENT OBSERVATION MIGRATION

## SCOPE SUMMARY
Migrating from 6-element to 7-element per-agent observation by adding:
- Element 5: `theoretical_machine_count` (machines qualified for current operation)
- Element 6: `free_machine_count` (qualified machines currently available)

Elements 1-4 and 7 remain unchanged.

---

## A. FILES REQUIRING UPDATES

### Critical Path (Must Update):
1. `utils/env_obs.py` - Observation builder
2. `environment.py` - Add helper methods for machine counts
3. `MARL/runner.py` - Update shape injection from 6→7
4. `MARL/common/arguments.py` - Update default obs_shape
5. `main.py` - Update any hardcoded references

### Verification Path (Review Only):
6. `MARL/common/replay_buffer.py` - Verify shape-agnostic
7. `MARL/agent/agent.py` - Verify input layer reads from args
8. `MARL/network/base_net.py` - Verify RNN input shape dynamic
9. `MARL/policy/qmix.py` - Verify network init reads from args
10. `MARL/common/rollout.py` - Verify observation collection is pass-through

### Testing Impact:
11. All test files that hardcode `obs_shape=6`

---

## B. FILE-BY-FILE DETAILED IMPACT

### **FILE 1: `utils/env_obs.py`**

**Functions to Modify:**
- `build_agent_obs()` (lines 42-139)

**Required Changes:**
- Add after line 121 (after `n_jobs_active_norm`):
  - Call `env.get_theoretical_machine_count(job)` → normalize → store as element 5
  - Call `env.get_free_machine_count(job)` → normalize → store as element 6
- Update line 132: Change array length validation from 6 to 7
- Update docstring (lines 1-21): Add elements 5 and 6 to layout description

**Dependencies:**
- Environment MUST provide:
  - `get_theoretical_machine_count(job)` → int (total qualified machines)
  - `get_free_machine_count(job)` → int (qualified + currently idle)
- Both must handle current operation lookup internally

**Logic That Will Break:**
- Final shape check at line 132 will raise if not updated
- Any test that validates exact observation content

---

### **FILE 2: `environment.py`**

**Functions to Add:**
- `get_theoretical_machine_count(self, job: JobAgent) -> int`
  - Location: After line 1470 (near `active_jobs_count()`)
  - Logic: 
    - Get `current_op = job.current_operation()`
    - Extract `op_type = current_op.type`
    - Query `self.workcenters_meta` for machines qualified for `op_type`
    - Return count
  - Error handling: Raise if job has no current operation or op_type invalid

- `get_free_machine_count(self, job: JobAgent) -> int`
  - Location: After `get_theoretical_machine_count()`
  - Logic:
    - Get theoretical count (call above method)
    - Filter for machines where `self._resource_free(machine_resource[i]) == True`
    - Return count of free qualified machines
  - Error handling: Same as above

**Attributes Impacted:**
- None (uses existing `workcenters_meta`, `machine_resources`)

**Dependencies:**
- Must work with current `WorkCenters` qualification logic
- Must align with mask generation in `_avail_row_for_job()`

**Risk:**
- If qualification logic is complex/scattered, need to centralize

---

### **FILE 3: `MARL/runner.py`**

**Functions to Modify:**
- `__init__()` (lines 887-898)

**Required Changes:**
- Line 894: Observation shape injection already reads from `env_info["obs_shape"]`
- **No code change needed** - shape auto-propagates from environment

**Verification Needed:**
- Confirm line 894 uses `env_info["obs_shape"]` not hardcoded 6

**Logic That Will Break:**
- None (already using environment's `get_env_info()`)

---

### **FILE 4: `environment.py` (Shape Definition)**

**Functions to Modify:**
- `__init__()` (line 139)

**Required Changes:**
- Line 139: Change `self.obs_dim_agent = 6` to `self.obs_dim_agent = 7`
- Update comment to reflect new canonical observation

**Dependencies:**
- This is the SOURCE OF TRUTH for observation shape
- Runner will propagate this to args automatically

**Logic That Will Break:**
- Any hardcoded size=6 assumptions in environment itself

---

### **FILE 5: `MARL/common/arguments.py`**

**Functions to Modify:**
- Argument defaults (around line 126)

**Required Changes:**
- Find: `parser.add_argument('--obs_shape', type=int, default=6)`
- Change to: `parser.add_argument('--obs_shape', type=int, default=7)`
- **NOTE**: This is only a CLI default fallback
- Environment override via `get_env_info()` takes precedence

**Risk:**
- If scripts bypass Runner and read args directly, they'll get wrong shape

---

### **FILE 6: `main.py`**

**Functions to Verify:**
- Lines 70-82 (shape printing after Runner init)

**Required Changes:**
- None - already prints from args after Runner injection

**Verification:**
- Confirm no hardcoded "obs_shape=6" assertions or prints before Runner

---

### **FILE 7: `MARL/common/replay_buffer.py`**

**Functions to Verify:**
- `sample()` (lines 82-315)
- `_as_agents_obs()` (lines 325-349)

**Analysis:**
- Line 102: Uses `n_agents, obs_dim = _infer_agents_obs(episodes)`
- Line 163-176: `_as_agents_obs()` pads/trims to `ensure_shape=(n_agents, obs_dim)`
- **Shape-agnostic design** - no hardcoded 6

**Required Changes:**
- **None** - buffer infers shape dynamically

**Risk:**
- If buffer was constructed with explicit `obs_dim=6`, it will fail
- Current code allows runtime inference (safe)

---

### **FILE 8: `MARL/agent/agent.py`**

**Functions to Verify:**
- Agent initialization

**Analysis:**
- Agent/policy reads `args.obs_shape` to construct input layers
- Runner already injects from environment

**Required Changes:**
- **None** - uses args.obs_shape dynamically

**Risk:**
- Any agent that hardcodes input size=6 in network definition

---

### **FILE 9: `MARL/network/base_net.py`**

**Functions to Verify:**
- `RNNAgent.__init__()` (reads input_shape from args)

**Analysis:**
- Network input shape is parameter-driven

**Required Changes:**
- **None** - reads from args

**Risk:**
- None identified

---

### **FILE 10: `MARL/policy/qmix.py`**

**Functions to Verify:**
- `__init__()` (lines 23-74)

**Analysis:**
- Line 34: `input_shape = self.obs_shape` (from args)
- Line 42-56: Validates `input_shape > 0` but doesn't check specific value

**Required Changes:**
- **None** - shape is parameter-driven

**Risk:**
- None identified

---

### **FILE 11: `MARL/common/rollout.py`**

**Functions to Verify:**
- Observation collection (pass-through to environment)

**Analysis:**
- Rollout calls `env._build_agent_obs(job)` or similar
- Does not construct observations itself

**Required Changes:**
- **None** - pass-through design

**Risk:**
- If rollout has debug assertions checking obs.shape[0] == 6

---

## C. WHAT MUST BE DELETED

### **In `utils/env_obs.py`:**
- **Nothing** - we're extending, not replacing
- Old 6-element logic becomes 7-element logic
- Docstring updated to reflect new schema

### **In `environment.py`:**
- **Nothing** - only adding helpers

### **Everywhere else:**
- Any hardcoded assertions like `assert obs.shape == (6,)`
- Any comments stating "canonical 6D observation"

---

## D. WHAT MUST BE ADDED

### **In `environment.py`:**
1. **Method**: `get_theoretical_machine_count(self, job: JobAgent) -> int`
   - Returns count of machines qualified for job's current operation
   
2. **Method**: `get_free_machine_count(self, job: JobAgent) -> int`
   - Returns count of qualified machines that are currently idle

3. **Update**: Line 139: `self.obs_dim_agent = 7`

### **In `utils/env_obs.py`:**
1. **Add** two new observation elements (lines 122-130)
2. **Update** shape validation from 6 to 7 (line 132)
3. **Update** docstring (lines 1-21)

### **In `MARL/common/arguments.py`:**
1. **Update** default: `--obs_shape` from 6 to 7

---

## E. COMPONENT-SPECIFIC CONFIRMATION

### **ReplayBuffer Storage Format:**
✅ **NO CHANGES NEEDED**
- Buffer stores observations as numpy arrays with inferred shape
- Shape validation is dynamic (`_as_agents_obs` pads/truncates)
- No hardcoded dimension assumptions

### **Agent Input Layers / Model Init:**
✅ **NO CHANGES NEEDED**
- Agents read `args.obs_shape` to construct input layers
- Runner injects `args.obs_shape = 7` from environment
- Networks are parameter-driven

### **Action Selection / Rollout Collection:**
✅ **NO CHANGES NEEDED**
- Rollout calls environment observation builders
- No local observation construction
- Pass-through design

### **Runner Logging / Metrics:**
✅ **NO CHANGES NEEDED**
- Observation shape already printed from args (after injection)
- No hardcoded references to 6

---

## F. RISKS & MITIGATION

### **Risk 1: Hardcoded obs.shape == 6 Assertions**
- **Location**: Test files, debug code
- **Impact**: Tests will fail
- **Mitigation**: Search codebase for `== 6` and update to `== 7`

### **Risk 2: Normalization Assumptions**
- **Location**: If machine counts are not normalized properly
- **Impact**: Network receives values outside [0, 1] range
- **Mitigation**: Add proper normalization in `build_agent_obs()`:
  - `theoretical_machine_count / max(1, total_machines_in_env)`
  - `free_machine_count / max(1, theoretical_machine_count)`

### **Risk 3: Machine Qualification Logic Complexity**
- **Location**: `WorkCenters` may have complex qualification rules
- **Impact**: Helper methods may be non-trivial to implement
- **Mitigation**: Review existing mask generation logic and reuse

### **Risk 4: Observation Collection Timing**
- **Location**: If observation built before job has current operation
- **Impact**: `get_theoretical_machine_count()` may fail
- **Mitigation**: Add defensive checks, return 0 if no current operation

### **Risk 5: State Vector Confusion**
- **Location**: State remains 4D, observation becomes 7D
- **Impact**: Developers may confuse the two
- **Mitigation**: Clear comments distinguishing per-agent obs vs global state

### **Risk 6: Fallback Logic Reading Old Shapes**
- **Location**: Code bypassing Runner (e.g., random baseline)
- **Impact**: May still use obs_shape=6 from args default
- **Mitigation**: Audit all environment construction sites

---

## G. SEARCH COMMANDS FOR VERIFICATION

Before implementation, run these searches:

```bash
# Find hardcoded 6 references
grep -rn "obs.*6" MARL/ utils/ tests/
grep -rn "shape.*6" MARL/ utils/ tests/
grep -rn "== 6" MARL/ utils/ tests/

# Find observation shape usage
grep -rn "obs_shape" MARL/ utils/ environment.py

# Find observation builders
grep -rn "build_agent_obs" MARL/ utils/ environment.py

# Find test files setting obs_shape
grep -rn "obs_shape.*=" tests/
```

---

## H. IMPLEMENTATION ORDER

**Phase 1A: Core Observation**
1. Add helper methods to `environment.py`
2. Update `utils/env_obs.py` observation builder
3. Update `environment.py` line 139 (obs_dim_agent = 7)

**Phase 1B: Configuration**
4. Update `MARL/common/arguments.py` default

**Phase 1C: Verification**
5. Run test suite
6. Fix any hardcoded 6 assertions
7. Verify all components read shape dynamically

**Phase 1D: Documentation**
8. Update docstrings
9. Update any README or architecture docs

---

## I. TESTING CHECKLIST

After implementation, verify:

- ✅ `env.get_env_info()["obs_shape"]` returns 7
- ✅ Runner injects `args.obs_shape = 7`
- ✅ Agent networks initialize with input_shape=7
- ✅ Observations sampled from buffer have shape (B, T, n_agents, 7)
- ✅ QMIX policy accepts 7-element observations
- ✅ No shape mismatch errors during episode rollout
- ✅ All tests pass with new observation

---

## J. SUMMARY

**Files Requiring Code Changes: 3**
1. `utils/env_obs.py` - Extend observation builder
2. `environment.py` - Add machine count helpers + update obs_dim_agent
3. `MARL/common/arguments.py` - Update default obs_shape

**Files Requiring Verification Only: 7**
- All agent/network/policy/buffer files (confirm shape-agnostic)

**Estimated Complexity: LOW-MEDIUM**
- Core change is localized to observation builder
- Shape propagation already uses dynamic `get_env_info()`
- No breaking changes to storage or network architecture

**Estimated Implementation Time: 2-3 hours**
- 1 hour: Add helpers + update observation builder
- 30 min: Update defaults and verify propagation
- 1 hour: Test suite updates and verification

---

**Status: READY FOR IMPLEMENTATION APPROVAL**

When approved, implementation will proceed in the following order:
1. Add environment helper methods
2. Update observation builder
3. Update shape constants
4. Verify and test
