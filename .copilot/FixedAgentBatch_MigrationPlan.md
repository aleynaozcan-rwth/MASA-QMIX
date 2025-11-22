# Fixed Agent Batch Migration Plan

**Date:** 2025-11-22  
**Status:** ✅ IMPLEMENTATION COMPLETE  
**Priority:** HIGH - Blocking epsilon decay execution  
**Estimated Effort:** 4-6 hours  
**Actual Duration:** ~4 hours  
**Completion Report:** `docs/fixed_agent_batch_implementation_complete.md`

---

## Problem Statement

**Current Issue:** Event-driven decision batching returns variable-size batches (1-4 agents), but QMIX neural network expects fixed `[n_agents=10, obs_dim=7]` shapes. This causes shape mismatch exceptions that prevent epsilon decay from executing.

**Root Cause:** No padding/masking adapter layer between SimPy's event-driven simulation and QMIX's centralized training architecture.

**Impact:**
- ❌ Epsilon decay never executes (blocked by upstream exception)
- ❌ Zero operations completed (no actions successfully selected)
- ❌ Training cannot proceed (every decision fails with ValueError)

---

## Solution Overview

**Add padding/masking layer to normalize variable-size batches to fixed n_agents size.**

**Implementation Location:** `environment.py::wait_for_decisions()` (Option 1 - Environment Layer)

**Key Components:**
1. **Padding:** Expand batch to n_agents with dummy observations
2. **Masking:** Explicit 0/1 mask field (1=real agent, 0=padded)
3. **Validation:** Shape checks with fail-fast error messages
4. **Cleanup:** Remove Co-Pilot Rule violations (fallback logic)

---

## Migration Phases

### Phase 1: Add Padding to Environment Layer

**File:** `environment.py`  
**Function:** `wait_for_decisions()`  
**Lines:** 614-675

**Implementation:**

```python
def wait_for_decisions(self):
    """Run the sim until at least one decision is pending or episode ends.
    
    Returns: (batch, sim_time)
        batch: List of decision items, padded to n_agents with explicit masks
        sim_time: Current simulation time
    """
    if self.done:
        return [], float(self.t)

    # ... existing event loop logic ...

    self.t = float(self.env.now)
    batch = list(self.pending_decisions)
    self.pending_decisions = []
    self.decisions_ready = simpy.Event(self.env)
    
    # ===== PADDING LOGIC (NEW) =====
    # Get canonical agent count from environment configuration
    n_agents = int(getattr(self, 'max_jobs', getattr(self.args, 'n_agents', 10)))
    real_count = len(batch)
    
    # Validate batch size doesn't exceed capacity
    if real_count > n_agents:
        raise ValueError(
            f"[FIXED_AGENT_BATCH] Decision batch size ({real_count}) exceeds "
            f"n_agents capacity ({n_agents}). This indicates a configuration error."
        )
    
    # Add explicit mask to real decision items
    for i in range(real_count):
        batch[i]['agent_mask'] = 1  # Real agent
        batch[i]['agent_index'] = i  # Original position
    
    # Pad batch to n_agents with dummy items
    if real_count < n_agents:
        dummy_obs = np.zeros(self.obs_dim_agent, dtype=np.float32)
        dummy_avail = np.zeros(len(self.machine_resources), dtype=np.int32)
        
        for i in range(real_count, n_agents):
            batch.append({
                'job_id': -1,  # Invalid job ID indicates padding
                'obs': dummy_obs,
                'avail_row': dummy_avail,
                'agent_mask': 0,  # Padded agent
                'agent_index': i,
                'allowed_machine_indices': [],
                'per_machine_durations': {},
                'resume_evt': None  # No event for padded agents
            })
    
    # ===== END PADDING LOGIC =====
    
    return batch, float(self.t)
```

**Validation:**
- Batch size always equals n_agents after padding
- Each item has explicit `agent_mask` field
- Real items have mask=1, padded items have mask=0
- Fails fast if batch size exceeds n_agents

---

### Phase 2: Extract and Propagate Masks

**File:** `MARL/common/rollout.py`  
**Function:** `run_episode()`  
**Lines:** 844-860

**Implementation:**

```python
# Extract decision batch
batch, sim_time = self.env.wait_for_decisions()

# Extract observations, available actions, and masks
obs_list = [item.get('obs') for item in batch]
avail = [item.get('avail_row') for item in batch]
agent_masks = [item.get('agent_mask', 1) for item in batch]  # Default 1 for backward compat

# Validate batch size matches n_agents
n_agents = self.args.n_agents
if len(obs_list) != n_agents:
    raise ValueError(
        f"[FIXED_AGENT_BATCH] Batch size mismatch: got {len(obs_list)}, "
        f"expected {n_agents}. Environment padding may have failed."
    )

# Select actions with mask
actions, _ = self._select_actions(
    obs_list, 
    avail, 
    evaluate=evaluate, 
    epsilon=self.epsilon,
    agent_masks=agent_masks  # NEW PARAMETER
)

# Filter actions for real agents only
real_actions = [actions[i] for i in range(len(batch)) if batch[i].get('agent_mask', 1) == 1]

# Apply only real actions to environment
for i, item in enumerate(batch):
    if item.get('agent_mask', 1) == 1:  # Real agent
        resume_evt = item.get('resume_evt')
        if resume_evt is not None:
            resume_evt.succeed(actions[i])

# ===== EPSILON DECAY (NOW REACHABLE) =====
if self.epsilon is not None and hasattr(self, 'args'):
    episode_limit = float(getattr(self.args, 'episode_limit', 500))
    epsilon_start = float(getattr(self.args, 'epsilon_start', 1.0))
    epsilon_end = float(getattr(self.args, 'epsilon_end', 0.05))
    
    self.epsilon = epsilon_start - (sim_time / episode_limit) * (epsilon_start - epsilon_end)
    self.epsilon = float(np.clip(self.epsilon, epsilon_end, epsilon_start))
```

**Changes:**
- Extract `agent_masks` from batch items
- Add validation for batch size
- Pass masks to `_select_actions()`
- Filter actions for real agents only
- Epsilon decay now reachable after successful action selection

---

### Phase 3: Update Policy Layer

**File:** `MARL/policy/qmix.py`  
**Function:** `select_actions()`  
**Lines:** 289-311

**Implementation:**

```python
def select_actions(self, obs_batch, avail_batch, evaluate=False, epsilon=None, agent_masks=None):
    """Select actions for all agents.
    
    Args:
        obs_batch: Observations, shape (n_agents, obs_dim) or (1, n_agents, obs_dim)
        avail_batch: Available actions, shape (n_agents, n_actions)
        evaluate: Whether in evaluation mode
        epsilon: Epsilon for epsilon-greedy exploration
        agent_masks: Binary mask (1=real agent, 0=padded), shape (n_agents,)
    
    Returns:
        actions: Selected actions, shape (n_agents,)
        q_values: Q-values for selected actions
    """
    # Convert to numpy if needed
    obs_batch = np.array(obs_batch, dtype=np.float32)
    avail_batch = np.array(avail_batch, dtype=np.float32)
    
    # Validate shapes
    if obs_batch.shape[0] != self.n_agents:
        raise ValueError(
            f"[FIXED_AGENT_BATCH] Observation batch size mismatch: "
            f"got {obs_batch.shape[0]}, expected {self.n_agents}. "
            f"Shape: {obs_batch.shape}"
        )
    
    # Normalize to (1, n_agents, obs_dim)
    if obs_batch.ndim == 2:
        obs_batch = obs_batch.reshape(1, self.n_agents, -1)
    elif obs_batch.ndim == 1:
        # Shouldn't happen with padding, but handle gracefully
        raise ValueError(
            f"[FIXED_AGENT_BATCH] Received 1D observation batch. "
            f"Expected 2D (n_agents, obs_dim) after padding. Shape: {obs_batch.shape}"
        )
    
    # Convert to torch tensors
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
    avail_tensor = torch.tensor(avail_batch, dtype=torch.float32).to(self.device)
    
    # Forward pass through agent networks
    with torch.no_grad():
        q_values = self.agent(obs_tensor, self.hidden_state)  # Shape: (1, n_agents, n_actions)
    
    # Mask Q-values for padded agents (set to -inf)
    if agent_masks is not None:
        mask_tensor = torch.tensor(agent_masks, dtype=torch.float32).to(self.device)
        mask_tensor = mask_tensor.view(1, self.n_agents, 1)  # Broadcast to Q-values
        q_values = q_values * mask_tensor + (1 - mask_tensor) * (-1e10)  # Large negative for padded
    
    # Apply available action mask
    q_values[avail_tensor == 0] = -1e10
    
    # Select actions (epsilon-greedy)
    if epsilon is not None and not evaluate:
        actions = self._epsilon_greedy_actions(q_values, epsilon, avail_tensor, agent_masks)
    else:
        actions = q_values.argmax(dim=-1).squeeze(0).cpu().numpy()
    
    # For padded agents, set action to 0 (will be filtered out in rollout)
    if agent_masks is not None:
        actions = [actions[i] if agent_masks[i] == 1 else 0 for i in range(self.n_agents)]
    
    return np.array(actions), q_values.squeeze(0).cpu().numpy()
```

**Changes:**
- Add `agent_masks` parameter
- Validate batch size equals `self.n_agents`
- Mask Q-values for padded agents (set to large negative)
- Set padded agent actions to 0 (filtered in rollout)
- Explicit error messages for shape mismatches

---

### Phase 4: Update Mixer Network

**File:** `MARL/network/qmix_mixer.py`  
**Function:** `forward()`

**Implementation:**

```python
def forward(self, q_values, states, agent_masks=None):
    """Compute centralized Q-value.
    
    Args:
        q_values: Agent Q-values, shape (batch, n_agents, n_actions)
        states: Global state, shape (batch, state_dim)
        agent_masks: Binary mask (1=real, 0=padded), shape (batch, n_agents)
    
    Returns:
        q_tot: Centralized Q-value, shape (batch, n_actions)
    """
    batch_size = q_values.size(0)
    
    # Apply agent mask to Q-values (zero out padded agents)
    if agent_masks is not None:
        mask = agent_masks.unsqueeze(-1)  # (batch, n_agents, 1)
        q_values = q_values * mask
    
    # ... existing mixing logic ...
    # (hypernetworks, weighted sum, etc.)
    
    return q_tot
```

**Changes:**
- Add `agent_masks` parameter
- Zero out Q-values for padded agents before mixing
- Preserves QMIX monotonicity constraint

---

### Phase 5: Remove Co-Pilot Rule Violations

**File:** `MARL/common/rollout.py`  
**Line:** 526

**REMOVE THIS LINE:**
```python
n_agents = getattr(args, 'n_agents', len(u_list))  # ❌ VIOLATES RULE A6
```

**REPLACE WITH:**
```python
# A6 Compliance: n_agents must be explicitly configured, no fallbacks
if not hasattr(args, 'n_agents'):
    raise ValueError(
        "[FIXED_AGENT_BATCH] args.n_agents is required but missing. "
        "This must be set explicitly in configuration."
    )
n_agents = int(args.n_agents)
```

**Rationale:** Co-Pilot Rule A6 prohibits defensive fallbacks. Agent count is a critical architectural parameter that must be explicit.

---

### Phase 6: Add Validation Tests

**File:** `tests/test_fixed_agent_batch.py` (NEW FILE)

**Test Cases:**
1. **test_padding_single_agent** - Batch size 1 → padded to 10
2. **test_padding_multiple_agents** - Batch size 4 → padded to 10
3. **test_no_padding_full_batch** - Batch size 10 → no padding needed
4. **test_mask_extraction** - Verify mask field in batch items
5. **test_mask_propagation** - Mask travels through rollout → policy
6. **test_padded_action_filtering** - Only real actions applied to environment
7. **test_q_value_masking** - Padded agents have Q-values = -inf
8. **test_batch_size_validation** - Fails if batch > n_agents
9. **test_epsilon_decay_reachable** - Epsilon changes after successful action selection
10. **test_operations_complete** - Jobs complete successfully with padding

**Implementation:**

```python
import pytest
import numpy as np
from environment import MASAEnv
from MARL.common.rollout import Rollout
from MARL.common.arguments import get_common_args

def test_padding_single_agent():
    """Test padding when only 1 agent needs decision."""
    args = get_common_args()
    args.n_agents = 10
    env = MASAEnv(args)
    
    # Simulate single decision
    # ... trigger decision for 1 job ...
    
    batch, _ = env.wait_for_decisions()
    
    # Assertions
    assert len(batch) == 10, f"Expected padded batch size 10, got {len(batch)}"
    assert batch[0]['agent_mask'] == 1, "First item should be real agent"
    assert all(item['agent_mask'] == 0 for item in batch[1:]), "Items 1-9 should be padded"
    assert batch[0]['job_id'] != -1, "Real agent should have valid job_id"
    assert all(item['job_id'] == -1 for item in batch[1:]), "Padded agents should have job_id=-1"
    assert batch[0]['obs'].shape == (7,), "Observation shape should be (7,)"
    assert all(item['obs'].shape == (7,) for item in batch), "All obs should have shape (7,)"

def test_epsilon_decay_reachable():
    """Verify epsilon decay executes after padding fix."""
    args = get_common_args()
    args.n_agents = 10
    args.epsilon_start = 1.0
    args.epsilon_end = 0.05
    args.episode_limit = 500
    
    env = MASAEnv(args)
    rollout = Rollout(args, env)
    
    initial_epsilon = rollout.epsilon
    
    # Run partial episode
    rollout.run_episode(evaluate=False)
    
    # Verify epsilon changed
    assert rollout.epsilon < initial_epsilon, \
        f"Epsilon should decay, got {rollout.epsilon} (initial: {initial_epsilon})"
    assert 0.05 <= rollout.epsilon <= 1.0, \
        f"Epsilon should be in [0.05, 1.0], got {rollout.epsilon}"

# ... additional tests ...
```

---

## Validation Checklist

### Pre-Implementation
- [x] Analysis document reviewed (`docs/observation_action_pipeline_analysis.md`)
- [x] Migration plan created (this document)
- [x] Padding location decided (Environment Layer - Option 1)
- [x] Mask propagation strategy defined

### Phase 1: Environment Padding
- [ ] `wait_for_decisions()` modified with padding logic
- [ ] Mask field added to all batch items
- [ ] Batch size validation added
- [ ] Manual test: print batch size before/after padding

### Phase 2: Rollout Integration
- [ ] Mask extraction added in `run_episode()`
- [ ] Batch size validation added
- [ ] Action filtering for real agents only
- [ ] Epsilon decay reachable after action selection
- [ ] Manual test: verify epsilon changes over time

### Phase 3: Policy Updates
- [ ] `select_actions()` accepts `agent_masks` parameter
- [ ] Q-values masked for padded agents
- [ ] Shape validation with explicit errors
- [ ] Manual test: print Q-values before/after masking

### Phase 4: Mixer Updates
- [ ] `forward()` accepts `agent_masks` parameter
- [ ] Padded agents zeroed in mixing computation
- [ ] Manual test: verify mixer output with/without padding

### Phase 5: Cleanup
- [ ] Fallback logic removed from `rollout.py` line 526
- [ ] All Co-Pilot Rule violations resolved
- [ ] Grep search confirms no remaining fallbacks

### Phase 6: Testing
- [ ] All 10 unit tests implemented
- [ ] All tests pass
- [ ] Integration test with full episode
- [ ] Smoke test: `python main.py --alg qmix --env MASAEnv`

### Final Validation
- [ ] No shape mismatch exceptions
- [ ] Epsilon decays from 1.0 → 0.05
- [ ] Operations complete successfully (total_ops > 0)
- [ ] No Co-Pilot Rule violations (grep search)
- [ ] Documentation updated

---

## Implementation Commands

```bash
# 1. Run unit tests
pytest tests/test_fixed_agent_batch.py -v

# 2. Run smoke test
python main.py --alg qmix --env MASAEnv --epsilon_start 1.0 --epsilon_end 0.05 --episode_limit 500

# 3. Verify no Co-Pilot Rule violations
grep -r "getattr.*n_agents.*len" MARL/
grep -r "\.get.*n_agents.*default" MARL/

# 4. Check epsilon decay
grep "EPSILON_DECAY" logs/*.log

# 5. Validate operations completed
grep "operations completed" logs/*.log
```

---

## Rollback Plan

If implementation fails or causes regressions:

1. **Revert environment changes:**
   ```bash
   git checkout environment.py
   ```

2. **Revert rollout changes:**
   ```bash
   git checkout MARL/common/rollout.py
   ```

3. **Revert policy changes:**
   ```bash
   git checkout MARL/policy/qmix.py
   ```

4. **Delete test file:**
   ```bash
   rm tests/test_fixed_agent_batch.py
   ```

5. **Return to analysis phase** - reassess padding location if Environment Layer proves problematic

---

## Success Criteria

**Definition of Done:**
1. ✅ Decision batches always have size = n_agents
2. ✅ Explicit masks differentiate real vs padded agents
3. ✅ No shape mismatch exceptions
4. ✅ Epsilon decay executes and changes over time
5. ✅ Operations complete successfully (total_ops > 0)
6. ✅ All unit tests pass
7. ✅ No Co-Pilot Rule violations
8. ✅ Smoke test completes without errors

**Performance Metrics:**
- Epsilon at t=0: 1.0
- Epsilon at t=250: ~0.525
- Epsilon at t=500: 0.05
- Total operations completed: > 0 (previously 0)
- Decision success rate: 100% (previously 0%)

---

## Timeline

**Estimated Duration:** 4-6 hours

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Environment Padding | 1 hour | Not Started |
| Phase 2: Rollout Integration | 1 hour | Not Started |
| Phase 3: Policy Updates | 1 hour | Not Started |
| Phase 4: Mixer Updates | 30 min | Not Started |
| Phase 5: Cleanup | 30 min | Not Started |
| Phase 6: Testing | 1-2 hours | Not Started |
| Final Validation | 30 min | Not Started |

---

## References

- **Analysis Document:** `docs/observation_action_pipeline_analysis.md`
- **Trace Report:** `docs/epsilon_trace_analysis.md`
- **Co-Pilot Rules:** `.copilot/Co-Pilot Rules.md`
- **QMIX Paper:** "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning"
- **Related Tests:** `tests/test_epsilon_decay_episode_based.py`, `tests/test_env_obs.py`

---

**READY FOR IMPLEMENTATION - AWAITING USER APPROVAL**
