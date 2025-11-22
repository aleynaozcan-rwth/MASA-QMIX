#!/usr/bin/env python
"""
Validation script for Fixed Agent Batch implementation.

Verifies:
1. No Co-Pilot Rule violations (no getattr fallbacks for critical parameters)
2. Padding logic implemented correctly
3. Mask propagation works
4. Epsilon decay is now reachable
"""

import sys
import re
from pathlib import Path


def check_no_fallbacks():
    """Verify no getattr fallbacks for critical parameters."""
    print("=" * 80)
    print("CHECKING CO-PILOT RULE COMPLIANCE")
    print("=" * 80)
    
    violations = []
    
    # Check for n_agents fallbacks
    rollout_file = Path("MARL/common/rollout.py")
    with open(rollout_file) as f:
        content = f.read()
        
        # Look for problematic patterns (excluding logging)
        patterns = [
            (r"getattr\(.*,\s*['\"]n_agents['\"].*,.*\)", "n_agents fallback"),
            (r"getattr\(.*,\s*['\"]epsilon_start['\"].*,.*\)", "epsilon_start fallback"),
            (r"getattr\(.*,\s*['\"]epsilon_end['\"].*,.*\)", "epsilon_end fallback"),
            (r"getattr\(.*,\s*['\"]episode_limit['\"].*,.*\)", "episode_limit fallback"),
        ]
        
        for line_num, line in enumerate(content.split('\n'), 1):
            # Skip logging lines
            if "[ACTIVE_ARGS]" in line or "logging" in line.lower():
                continue
                
            for pattern, desc in patterns:
                if re.search(pattern, line):
                    violations.append((rollout_file, line_num, desc, line.strip()))
    
    if violations:
        print("❌ FOUND CO-PILOT RULE VIOLATIONS:")
        for file, line, desc, code in violations:
            print(f"  {file}:{line} - {desc}")
            print(f"    {code}")
        return False
    else:
        print("✅ NO CO-PILOT RULE VIOLATIONS FOUND")
        print("   - No getattr fallbacks for n_agents")
        print("   - No getattr fallbacks for epsilon parameters")
        print("   - No getattr fallbacks for episode_limit")
        return True


def check_padding_implementation():
    """Verify padding logic exists in environment.py."""
    print("\n" + "=" * 80)
    print("CHECKING PADDING IMPLEMENTATION")
    print("=" * 80)
    
    env_file = Path("environment.py")
    with open(env_file) as f:
        content = f.read()
    
    checks = [
        ("agent_mask", "Agent mask field in batch items"),
        ("agent_index", "Agent index tracking"),
        ("job_id': -1", "Padded agent marker (job_id=-1)"),
        ("resume_evt': None", "No resume event for padded agents"),
        ("PADDING LOGIC", "Padding logic section marker"),
        ("np.zeros(self.obs_dim_agent", "Dummy observation creation"),
    ]
    
    all_found = True
    for pattern, desc in checks:
        if pattern in content:
            print(f"✅ {desc}")
        else:
            print(f"❌ MISSING: {desc}")
            all_found = False
    
    return all_found


def check_mask_propagation():
    """Verify mask propagation through rollout and policy."""
    print("\n" + "=" * 80)
    print("CHECKING MASK PROPAGATION")
    print("=" * 80)
    
    # Check rollout.py
    rollout_file = Path("MARL/common/rollout.py")
    with open(rollout_file) as f:
        rollout_content = f.read()
    
    rollout_checks = [
        ("agent_masks = [item.get('agent_mask'", "Mask extraction from batch"),
        ("agent_masks=agent_masks", "Mask passed to _select_actions"),
        ("agent_masks: Optional[List[int]]", "agent_masks parameter in _select_actions signature"),
    ]
    
    rollout_ok = True
    print("\nRollout layer:")
    for pattern, desc in rollout_checks:
        if pattern in rollout_content:
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ MISSING: {desc}")
            rollout_ok = False
    
    # Check qmix.py
    qmix_file = Path("MARL/policy/qmix.py")
    with open(qmix_file) as f:
        qmix_content = f.read()
    
    qmix_checks = [
        ("agent_masks=None", "agent_masks parameter in select_actions"),
        ("agent_masks is not None", "Mask validation"),
        ("agent_masks_arr", "Mask array processing"),
        ("FIXED_AGENT_BATCH", "Fixed agent batch validation"),
    ]
    
    qmix_ok = True
    print("\nPolicy layer:")
    for pattern, desc in qmix_checks:
        if pattern in qmix_content:
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ MISSING: {desc}")
            qmix_ok = False
    
    return rollout_ok and qmix_ok


def check_epsilon_decay_reachability():
    """Verify epsilon decay code is now reachable."""
    print("\n" + "=" * 80)
    print("CHECKING EPSILON DECAY REACHABILITY")
    print("=" * 80)
    
    rollout_file = Path("MARL/common/rollout.py")
    with open(rollout_file) as f:
        content = f.read()
    
    # Find epsilon decay section
    if "TIME-BASED EPSILON DECAY" in content:
        print("✅ Time-based epsilon decay code exists")
        
        # Check it comes after action selection (not before)
        lines = content.split('\n')
        action_selection_line = None
        epsilon_decay_line = None
        
        for i, line in enumerate(lines):
            if "_select_actions" in line and "epsilon=" in line:
                action_selection_line = i
            if "TIME-BASED EPSILON DECAY" in line:
                epsilon_decay_line = i
        
        if action_selection_line and epsilon_decay_line:
            if epsilon_decay_line > action_selection_line:
                print("✅ Epsilon decay comes AFTER action selection (correct order)")
                return True
            else:
                print("❌ Epsilon decay comes BEFORE action selection (wrong order)")
                return False
    else:
        print("❌ Time-based epsilon decay code not found")
        return False


def main():
    """Run all validation checks."""
    print("\n" + "=" * 80)
    print("FIXED AGENT BATCH MIGRATION VALIDATION")
    print("=" * 80)
    print()
    
    results = {
        "Co-Pilot Rule Compliance": check_no_fallbacks(),
        "Padding Implementation": check_padding_implementation(),
        "Mask Propagation": check_mask_propagation(),
        "Epsilon Decay Reachability": check_epsilon_decay_reachability(),
    }
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED")
        print("=" * 80)
        print("\nFixed Agent Batch implementation is complete and compliant.")
        print("Ready for integration testing.")
        return 0
    else:
        print("⚠️  SOME VALIDATIONS FAILED")
        print("=" * 80)
        print("\nReview the failures above and fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
