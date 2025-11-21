#!/usr/bin/env python3
"""
Static verification for Phase 2 fail-fast changes.
Validates that critical issues A1, A3, A5 have been correctly fixed.

Run: python tests/verify_phase2_changes.py
"""

import re
import sys
from pathlib import Path

# Define repository root
REPO_ROOT = Path(__file__).parent.parent

def read_file_content(filepath):
    """Read and return file content."""
    full_path = REPO_ROOT / filepath
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ FAIL: File not found: {filepath}")
        sys.exit(1)

def check_pattern_present(content, pattern, description):
    """Check if pattern is present in content."""
    if re.search(pattern, content, re.MULTILINE | re.DOTALL):
        print(f"✅ PASS: {description}")
        return True
    else:
        print(f"❌ FAIL: {description}")
        return False

def check_pattern_absent(content, pattern, description):
    """Check if pattern is absent in content."""
    if not re.search(pattern, content, re.MULTILINE | re.DOTALL):
        print(f"✅ PASS: {description}")
        return True
    else:
        print(f"❌ FAIL: {description}")
        return False

def verify_phase2():
    """Verify all Phase 2 changes."""
    print("=" * 70)
    print("Phase 2 Fail-Fast Verification")
    print("=" * 70)
    
    checks_passed = 0
    checks_total = 0
    
    # ========================================================================
    # A1: QMIX avail_batch length validation
    # ========================================================================
    print("\n[A1] QMIX avail_batch length validation (qmix.py)")
    print("-" * 70)
    
    qmix_content = read_file_content("MARL/policy/qmix.py")
    
    # A1.1: Check that explicit validation exists
    checks_total += 1
    pattern = r"if\s+avail_batch\s+is\s+not\s+None:.*?if\s+len\(avail_batch\)\s*!=\s*q_vals\.shape\[0\]"
    if check_pattern_present(qmix_content, pattern, "A1.1: Explicit avail_batch length check exists"):
        checks_passed += 1
    
    # A1.2: Check that ValueError is raised with descriptive message
    checks_total += 1
    pattern = r"raise\s+ValueError\s*\(\s*\n?\s*f['\"].*avail_batch.*length.*mismatch"
    if check_pattern_present(qmix_content, pattern, "A1.2: ValueError raised on length mismatch"):
        checks_passed += 1
    
    # A1.3: Check that error message includes both got and expected values
    checks_total += 1
    pattern = r"got\s+\{len\(avail_batch\)\}.*expected\s+\{q_vals\.shape\[0\]\}"
    if check_pattern_present(qmix_content, pattern, "A1.3: Error message includes got/expected values"):
        checks_passed += 1
    
    # A1.4: Check that validation happens before the loop
    checks_total += 1
    pattern = r"if\s+avail_batch\s+is\s+not\s+None:.*?raise\s+ValueError.*?for\s+a_idx\s+in\s+range\(q_vals\.shape\[0\]\)"
    if check_pattern_present(qmix_content, pattern, "A1.4: Validation happens before agent loop"):
        checks_passed += 1
    
    # ========================================================================
    # A3: Runner all-ones mask fallback removal
    # ========================================================================
    print("\n[A3] Runner all-ones mask fallback removal (runner.py)")
    print("-" * 70)
    
    runner_content = read_file_content("MARL/runner.py")
    
    # A3.1: Check that permissive all-ones fallback is removed
    checks_total += 1
    # Look for the old pattern where we append [1] * something as a fallback
    pattern = r"avail_batch\.append\(\[1\]\s*\*"
    if check_pattern_absent(runner_content, pattern, "A3.1: No more [1] * N fallback patterns"):
        checks_passed += 1
    
    # A3.2: Check that RuntimeError is raised instead
    checks_total += 1
    pattern = r"raise\s+RuntimeError\(\s*['\"].*Cannot\s+determine\s+availability\s+mask"
    if check_pattern_present(runner_content, pattern, "A3.2: RuntimeError raised when mask cannot be determined"):
        checks_passed += 1
    
    # A3.3: Check that error message includes agent_idx for debugging
    checks_total += 1
    pattern = r"for\s+agent\s+\{agent_idx\}"
    if check_pattern_present(runner_content, pattern, "A3.3: Error message includes agent_idx"):
        checks_passed += 1
    
    # A3.4: Check that error message mentions both avail_row and build_machine_major_mask
    checks_total += 1
    pattern = r"avail_row.*build_machine_major_mask"
    if check_pattern_present(runner_content, pattern, "A3.4: Error mentions both mask sources"):
        checks_passed += 1
    
    # A3.5: Check that the old "final deterministic permissive fallback" comment is gone
    checks_total += 1
    pattern = r"final\s+deterministic\s+permissive\s+fallback"
    if check_pattern_absent(runner_content, pattern, "A3.5: Old permissive fallback comment removed"):
        checks_passed += 1
    
    # ========================================================================
    # A5: _compute_utilization_summary validation
    # ========================================================================
    print("\n[A5] _compute_utilization_summary validation (environment.py)")
    print("-" * 70)
    
    env_content = read_file_content("environment.py")
    
    # A5.1: Check that result type is validated as dict
    checks_total += 1
    pattern = r"if\s+not\s+isinstance\(util,\s*dict\)"
    if check_pattern_present(env_content, pattern, "A5.1: Type validation - util must be dict"):
        checks_passed += 1
    
    # A5.2: Check that RuntimeError is raised on wrong type
    checks_total += 1
    pattern = r"raise\s+RuntimeError\s*\(\s*\n?\s*f['\"].*_compute_utilization_summary.*returned"
    if check_pattern_present(env_content, pattern, "A5.2: RuntimeError raised on wrong return type"):
        checks_passed += 1
    
    # A5.3: Check that required keys are validated
    checks_total += 1
    pattern = r"if\s+['\"]per_machine_utilization['\"].*not\s+in\s+util.*or.*['\"]per_operator_utilization['\"].*not\s+in\s+util"
    if check_pattern_present(env_content, pattern, "A5.3: Required keys validated"):
        checks_passed += 1
    
    # A5.4: Check that error message lists actual keys
    checks_total += 1
    pattern = r"Got\s+keys:.*list\(util\.keys\(\)\)"
    if check_pattern_present(env_content, pattern, "A5.4: Error message shows actual keys"):
        checks_passed += 1
    
    # A5.5: Check that error message lists expected keys
    checks_total += 1
    pattern = r"expected:.*per_machine_utilization.*per_operator_utilization"
    if check_pattern_present(env_content, pattern, "A5.5: Error message shows expected keys"):
        checks_passed += 1
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2 VERIFICATION SUMMARY: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)
    
    if checks_passed == checks_total:
        print("🎉 ALL VERIFICATIONS PASSED!")
        return 0
    else:
        print(f"❌ {checks_total - checks_passed} verification(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(verify_phase2())
