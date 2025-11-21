#!/usr/bin/env python3
"""
Phase 1 Code Verification - Static Analysis
Verifies that the fail-fast changes were applied correctly by checking the code directly.
"""
import os
import re

def check_file_content(filepath, pattern, should_exist=False, description=""):
    """Check if a pattern exists/doesn't exist in a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    found = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    
    if should_exist and not found:
        print(f"❌ FAIL: {description}")
        print(f"   Expected pattern NOT found in {filepath}")
        return False
    elif not should_exist and found:
        print(f"❌ FAIL: {description}")
        print(f"   Unwanted pattern FOUND in {filepath}")
        print(f"   Match: {found.group(0)[:100]}...")
        return False
    else:
        print(f"✅ PASS: {description}")
        return True

print("="*70)
print("PHASE 1 CODE VERIFICATION - Static Analysis")
print("="*70)

passed = 0
failed = 0

# ============================================================================
# A6: Verify RolloutWorker._select_actions no longer has "pick 0" fallback
# ============================================================================
print("\n[A6] RolloutWorker._select_actions verification:")
print("-"*70)

test = check_file_content(
    'MARL/common/rollout.py',
    r'# no info.*pick 0',
    should_exist=False,
    description="A6.1: Old comment '# no info — pick 0' removed"
)
passed += test; failed += not test

test = check_file_content(
    'MARL/common/rollout.py',
    r'if allowed is None:[\s\n]*actions\.append\(0\)',
    should_exist=False,
    description="A6.2: No 'append(0)' when allowed is None"
)
passed += test; failed += not test

test = check_file_content(
    'MARL/common/rollout.py',
    r'if allowed is None:[\s\n]*raise ValueError\(',
    should_exist=True,
    description="A6.3: New ValueError raise when allowed is None"
)
passed += test; failed += not test

test = check_file_content(
    'MARL/common/rollout.py',
    r'missing availability info in observation',
    should_exist=True,
    description="A6.4: Error message mentions 'missing availability info'"
)
passed += test; failed += not test

# ============================================================================
# A4: Verify Environment.pop_decision_reward has fail-fast infeasibility check
# ============================================================================
print("\n[A4] Environment.pop_decision_reward verification:")
print("-"*70)

test = check_file_content(
    'environment.py',
    r'# Fail-fast infeasibility detection',
    should_exist=True,
    description="A4.1: Comment indicates fail-fast infeasibility detection"
)
passed += test; failed += not test

test = check_file_content(
    'environment.py',
    r'except \(ValueError, TypeError, IndexError\) as e:[\s\n]*raise RuntimeError',
    should_exist=True,
    description="A4.2: RuntimeError raised on infeasibility check failure"
)
passed += test; failed += not test

test = check_file_content(
    'environment.py',
    r'Infeasibility check failed in reward computation',
    should_exist=True,
    description="A4.3: Error message mentions 'Infeasibility check failed'"
)
passed += test; failed += not test

test = check_file_content(
    'environment.py',
    r'elif chosen != -1:[\s\n]*# If avail is None.*[\s\n]*raise RuntimeError',
    should_exist=True,
    description="A4.4: RuntimeError when action chosen without avail_row"
)
passed += test; failed += not test

test = check_file_content(
    'environment.py',
    r'avail_row is None.*Cannot validate feasibility',
    should_exist=True,
    description="A4.5: Error message for missing avail_row"
)
passed += test; failed += not test

# ============================================================================
# A2: Verify QMIX.select_actions has no nested fallbacks
# ============================================================================
print("\n[A2] QMIX.select_actions verification:")
print("-"*70)

test = check_file_content(
    'MARL/policy/qmix.py',
    r'avail_batch\[0\]',
    should_exist=False,
    description="A2.1: No fallback to avail_batch[0]"
)
passed += test; failed += not test

test = check_file_content(
    'MARL/policy/qmix.py',
    r'# Fail-fast: if avail_batch is provided, conversion must succeed',
    should_exist=True,
    description="A2.2: Comment indicates fail-fast conversion"
)
passed += test; failed += not test

test = check_file_content(
    'MARL/policy/qmix.py',
    r'allowed = list\(_np\.asarray\(avail_batch\[a_idx\], dtype=_np\.int32\)\)',
    should_exist=True,
    description="A2.3: Direct conversion without try/except"
)
passed += test; failed += not test

test = check_file_content(
    'MARL/policy/qmix.py',
    r'# Fail-fast: mask processing must succeed',
    should_exist=True,
    description="A2.4: Comment indicates fail-fast mask processing"
)
passed += test; failed += not test

test = check_file_content(
    'MARL/policy/qmix.py',
    r'except Exception:[\s\n]*act = int\(_np\.argmax\(masked_q\)\)',
    should_exist=False,
    description="A2.5: No exception handler falling back to argmax"
)
passed += test; failed += not test

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
print(f"Total Checks: {passed + failed}")
print(f"✅ Passed:     {passed}")
print(f"❌ Failed:     {failed}")
print("="*70)

if failed == 0:
    print("\n🎉 ALL VERIFICATIONS PASSED!")
    print("\nPhase 1 changes are correctly applied:")
    print("  ✓ A6: RolloutWorker no longer has 'pick 0' fallback")
    print("  ✓ A4: Environment reward computation is fail-fast")
    print("  ✓ A2: QMIX select_actions has no nested fallbacks")
    print("\nNext steps:")
    print("  1. Install torch: pip install torch")
    print("  2. Run actual training to verify integration")
    print("  3. Proceed to Phase 2 (A1, A3, A5) when ready")
    exit(0)
else:
    print(f"\n⚠️  {failed} verification(s) failed!")
    print("Review the code changes above.")
    exit(1)
