#!/usr/bin/env python3
"""
Verification script for Quick Wins + Phase A Mini-Subset changes.

Tests that:
1. Device fallback removed (C12)
2. Buffer.sample() raises on empty (C9)
3. Reward finite validation added (C10)
4. Granular action fallback removed (C11)
5. State shape validation added (A)
6. Observation shape validation added (A)
7. Availability mask validation added (B)
8. Hidden state reset at episode start (C)
"""

import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

def read_file(filepath):
    with open(REPO_ROOT / filepath, 'r') as f:
        return f.read()

def check_present(content, pattern, description):
    if re.search(pattern, content, re.MULTILINE | re.DOTALL):
        print(f"✅ PASS: {description}")
        return True
    else:
        print(f"❌ FAIL: {description}")
        return False

def check_absent(content, pattern, description):
    if not re.search(pattern, content, re.MULTILINE | re.DOTALL):
        print(f"✅ PASS: {description}")
        return True
    else:
        print(f"❌ FAIL: {description}")
        return False

def main():
    print("=" * 70)
    print("Quick Wins + Phase A Mini-Subset Verification")
    print("=" * 70)
    
    passed = 0
    total = 0
    
    # ========================================================================
    # Quick Win C12: Device fallback removed
    # ========================================================================
    print("\n[C12] Device fallback removed (rollout.py)")
    print("-" * 70)
    
    rollout = read_file("MARL/common/rollout.py")
    
    total += 1
    if check_absent(rollout, r"self\.device\s*=\s*['\"]cpu['\"]", "C12.1: No hardcoded 'cpu' fallback"):
        passed += 1
    
    total += 1
    if check_present(rollout, r"raise ValueError\(.*device must be provided", "C12.2: Raises if device not provided"):
        passed += 1
    
    # ========================================================================
    # Quick Win C9: Buffer.sample() raises
    # ========================================================================
    print("\n[C9] Buffer.sample() raises instead of None (replay_buffer.py)")
    print("-" * 70)
    
    buffer = read_file("MARL/common/replay_buffer.py")
    
    total += 1
    if check_absent(buffer, r"return None\s*#.*PROBLEM", "C9.1: No 'return None' with problem comment"):
        passed += 1
    
    total += 1
    if check_present(buffer, r"raise ValueError\(.*ReplayBuffer is empty", "C9.2: Raises ValueError when empty"):
        passed += 1
    
    total += 1
    if check_present(buffer, r"-> Dict\[str, np\.ndarray\]:", "C9.3: Return type changed to Dict (not Optional)"):
        passed += 1
    
    # ========================================================================
    # Quick Win C10: Reward finite validation
    # ========================================================================
    print("\n[C10] Reward finite validation (environment.py)")
    print("-" * 70)
    
    env = read_file("environment.py")
    
    total += 1
    if check_present(env, r"if not np\.isfinite\(R_global\)", "C10.1: R_global finite check"):
        passed += 1
    
    total += 1
    if check_present(env, r"raise RuntimeError\(.*Invalid R_global computed", "C10.2: Raises on invalid R_global"):
        passed += 1
    
    total += 1
    if check_present(env, r"if not np\.isfinite\(R_local_mean\)", "C10.3: R_local_mean finite check"):
        passed += 1
    
    total += 1
    if check_present(env, r"if not np\.isfinite\(R_total\)", "C10.4: R_total finite check"):
        passed += 1
    
    # ========================================================================
    # Quick Win C11: Granular action fallback removed
    # ========================================================================
    print("\n[C11] Granular action fallback removed (rollout.py)")
    print("-" * 70)
    
    total += 1
    if check_absent(rollout, r"avail_batch\.append\(\[1\]\s*\*.*permissive", "C11.1: No [1]*N permissive fallback"):
        passed += 1
    
    total += 1
    if check_present(rollout, r"raise RuntimeError\(.*Cannot determine availability mask for granular", "C11.2: Raises on missing granular mask"):
        passed += 1
    
    # ========================================================================
    # Phase A(A): State shape validation
    # ========================================================================
    print("\n[A(A)] State shape validation (environment.py)")
    print("-" * 70)
    
    total += 1
    if check_present(env, r"expected_state_shape.*getattr.*state_shape", "A.1: Checks expected_state_shape"):
        passed += 1
    
    total += 1
    if check_present(env, r"raise ValueError\(.*State vector shape mismatch", "A.2: Raises on state shape mismatch"):
        passed += 1
    
    # ========================================================================
    # Phase A(A): Observation shape validation
    # ========================================================================
    print("\n[A(A)] Observation shape validation (rollout.py)")
    print("-" * 70)
    
    total += 1
    if check_present(rollout, r"if obs_dim is None:.*raise ValueError.*obs_shape must be explicitly specified", "A.3: Requires explicit obs_shape"):
        passed += 1
    
    total += 1
    if check_present(rollout, r"if o_tmp\.shape\[0\]\s*!=\s*n_agents:.*raise ValueError", "A.4: Validates agent count matches"):
        passed += 1
    
    total += 1
    if check_present(rollout, r"if o_tmp\.shape\[1\]\s*!=\s*obs_dim:.*raise ValueError", "A.5: Validates obs_dim matches"):
        passed += 1
    
    total += 1
    if check_absent(rollout, r"col_pad\s*=\s*np\.zeros.*obs_dim\s*-", "A.6: No observation padding logic"):
        passed += 1
    
    # ========================================================================
    # Phase A(B): Availability mask validation
    # ========================================================================
    print("\n[A(B)] Availability mask length validation (rollout.py)")
    print("-" * 70)
    
    total += 1
    if check_present(rollout, r"if n_actions is None:.*raise ValueError.*n_actions must be explicitly specified", "B.1: Requires explicit n_actions"):
        passed += 1
    
    total += 1
    if check_present(rollout, r"if a_tmp\.shape\[1\]\s*!=\s*n_actions:.*raise ValueError", "B.2: Validates mask length == n_actions"):
        passed += 1
    
    total += 1
    if check_present(rollout, r"if a_tmp\.shape\[0\]\s*!=\s*n_agents:.*raise ValueError", "B.3: Validates avail_batch agent count"):
        passed += 1
    
    total += 1
    if check_absent(rollout, r"pad_rows\s*=\s*np\.zeros.*n_agents\s*-.*avail", "B.4: No availability mask padding"):
        passed += 1
    
    # ========================================================================
    # Phase A(C): Hidden state management
    # ========================================================================
    print("\n[A(C)] Hidden state reset at episode start (rollout.py)")
    print("-" * 70)
    
    total += 1
    if check_present(rollout, r"Phase A\(C\):.*Reset RNN hidden state", "C.1: Comment indicates hidden reset"):
        passed += 1
    
    total += 1
    if check_present(rollout, r"if hasattr\(self\.agents.*policy.*init_hidden", "C.2: Checks for policy.init_hidden"):
        passed += 1
    
    total += 1
    if check_present(rollout, r"self\.agents\.policy\.init_hidden\(episode_num=1\)", "C.3: Calls init_hidden at episode start"):
        passed += 1
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY: {passed}/{total} checks passed")
    print("=" * 70)
    
    if passed == total:
        print("🎉 ALL VERIFICATIONS PASSED!")
        return 0
    else:
        print(f"❌ {total - passed} verification(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
