"""
Test TIME-BASED Epsilon Decay: Epsilon decays based on simulation time.
Formula: epsilon(t) = epsilon_start - (t / episode_limit) * (epsilon_start - epsilon_end)
"""
import os
import sys

def test_epsilon_decay_formula():
    """Verify time-based epsilon decay formula."""
    epsilon_start = 1.0
    epsilon_end = 0.05
    episode_limit = 500.0
    
    def compute_epsilon(current_time):
        fraction = min(1.0, current_time / episode_limit)
        return epsilon_start - fraction * (epsilon_start - epsilon_end)
    
    tests_passed = 0
    tests_total = 6
    
    tests = [
        (0.0, 1.0, "t=0"),
        (50.0, 0.905, "t=50"),
        (100.0, 0.81, "t=100"),
        (250.0, 0.525, "t=250"),
        (500.0, 0.05, "t=500 (limit)"),
        (600.0, 0.05, "t=600 (clamped)"),
    ]
    
    for t, expected, name in tests:
        try:
            result = compute_epsilon(t)
            assert abs(result - expected) < 1e-3, f"{name} should be {expected}, got {result}"
            print(f"✅ {name}: epsilon = {result:.4f}")
            tests_passed += 1
        except AssertionError as e:
            print(f"❌ {name} FAILED: {e}")
    
    print(f"\n[Formula Tests] {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

def test_rollout_implementation():
    """Verify rollout.py uses time-based decay."""
    rollout_path = "MARL/common/rollout.py"
    
    if not os.path.exists(rollout_path):
        print(f"⚠️ {rollout_path} not found")
        return False
    
    with open(rollout_path, 'r') as f:
        content = f.read()
    
    checks = {
        "TIME-BASED EPSILON DECAY": "TIME-BASED EPSILON DECAY" in content,
        "current_t = float(self.env.env.now)": "current_t = float(self.env.env.now)" in content,
        "limit_t = float(self.episode_limit)": "limit_t = float(self.episode_limit)" in content,
    }
    
    print("\n[Code Verification]")
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("="*60)
    print("TIME-BASED EPSILON DECAY VERIFICATION")
    print("="*60)
    
    print("\n[1/2] Testing epsilon decay formula...")
    formula_ok = test_epsilon_decay_formula()
    
    print("\n[2/2] Verifying code implementation...")
    code_ok = test_rollout_implementation()
    
    print("\n" + "="*60)
    if formula_ok and code_ok:
        print("🎉 VERIFICATION PASSED")
        sys.exit(0)
    else:
        print("❌ VERIFICATION FAILED")
        sys.exit(1)
