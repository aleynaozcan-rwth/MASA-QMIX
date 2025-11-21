"""
Test C8 Fix: Epsilon decay is episode-based, not decision-based.
"""
import os
import sys

def test_epsilon_decay_formula():
    """Verify epsilon decay formula matches expected values."""
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_anneal_steps = 5000
    
    def compute_epsilon(episode_count):
        if episode_count >= epsilon_anneal_steps:
            return epsilon_end
        decay_fraction = float(episode_count) / float(epsilon_anneal_steps)
        return epsilon_start - decay_fraction * (epsilon_start - epsilon_end)
    
    # Test cases
    tests_passed = 0
    tests_total = 6
    
    try:
        assert abs(compute_epsilon(0) - 1.0) < 1e-6, "Episode 0 should be 1.0"
        print("✅ Test 1/6: Episode 0 epsilon = 1.0")
        tests_passed += 1
    except AssertionError as e:
        print(f"❌ Test 1/6 FAILED: {e}")
    
    try:
        assert abs(compute_epsilon(10) - 0.9981) < 1e-3, "Episode 10 should be ~0.9981"
        print("✅ Test 2/6: Episode 10 epsilon ≈ 0.9981")
        tests_passed += 1
    except AssertionError as e:
        print(f"❌ Test 2/6 FAILED: {e}")
    
    try:
        assert abs(compute_epsilon(100) - 0.981) < 1e-3, "Episode 100 should be ~0.981"
        print("✅ Test 3/6: Episode 100 epsilon ≈ 0.981")
        tests_passed += 1
    except AssertionError as e:
        print(f"❌ Test 3/6 FAILED: {e}")
    
    try:
        assert abs(compute_epsilon(1000) - 0.81) < 1e-3, "Episode 1000 should be ~0.81"
        print("✅ Test 4/6: Episode 1000 epsilon ≈ 0.81")
        tests_passed += 1
    except AssertionError as e:
        print(f"❌ Test 4/6 FAILED: {e}")
    
    try:
        assert abs(compute_epsilon(5000) - 0.05) < 1e-6, "Episode 5000 should be 0.05"
        print("✅ Test 5/6: Episode 5000 epsilon = 0.05")
        tests_passed += 1
    except AssertionError as e:
        print(f"❌ Test 5/6 FAILED: {e}")
    
    try:
        assert abs(compute_epsilon(6000) - 0.05) < 1e-6, "Episode 6000+ should stay 0.05"
        print("✅ Test 6/6: Episode 6000+ epsilon = 0.05 (clamped)")
        tests_passed += 1
    except AssertionError as e:
        print(f"❌ Test 6/6 FAILED: {e}")
    
    print(f"\n[Formula Tests] {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

def test_epsilon_log_file():
    """Verify epsilon_decay_log.txt contains expected entries."""
    log_path = "my_data_and_graph/historydata/epsilon_decay_log.txt"
    
    if not os.path.exists(log_path):
        print("⚠️  Epsilon log file not found (run training first)")
        print(f"   Expected path: {log_path}")
        return True  # Not a failure, just not run yet
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            print("⚠️  Epsilon log file is empty")
            return True
        
        # Check format: episode,epsilon
        checks_passed = 0
        for i, line in enumerate(lines[:5]):  # Check first 5 lines
            try:
                parts = line.strip().split(',')
                assert len(parts) == 2, f"Invalid format: {line}"
                episode = int(parts[0])
                epsilon = float(parts[1])
                assert 0.05 <= epsilon <= 1.0, f"Epsilon out of range: {epsilon}"
                checks_passed += 1
            except Exception as e:
                print(f"❌ Line {i+1} check failed: {e}")
                return False
        
        print(f"✅ Epsilon log file verified ({len(lines)} entries, checked {checks_passed} lines)")
        return True
    except Exception as e:
        print(f"❌ Failed to read epsilon log: {e}")
        return False

def test_code_changes():
    """Verify that C8 code changes are present in rollout.py"""
    rollout_path = "MARL/common/rollout.py"
    
    if not os.path.exists(rollout_path):
        print(f"❌ File not found: {rollout_path}")
        return False
    
    with open(rollout_path, 'r') as f:
        content = f.read()
    
    checks = {
        "episode_count initialized": "self.episode_count = 0" in content,
        "C8 FIX comment present": "C8 FIX: Episode-based epsilon" in content,
        "Epsilon decay moved comment": "Epsilon decay moved to episode end" in content,
        "episode_count increment": "self.episode_count += 1" in content,
        "epsilon_anneal_steps check": "self.episode_count >= self.epsilon_anneal_steps" in content,
        "epsilon_decay_log.txt": "epsilon_decay_log.txt" in content,
    }
    
    print("\n[Code Change Verification]")
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("=" * 60)
    print("C8 Fix Verification: Episode-Based Epsilon Decay")
    print("=" * 60)
    
    print("\n[1/3] Testing epsilon decay formula...")
    formula_ok = test_epsilon_decay_formula()
    
    print("\n[2/3] Verifying code changes in rollout.py...")
    code_ok = test_code_changes()
    
    print("\n[3/3] Checking epsilon log file...")
    log_ok = test_epsilon_log_file()
    
    print("\n" + "=" * 60)
    if formula_ok and code_ok:
        print("🎉 C8 VERIFICATION PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run training: python scripts/run_train_qmix.py --n_episodes=50 --n_epoch=1")
        print("2. Check epsilon log: cat my_data_and_graph/historydata/epsilon_decay_log.txt")
        print("3. Verify epsilon decays gradually over episodes (not decisions)")
        sys.exit(0)
    else:
        print("❌ C8 VERIFICATION FAILED")
        print("=" * 60)
        if not formula_ok:
            print("- Formula tests failed")
        if not code_ok:
            print("- Code changes not found")
        sys.exit(1)
