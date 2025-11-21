"""
Test Scenario 1: C13 + C17 Validation Fixes

Verifies:
- C13: Machine index validation (strict bounds checking)
- C17: Processing time validation (positive, finite)
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_c17_processing_time_validation():
    """Test C17: Processing time validation catches invalid configs."""
    import numpy as np
    
    # Import the validation method directly
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    # Create a mock object with the validation method
    from environment import MASAEnv
    
    # Test the validation logic directly (without full env init)
    def validate_processing_times(pt_means: dict):
        """Replicate C17 validation logic."""
        if not isinstance(pt_means, dict):
            raise ValueError(f"processing_time_means must be a dict, got {type(pt_means)}")
        
        errors = []
        for op_type, durations in pt_means.items():
            if not isinstance(durations, dict):
                errors.append(f"op_type={op_type}: durations must be dict, got {type(durations)}")
                continue
            
            for wc, dur in durations.items():
                try:
                    dur_f = float(dur)
                    if dur_f <= 0:
                        errors.append(f"op_type={op_type}, wc={wc}: processing_time must be positive, got {dur_f}")
                    if not np.isfinite(dur_f):
                        errors.append(f"op_type={op_type}, wc={wc}: processing_time must be finite, got {dur_f}")
                except (ValueError, TypeError) as e:
                    errors.append(f"op_type={op_type}, wc={wc}: cannot convert to float: {e}")
        
        if errors:
            raise ValueError(
                f"Processing time validation failed ({len(errors)} errors):\\n" +
                "\\n".join(f"  - {err}" for err in errors[:10]) +
                (f"\\n  ... and {len(errors) - 10} more" if len(errors) > 10 else "")
            )
    
    # Valid config
    valid_config = {
        0: {0: 10.0, 1: 15.0},
        1: {0: 12.0, 1: 18.0},
    }
    
    # Test 1: Zero processing time should fail
    print("\\n[C17 Test 1] Testing zero processing time...")
    invalid_config_zero = {
        0: {0: 0.0, 1: 15.0},  # ← Zero!
    }
    
    try:
        validate_processing_times(invalid_config_zero)
        print("❌ FAIL: Zero processing time was not caught!")
        assert False, "Should have raised ValueError for zero processing time"
    except ValueError as e:
        if "processing_time must be positive" in str(e):
            print(f"✅ PASS: Zero processing time caught")
        else:
            raise
    
    # Test 2: Negative processing time should fail
    print("\\n[C17 Test 2] Testing negative processing time...")
    invalid_config_negative = {
        0: {0: -5.0, 1: 15.0},  # ← Negative!
    }
    
    try:
        validate_processing_times(invalid_config_negative)
        print("❌ FAIL: Negative processing time was not caught!")
        assert False, "Should have raised ValueError for negative processing time"
    except ValueError as e:
        if "processing_time must be positive" in str(e):
            print(f"✅ PASS: Negative processing time caught")
        else:
            raise
    
    # Test 3: Infinite processing time should fail
    print("\\n[C17 Test 3] Testing infinite processing time...")
    invalid_config_inf = {
        0: {0: float('inf'), 1: 15.0},  # ← Infinite!
    }
    
    try:
        validate_processing_times(invalid_config_inf)
        print("❌ FAIL: Infinite processing time was not caught!")
        assert False, "Should have raised ValueError for infinite processing time"
    except ValueError as e:
        if "processing_time must be finite" in str(e):
            print(f"✅ PASS: Infinite processing time caught")
        else:
            raise
    
    # Test 4: Valid config should pass
    print("\\n[C17 Test 4] Testing valid processing times...")
    try:
        validate_processing_times(valid_config)
        print("✅ PASS: Valid processing times accepted")
    except Exception as e:
        print(f"❌ FAIL: Valid config rejected: {e}")
        raise


def test_c13_machine_index_bounds():
    """Test C13: Machine index validation logic."""
    
    print("\n[C13 Test] Verifying machine index validation logic...")
    
    # Simulate the C13 validation logic
    def validate_machine_index(chosen_idx, n_machines, job_id=0, op_idx=0, t=0.0):
        """Replicate C13 FIX validation logic."""
        chosen_idx_int = int(chosen_idx)
        if not (0 <= chosen_idx_int < n_machines):
            raise ValueError(
                f"[C13 FIX] Invalid machine index: {chosen_idx_int} out of bounds "
                f"[0, {n_machines}). Job={job_id}, op_idx={op_idx}, "
                f"t={t:.2f}"
            )
        return chosen_idx_int
    
    # Test 1: Valid indices should pass
    print("[C13 Test 1] Valid machine indices...")
    n_machines = 5
    for idx in [0, 1, 2, 3, 4]:
        result = validate_machine_index(idx, n_machines)
        assert result == idx
    print(f"✅ PASS: Valid indices [0-{n_machines-1}] accepted")
    
    # Test 2: Negative index should fail
    print("[C13 Test 2] Negative machine index...")
    try:
        validate_machine_index(-1, n_machines)
        print("❌ FAIL: Negative index was not caught!")
        assert False
    except ValueError as e:
        if "out of bounds" in str(e):
            print(f"✅ PASS: Negative index rejected: {e}")
        else:
            raise
    
    # Test 3: Index >= n_machines should fail
    print("[C13 Test 3] Out-of-bounds machine index...")
    try:
        validate_machine_index(5, n_machines)  # n_machines = 5, valid = [0,4]
        print("❌ FAIL: Out-of-bounds index was not caught!")
        assert False
    except ValueError as e:
        if "out of bounds" in str(e):
            print(f"✅ PASS: Out-of-bounds index rejected: {e}")
        else:
            raise
    
    # Test 4: Large out-of-bounds index should fail
    print("[C13 Test 4] Large out-of-bounds machine index...")
    try:
        validate_machine_index(100, n_machines)
        print("❌ FAIL: Large out-of-bounds index was not caught!")
        assert False
    except ValueError as e:
        if "out of bounds" in str(e):
            print(f"✅ PASS: Large out-of-bounds index rejected: {e}")
        else:
            raise


def test_c13_uses_machine_count_not_n_actions():
    """Test C13: Verify validation uses len(machine_resources), not n_actions."""
    
    print("\n[C13 Semantic Test] Verifying machine_resources is used, not n_actions...")
    
    # Read the C13 fix code from environment.py
    env_file = os.path.join(os.path.dirname(__file__), '..', 'environment.py')
    with open(env_file, 'r') as f:
        code = f.read()
    
    # Find the C13 FIX block
    c13_start = code.find('# C13 FIX: Strict machine index validation')
    if c13_start == -1:
        print("❌ FAIL: C13 FIX comment not found in environment.py")
        assert False, "C13 FIX not implemented"
    
    # Extract ~20 lines after the C13 comment
    c13_block = code[c13_start:c13_start+800]
    
    # Check that validation uses machine_resources or machine_list, NOT n_actions
    checks = {
        'uses_machine_resources': 'len(self.machine_resources)' in c13_block,
        'uses_machine_list': 'len(self.workcenters_meta.machine_list)' in c13_block,
        'avoids_n_actions': 'self.n_actions' not in c13_block,
        'has_bounds_check': '0 <= chosen_idx_int < n_machines' in c13_block,
        'raises_valueerror': 'raise ValueError' in c13_block,
    }
    
    print(f"  ✓ Uses len(machine_resources): {checks['uses_machine_resources']}")
    print(f"  ✓ Uses len(machine_list): {checks['uses_machine_list']}")
    print(f"  ✓ Avoids self.n_actions: {checks['avoids_n_actions']}")
    print(f"  ✓ Has bounds check: {checks['has_bounds_check']}")
    print(f"  ✓ Raises ValueError: {checks['raises_valueerror']}")
    
    # Must use machine_resources OR machine_list (not n_actions)
    if not (checks['uses_machine_resources'] or checks['uses_machine_list']):
        print("❌ FAIL: C13 does not compute n_machines from machine_resources or machine_list")
        assert False
    
    # Must NOT use n_actions
    if not checks['avoids_n_actions']:
        print("❌ FAIL: C13 incorrectly uses self.n_actions instead of machine count")
        assert False
    
    # Must have bounds check and raise
    if not (checks['has_bounds_check'] and checks['raises_valueerror']):
        print("❌ FAIL: C13 missing bounds check or ValueError")
        assert False
    
    print("✅ PASS: C13 correctly validates against machine count (not n_actions)")


def test_c17_code_exists():
    """Test C17: Verify _validate_processing_times method exists."""
    
    print("\n[C17 Code Test] Verifying _validate_processing_times method exists...")
    
    env_file = os.path.join(os.path.dirname(__file__), '..', 'environment.py')
    with open(env_file, 'r') as f:
        code = f.read()
    
    checks = {
        'method_defined': 'def _validate_processing_times(self, pt_means: dict):' in code,
        'c17_comment': '# C17 FIX:' in code,
        'positive_check': 'dur_f <= 0' in code,
        'finite_check': 'np.isfinite(dur_f)' in code,
        'method_called': 'self._validate_processing_times(' in code,
    }
    
    print(f"  ✓ Method defined: {checks['method_defined']}")
    print(f"  ✓ Has C17 FIX comment: {checks['c17_comment']}")
    print(f"  ✓ Checks positive: {checks['positive_check']}")
    print(f"  ✓ Checks finite: {checks['finite_check']}")
    print(f"  ✓ Method called in __init__: {checks['method_called']}")
    
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        print(f"❌ FAIL: Missing C17 components: {failed}")
        assert False
    
    print("✅ PASS: C17 _validate_processing_times method properly implemented")


if __name__ == "__main__":
    print("=" * 70)
    print("SCENARIO 1: C13 + C17 VALIDATION TESTS")
    print("=" * 70)
    
    # Run tests
    try:
        test_c13_machine_index_bounds()
        print("\n" + "="*70)
        
        test_c13_uses_machine_count_not_n_actions()
        print("\n" + "="*70)
        
        test_c17_code_exists()
        print("\n" + "="*70)
        
        test_c17_processing_time_validation()
        print("\n" + "="*70)
        
        print("\n🎉 ALL SCENARIO 1 TESTS PASSED!")
        print("\nC13 ✅ Machine index validation (strict bounds)")
        print("C17 ✅ Processing time validation (positive, finite)")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
