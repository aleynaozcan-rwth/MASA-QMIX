#!/usr/bin/env python3
"""Test task generator fix - ensures no duplicate operations within a job."""

import sys
import random
sys.path.insert(0, '/home/cc253232/MASA-QMIX')

from utils.task_generator import TaskGenerator
from utils.workcenter import WorkCenters
from utils.job import Jobs

def test_no_duplicates_in_job():
    """Test that a single job doesn't contain duplicate operation types."""
    
    # Create task generator with fixed seed for reproducibility
    seed = 42
    py_rng = random.Random(seed)
    task_gen = TaskGenerator(py_rng=py_rng)
    
    print(f"Testing with seed={seed}")
    print(f"Available operations: {len(task_gen.jobs.jobs_object_list)}")
    
    # Generate multiple jobs with different operation counts
    for job_num in range(10):
        num_ops = random.randint(2, 5)
        print(f"\n--- Job {job_num} (requesting {num_ops} ops) ---")
        
        try:
            ops_sequence = task_gen.generate_constrained_task(num_ops=num_ops)
            
            # Extract operation IDs
            op_ids = [op.index_id for op in ops_sequence]
            print(f"Generated ops: {op_ids}")
            
            # Check for duplicates
            if len(op_ids) != len(set(op_ids)):
                print(f"❌ ERROR: Duplicate operations found in job {job_num}!")
                print(f"   Operations: {op_ids}")
                print(f"   Unique: {set(op_ids)}")
                duplicates = [op for op in op_ids if op_ids.count(op) > 1]
                print(f"   Duplicates: {set(duplicates)}")
                return False
            else:
                print(f"✓ OK: All operations unique")
                
        except Exception as e:
            print(f"❌ ERROR generating job {job_num}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED: No duplicate operations within any job")
    print("="*50)
    return True

if __name__ == '__main__':
    success = test_no_duplicates_in_job()
    sys.exit(0 if success else 1)
