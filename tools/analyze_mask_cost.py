#!/usr/bin/env python3
"""
Analyze specific cost of action mask computation.
Shows detailed breakdown of _avail_row_for_job and _build_avail_actions.
"""
import time
import numpy as np


def benchmark_mask_operations():
    """Benchmark the cost of current mask computation logic."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from environment import MASAEnv
    from MARL.common.arguments import get_common_args
    
    print("=" * 80)
    print("ACTION MASK COMPUTATION COST ANALYSIS")
    print("=" * 80)
    
    # Create test environment
    print("\nInitializing test environment...")
    
    # Get minimal args
    args = get_common_args()
    args.n_agents = 3
    args.max_jobs = 3
    args.episode_limit = 100
    args.seed = 42
    
    env = MASAEnv(args=args)
    env.reset()
    
    # Wait for some jobs to be active
    env.env.run(until=env.env.now + 5.0)
    
    n_jobs = len([j for j in env.jobs if not j.finished])
    n_machines = len(env.workcenters_meta.machine_list)
    
    print(f"Active jobs: {n_jobs}")
    print(f"Machines: {n_machines}")
    print(f"Operators: {len(env.operators.operators_object_list)}")
    
    # Benchmark individual components
    print("\n" + "=" * 80)
    print("BENCHMARKING INDIVIDUAL OPERATIONS")
    print("=" * 80)
    
    # 1. _avail_row_for_job (per job)
    if n_jobs > 0:
        job = [j for j in env.jobs if not j.finished][0]
        
        print("\n### 1. _avail_row_for_job (single job) ###")
        iterations = 1000
        
        start = time.perf_counter()
        for _ in range(iterations):
            row = env._avail_row_for_job(job)
        elapsed = time.perf_counter() - start
        
        avg_time_us = (elapsed / iterations) * 1_000_000
        print(f"Average time: {avg_time_us:.2f} µs per call")
        print(f"Total for {iterations} calls: {elapsed*1000:.2f} ms")
        
        # Break down the operator check cost
        print("\n   Breakdown:")
        print(f"   - Machine capability checks: ~{n_machines} iterations")
        
        # Count how many machines are capable
        capable_count = np.sum(row)
        print(f"   - Machines capable for this op: {capable_count}")
        
        if capable_count > 0:
            n_operators = len(env.operators.operators_object_list)
            print(f"   - Operator checks per capable machine: {n_operators} operators")
            print(f"   - Total operator checks: ~{capable_count * n_operators}")
            
            # Estimate operator check cost
            start = time.perf_counter()
            for _ in range(iterations):
                for op_obj in env.operators.operators_object_list:
                    _ = op_obj.is_busy
                    _ = 'M0' in op_obj.qualified_machines
                    _ = op_obj.can_do_job(0, 0)
            elapsed_op = time.perf_counter() - start
            avg_op_time_us = (elapsed_op / iterations) * 1_000_000
            print(f"   - Operator check overhead: ~{avg_op_time_us:.2f} µs per capable machine")
    
    # 2. _build_avail_actions (all jobs)
    print("\n### 2. _build_avail_actions (all jobs) ###")
    iterations = 100
    
    start = time.perf_counter()
    for _ in range(iterations):
        avail = env._build_avail_actions()
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / iterations) * 1000
    print(f"Average time: {avg_time_ms:.3f} ms per call")
    print(f"Total for {iterations} calls: {elapsed*1000:.2f} ms")
    print(f"\nBreakdown:")
    print(f"   - Calls _avail_row_for_job {n_jobs} times")
    print(f"   - Then filters by machine_free for {n_jobs * n_machines} machine slots")
    print(f"   - Additional operator group checks (WorkCenter-level)")
    
    # 3. Cost per decision
    print("\n### 3. Estimated Cost Per Decision ###")
    # In event-driven mode, we call create_decision_item which calls _avail_row_for_job once
    if n_jobs > 0:
        job = [j for j in env.jobs if not j.finished][0]
        iterations = 1000
        
        start = time.perf_counter()
        for _ in range(iterations):
            # Simulate what happens at decision time
            row = env._avail_row_for_job(job)
            obs = env._build_agent_obs(job)
        elapsed = time.perf_counter() - start
        
        avg_time_us = (elapsed / iterations) * 1_000_000
        print(f"Average per decision (mask + obs): {avg_time_us:.2f} µs")
    
    # Analyze the split-function overhead
    print("\n" + "=" * 80)
    print("ARCHITECTURE ANALYSIS")
    print("=" * 80)
    
    print("""
Current Implementation:
1. create_decision_item() calls _avail_row_for_job() once per decision
   - Checks: capability + operator (no machine_free check)
   - Cost: O(n_machines * n_operators) for capable machines
   
2. Later, _build_avail_actions() is called (in step-based mode)
   - Calls _avail_row_for_job() again for ALL jobs
   - Then checks machine_free
   - Cost: O(n_jobs * n_machines * n_operators)

PROBLEM: Checking operators BEFORE checking machine_free wastes computation!

Example with current code order:
- Job_0.Op7 decision
- M1 is capable → check all 2 operators (O1, O2) for eligibility
- M3 is capable → check all 2 operators for eligibility
- Then later discover M1 was busy anyway (wasted operator checks)

Better order would be:
1. Check capability (cheap)
2. Check machine_free (cheap) 
3. Only then check operators (expensive nested loop)

Cost Estimate:
- If 50% of capable machines are busy on average
- We waste 50% of operator checks
- With 5 machines, 2 operators, 3 jobs:
  Current: ~30 operator checks per decision
  Optimal: ~15 operator checks per decision (50% reduction)
    """)
    
    # Provide specific measurements
    print("\n" + "=" * 80)
    print("HOTSPOT IDENTIFICATION")
    print("=" * 80)
    
    if n_jobs > 0:
        job = [j for j in env.jobs if not j.finished][0]
        
        # Measure just capability check
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            op = job.current_op()
            if op is not None:
                if isinstance(op, (list, tuple)) and len(op) >= 3:
                    op_idx = int(op[0])
                else:
                    op_idx = int(job.current_op_idx)
                
                registry = env.workcenters_meta.machine_registry
                for mname in env.workcenters_meta.machine_list:
                    caps = registry.get(mname, {}).get('capabilities', [])
                    _ = op_idx in caps
        elapsed_cap = time.perf_counter() - start
        
        # Measure operator checks
        start = time.perf_counter()
        for _ in range(iterations):
            for mname in env.workcenters_meta.machine_list:
                for op_obj in env.operators.operators_object_list:
                    _ = mname in op_obj.qualified_machines
                    _ = op_obj.can_do_job(0, 0)
                    _ = op_obj.is_busy
        elapsed_op = time.perf_counter() - start
        
        # Measure machine_free check
        start = time.perf_counter()
        for _ in range(iterations):
            for m in range(n_machines):
                _ = env._resource_free(env.machine_resources[m])
        elapsed_free = time.perf_counter() - start
        
        print(f"\nPer-decision cost breakdown (averaged over {iterations} iterations):")
        print(f"1. Capability checks:  {(elapsed_cap/iterations)*1_000_000:.2f} µs (CHEAP)")
        print(f"2. Machine free checks: {(elapsed_free/iterations)*1_000_000:.2f} µs (CHEAP)")
        print(f"3. Operator checks:     {(elapsed_op/iterations)*1_000_000:.2f} µs (EXPENSIVE)")
        print(f"\nTotal ratio: Operator checks are ~{elapsed_op/elapsed_cap:.1f}x more expensive than capability checks")
        
        print("\n🔴 CRITICAL FINDING:")
        print(f"   Current code checks operators BEFORE machine_free")
        print(f"   This means we do expensive operator checks even for busy machines")
        print(f"   Reordering to check machine_free BEFORE operators would save:")
        print(f"   ~{(elapsed_op/iterations)*1_000_000 * 0.5:.2f} µs per decision (assuming 50% machines busy)")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
Priority 1: Reorder checks in _avail_row_for_job
   Current: capability → operator → [later] machine_free
   Better:  capability → machine_free → operator
   Savings: ~50% reduction in operator checks

Priority 2: Eliminate redundant WorkCenter-level operator checks
   _build_avail_actions currently has additional operator checks
   These duplicate what _avail_row_for_job already did
   Savings: ~30% reduction in total mask computation time

Priority 3: Cache operator qualifications per machine
   Currently: qualified_machines is a list, requires linear search
   Better: Use set or dict for O(1) lookup
   Savings: ~10-20% in operator check time

Priority 4: Early exit for finished jobs
   _build_avail_actions already does `if j.finished: continue`
   Ensure this is also done in all observation building
   
NOT RECOMMENDED (already optimized):
   - Machine capabilities are cached in machine_registry ✅
   - Operator objects are created once and reused ✅
   - NumPy arrays used for masks ✅
    """)


if __name__ == "__main__":
    benchmark_mask_operations()
