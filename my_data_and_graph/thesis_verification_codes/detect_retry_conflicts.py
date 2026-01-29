#!/usr/bin/env python3
"""
Detect real conflicts: When the same job-operation appears at multiple decision points,
it means the job failed to get resources and had to retry (real conflict).
"""

import re
from collections import defaultdict

TIMELINE_FILE = 'my_data_and_graph/historydata/scheduling_timeline.txt'

def parse_decision_points_and_allocations():
    """
    Parse timeline to track:
    1. When each (job, operation) reaches a decision point
    2. Whether it successfully got allocated (started)
    """
    
    episodes = []
    current_episode = None
    decision_points = []  # (timestamp, job_id, operation)
    allocations = []  # (timestamp, job_id, operation, machine)
    in_lifecycle = False
    in_timeline = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                if current_episode is not None:
                    episodes.append({
                        'episode': current_episode,
                        'decision_points': decision_points.copy(),
                        'allocations': allocations.copy()
                    })
                
                match = re.search(r'EPISODE (\d+)', line)
                current_episode = int(match.group(1)) if match else None
                decision_points = []
                allocations = []
                in_lifecycle = False
                in_timeline = False
                continue
            
            if 'LIFECYCLE TRACE START' in line:
                in_lifecycle = True
                continue
            if 'LIFECYCLE TRACE END' in line:
                in_lifecycle = False
                continue
            if '=== TIMELINE ===' in line:
                in_timeline = True
                continue
            
            # Decision points from completions
            if not in_lifecycle and '[t=' in line and 'finished' in line and 'next queued' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'Job_(\d+)\.Op(\d+) finished', line)
                if time_match and job_match:
                    timestamp = float(time_match.group(1))
                    job_id = int(job_match.group(1))
                    completed_op = int(job_match.group(2))
                    next_op = completed_op + 1
                    decision_points.append((timestamp, job_id, next_op))
                continue
            
            # Allocations (when operation actually starts)
            if in_timeline and '[t=' in line and 'started on' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'Job_(\d+)\.Op(\d+) started', line)
                machine_match = re.search(r'started on (M\d+)', line)
                if time_match and job_match and machine_match:
                    timestamp = float(time_match.group(1))
                    job_id = int(job_match.group(1))
                    operation = int(job_match.group(2))
                    machine = machine_match.group(1)
                    allocations.append((timestamp, job_id, operation, machine))
                continue
    
    # Last episode
    if current_episode is not None:
        episodes.append({
            'episode': current_episode,
            'decision_points': decision_points.copy(),
            'allocations': allocations.copy()
        })
    
    return episodes


def detect_retry_conflicts(episodes):
    """
    Detect conflicts: same (job, operation) appearing at multiple decision points
    before getting allocated. This means the job tried and failed to get resources.
    """
    
    total_conflicts = 0
    episodes_with_conflicts = 0
    conflict_examples = []
    retry_stats = defaultdict(int)  # Track how many times jobs retry
    
    for ep_data in episodes:
        ep_num = ep_data['episode']
        dps = ep_data['decision_points']
        allocs = ep_data['allocations']
        
        # Build allocation lookup: (job, op) -> timestamp when it started
        allocation_times = {}
        for timestamp, job_id, operation, machine in allocs:
            allocation_times[(job_id, operation)] = timestamp
        
        # Group decision points by (job, operation)
        job_op_decisions = defaultdict(list)
        for timestamp, job_id, operation in dps:
            job_op_decisions[(job_id, operation)].append(timestamp)
        
        # Find conflicts: (job, op) with multiple decision points before allocation
        episode_conflicts = 0
        for (job_id, operation), timestamps in job_op_decisions.items():
            if len(timestamps) > 1:
                # This job-operation reached decision point multiple times
                # Check if it's before allocation
                alloc_time = allocation_times.get((job_id, operation), float('inf'))
                
                # Count how many decision points happened before allocation
                failed_attempts = [t for t in timestamps if t < alloc_time]
                
                if len(failed_attempts) > 1:
                    # Multiple decision points before allocation = CONFLICT!
                    episode_conflicts += 1
                    total_conflicts += 1
                    retry_count = len(failed_attempts) - 1  # First one isn't a retry
                    retry_stats[retry_count] += 1
                    
                    if len(conflict_examples) < 15:
                        conflict_examples.append({
                            'episode': ep_num,
                            'job': job_id,
                            'operation': operation,
                            'decision_times': failed_attempts,
                            'allocation_time': alloc_time if alloc_time != float('inf') else None,
                            'retry_count': retry_count
                        })
        
        if episode_conflicts > 0:
            episodes_with_conflicts += 1
    
    return total_conflicts, episodes_with_conflicts, retry_stats, conflict_examples


def main():
    print("="*70)
    print("REAL CONFLICT DETECTION: FAILED RESOURCE ALLOCATION ATTEMPTS")
    print("="*70)
    
    print("\nParsing timeline data...")
    episodes = parse_decision_points_and_allocations()
    print(f"Parsed {len(episodes)} episodes")
    
    print("\nAnalyzing retry conflicts...")
    total_conflicts, ep_with_conflicts, retry_stats, examples = detect_retry_conflicts(episodes)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nTotal episodes: {len(episodes):,}")
    print(f"Episodes with retry conflicts: {ep_with_conflicts:,} ({100*ep_with_conflicts/len(episodes):.1f}%)")
    print(f"Total retry conflicts detected: {total_conflicts:,}")
    print(f"Average conflicts per episode: {total_conflicts/len(episodes):.2f}")
    
    print("\n" + "-"*70)
    print("RETRY DISTRIBUTION")
    print("-"*70)
    print("Number of retries → Occurrences:")
    for retry_count in sorted(retry_stats.keys()):
        print(f"  {retry_count} retry(ies): {retry_stats[retry_count]:,} conflicts")
    
    if examples:
        print("\n" + "="*70)
        print(f"CONFLICT EXAMPLES (First {len(examples)})")
        print("="*70)
        for i, ex in enumerate(examples, 1):
            print(f"\n{i}. Episode {ex['episode']}: Job_{ex['job']} Op{ex['operation']}")
            print(f"   Decision attempts at: {[f't={t:.2f}' for t in ex['decision_times']]}")
            if ex['allocation_time']:
                print(f"   Finally allocated at: t={ex['allocation_time']:.2f}")
            else:
                print(f"   Never allocated (job might have completed via different path)")
            print(f"   → {ex['retry_count']} retry(ies) due to resource conflicts")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"\n✓ {100*ep_with_conflicts/len(episodes):.1f}% of episodes have resource allocation conflicts")
    print(f"✓ {total_conflicts:,} total instances where jobs had to retry due to unavailable resources")
    print(f"✓ Average {total_conflicts/len(episodes):.2f} conflicts per episode")
    print("\nThese retries demonstrate:")
    print("  - Real resource contention between agents")
    print("  - Jobs competing for same machines/operators")
    print("  - Need for multi-agent coordination to minimize conflicts")
    print("  - QMIX learns optimal resource allocation strategies to reduce retries")


if __name__ == '__main__':
    main()
