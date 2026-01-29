#!/usr/bin/env python3
"""
Analyze conflicts at parallel decision points.
A conflict occurs when the same job's same operation appears at multiple decision points,
indicating that a resource was taken away (preemption).
"""

import re
from collections import defaultdict

TIMELINE_FILE = 'my_data_and_graph/historydata/scheduling_timeline.txt'

def parse_decision_points_with_details():
    """Parse timeline and extract detailed information about each decision point."""
    
    episodes = []
    current_episode = None
    decision_points = []  # List of (timestamp, job_id, operation, event_type)
    in_lifecycle = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Episode marker
            if '=== EPISODE' in line:
                if current_episode is not None and decision_points:
                    episodes.append({
                        'episode': current_episode,
                        'decision_points': decision_points.copy()
                    })
                
                match = re.search(r'EPISODE (\d+)', line)
                current_episode = int(match.group(1)) if match else None
                decision_points = []
                in_lifecycle = False
                continue
            
            if 'LIFECYCLE TRACE START' in line:
                in_lifecycle = True
                continue
            if 'LIFECYCLE TRACE END' in line:
                in_lifecycle = False
                continue
            
            # Job arrivals (decision point for new job)
            if in_lifecycle and '[t=' in line and 'New job' in line and 'arrived' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'job (\d+) arrived', line)
                if time_match and job_match:
                    timestamp = time_match.group(1)
                    job_id = int(job_match.group(1))
                    decision_points.append((timestamp, job_id, 'Op1', 'arrival'))
                continue
            
            # Operation completions (decision point for next operation)
            if not in_lifecycle and '[t=' in line and 'finished' in line and 'next queued' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'Job_(\d+)\.Op(\d+) finished', line)
                if time_match and job_match:
                    timestamp = time_match.group(1)
                    job_id = int(job_match.group(1))
                    completed_op = int(job_match.group(2))
                    next_op = completed_op + 1  # Assuming sequential operations
                    decision_points.append((timestamp, job_id, f'Op{next_op}', 'completion'))
                continue
    
    # Save last episode
    if current_episode is not None and decision_points:
        episodes.append({
            'episode': current_episode,
            'decision_points': decision_points.copy()
        })
    
    return episodes


def analyze_conflicts_at_parallel_points(episodes):
    """
    Analyze conflicts at parallel decision points.
    A conflict is when the same (job, operation) pair appears multiple times,
    indicating preemption/resource reallocation.
    """
    
    total_parallel_dps = 0
    parallel_with_conflicts = 0
    conflict_details = []
    
    for ep_data in episodes:
        ep_num = ep_data['episode']
        dps = ep_data['decision_points']
        
        # Group by timestamp
        timestamp_groups = defaultdict(list)
        for timestamp, job_id, operation, event_type in dps:
            timestamp_groups[timestamp].append((job_id, operation, event_type))
        
        # Find parallel decision points (same timestamp, multiple DPs)
        for timestamp, dp_list in timestamp_groups.items():
            if len(dp_list) > 1:  # Parallel decision point
                total_parallel_dps += 1
                
                # Check for conflicts: same (job, operation) appearing multiple times
                job_op_pairs = [(job_id, operation) for job_id, operation, _ in dp_list]
                
                # Count occurrences
                pair_counts = defaultdict(int)
                for pair in job_op_pairs:
                    pair_counts[pair] += 1
                
                # If any pair appears more than once, it's a conflict
                has_conflict = any(count > 1 for count in pair_counts.values())
                
                if has_conflict:
                    parallel_with_conflicts += 1
                    conflict_pairs = [pair for pair, count in pair_counts.items() if count > 1]
                    conflict_details.append({
                        'episode': ep_num,
                        'timestamp': timestamp,
                        'total_dps': len(dp_list),
                        'conflict_pairs': conflict_pairs,
                        'all_dps': dp_list
                    })
    
    return total_parallel_dps, parallel_with_conflicts, conflict_details


def check_job_operation_repetitions(episodes):
    """
    Check if the same (job, operation) pair appears at multiple decision points
    in the same episode, indicating resource preemption.
    """
    
    total_episodes_with_preemption = 0
    total_preemptions = 0
    preemption_examples = []
    
    for ep_data in episodes:
        ep_num = ep_data['episode']
        dps = ep_data['decision_points']
        
        # Track all (job, operation) pairs and their timestamps
        job_op_timestamps = defaultdict(list)
        for timestamp, job_id, operation, event_type in dps:
            job_op_timestamps[(job_id, operation)].append((timestamp, event_type))
        
        # Check for repetitions
        episode_preemptions = 0
        for (job_id, operation), timestamp_list in job_op_timestamps.items():
            if len(timestamp_list) > 1:
                # This job-operation pair appeared multiple times = preemption!
                episode_preemptions += 1
                total_preemptions += 1
                
                if len(preemption_examples) < 10:  # Collect first 10 examples
                    preemption_examples.append({
                        'episode': ep_num,
                        'job': job_id,
                        'operation': operation,
                        'occurrences': len(timestamp_list),
                        'timestamps': [ts for ts, _ in timestamp_list]
                    })
        
        if episode_preemptions > 0:
            total_episodes_with_preemption += 1
    
    return total_episodes_with_preemption, total_preemptions, preemption_examples


def main():
    print("="*70)
    print("PARALLEL DECISION POINT CONFLICT ANALYSIS")
    print("="*70)
    
    print("\nParsing timeline data...")
    episodes = parse_decision_points_with_details()
    print(f"Parsed {len(episodes)} episodes")
    
    # Analyze conflicts at parallel points
    print("\n" + "="*70)
    print("METHOD 1: Conflicts Within Parallel Decision Points")
    print("="*70)
    total_parallel, parallel_conflicts, conflict_details = analyze_conflicts_at_parallel_points(episodes)
    
    print(f"\nTotal parallel decision point timestamps: {total_parallel}")
    print(f"Parallel timestamps with conflicts: {parallel_conflicts}")
    if total_parallel > 0:
        print(f"Conflict ratio: {100*parallel_conflicts/total_parallel:.1f}%")
    
    if conflict_details:
        print(f"\nFirst {min(5, len(conflict_details))} conflict examples:")
        for i, detail in enumerate(conflict_details[:5], 1):
            print(f"\n  {i}. Episode {detail['episode']}, t={detail['timestamp']}")
            print(f"     Total DPs at this timestamp: {detail['total_dps']}")
            print(f"     Conflicting (job, operation) pairs: {detail['conflict_pairs']}")
    
    # Analyze job-operation repetitions (preemption)
    print("\n" + "="*70)
    print("METHOD 2: Job-Operation Preemption Detection")
    print("="*70)
    episodes_with_preempt, total_preempt, examples = check_job_operation_repetitions(episodes)
    
    print(f"\nTotal episodes: {len(episodes)}")
    print(f"Episodes with preemption: {episodes_with_preempt} ({100*episodes_with_preempt/len(episodes):.1f}%)")
    print(f"Total preemption events: {total_preempt}")
    print(f"Average preemptions per episode: {total_preempt/len(episodes):.2f}")
    
    if examples:
        print(f"\nFirst {len(examples)} preemption examples:")
        for i, ex in enumerate(examples, 1):
            print(f"\n  {i}. Episode {ex['episode']}: Job_{ex['job']} {ex['operation']}")
            print(f"     Appeared {ex['occurrences']} times at: {ex['timestamps']}")
            print(f"     → Resource was taken away and job had to reschedule")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\nIf preemptions exist, they indicate:")
    print("  - Resource contention between agents")
    print("  - Dynamic rescheduling due to conflicts")
    print("  - Multi-agent coordination challenges")
    print("\nThese preemptions at parallel decision points demonstrate")
    print("the need for MARL credit assignment and value decomposition.")


if __name__ == '__main__':
    main()
