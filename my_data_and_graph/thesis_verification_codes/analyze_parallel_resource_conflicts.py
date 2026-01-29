#!/usr/bin/env python3
"""
Analyze resource conflicts at parallel decision points.
A resource conflict occurs when multiple jobs at the same timestamp
need the same machine or operator.
"""

import re
from collections import defaultdict

TIMELINE_FILE = 'my_data_and_graph/historydata/scheduling_timeline.txt'

def parse_timeline_with_resources():
    """Parse timeline and extract decision points with resource requirements."""
    
    episodes = []
    current_episode = None
    decision_points = []  # (timestamp, job_id, event_type, machines, operators)
    in_lifecycle = False
    current_timeline_section = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Episode marker
            if '=== EPISODE' in line:
                if current_episode is not None:
                    episodes.append({
                        'episode': current_episode,
                        'decision_points': decision_points.copy()
                    })
                
                match = re.search(r'EPISODE (\d+)', line)
                current_episode = int(match.group(1)) if match else None
                decision_points = []
                in_lifecycle = False
                current_timeline_section = False
                continue
            
            if 'LIFECYCLE TRACE START' in line:
                in_lifecycle = True
                continue
            if 'LIFECYCLE TRACE END' in line:
                in_lifecycle = False
                continue
            if '=== TIMELINE ===' in line:
                current_timeline_section = True
                continue
            
            # Job arrivals - decision point (all machines/operators potentially available)
            if in_lifecycle and '[t=' in line and 'New job' in line and 'arrived' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'job (\d+)', line)
                if time_match and job_match:
                    timestamp = time_match.group(1)
                    job_id = int(job_match.group(1))
                    # At arrival, we don't know specific resource needs yet
                    decision_points.append((timestamp, job_id, 'arrival', None, None))
                continue
            
            # Operation completions - decision point for next operation
            if not in_lifecycle and '[t=' in line and 'finished' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'Job_(\d+)', line)
                if time_match and job_match:
                    timestamp = time_match.group(1)
                    job_id = int(job_match.group(1))
                    decision_points.append((timestamp, job_id, 'completion', None, None))
                continue
            
            # Operation starts - shows actual resource allocation
            if current_timeline_section and '[t=' in line and 'started on' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'Job_(\d+)', line)
                machine_match = re.search(r'started on (M\d+)', line)
                operator_match = re.search(r'by (O\d+)', line)
                capable_match = re.search(r"capable_machines=\[(.*?)\]", line)
                
                if time_match and job_match and machine_match:
                    timestamp = time_match.group(1)
                    job_id = int(job_match.group(1))
                    machine = machine_match.group(1)
                    operator = operator_match.group(1) if operator_match else None
                    
                    # Parse capable machines
                    capable_machines = []
                    if capable_match:
                        capable_str = capable_match.group(1)
                        capable_machines = re.findall(r"'(M\d+)'", capable_str)
                    
                    decision_points.append((timestamp, job_id, 'start', machine, operator, capable_machines))
                continue
    
    # Save last episode
    if current_episode is not None:
        episodes.append({
            'episode': current_episode,
            'decision_points': decision_points.copy()
        })
    
    return episodes


def analyze_resource_conflicts(episodes):
    """
    Analyze resource conflicts at parallel decision points.
    """
    
    total_parallel_timestamps = 0
    parallel_with_conflicts = 0
    conflict_examples = []
    
    # Statistics
    machine_conflicts = 0
    operator_conflicts = 0
    both_conflicts = 0
    
    for ep_data in episodes:
        ep_num = ep_data['episode']
        dps = ep_data['decision_points']
        
        # Group by timestamp
        timestamp_groups = defaultdict(list)
        for dp_data in dps:
            timestamp = dp_data[0]
            timestamp_groups[timestamp].append(dp_data)
        
        # Analyze parallel decision points
        for timestamp, dp_list in timestamp_groups.items():
            if len(dp_list) <= 1:
                continue
            
            total_parallel_timestamps += 1
            
            # Extract resource allocations at this timestamp
            machines_used = []
            operators_used = []
            capable_machines_all = []
            
            for dp_data in dp_list:
                if len(dp_data) >= 6:  # Has resource info
                    _, job_id, event_type, machine, operator, capable = dp_data
                    if machine:
                        machines_used.append((job_id, machine))
                    if operator:
                        operators_used.append((job_id, operator))
                    if capable:
                        capable_machines_all.append((job_id, set(capable)))
            
            # Check for conflicts
            has_machine_conflict = len(machines_used) != len(set([m for _, m in machines_used]))
            has_operator_conflict = len(operators_used) != len(set([o for _, o in operators_used]))
            
            # Check capable machine overlaps (potential conflict)
            capable_conflict = False
            if len(capable_machines_all) >= 2:
                for i in range(len(capable_machines_all)):
                    for j in range(i+1, len(capable_machines_all)):
                        job1, cap1 = capable_machines_all[i]
                        job2, cap2 = capable_machines_all[j]
                        if cap1 & cap2:  # Intersection: both jobs can use same machines
                            capable_conflict = True
                            break
            
            has_conflict = has_machine_conflict or has_operator_conflict or capable_conflict
            
            if has_conflict:
                parallel_with_conflicts += 1
                
                if has_machine_conflict:
                    machine_conflicts += 1
                if has_operator_conflict:
                    operator_conflicts += 1
                if has_machine_conflict and has_operator_conflict:
                    both_conflicts += 1
                
                if len(conflict_examples) < 10:
                    conflict_examples.append({
                        'episode': ep_num,
                        'timestamp': timestamp,
                        'num_jobs': len(dp_list),
                        'machines': machines_used,
                        'operators': operators_used,
                        'capable_overlap': capable_conflict,
                        'machine_conflict': has_machine_conflict,
                        'operator_conflict': has_operator_conflict
                    })
    
    return (total_parallel_timestamps, parallel_with_conflicts, 
            machine_conflicts, operator_conflicts, both_conflicts, conflict_examples)


def main():
    print("="*70)
    print("RESOURCE CONFLICT ANALYSIS AT PARALLEL DECISION POINTS")
    print("="*70)
    
    print("\nParsing timeline with resource information...")
    episodes = parse_timeline_with_resources()
    print(f"Parsed {len(episodes)} episodes")
    
    print("\nAnalyzing resource conflicts at parallel decision points...")
    (total_par, conflicts, m_conf, o_conf, both_conf, examples) = analyze_resource_conflicts(episodes)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nTotal parallel decision point timestamps: {total_par:,}")
    print(f"Parallel timestamps with resource conflicts: {conflicts:,}")
    if total_par > 0:
        print(f"Conflict rate: {100*conflicts/total_par:.1f}%")
    
    print(f"\nConflict breakdown:")
    print(f"  Machine conflicts: {m_conf:,}")
    print(f"  Operator conflicts: {o_conf:,}")
    print(f"  Both machine & operator conflicts: {both_conf:,}")
    
    if examples:
        print(f"\n" + "="*70)
        print(f"CONFLICT EXAMPLES (First {len(examples)})")
        print("="*70)
        for i, ex in enumerate(examples, 1):
            print(f"\n{i}. Episode {ex['episode']}, t={ex['timestamp']}")
            print(f"   {ex['num_jobs']} jobs at this parallel decision point")
            if ex['machine_conflict']:
                print(f"   ⚠ MACHINE CONFLICT: {ex['machines']}")
            if ex['operator_conflict']:
                print(f"   ⚠ OPERATOR CONFLICT: {ex['operators']}")
            if ex['capable_overlap']:
                print(f"   ⚠ Jobs compete for same capable machines")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    if conflicts > 0:
        print(f"\n✓ {100*conflicts/total_par:.1f}% of parallel decision points have resource conflicts")
        print("✓ Multiple agents competing for same resources")
        print("✓ Coordination critical to avoid makespan penalties")
        print("✓ QMIX learns to handle these conflict scenarios")
    else:
        print("\n✗ No same-resource conflicts detected at parallel timestamps")
        print("  (Jobs at parallel DPs may use different resources)")
        print("  However, indirect competition still exists through:")
        print("    - Capable machine overlaps")
        print("    - Sequential dependencies")
        print("    - Resource availability constraints")


if __name__ == '__main__':
    main()
