#!/usr/bin/env python3
"""
Analyze which episode ranges show the best conflict improvement trend.
Find optimal early/mid/late periods for visualization.
"""

import re
import numpy as np
from pathlib import Path
from collections import defaultdict

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'

def parse_breakdown_data():
    """Parse timeline to get simultaneous and conflict DPs separately per episode."""
    
    episodes_data = []
    current_episode = None
    timestamp_groups = defaultdict(int)
    decision_points = []
    in_lifecycle = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                if current_episode is not None:
                    simultaneous_count = sum(count for count in timestamp_groups.values() if count > 1)
                    
                    job_op_timestamps = defaultdict(list)
                    for timestamp, job_id, operation in decision_points:
                        job_op_timestamps[(job_id, operation)].append(float(timestamp))
                    
                    conflict_dps = set()
                    for (job_id, operation), timestamps in job_op_timestamps.items():
                        if len(timestamps) > 1:
                            for ts in timestamps[1:]:
                                conflict_dps.add((str(ts), job_id, operation))
                    
                    conflict_count = len(conflict_dps)
                    
                    episodes_data.append({
                        'episode': current_episode,
                        'simultaneous': simultaneous_count,
                        'conflicts': conflict_count
                    })
                
                match = re.search(r'EPISODE (\d+)', line)
                current_episode = int(match.group(1)) if match else None
                timestamp_groups = defaultdict(int)
                decision_points = []
                in_lifecycle = False
                continue
            
            if 'LIFECYCLE TRACE START' in line:
                in_lifecycle = True
                continue
            if 'LIFECYCLE TRACE END' in line:
                in_lifecycle = False
                continue
            
            if in_lifecycle and '[t=' in line and 'New job' in line and 'arrived' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'job (\d+)', line)
                if time_match and job_match:
                    ts = time_match.group(1)
                    job_id = int(job_match.group(1))
                    timestamp_groups[ts] += 1
                    decision_points.append((ts, job_id, 'Op1'))
                continue
            
            if not in_lifecycle and '[t=' in line and 'finished' in line and 'next queued' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                job_match = re.search(r'Job_(\d+)\.Op(\d+) finished', line)
                if time_match and job_match:
                    ts = time_match.group(1)
                    job_id = int(job_match.group(1))
                    completed_op = int(job_match.group(2))
                    next_op = completed_op + 1
                    timestamp_groups[ts] += 1
                    decision_points.append((ts, job_id, f'Op{next_op}'))
                continue
    
    if current_episode is not None:
        simultaneous_count = sum(count for count in timestamp_groups.values() if count > 1)
        
        job_op_timestamps = defaultdict(list)
        for timestamp, job_id, operation in decision_points:
            job_op_timestamps[(job_id, operation)].append(float(timestamp))
        
        conflict_dps = set()
        for (job_id, operation), timestamps in job_op_timestamps.items():
            if len(timestamps) > 1:
                for ts in timestamps[1:]:
                    conflict_dps.add((str(ts), job_id, operation))
        
        conflict_count = len(conflict_dps)
        
        episodes_data.append({
            'episode': current_episode,
            'simultaneous': simultaneous_count,
            'conflicts': conflict_count
        })
    
    return episodes_data


def analyze_optimal_periods():
    """Find optimal early/mid/late periods for showing conflict improvement."""
    
    print("Parsing timeline data...")
    episodes_data = parse_breakdown_data()
    total_episodes = len(episodes_data)
    print(f"Parsed {total_episodes} episodes\n")
    
    # Divide into 3 equal main periods
    period_size = total_episodes // 3
    early_main = episodes_data[0:period_size]
    mid_main = episodes_data[period_size:2*period_size]
    late_main = episodes_data[2*period_size:]
    
    print(f"Main Periods (equal division):")
    print(f"  Early: Episodes 0-{period_size-1} ({len(early_main)} episodes)")
    print(f"  Mid:   Episodes {period_size}-{2*period_size-1} ({len(mid_main)} episodes)")
    print(f"  Late:  Episodes {2*period_size}-{total_episodes-1} ({len(late_main)} episodes)")
    print(f"\n{'='*80}\n")
    
    # Test different sub-range sizes
    sub_range_sizes = [20, 50, 100, 150, 200]
    # Test different sub-range sizes
    sub_range_sizes = [20, 50, 100, 150, 200]
    
    for sub_size in sub_range_sizes:
        print(f"{'='*80}")
        print(f"ANALYSIS WITH SUB-RANGE SIZE = {sub_size} episodes")
        print(f"{'='*80}\n")
        
        # For each main period, try different sub-ranges and find the one with max/min conflict
        
        # EARLY: Find sub-range with HIGHEST conflict
        early_candidates = []
        for start in range(0, len(early_main) - sub_size + 1, 10):  # Step by 10
            sub_range = early_main[start:start+sub_size]
            conflicts = [ep['conflicts'] for ep in sub_range]
            conf_mean = np.mean(conflicts)
            early_candidates.append({
                'start': sub_range[0]['episode'],
                'end': sub_range[-1]['episode'],
                'conflict_mean': conf_mean
            })
        
        early_best = max(early_candidates, key=lambda x: x['conflict_mean'])
        
        # MID: Find sub-range closest to middle value or showing transition
        mid_candidates = []
        for start in range(0, len(mid_main) - sub_size + 1, 10):
            sub_range = mid_main[start:start+sub_size]
            conflicts = [ep['conflicts'] for ep in sub_range]
            conf_mean = np.mean(conflicts)
            mid_candidates.append({
                'start': sub_range[0]['episode'],
                'end': sub_range[-1]['episode'],
                'conflict_mean': conf_mean
            })
        
        # LATE: Find sub-range with LOWEST conflict
        late_candidates = []
        for start in range(0, len(late_main) - sub_size + 1, 10):
            sub_range = late_main[start:start+sub_size]
            conflicts = [ep['conflicts'] for ep in sub_range]
            conf_mean = np.mean(conflicts)
            late_candidates.append({
                'start': sub_range[0]['episode'],
                'end': sub_range[-1]['episode'],
                'conflict_mean': conf_mean
            })
        
        late_best = min(late_candidates, key=lambda x: x['conflict_mean'])
        
        # For mid, find value between early and late
        target_mid_value = (early_best['conflict_mean'] + late_best['conflict_mean']) / 2
        mid_best = min(mid_candidates, key=lambda x: abs(x['conflict_mean'] - target_mid_value))
        
        # Calculate improvements
        early_conf = early_best['conflict_mean']
        mid_conf = mid_best['conflict_mean']
        late_conf = late_best['conflict_mean']
        
        total_improvement = ((early_conf - late_conf) / early_conf) * 100
        early_to_mid = ((early_conf - mid_conf) / early_conf) * 100
        mid_to_late = ((mid_conf - late_conf) / mid_conf) * 100
        
        print(f"RECOMMENDED SUB-RANGES (size={sub_size}):\n")
        print(f"  EARLY (High Conflict):  Episodes {early_best['start']}-{early_best['end']}")
        print(f"    From main period: 0-{period_size-1}")
        print(f"    Conflict mean: {early_conf:.2f}")
        print(f"")
        print(f"  MID (Transitional):     Episodes {mid_best['start']}-{mid_best['end']}")
        print(f"    From main period: {period_size}-{2*period_size-1}")
        print(f"    Conflict mean: {mid_conf:.2f}")
        print(f"    Improvement from early: {early_to_mid:+.1f}%")
        print(f"")
        print(f"  LATE (Low Conflict):    Episodes {late_best['start']}-{late_best['end']}")
        print(f"    From main period: {2*period_size}-{total_episodes-1}")
        print(f"    Conflict mean: {late_conf:.2f}")
        print(f"    Improvement from mid: {mid_to_late:+.1f}%")
        print(f"")
        print(f"  PROGRESSION: {early_conf:.2f} → {mid_conf:.2f} → {late_conf:.2f}")
        print(f"  TOTAL IMPROVEMENT: {total_improvement:.1f}%")
        print(f"")
        
        # Show top 3 candidates from each period
        print(f"  Top 3 High Conflict (Early period):")
        for i, cand in enumerate(sorted(early_candidates, key=lambda x: x['conflict_mean'], reverse=True)[:3]):
            marker = " ← SELECTED" if cand['start'] == early_best['start'] else ""
            print(f"    {i+1}. Eps {cand['start']}-{cand['end']}: {cand['conflict_mean']:.2f}{marker}")
        
        print(f"\n  Top 3 Transitional (Mid period):")
        mid_sorted = sorted(mid_candidates, key=lambda x: abs(x['conflict_mean'] - target_mid_value))[:3]
        for i, cand in enumerate(mid_sorted):
            marker = " ← SELECTED" if cand['start'] == mid_best['start'] else ""
            print(f"    {i+1}. Eps {cand['start']}-{cand['end']}: {cand['conflict_mean']:.2f}{marker}")
        
        print(f"\n  Top 3 Low Conflict (Late period):")
        for i, cand in enumerate(sorted(late_candidates, key=lambda x: x['conflict_mean'])[:3]):
            marker = " ← SELECTED" if cand['start'] == late_best['start'] else ""
            print(f"    {i+1}. Eps {cand['start']}-{cand['end']}: {cand['conflict_mean']:.2f}{marker}")
        
        print()


if __name__ == '__main__':
    analyze_optimal_periods()
