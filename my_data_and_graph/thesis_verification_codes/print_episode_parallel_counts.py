#!/usr/bin/env python3
"""
Print per-episode parallel decision point counts from scheduling timeline
"""

import re
from pathlib import Path
from collections import defaultdict

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'

def count_parallel_dps_per_episode():
    """Count parallel decision points for each episode."""
    
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
                    # Calculate parallel DPs (simultaneity at same timestamp)
                    parallel_count = sum(count for count in timestamp_groups.values() if count > 1)
                    
                    # Detect conflicts (job-op appearing multiple times)
                    job_op_timestamps = defaultdict(list)
                    for timestamp, job_id, operation in decision_points:
                        job_op_timestamps[(job_id, operation)].append(float(timestamp))
                    
                    # Count conflict DPs (retry attempts)
                    conflict_dps = set()
                    for (job_id, operation), timestamps in job_op_timestamps.items():
                        if len(timestamps) > 1:
                            for ts in timestamps[1:]:
                                conflict_dps.add((str(ts), job_id, operation))
                    
                    # Total parallel DPs
                    total_parallel = parallel_count + len(conflict_dps)
                    
                    episodes_data.append({
                        'episode': current_episode,
                        'parallel_dps': parallel_count,
                        'conflict_dps': len(conflict_dps),
                        'total_parallel_dps': total_parallel
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
    
    # Last episode
    if current_episode is not None:
        parallel_count = sum(count for count in timestamp_groups.values() if count > 1)
        
        job_op_timestamps = defaultdict(list)
        for timestamp, job_id, operation in decision_points:
            job_op_timestamps[(job_id, operation)].append(float(timestamp))
        
        conflict_dps = set()
        for (job_id, operation), timestamps in job_op_timestamps.items():
            if len(timestamps) > 1:
                for ts in timestamps[1:]:
                    conflict_dps.add((str(ts), job_id, operation))
        
        total_parallel = parallel_count + len(conflict_dps)
        
        episodes_data.append({
            'episode': current_episode,
            'parallel_dps': parallel_count,
            'conflict_dps': len(conflict_dps),
            'total_parallel_dps': total_parallel
        })
    
    return episodes_data


if __name__ == '__main__':
    import csv
    
    print("Parsing parallel DPs per episode...")
    episodes = count_parallel_dps_per_episode()
    
    # Save to CSV
    output_path = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/episode_parallel_counts.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Total_Parallel_DPs', 'Simultaneous_DPs', 'Conflict_DPs'])
        
        for ep in episodes:
            writer.writerow([
                ep['episode'],
                ep['total_parallel_dps'],
                ep['parallel_dps'],
                ep['conflict_dps']
            ])
    
    print(f"\n✓ Saved to CSV: {output_path}")
    print(f"\nTotal episodes: {len(episodes)}")
    print(f"Total parallel DPs (sum): {sum(ep['total_parallel_dps'] for ep in episodes)}")
    print(f"Average per episode: {sum(ep['total_parallel_dps'] for ep in episodes) / len(episodes):.2f}")
    
    # Show some examples
    print("\nExample entries:")
    print(f"Episode {episodes[0]['episode']}: {episodes[0]['total_parallel_dps']} parallel DP ({episodes[0]['parallel_dps']} simultaneous + {episodes[0]['conflict_dps']} conflicts)")
    print(f"Episode {episodes[1]['episode']}: {episodes[1]['total_parallel_dps']} parallel DP ({episodes[1]['parallel_dps']} simultaneous + {episodes[1]['conflict_dps']} conflicts)")
    print(f"Episode {episodes[391]['episode']}: {episodes[391]['total_parallel_dps']} parallel DP ({episodes[391]['parallel_dps']} simultaneous + {episodes[391]['conflict_dps']} conflicts)")
    print(f"Episode {episodes[1294]['episode']}: {episodes[1294]['total_parallel_dps']} parallel DP ({episodes[1294]['parallel_dps']} simultaneous + {episodes[1294]['conflict_dps']} conflicts)")
