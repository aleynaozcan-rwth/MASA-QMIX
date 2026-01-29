#!/usr/bin/env python3
"""
Analyze Job Arrivals vs Departures per Episode
Parse scheduling timeline to extract:
- How many jobs arrived in each episode
- How many jobs completed (departed) in each episode
"""

import re
from collections import defaultdict
from pathlib import Path

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'

def parse_arrivals_departures():
    """Parse arrivals and departures for each episode."""
    
    episode_arrivals = defaultdict(int)
    episode_departures = defaultdict(int)
    
    # Track job completion status: job_id -> {operations completed}
    job_status = defaultdict(lambda: {'total_ops': 0, 'completed_ops': 0, 'arrived_in_episode': None})
    
    current_episode = None
    in_lifecycle = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Episode marker
            if '=== EPISODE' in line and '===' in line:
                match = re.search(r'EPISODE (\d+)', line)
                if match:
                    current_episode = int(match.group(1))
                    in_lifecycle = False
                continue
            
            if current_episode is None:
                continue
            
            # Lifecycle trace markers
            if 'JOB AGENT LIFECYCLE TRACE START' in line:
                in_lifecycle = True
                continue
            if 'JOB AGENT LIFECYCLE TRACE END' in line:
                in_lifecycle = False
                continue
            
            # Job arrivals (in lifecycle trace) - format: "New job X arrived with Y ops"
            if in_lifecycle and 'New job' in line and 'arrived with' in line:
                job_match = re.search(r'New job (\d+) arrived with (\d+) ops', line)
                
                if job_match:
                    job_id = int(job_match.group(1))
                    num_ops = int(job_match.group(2))
                    
                    episode_arrivals[current_episode] += 1
                    job_status[job_id]['total_ops'] = num_ops
                    job_status[job_id]['arrived_in_episode'] = current_episode
                continue
            
            # Job completions (format: "Job_X completed all operations")
            if 'completed all operations' in line:
                job_match = re.search(r'Job_(\d+) completed all operations', line)
                
                if job_match:
                    job_id = int(job_match.group(1))
                    
                    # Job departed in current episode
                    if current_episode is not None:
                        episode_departures[current_episode] += 1
                continue
    
    return episode_arrivals, episode_departures

print("Parsing scheduling timeline for arrivals and departures...")
arrivals, departures = parse_arrivals_departures()

# Get sorted episode numbers
episodes = sorted(set(list(arrivals.keys()) + list(departures.keys())))

print(f"\nFound {len(episodes)} episodes with arrival/departure data")
print(f"Episode range: {min(episodes)} to {max(episodes)}")

# Print summary statistics
print("\n" + "="*80)
print("EPISODE-BY-EPISODE SUMMARY")
print("="*80)
print(f"{'Episode':<10} {'Arrivals':<12} {'Departures':<12} {'Ratio':<12} {'Status'}")
print("-"*80)

total_arrivals = 0
total_departures = 0

for ep in episodes[:50]:  # First 50 episodes
    arr = arrivals[ep]
    dep = departures[ep]
    ratio = dep / arr if arr > 0 else 0.0
    
    total_arrivals += arr
    total_departures += dep
    
    if ratio == 1.0:
        status = "✓ All completed"
    elif ratio > 0.8:
        status = "~ Most completed"
    elif ratio > 0.5:
        status = "⚠ Some pending"
    else:
        status = "✗ Many pending"
    
    print(f"{ep:<10} {arr:<12} {dep:<12} {ratio:<12.2f} {status}")

if len(episodes) > 50:
    print(f"... ({len(episodes) - 50} more episodes)")

print("-"*80)
print(f"{'TOTAL':<10} {total_arrivals:<12} {total_departures:<12} {total_departures/total_arrivals:<12.2f}")

# Overall statistics
all_arrivals = sum(arrivals.values())
all_departures = sum(departures.values())
overall_ratio = all_departures / all_arrivals if all_arrivals > 0 else 0.0

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Total Episodes:     {len(episodes)}")
print(f"Total Arrivals:     {all_arrivals}")
print(f"Total Departures:   {all_departures}")
print(f"Overall Ratio:      {overall_ratio:.3f} ({overall_ratio*100:.1f}%)")
print(f"Avg Arrivals/Ep:    {all_arrivals/len(episodes):.2f}")
print(f"Avg Departures/Ep:  {all_departures/len(episodes):.2f}")

# Save to file for plotting
output_file = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/arrivals_departures_per_episode.txt')
with open(output_file, 'w') as f:
    f.write("Episode,Arrivals,Departures,Ratio\n")
    for ep in episodes:
        arr = arrivals[ep]
        dep = departures[ep]
        ratio = dep / arr if arr > 0 else 0.0
        f.write(f"{ep},{arr},{dep},{ratio:.4f}\n")

print(f"\n✓ Saved data to: {output_file}")
print("Ready for plotting!")
