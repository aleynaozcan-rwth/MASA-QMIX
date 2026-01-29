#!/usr/bin/env python3
"""
Analyze Operation Completions vs Arrivals per Episode
More granular than job-level analysis:
- Track total operations arrived (sum of all operations in arrived jobs)
- Track total operations completed
"""

import re
from collections import defaultdict
from pathlib import Path

TIMELINE_FILE = '/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'

def parse_operation_arrivals_completions():
    """Parse operation arrivals and completions for each episode."""
    
    episode_op_arrivals = defaultdict(int)
    episode_op_completions = defaultdict(int)
    
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
            
            # Job arrivals - count total operations
            if in_lifecycle and 'New job' in line and 'arrived with' in line:
                job_match = re.search(r'New job (\d+) arrived with (\d+) ops', line)
                
                if job_match:
                    num_ops = int(job_match.group(2))
                    episode_op_arrivals[current_episode] += num_ops
                continue
            
            # Operation completions - format: "Job_X.OpY finished"
            if 'finished' in line and 'next queued' in line:
                op_match = re.search(r'Job_\d+\.Op\d+ finished', line)
                
                if op_match:
                    episode_op_completions[current_episode] += 1
                continue
    
    return episode_op_arrivals, episode_op_completions

print("Parsing scheduling timeline for operation arrivals and completions...")
op_arrivals, op_completions = parse_operation_arrivals_completions()

# Get sorted episode numbers
episodes = sorted(set(list(op_arrivals.keys()) + list(op_completions.keys())))

print(f"\nFound {len(episodes)} episodes with operation data")
print(f"Episode range: {min(episodes)} to {max(episodes)}")

# Print summary statistics
print("\n" + "="*90)
print("EPISODE-BY-EPISODE SUMMARY (OPERATION LEVEL)")
print("="*90)
print(f"{'Episode':<10} {'Op Arrivals':<15} {'Op Completions':<18} {'Ratio':<12} {'Status'}")
print("-"*90)

total_op_arrivals = 0
total_op_completions = 0

for ep in episodes[:50]:  # First 50 episodes
    arr = op_arrivals[ep]
    comp = op_completions[ep]
    ratio = comp / arr if arr > 0 else 0.0
    
    total_op_arrivals += arr
    total_op_completions += comp
    
    if ratio >= 0.9:
        status = "✓ Excellent"
    elif ratio >= 0.7:
        status = "✓ Good"
    elif ratio >= 0.5:
        status = "~ Moderate"
    elif ratio >= 0.3:
        status = "⚠ Low"
    else:
        status = "✗ Very Low"
    
    print(f"{ep:<10} {arr:<15} {comp:<18} {ratio:<12.2f} {status}")

if len(episodes) > 50:
    print(f"... ({len(episodes) - 50} more episodes)")

print("-"*90)
print(f"{'TOTAL':<10} {total_op_arrivals:<15} {total_op_completions:<18} {total_op_completions/total_op_arrivals:<12.2f}")

# Overall statistics
all_op_arrivals = sum(op_arrivals.values())
all_op_completions = sum(op_completions.values())
overall_ratio = all_op_completions / all_op_arrivals if all_op_arrivals > 0 else 0.0

print("\n" + "="*90)
print("OVERALL STATISTICS (OPERATION LEVEL)")
print("="*90)
print(f"Total Episodes:          {len(episodes)}")
print(f"Total Op Arrivals:       {all_op_arrivals}")
print(f"Total Op Completions:    {all_op_completions}")
print(f"Overall Ratio:           {overall_ratio:.3f} ({overall_ratio*100:.1f}%)")
print(f"Avg Op Arrivals/Ep:      {all_op_arrivals/len(episodes):.2f}")
print(f"Avg Op Completions/Ep:   {all_op_completions/len(episodes):.2f}")

# Save to file for plotting
output_file = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/operation_arrivals_completions_per_episode.txt')
with open(output_file, 'w') as f:
    f.write("Episode,OpArrivals,OpCompletions,Ratio\n")
    for ep in episodes:
        arr = op_arrivals[ep]
        comp = op_completions[ep]
        ratio = comp / arr if arr > 0 else 0.0
        f.write(f"{ep},{arr},{comp},{ratio:.4f}\n")

print(f"\n✓ Saved operation-level data to: {output_file}")
print("Ready for plotting!")
