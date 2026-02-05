"""Extract job and operation completion ratios from scheduling_timeline.txt."""

import os
import re
import pandas as pd


def extract_completion_ratios(timeline_path: str, output_path: str):
    """Extract episode-level completion ratios from scheduling timeline.
    
    Parses scheduling_timeline.txt using episode markers:
    - === EPISODE X === : Episode boundary
    - New job X arrived with Y ops : Track jobs and ops arrived
    - Job_X completed all operations : Track completed jobs
    - total_jobs_generated / total_operations_executed : Summary data
    
    Saves results to completion_ratios.csv.
    """
    if not os.path.exists(timeline_path):
        print(f"[ERROR] File not found: {timeline_path}")
        return
    
    print(f"[extract_completion_ratios] Reading {timeline_path}...")
    
    episodes_data = []
    current_episode = None
    episode_jobs_arrived = 0
    episode_ops_arrived = 0
    episode_jobs_completed = 0
    episode_ops_completed = 0
    in_timeline_section = False  # Flag to track if we're in TIMELINE section
    
    with open(timeline_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect episode boundary: === EPISODE X ===
            match = re.match(r'=== EPISODE (\d+) ===', line)
            if match:
                # Save previous episode if exists AND has data
                if current_episode is not None and (episode_jobs_arrived > 0 or episode_jobs_completed > 0):
                    job_ratio = (episode_jobs_completed / episode_jobs_arrived * 100) if episode_jobs_arrived > 0 else 0.0
                    op_ratio = (episode_ops_completed / episode_ops_arrived * 100) if episode_ops_arrived > 0 else 0.0
                    
                    episodes_data.append({
                        'episode': current_episode,  # Use actual episode number from timeline
                        'jobs_arrived': episode_jobs_arrived,
                        'jobs_completed': episode_jobs_completed,
                        'job_completion_ratio': job_ratio,
                        'ops_arrived': episode_ops_arrived,
                        'ops_completed': episode_ops_completed,
                        'op_completion_ratio': op_ratio
                    })
                
                # Start new episode - reset counters FIRST
                current_episode = int(match.group(1))
                episode_jobs_arrived = 0
                episode_ops_arrived = 0
                episode_jobs_completed = 0
                episode_ops_completed = 0
                in_timeline_section = False
                continue
            
            # Skip if no episode started yet
            if current_episode is None:
                continue
            
            # Check if we entered TIMELINE section (skip LIFECYCLE TRACE section)
            if '=== TIMELINE ===' in line:
                in_timeline_section = True
                continue
            
            # Only count arrivals and completions from TIMELINE section (not from LIFECYCLE TRACE)
            if in_timeline_section:
                # Track job arrivals: "New job X arrived with Y ops"
                match = re.search(r'New job \w+ arrived with (\d+) ops', line)
                if match:
                    num_ops = int(match.group(1))
                    episode_jobs_arrived += 1
                    episode_ops_arrived += num_ops
                    continue
                
                # Track job completions: "Job_X completed all operations"
                if 'completed all operations' in line:
                    episode_jobs_completed += 1
                    continue
            
            # Read operations executed from summary: "total_operations_executed = X"
            match = re.search(r'total_operations_executed = (\d+)', line)
            if match:
                # This is completed operations count from summary
                summary_ops = int(match.group(1))
                episode_ops_completed = summary_ops
                continue
    
    # Don't forget the last episode
    if current_episode is not None:
        job_ratio = (episode_jobs_completed / episode_jobs_arrived * 100) if episode_jobs_arrived > 0 else 0.0
        op_ratio = (episode_ops_completed / episode_ops_arrived * 100) if episode_ops_arrived > 0 else 0.0
        
        episodes_data.append({
            'episode': current_episode,  # Use actual episode number
            'jobs_arrived': episode_jobs_arrived,
            'jobs_completed': episode_jobs_completed,
            'job_completion_ratio': job_ratio,
            'ops_arrived': episode_ops_arrived,
            'ops_completed': episode_ops_completed,
            'op_completion_ratio': op_ratio
        })
    
    # Create DataFrame and save
    if episodes_data:
        df = pd.DataFrame(episodes_data)
        df.to_csv(output_path, index=False)
        print(f"[extract_completion_ratios] Saved {len(df)} episodes to {output_path}")
        print(f"\n[extract_completion_ratios] Sample data (first 10 episodes):")
        print(df.head(10).to_string(index=False))
        print(f"\n[extract_completion_ratios] Statistics:")
        print(f"  Avg Job Completion: {df['job_completion_ratio'].mean():.2f}%")
        print(f"  Avg Op Completion: {df['op_completion_ratio'].mean():.2f}%")
        print(f"  Total episodes: {len(df)}")
    else:
        print("[extract_completion_ratios] No episode data found!")


def extract_completion_ratios_v2(timeline_path: str, output_path: str):
    """Alternative extraction method using Lifecycle markers."""
    pass  # No longer needed, main function is accurate now


if __name__ == '__main__':
    timeline_file = 'my_data_and_graph/historydata/scheduling_timeline.txt'
    output_file = 'my_data_and_graph/historydata/completion_ratios.csv'
    
    print("=" * 70)
    print("  COMPLETION RATIO EXTRACTION FROM SCHEDULING TIMELINE")
    print("=" * 70)
    print()
    
    extract_completion_ratios(timeline_file, output_file)
