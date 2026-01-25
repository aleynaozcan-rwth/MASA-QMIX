#!/usr/bin/env python3
"""
Analyze conflicts from decision logs where action=-1 indicates
the agent couldn't get its desired machine (conflict loser).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path

def analyze_conflicts_from_decisions(decision_file='my_data_and_graph/historydata/decision_observation_metrics.csv'):
    """
    Analyze conflicts using action=-1 as indicator.
    action=-1 means the job couldn't execute its desired action (conflict).
    """
    print("Loading decision log...")
    df = pd.read_csv(decision_file)
    
    total_decisions = len(df)
    conflicts = df[df['action_idx'] == -1].copy()
    num_conflicts = len(conflicts)
    
    print(f"\n{'='*80}")
    print("CONFLICT ANALYSIS FROM DECISION LOGS")
    print(f"{'='*80}")
    print(f"\nTotal decisions made: {total_decisions:,}")
    print(f"Conflict decisions (action=-1): {num_conflicts:,}")
    print(f"Conflict rate: {100*num_conflicts/total_decisions:.2f}%")
    
    # Analyze parallel decisions at same time
    print("\n" + "-"*80)
    print("PARALLEL DECISION ANALYSIS")
    print("-"*80)
    
    # Group by decision time
    time_groups = df.groupby('decision_time')
    
    parallel_points = []
    conflict_points = []
    
    for t, group in time_groups:
        num_agents = len(group)
        num_conflicts = (group['action_idx'] == -1).sum()
        
        if num_agents > 1:
            parallel_points.append({
                'time': t,
                'num_agents': num_agents,
                'num_conflicts': num_conflicts,
                'conflict_rate': num_conflicts / num_agents
            })
        
        if num_conflicts > 0:
            conflict_points.append({
                'time': t,
                'num_agents': num_agents,
                'num_conflicts': num_conflicts,
                'jobs': group[group['action_idx'] == -1]['job_id'].tolist()
            })
    
    print(f"Parallel decision points (>1 agent): {len(parallel_points)}")
    print(f"Decision points with conflicts: {len(conflict_points)}")
    
    if parallel_points:
        avg_agents = np.mean([p['num_agents'] for p in parallel_points])
        max_agents = max([p['num_agents'] for p in parallel_points])
        print(f"Average agents per parallel decision: {avg_agents:.2f}")
        print(f"Maximum agents in one decision point: {max_agents}")
    
    # Analyze which jobs have most conflicts
    print("\n" + "-"*80)
    print("JOB-WISE CONFLICT ANALYSIS")
    print("-"*80)
    
    job_conflicts = conflicts.groupby('job_id').size().sort_values(ascending=False)
    job_total_decisions = df.groupby('job_id').size()
    
    print(f"\nTop 10 jobs with most conflicts:")
    for i, (job_id, count) in enumerate(job_conflicts.head(10).items(), 1):
        total = job_total_decisions[job_id]
        rate = 100 * count / total
        print(f"  {i}. Job {job_id}: {count} conflicts out of {total} decisions ({rate:.1f}%)")
    
    # Time distribution
    print("\n" + "-"*80)
    print("TEMPORAL CONFLICT DISTRIBUTION")
    print("-"*80)
    
    max_time = conflicts['decision_time'].max()
    min_time = conflicts['decision_time'].min()
    
    # Create sensible bins based on actual time range
    if max_time <= 50:
        time_bins = [min_time, 10, 20, 30, max_time] if max_time > 30 else [min_time, 10, 20, max_time]
    elif max_time <= 100:
        time_bins = [min_time, 25, 50, 75, max_time]
    else:
        time_bins = [min_time, 50, 100, 200, max_time]
    
    # Remove duplicate bins
    time_bins = sorted(list(set([b for b in time_bins if min_time <= b <= max_time])))
    
    if len(time_bins) >= 2:
        conflict_time_dist = pd.cut(conflicts['decision_time'], bins=time_bins).value_counts().sort_index()
        
        print(f"\nConflicts by time range (total range: {min_time:.1f} to {max_time:.1f}):")
        for interval, count in conflict_time_dist.items():
            print(f"  {interval}: {count} conflicts")
    
    return conflicts, parallel_points, conflict_points, df

def generate_conflict_report(conflicts, parallel_points, conflict_points, total_df, output_file):
    """Generate detailed text report."""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RESOURCE CONFLICT ANALYSIS - EVIDENCE FOR THESIS\n")
        f.write("Based on Decision Logs (action=-1 indicates conflict)\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total decisions made: {len(total_df):,}\n")
        f.write(f"Conflict decisions (action=-1): {len(conflicts):,}\n")
        f.write(f"Conflict rate: {100*len(conflicts)/len(total_df):.2f}%\n\n")
        
        f.write(f"Parallel decision points: {len(parallel_points)}\n")
        if parallel_points:
            avg_agents = np.mean([p['num_agents'] for p in parallel_points])
            max_agents = max([p['num_agents'] for p in parallel_points])
            f.write(f"Average agents per parallel decision: {avg_agents:.2f}\n")
            f.write(f"Maximum agents deciding simultaneously: {max_agents}\n")
        
        f.write("\n")
        f.write("CONFLICT EVIDENCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Decision points with conflicts: {len(conflict_points)}\n\n")
        
        f.write("Sample conflict events (First 20):\n")
        for i, cp in enumerate(conflict_points[:20], 1):
            f.write(f"  {i}. Time {cp['time']:.2f}: {cp['num_conflicts']} agents ")
            f.write(f"had conflicts out of {cp['num_agents']} deciding ")
            f.write(f"(Jobs: {cp['jobs']})\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("THESIS INTERPRETATION\n")
        f.write("="*80 + "\n")
        f.write(f"""
The decision logs provide direct evidence of resource conflicts in the
multi-agent system:

1. CONFLICT DETECTION: {len(conflicts):,} instances where agents received
   action=-1, indicating they could not execute their preferred action due
   to resource contention.

2. CONFLICT RATE: {100*len(conflicts)/len(total_df):.2f}% of all decisions resulted in conflicts,
   demonstrating significant resource competition that requires resolution.

3. PARALLEL DECISIONS: {len(parallel_points)} time steps with multiple
   simultaneous agent decisions, validating the parallel MARL framework.

4. LEARNING SIGNAL: Each conflict (action=-1) provides a clear learning
   signal to the agent, enabling the system to learn conflict avoidance
   strategies over time.

This empirical evidence directly supports the parallel conflict resolution
mechanism described in the thesis methodology.
""")
    
    print(f"\nGenerated report: {output_file}")

def plot_conflict_analysis(conflicts, parallel_points, output_dir):
    """Generate visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot 1: Conflict rate over time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bin conflicts by time
    time_bins = np.linspace(0, conflicts['decision_time'].max(), 50)
    conflict_hist, _ = np.histogram(conflicts['decision_time'], bins=time_bins)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    ax.plot(bin_centers, conflict_hist, 'r-', linewidth=2, label='Conflicts (action=-1)')
    ax.fill_between(bin_centers, 0, conflict_hist, alpha=0.3, color='red')
    
    ax.set_xlabel('Simulation Time', fontsize=12)
    ax.set_ylabel('Number of Conflicts', fontsize=12)
    ax.set_title('Resource Conflicts Over Time\nDirect Evidence from Decision Logs',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'conflict_timeline.pdf', dpi=300)
    print(f"Saved: conflict_timeline.pdf")
    plt.close()
    
    # Plot 2: Parallel decision distribution
    if parallel_points:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        agent_counts = [p['num_agents'] for p in parallel_points]
        
        counts, bins, patches = ax.hist(agent_counts, bins=range(2, max(agent_counts)+2),
                                        color='#3498db', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Number of Simultaneous Agents', fontsize=12)
        ax.set_ylabel('Frequency (Decision Points)', fontsize=12)
        ax.set_title(f'Parallel Decision Making Evidence\n{len(parallel_points)} decision points with multiple agents',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(2, max(agent_counts)+1))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parallel_decisions.pdf', dpi=300)
        print(f"Saved: parallel_decisions.pdf")
        plt.close()
    
    # Plot 3: Conflict rate by number of parallel agents
    if parallel_points:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by number of agents
        agent_conflict_rates = defaultdict(list)
        for p in parallel_points:
            if p['num_agents'] >= 2:
                agent_conflict_rates[p['num_agents']].append(p['conflict_rate'])
        
        agents = sorted(agent_conflict_rates.keys())
        avg_rates = [np.mean(agent_conflict_rates[a]) * 100 for a in agents]
        
        bars = ax.bar(agents, avg_rates, color='#e74c3c', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Number of Simultaneous Agents', fontsize=12)
        ax.set_ylabel('Average Conflict Rate (%)', fontsize=12)
        ax.set_title('Conflict Rate vs. Parallelism\nMore agents → Higher conflict probability',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(agents)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'conflict_rate_vs_parallelism.pdf', dpi=300)
        print(f"Saved: conflict_rate_vs_parallelism.pdf")
        plt.close()

def main():
    print("="*80)
    print("CONFLICT ANALYSIS FROM DECISION LOGS")
    print("Using action=-1 as conflict indicator")
    print("="*80)
    print()
    
    decision_file = 'my_data_and_graph/historydata/decision_observation_metrics.csv'
    
    if not Path(decision_file).exists():
        print(f"Error: {decision_file} not found!")
        return
    
    # Analyze
    conflicts, parallel_points, conflict_points, total_df = analyze_conflicts_from_decisions(decision_file)
    
    # Generate outputs
    output_dir = Path('my_data_and_graph/conflict_evidence')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*80)
    print("Generating report and plots...")
    print("="*80)
    
    generate_conflict_report(conflicts, parallel_points, conflict_points, total_df,
                           output_dir / 'conflict_analysis_report.txt')
    
    plot_conflict_analysis(conflicts, parallel_points, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nKey Findings for Thesis:")
    print(f"  ✓ {len(conflicts):,} conflict instances detected (action=-1)")
    print(f"  ✓ {100*len(conflicts)/len(total_df):.2f}% conflict rate")
    print(f"  ✓ {len(parallel_points)} parallel decision points")
    print(f"  ✓ {len(conflict_points)} decision points with conflicts")
    print("\nThis provides DIRECT EVIDENCE of resource conflicts requiring resolution!")

if __name__ == '__main__':
    main()
