#!/usr/bin/env python3
"""
Analyze how parallel decision points change during training.
If agents learn to avoid conflicts, parallel DPs should decrease over time.
"""

import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# Use Agg backend
import matplotlib
matplotlib.use('Agg')

TIMELINE_FILE = 'my_data_and_graph/historydata/scheduling_timeline.txt'
OUTPUT_DIR = Path('my_data_and_graph/historydata/plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_parallel_dps_over_training():
    """Parse and track parallel DPs across training episodes."""
    
    episodes = []
    current_episode = None
    timestamp_groups = defaultdict(int)
    in_lifecycle = False
    
    with open(TIMELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '=== EPISODE' in line:
                if current_episode is not None:
                    # Calculate parallel DPs for this episode
                    parallel_count = sum(count for count in timestamp_groups.values() if count > 1)
                    total_count = sum(timestamp_groups.values())
                    
                    episodes.append({
                        'episode': current_episode,
                        'parallel_dps': parallel_count,
                        'total_dps': total_count,
                        'parallel_ratio': parallel_count / total_count if total_count > 0 else 0
                    })
                
                match = re.search(r'EPISODE (\d+)', line)
                current_episode = int(match.group(1)) if match else None
                timestamp_groups = defaultdict(int)
                in_lifecycle = False
                continue
            
            if 'LIFECYCLE TRACE START' in line:
                in_lifecycle = True
                continue
            if 'LIFECYCLE TRACE END' in line:
                in_lifecycle = False
                continue
            
            # Job arrivals
            if in_lifecycle and '[t=' in line and 'New job' in line and 'arrived' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                if time_match:
                    timestamp_groups[time_match.group(1)] += 1
                continue
            
            # Completions
            if not in_lifecycle and '[t=' in line and 'finished' in line and 'next queued' in line:
                time_match = re.search(r'\[t=([\d.]+)\]', line)
                if time_match:
                    timestamp_groups[time_match.group(1)] += 1
                continue
    
    # Last episode
    if current_episode is not None:
        parallel_count = sum(count for count in timestamp_groups.values() if count > 1)
        total_count = sum(timestamp_groups.values())
        episodes.append({
            'episode': current_episode,
            'parallel_dps': parallel_count,
            'total_dps': total_count,
            'parallel_ratio': parallel_count / total_count if total_count > 0 else 0
        })
    
    return episodes


def analyze_training_progression(episodes, window_size=50):
    """Analyze if parallel DPs decrease during training (learning effect)."""
    
    episode_nums = np.array([ep['episode'] for ep in episodes])
    parallel_dps = np.array([ep['parallel_dps'] for ep in episodes])
    total_dps = np.array([ep['total_dps'] for ep in episodes])
    parallel_ratios = np.array([ep['parallel_ratio'] for ep in episodes])
    
    # Split into training phases
    n_episodes = len(episodes)
    early = n_episodes // 3
    mid = 2 * n_episodes // 3
    
    early_data = parallel_dps[:early]
    mid_data = parallel_dps[early:mid]
    late_data = parallel_dps[mid:]
    
    early_ratio = parallel_ratios[:early]
    mid_ratio = parallel_ratios[early:mid]
    late_ratio = parallel_ratios[mid:]
    
    print("="*70)
    print("TRAINING PROGRESSION ANALYSIS")
    print("="*70)
    
    print(f"\nTotal episodes: {n_episodes}")
    print(f"Early training: Episodes 0-{early-1}")
    print(f"Mid training: Episodes {early}-{mid-1}")
    print(f"Late training: Episodes {mid}-{n_episodes-1}")
    
    print("\n" + "-"*70)
    print("PARALLEL DPs PER EPISODE")
    print("-"*70)
    print(f"Early training:  Mean={np.mean(early_data):.2f} ± {np.std(early_data):.2f}")
    print(f"Mid training:    Mean={np.mean(mid_data):.2f} ± {np.std(mid_data):.2f}")
    print(f"Late training:   Mean={np.mean(late_data):.2f} ± {np.std(late_data):.2f}")
    
    change_early_to_late = np.mean(late_data) - np.mean(early_data)
    pct_change = (change_early_to_late / np.mean(early_data)) * 100 if np.mean(early_data) > 0 else 0
    
    print(f"\nChange from early to late: {change_early_to_late:+.2f} ({pct_change:+.1f}%)")
    
    print("\n" + "-"*70)
    print("PARALLEL RATIO (% of total DPs)")
    print("-"*70)
    print(f"Early training:  {np.mean(early_ratio)*100:.1f}% ± {np.std(early_ratio)*100:.1f}%")
    print(f"Mid training:    {np.mean(mid_ratio)*100:.1f}% ± {np.std(mid_ratio)*100:.1f}%")
    print(f"Late training:   {np.mean(late_ratio)*100:.1f}% ± {np.std(late_ratio)*100:.1f}%")
    
    ratio_change = (np.mean(late_ratio) - np.mean(early_ratio)) * 100
    print(f"\nChange from early to late: {ratio_change:+.1f} percentage points")
    
    # Trend analysis
    print("\n" + "-"*70)
    print("TREND ANALYSIS")
    print("-"*70)
    
    # Linear regression on moving average
    if len(parallel_dps) > 100:
        smoothed = gaussian_filter1d(parallel_dps, sigma=10)
        coeffs = np.polyfit(episode_nums, smoothed, 1)
        slope = coeffs[0]
        print(f"Linear trend slope: {slope:.6f} parallel DPs per episode")
        
        if slope < -0.001:
            print("→ DECREASING trend: Agents learning to reduce conflicts ✓")
        elif slope > 0.001:
            print("→ INCREASING trend: Conflicts not being resolved")
        else:
            print("→ STABLE: No significant trend")
    
    return episode_nums, parallel_dps, parallel_ratios


def create_training_progression_plot(episode_nums, parallel_dps, parallel_ratios):
    """Create visualization showing how parallel DPs change during training."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Parallel DPs count
    ax1.scatter(episode_nums, parallel_dps, s=5, alpha=0.3, color='#DC143C', label='Raw Data')
    
    # Moving average
    if len(parallel_dps) >= 10:
        smoothed = gaussian_filter1d(parallel_dps, sigma=10)
        ax1.plot(episode_nums, smoothed, linewidth=3, color='darkred', 
                label='Moving Average (σ=10)', zorder=10)
    
    # Mean lines for phases
    n = len(episode_nums)
    early_end = n // 3
    mid_end = 2 * n // 3
    
    early_mean = np.mean(parallel_dps[:early_end])
    mid_mean = np.mean(parallel_dps[early_end:mid_end])
    late_mean = np.mean(parallel_dps[mid_end:])
    
    ax1.axhline(early_mean, 0, 0.33, color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Early Mean: {early_mean:.2f}')
    ax1.axhline(mid_mean, 0.33, 0.67, color='yellow', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Mid Mean: {mid_mean:.2f}')
    ax1.axhline(late_mean, 0.67, 1.0, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Late Mean: {late_mean:.2f}')
    
    # Phase boundaries
    ax1.axvline(episode_nums[early_end], color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(episode_nums[mid_end], color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Training Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Parallel Decision Points', fontsize=12, fontweight='bold')
    ax1.set_title('Learning Effect: Parallel DPs Across Training', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, episode_nums[-1])
    
    # Plot 2: Parallel ratio
    ax2.scatter(episode_nums, parallel_ratios * 100, s=5, alpha=0.3, color='steelblue', label='Raw Data')
    
    if len(parallel_ratios) >= 10:
        smoothed_ratio = gaussian_filter1d(parallel_ratios, sigma=10)
        ax2.plot(episode_nums, smoothed_ratio * 100, linewidth=3, color='darkblue', 
                label='Moving Average (σ=10)', zorder=10)
    
    early_ratio_mean = np.mean(parallel_ratios[:early_end]) * 100
    mid_ratio_mean = np.mean(parallel_ratios[early_end:mid_end]) * 100
    late_ratio_mean = np.mean(parallel_ratios[mid_end:]) * 100
    
    ax2.axhline(early_ratio_mean, 0, 0.33, color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Early Mean: {early_ratio_mean:.1f}%')
    ax2.axhline(mid_ratio_mean, 0.33, 0.67, color='yellow', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Mid Mean: {mid_ratio_mean:.1f}%')
    ax2.axhline(late_ratio_mean, 0.67, 1.0, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Late Mean: {late_ratio_mean:.1f}%')
    
    ax2.axvline(episode_nums[early_end], color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(episode_nums[mid_end], color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Training Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Parallel DP Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Coordination Efficiency: Parallel Ratio Across Training', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, episode_nums[-1])
    ax2.set_ylim(0, max(parallel_ratios * 100) * 1.1)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'parallel_dps_training_progression.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved training progression plot: {output_path}")
    plt.close()


def main():
    print("Parsing timeline for training progression analysis...")
    episodes = parse_parallel_dps_over_training()
    print(f"Parsed {len(episodes)} episodes\n")
    
    episode_nums, parallel_dps, parallel_ratios = analyze_training_progression(episodes)
    
    create_training_progression_plot(episode_nums, parallel_dps, parallel_ratios)
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nIf parallel DPs DECREASE during training:")
    print("  ✓ Agents learning to avoid resource conflicts")
    print("  ✓ Better coordination and decision-making")
    print("  ✓ Evidence of MARL learning effectiveness")
    print("\nIf parallel DPs remain STABLE or INCREASE:")
    print("  → Conflicts are inherent to problem structure")
    print("  → Initial job arrivals dominate parallel DPs")
    print("  → Learning may focus on other aspects (makespan, utilization)")


if __name__ == '__main__':
    main()
