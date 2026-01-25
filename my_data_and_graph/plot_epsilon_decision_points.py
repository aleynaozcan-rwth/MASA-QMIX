"""
Epsilon Decay Visualization Based on Decision Points

This module provides visualization functions for analyzing epsilon decay dynamics
in the context of event-driven decision making, where decision points are triggered
by execution events rather than fixed time steps.

Key Insights Visualized:
- Epsilon decay per decision point (not per step)
- Decision point frequency variations across episodes
- Relationship between episode dynamics and decision point density
- Adaptive exploration behavior under stochastic arrivals
"""

import os
import warnings
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import scipy, if not available use numpy alternatives
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] scipy not available, using numpy alternatives for linear regression")


def _ensure_plots_dir(history_dir: str) -> str:
    """Ensure plots directory exists."""
    plots_dir = os.path.join(history_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_epsilon_decay_by_decision_points(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot epsilon decay dynamics based on decision points within and across episodes.
    
    This visualization shows how the decision-point-based epsilon decay mechanism
    adapts to execution dynamics (variable decision point frequency, arrival rates, etc.).
    
    Creates a multi-panel plot showing:
    - Epsilon decay per decision point (not step)
    - Decision point frequency per episode
    - Episode duration vs decision point count
    - Relationship between arrival rate and decision point density
    
    Args:
        history_dir: Directory containing training_metrics.csv
    """
    plots_dir = _ensure_plots_dir(history_dir)
    out_path = os.path.join(plots_dir, 'epsilon_decay_decision_points.png')
    
    csv_path = os.path.join(history_dir, 'training_metrics.csv')
    
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Required file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        required_cols = ['episode', 'train_step', 'epsilon']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Aggregate by episode to get decision point statistics
        episode_groups = df.groupby('episode').agg({
            'train_step': ['first', 'last', 'count'],  # count = decision points in episode
            'epsilon': ['first', 'last', 'mean']
        }).reset_index()
        
        # Flatten multi-index columns
        episode_groups.columns = ['episode', 'dp_first', 'dp_last', 'dp_count', 
                                   'eps_first', 'eps_last', 'eps_mean']
        
        # Create 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        
        # ============================================================
        # Plot 1: Epsilon Decay by Cumulative Decision Points
        # ============================================================
        ax = axes[0, 0]
        cumulative_dps = episode_groups['dp_count'].cumsum()
        ax.plot(cumulative_dps, episode_groups['eps_mean'], 
                linewidth=2.5, color='#2E86AB', marker='o', markersize=4, alpha=0.8,
                label='Epsilon (by decision point)')
        
        # Phase bands
        ax.axhspan(0.5, 1.0, alpha=0.08, color='red', label='High Exploration (ε>0.5)')
        ax.axhspan(0.2, 0.5, alpha=0.08, color='orange', label='Medium Exploration (0.2<ε≤0.5)')
        ax.axhspan(0.05, 0.2, alpha=0.08, color='yellow', label='Low Exploration (0.05<ε≤0.2)')
        ax.axhline(y=0.05, color='green', linestyle='--', linewidth=2.5, label='Min ε = 0.05 (Exploitation)')
        
        # Find exploitation start
        exploitation_eps = episode_groups[episode_groups['eps_mean'] <= 0.051]
        if len(exploitation_eps) > 0:
            exploit_start_idx = exploitation_eps.index[0]
            exploit_dp = cumulative_dps.iloc[exploit_start_idx]
            ax.axvline(x=exploit_dp, color='green', linestyle=':', alpha=0.6, linewidth=2)
            ax.text(exploit_dp, 0.55, f'  Exploitation\n  starts @ DP {int(exploit_dp)}',
                    rotation=0, va='center', ha='left', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        ax.set_xlabel('Cumulative Decision Points', fontsize=12, fontweight='bold')
        ax.set_ylabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        ax.set_title('Decision-Point-Based Epsilon Decay\n(Event-Driven Exploration Schedule)', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.05)
        
        # ============================================================
        # Plot 2: Decision Point Frequency per Episode
        # ============================================================
        ax = axes[0, 1]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(episode_groups)))
        ax.bar(episode_groups['episode'], episode_groups['dp_count'], 
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add moving average
        if len(episode_groups) > 10:
            window = min(20, len(episode_groups) // 3)
            ma = episode_groups['dp_count'].rolling(window=window, center=True).mean()
            ax.plot(episode_groups['episode'], ma, 
                    color='red', linewidth=3, linestyle='--', label=f'{window}-episode MA', zorder=10)
            ax.legend(loc='upper right', fontsize=10)
        
        # Add mean line
        mean_dp = episode_groups['dp_count'].mean()
        ax.axhline(y=mean_dp, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'Mean: {mean_dp:.1f} DPs')
        
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
        ax.set_title('Decision Point Frequency Across Episodes\n(Shows Execution Dynamics Variability)', 
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)
        
        # ============================================================
        # Plot 3: Epsilon Evolution Across Episodes
        # ============================================================
        ax = axes[1, 0]
        ax.plot(episode_groups['episode'], episode_groups['eps_mean'], 
                linewidth=2, color='purple', marker='s', markersize=4, alpha=0.7,
                label='Mean Epsilon per Episode')
        ax.fill_between(episode_groups['episode'], 
                        episode_groups['eps_first'], 
                        episode_groups['eps_last'],
                        alpha=0.2, color='purple', label='Episode ε range')
        
        # Phase transitions
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        ax.set_title('Epsilon Evolution Across Training Episodes', 
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # ============================================================
        # Plot 4: Decision Points vs Episode Number (with epsilon coloring)
        # ============================================================
        ax = axes[1, 1]
        
        scatter2 = ax.scatter(episode_groups['episode'], episode_groups['dp_count'],
                             c=episode_groups['eps_mean'], cmap='RdYlGn_r', 
                             s=80, alpha=0.7, edgecolors='black', linewidth=0.8,
                             vmin=0, vmax=1)
        
        # Add smoothed trend
        if len(episode_groups) > 10:
            window = min(30, len(episode_groups) // 3)
            dp_ma = episode_groups['dp_count'].rolling(window=window, center=True).mean()
            ax.plot(episode_groups['episode'], dp_ma, 
                    color='blue', linewidth=3, linestyle='-', alpha=0.9,
                    label=f'Smoothed DP Count ({window}-ep MA)')
            ax.legend(loc='best', fontsize=10)
        
        cbar2 = plt.colorbar(scatter2, ax=ax)
        cbar2.set_label('Mean Epsilon Level', fontsize=11, fontweight='bold')
        
        # Add horizontal reference
        mean_dp_count = episode_groups['dp_count'].mean()
        ax.axhline(y=mean_dp_count, color='purple', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'Mean: {mean_dp_count:.1f} DPs')
        
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
        ax.set_title('Decision Points Over Training\n(Color = Exploration level)', 
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # ============================================================
        # Add comprehensive summary statistics
        # ============================================================
        total_dps = episode_groups['dp_count'].sum()
        avg_dps_per_ep = episode_groups['dp_count'].mean()
        std_dps_per_ep = episode_groups['dp_count'].std()
        total_episodes = len(episode_groups)
        
        initial_eps = episode_groups['eps_first'].iloc[0]
        final_eps = episode_groups['eps_last'].iloc[-1]
        eps_decay_per_dp = (initial_eps - final_eps) / total_dps if total_dps > 0 else 0
        
        summary_text = f"""DECISION POINT STATISTICS (Event-Driven Mechanism)
{'='*55}
Total Episodes: {total_episodes:,}
Total Decision Points: {int(total_dps):,}

Decision Points per Episode:
  • Mean: {avg_dps_per_ep:.1f} ± {std_dps_per_ep:.1f}
  • Min: {episode_groups['dp_count'].min():.0f}
  • Max: {episode_groups['dp_count'].max():.0f}

Epsilon Decay:
  • Initial ε: {initial_eps:.4f}
  • Final ε: {final_eps:.4f}
  • Decay per DP: {eps_decay_per_dp:.7f}
  
Key Insight: Decision-point-based decay adapts to
execution dynamics, ensuring consistent exploration
regardless of arrival rate fluctuations.
"""
        
        fig.text(0.5, 0.005, summary_text, fontsize=9, 
                verticalalignment='bottom', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))
        
        plt.suptitle('Epsilon Decay: Decision-Point-Based Execution Dynamics Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0.15, 1, 0.98])
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[plot_epsilon_decay_by_decision_points] ✓ Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_epsilon_decay_by_decision_points] Error: {e}')
        import traceback
        traceback.print_exc()


def plot_episode_decision_point_distribution(history_dir: str = 'my_data_and_graph/historydata') -> None:
    """Plot distribution of decision points across episodes to show execution variability.
    
    Creates histogram and violin plots showing:
    - Distribution of decision points per episode
    - Temporal patterns in decision point occurrence
    - Comparison of early vs late training episodes
    """
    plots_dir = _ensure_plots_dir(history_dir)
    out_path = os.path.join(plots_dir, 'decision_point_distribution.png')
    
    csv_path = os.path.join(history_dir, 'training_metrics.csv')
    
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Required file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        if 'episode' not in df.columns or 'train_step' not in df.columns:
            raise ValueError("Missing required columns")
        
        # Count decision points per episode
        dp_counts = df.groupby('episode').size().reset_index(name='dp_count')
        
        # Divide into quarters for comparison
        n_episodes = len(dp_counts)
        quarter_size = n_episodes // 4
        
        dp_counts['phase'] = pd.cut(dp_counts.index, 
                                     bins=[0, quarter_size, 2*quarter_size, 3*quarter_size, n_episodes],
                                     labels=['Q1: Early', 'Q2: Mid-Early', 'Q3: Mid-Late', 'Q4: Late'],
                                     include_lowest=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Histogram with KDE
        ax = axes[0]
        ax.hist(dp_counts['dp_count'], bins=30, alpha=0.7, color='steelblue', 
                edgecolor='black', linewidth=1.2, density=True, label='Histogram')
        
        # Add KDE if scipy available
        if SCIPY_AVAILABLE:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(dp_counts['dp_count'])
            x_range = np.linspace(dp_counts['dp_count'].min(), dp_counts['dp_count'].max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=3, label='KDE', alpha=0.8)
        
        # Add mean and median lines
        mean_dp = dp_counts['dp_count'].mean()
        median_dp = dp_counts['dp_count'].median()
        ax.axvline(mean_dp, color='orange', linestyle='--', linewidth=2.5, label=f'Mean: {mean_dp:.1f}')
        ax.axvline(median_dp, color='green', linestyle=':', linewidth=2.5, label=f'Median: {median_dp:.1f}')
        
        ax.set_xlabel('Decision Points per Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Decision Points per Episode', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Violin plot by phase
        ax = axes[1]
        parts = ax.violinplot([dp_counts[dp_counts['phase'] == phase]['dp_count'].values 
                               for phase in dp_counts['phase'].cat.categories],
                              positions=range(4), widths=0.7, showmeans=True, showmedians=True)
        
        # Color the violins
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(4))
        ax.set_xticklabels(dp_counts['phase'].cat.categories, rotation=15, ha='right')
        ax.set_ylabel('Decision Points per Episode', fontsize=12, fontweight='bold')
        ax.set_title('Decision Points Distribution by Training Phase', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[plot_episode_decision_point_distribution] ✓ Saved {out_path}', flush=True)
        
    except Exception as e:
        warnings.warn(f'[plot_episode_decision_point_distribution] Error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    """Generate all decision-point-based epsilon decay visualizations."""
    import sys
    
    history_dir = 'my_data_and_graph/historydata'
    if len(sys.argv) > 1:
        history_dir = sys.argv[1]
    
    print(f"\n{'='*60}")
    print("Decision-Point-Based Epsilon Decay Visualization")
    print(f"{'='*60}\n")
    print(f"Reading data from: {history_dir}\n")
    
    # Generate main decision-point analysis
    print("Generating epsilon decay by decision points analysis...")
    plot_epsilon_decay_by_decision_points(history_dir)
    
    # Generate distribution analysis
    print("Generating decision point distribution analysis...")
    plot_episode_decision_point_distribution(history_dir)
    
    print(f"\n{'='*60}")
    print("✓ All visualizations complete!")
    print(f"{'='*60}\n")
