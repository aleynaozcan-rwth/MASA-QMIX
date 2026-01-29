#!/usr/bin/env python3
"""
Observation ve State kalitesini kontrol eden diagnostic tool.
Bu script observation ve state vektörlerinin sağlıklı olup olmadığını analiz eder.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Historydata path
HISTORY_DIR = "my_data_and_graph/historydata"
OBS_FILE = os.path.join(HISTORY_DIR, "decision_observation_metrics.csv")
STATE_FILE = os.path.join(HISTORY_DIR, "decision_state_metrics.csv")

def check_observation_quality():
    """Observation vektörlerinin kalitesini analiz et."""
    print("=" * 80)
    print("📊 OBSERVATION QUALITY CHECK")
    print("=" * 80)
    
    if not os.path.exists(OBS_FILE):
        print(f"❌ ERROR: {OBS_FILE} bulunamadı!")
        return
    
    # Load data
    df = pd.read_csv(OBS_FILE)
    
    print(f"\n✅ {len(df)} observation log bulundu")
    print(f"   Unique jobs: {df['job_id'].nunique()}")
    print(f"   Decision time range: {df['decision_time'].min():.1f} → {df['decision_time'].max():.1f}")
    
    # Check individual elements
    obs_cols = [
        'obs_current_op_type',
        'obs_total_operations', 
        'obs_remaining_operations',
        'obs_wait_time',
        'obs_theoretical_machine_count',
        'obs_free_machine_count',
        'obs_n_jobs_active',
        'finished_flag'
    ]
    
    print("\n" + "-" * 80)
    print("OBSERVATION ELEMENT STATISTICS (0-1 normalized)")
    print("-" * 80)
    
    issues = []
    
    for col in obs_cols:
        if col not in df.columns:
            print(f"❌ Missing column: {col}")
            issues.append(f"Missing: {col}")
            continue
        
        data = df[col].dropna()
        if len(data) == 0:
            print(f"❌ {col}: NO DATA")
            issues.append(f"No data: {col}")
            continue
        
        mean = data.mean()
        std = data.std()
        min_val = data.min()
        max_val = data.max()
        unique_vals = data.nunique()
        zero_ratio = (data == 0).sum() / len(data)
        
        # Status indicators
        status = "✅"
        warnings = []
        
        # Check for constant values (no variance)
        if std < 0.01:
            status = "⚠️"
            warnings.append("CONSTANT")
            issues.append(f"{col}: constant (std={std:.4f})")
        
        # Check if all zeros
        if zero_ratio > 0.99:
            status = "❌"
            warnings.append("ALL_ZEROS")
            issues.append(f"{col}: all zeros")
        
        # Check if stuck at min/max
        if max_val <= 0.01:
            status = "⚠️"
            warnings.append("TOO_SMALL")
        if min_val >= 0.99:
            status = "⚠️"
            warnings.append("SATURATED")
        
        # Check diversity
        if unique_vals < 5:
            status = "⚠️"
            warnings.append(f"LOW_DIVERSITY({unique_vals})")
        
        warning_str = f" [{', '.join(warnings)}]" if warnings else ""
        
        print(f"{status} {col:30s}  mean={mean:.3f}  std={std:.3f}  "
              f"range=[{min_val:.3f}, {max_val:.3f}]  unique={unique_vals:4d}{warning_str}")
    
    # Summary
    print("\n" + "=" * 80)
    if len(issues) == 0:
        print("✅ ALL OBSERVATION ELEMENTS LOOK HEALTHY!")
    else:
        print(f"⚠️  FOUND {len(issues)} POTENTIAL ISSUES:")
        for issue in issues:
            print(f"   • {issue}")
    print("=" * 80)
    
    return issues


def check_state_quality():
    """State vektörünün kalitesini analiz et."""
    print("\n\n")
    print("=" * 80)
    print("🌍 STATE QUALITY CHECK")
    print("=" * 80)
    
    if not os.path.exists(STATE_FILE):
        print(f"❌ ERROR: {STATE_FILE} bulunamadı!")
        return
    
    # Load data
    df = pd.read_csv(STATE_FILE)
    
    print(f"\n✅ {len(df)} state log bulundu")
    print(f"   Decision time range: {df['decision_time'].min():.1f} → {df['decision_time'].max():.1f}")
    
    # State columns (10 elements)
    state_cols = [
        'state_n_jobs_arrived',
        'state_n_jobs_processing',
        'state_n_jobs_waiting',
        'state_n_ops_arrived',
        'state_n_ops_processing',
        'state_n_ops_waiting',
        'state_avg_machine_util',
        'state_avg_operator_util',
        'state_global_avg_wait',
        'state_episode_time_fraction'
    ]
    
    print("\n" + "-" * 80)
    print("STATE ELEMENT STATISTICS (0-1 normalized)")
    print("-" * 80)
    
    issues = []
    
    for col in state_cols:
        if col not in df.columns:
            print(f"❌ Missing column: {col}")
            issues.append(f"Missing: {col}")
            continue
        
        data = df[col].dropna()
        if len(data) == 0:
            print(f"❌ {col}: NO DATA")
            issues.append(f"No data: {col}")
            continue
        
        mean = data.mean()
        std = data.std()
        min_val = data.min()
        max_val = data.max()
        unique_vals = data.nunique()
        zero_ratio = (data == 0).sum() / len(data)
        
        # Status indicators
        status = "✅"
        warnings = []
        
        # Check for constant values
        if std < 0.01:
            status = "⚠️"
            warnings.append("CONSTANT")
            issues.append(f"{col}: constant (std={std:.4f})")
        
        # Check if all zeros
        if zero_ratio > 0.99:
            status = "❌"
            warnings.append("ALL_ZEROS")
            issues.append(f"{col}: all zeros")
        
        # Check if stuck at limits
        if max_val <= 0.01:
            status = "⚠️"
            warnings.append("TOO_SMALL")
        if min_val >= 0.99:
            status = "⚠️"
            warnings.append("SATURATED")
        
        # Check diversity
        if unique_vals < 5:
            status = "⚠️"
            warnings.append(f"LOW_DIVERSITY({unique_vals})")
        
        warning_str = f" [{', '.join(warnings)}]" if warnings else ""
        
        print(f"{status} {col:32s}  mean={mean:.3f}  std={std:.3f}  "
              f"range=[{min_val:.3f}, {max_val:.3f}]  unique={unique_vals:4d}{warning_str}")
    
    # Summary
    print("\n" + "=" * 80)
    if len(issues) == 0:
        print("✅ ALL STATE ELEMENTS LOOK HEALTHY!")
    else:
        print(f"⚠️  FOUND {len(issues)} POTENTIAL ISSUES:")
        for issue in issues:
            print(f"   • {issue}")
    print("=" * 80)
    
    return issues


def check_action_correlation():
    """Action seçimlerinin observation ile korelasyonunu kontrol et."""
    print("\n\n")
    print("=" * 80)
    print("🎯 ACTION-OBSERVATION CORRELATION CHECK")
    print("=" * 80)
    
    if not os.path.exists(OBS_FILE):
        print(f"❌ ERROR: {OBS_FILE} bulunamadı!")
        return
    
    df = pd.read_csv(OBS_FILE)
    
    # Filter out invalid actions (-1)
    valid_actions = df[df['action_idx'] >= 0].copy()
    
    print(f"\n✅ {len(valid_actions)} valid action bulundu (action_idx >= 0)")
    print(f"   Invalid actions (action_idx=-1): {len(df) - len(valid_actions)}")
    
    if len(valid_actions) == 0:
        print("❌ NO VALID ACTIONS FOUND!")
        return
    
    # Check action distribution
    action_counts = valid_actions['action_idx'].value_counts().sort_index()
    print("\n📊 Action Distribution:")
    for action, count in action_counts.items():
        percentage = count / len(valid_actions) * 100
        bar = '█' * int(percentage / 2)
        print(f"   Action {action}: {count:6d} ({percentage:5.1f}%)  {bar}")
    
    # Check if actions are too uniform (random) or too concentrated (stuck)
    action_entropy = -(action_counts / len(valid_actions) * 
                       np.log(action_counts / len(valid_actions) + 1e-9)).sum()
    max_entropy = np.log(len(action_counts))
    normalized_entropy = action_entropy / max_entropy if max_entropy > 0 else 0
    
    print(f"\n🎲 Action Entropy: {normalized_entropy:.3f} (0=stuck, 1=random)")
    
    if normalized_entropy < 0.3:
        print("   ⚠️  Actions are too concentrated (agent might be stuck)")
    elif normalized_entropy > 0.9:
        print("   ⚠️  Actions are too uniform (agent might be random)")
    else:
        print("   ✅ Action distribution looks reasonable")
    
    # Check correlation between wait_time and actions
    if 'obs_wait_time' in valid_actions.columns:
        high_wait = valid_actions[valid_actions['obs_wait_time'] > 0.5]
        low_wait = valid_actions[valid_actions['obs_wait_time'] <= 0.5]
        
        if len(high_wait) > 0 and len(low_wait) > 0:
            high_wait_actions = high_wait['action_idx'].value_counts(normalize=True)
            low_wait_actions = low_wait['action_idx'].value_counts(normalize=True)
            
            print("\n⏱️  Wait Time vs Action Pattern:")
            print("   High wait_time jobs prefer different actions?")
            # Simple check: are the distributions different?
            if len(high_wait_actions) > 0 and len(low_wait_actions) > 0:
                # Compare top action
                top_high = high_wait_actions.idxmax()
                top_low = low_wait_actions.idxmax()
                if top_high != top_low:
                    print(f"   ✅ YES: High-wait prefers action {top_high}, Low-wait prefers action {top_low}")
                else:
                    print(f"   ⚠️  NO: Both prefer action {top_high} (agent might not use wait_time signal)")
    
    print("=" * 80)


def generate_diagnostic_plots():
    """Observation/state kalitesi için görsel analiz."""
    print("\n\n")
    print("=" * 80)
    print("📈 GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)
    
    plots_dir = os.path.join(HISTORY_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(OBS_FILE):
        print(f"❌ ERROR: {OBS_FILE} bulunamadı!")
        return
    
    df = pd.read_csv(OBS_FILE)
    
    # Sample to avoid memory issues (take every Nth row)
    if len(df) > 10000:
        step = len(df) // 10000
        df_plot = df.iloc[::step].copy()
        print(f"   Sampling {len(df_plot)} points from {len(df)} total")
    else:
        df_plot = df
    
    # Plot 1: Observation elements over time
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Observation Elements Over Time', fontsize=16, fontweight='bold')
    
    obs_cols = [
        'obs_current_op_type',
        'obs_total_operations', 
        'obs_remaining_operations',
        'obs_wait_time',
        'obs_theoretical_machine_count',
        'obs_free_machine_count',
        'obs_n_jobs_active',
        'finished_flag'
    ]
    
    for idx, col in enumerate(obs_cols):
        ax = axes[idx // 2, idx % 2]
        if col in df_plot.columns:
            ax.plot(df_plot['decision_time'], df_plot[col], alpha=0.5, linewidth=0.5)
            ax.set_title(col.replace('obs_', ''))
            ax.set_xlabel('Decision Time')
            ax.set_ylabel('Value [0-1]')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
        else:
            ax.text(0.5, 0.5, f'{col}\nNOT FOUND', ha='center', va='center')
    
    plt.tight_layout()
    out_path = os.path.join(plots_dir, 'observation_quality.png')
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"   ✅ Saved: {out_path}")
    
    # Plot 2: Observation distribution histograms
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Observation Value Distributions', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(obs_cols):
        ax = axes[idx // 2, idx % 2]
        if col in df_plot.columns:
            data = df_plot[col].dropna()
            ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col.replace("obs_", "")}\nmean={data.mean():.3f}, std={data.std():.3f}')
            ax.set_xlabel('Value [0-1]')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{col}\nNOT FOUND', ha='center', va='center')
    
    plt.tight_layout()
    out_path = os.path.join(plots_dir, 'observation_distributions.png')
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"   ✅ Saved: {out_path}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "OBSERVATION & STATE QUALITY CHECKER" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Run checks
    obs_issues = check_observation_quality()
    state_issues = check_state_quality()
    check_action_correlation()
    generate_diagnostic_plots()
    
    # Final summary
    print("\n\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "FINAL SUMMARY" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    
    total_issues = len(obs_issues) + len(state_issues)
    
    if total_issues == 0:
        print("\n✅ ✅ ✅  ALL CHECKS PASSED!  ✅ ✅ ✅")
        print("\nObservation ve State vektörleri sağlıklı görünüyor.")
        print("Eğitim sorunları observation/state'ten kaynaklanmıyor.")
    else:
        print(f"\n⚠️  TOPLAM {total_issues} SORUN BULUNDU")
        print("\nObservation/State vektörlerinde problemler var!")
        print("Bu eğitim sorunlarının bir nedeni olabilir.")
        print("\nÖnerilen düzeltmeler için yukarıdaki detaylara bakın.")
    
    print("\n📊 Diagnostic plots kaydedildi:")
    print(f"   • {HISTORY_DIR}/plots/observation_quality.png")
    print(f"   • {HISTORY_DIR}/plots/observation_distributions.png")
    print()
