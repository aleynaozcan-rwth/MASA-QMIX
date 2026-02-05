#!/usr/bin/env python3
"""
Machine Utilization Plot - Per Machine Utilization Over Training Epochs
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Machine capabilities from utils/workcenter.py
MACHINE_CAPABILITIES = {
    "0": 5,  # M0: Op1, Op2, Op3, Op5, Op9
    "1": 3,  # M1: Op4, Op5, Op8
    "2": 5,  # M2: Op1, Op2, Op4, Op6, Op9
    "3": 4,  # M3: Op1, Op3, Op7, Op8
    "4": 6,  # M4: Op1, Op3, Op4, Op6, Op8, Op9
}

def plot_machine_utilization(history_dir='my_data_and_graph/historydata'):
    """Generate per-machine utilization plot from run_summary.json"""
    
    path = os.path.join(history_dir, "run_summary.json")
    if not os.path.exists(path):
        print(f"❌ {path} not found!")
        return
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")
        return
    
    evols = data.get("evolutions", []) if isinstance(data, dict) else []
    if not evols:
        print("❌ No evolutions data found!")
        return

    # Extract machine utilization data
    x = list(range(len(evols)))
    machine_ids = []
    
    # Collect all machine IDs
    for e in evols:
        pm = e.get('per_machine_utilization', {}) or {}
        for k in pm.keys():
            ks = str(k)
            if ks not in machine_ids:
                machine_ids.append(ks)
    
    if not machine_ids:
        print("❌ No machine utilization data found!")
        return
    
    # Sort machine IDs for consistent ordering
    machine_ids = sorted(machine_ids, key=lambda x: int(x) if x.isdigit() else x)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    
    # Color map
    cmap = plt.get_cmap('tab10')
    
    # Plot each machine with moving average
    import pandas as pd
    window = min(20, len(evols) // 5) if len(evols) > 10 else 1
    
    for i, mid in enumerate(machine_ids):
        vals = []
        for e in evols:
            pm = e.get('per_machine_utilization', {}) or {}
            v = pm.get(int(mid), None) if mid.isdigit() else pm.get(mid, None)
            if v is None:
                v = pm.get(str(mid), 0.0)
            vals.append(float(v) * 100)  # Convert to percentage
        
        # Get capability count for this machine
        capability = MACHINE_CAPABILITIES.get(mid, 0)
        
        # Calculate and plot moving average only
        if len(vals) > 10:
            vals_series = pd.Series(vals)
            vals_ma = vals_series.rolling(window=window, center=True, min_periods=1).mean()
            ax.plot(x, vals_ma, label=f"Machine {mid} ({capability} ops)", 
                    color=cmap(i % 10), linewidth=2.0, alpha=0.9)
        else:
            ax.plot(x, vals, label=f"Machine {mid} ({capability} ops)", 
                    color=cmap(i % 10), linewidth=1.5, alpha=0.8)
    
    # Add exploitation start marker
    exploitation_start = 118
    ax.axvline(x=exploitation_start, color='orange', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'Exploitation Start (epoch {exploitation_start})')
    
    # Add note about moving average in title or as separate legend entry
    if len(evols) > 10:
        ax.plot([], [], ' ', label=f'({window}-epoch Moving Average)')
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Machine Utilization (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(evols))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, framealpha=0.95, ncol=2)
    
    # Calculate before/after exploitation utilization
    exploitation_start = 118
    textstr = "Avg Utilization: Before and After Exploitation\n"
    
    before_utils = []
    after_utils = []
    
    for mid in machine_ids:
        vals_before = []
        vals_after = []
        
        for idx, e in enumerate(evols):
            pm = e.get('per_machine_utilization', {}) or {}
            v = pm.get(int(mid), None) if mid.isdigit() else pm.get(mid, None)
            if v is None:
                v = pm.get(str(mid), 0.0)
            
            if idx <= exploitation_start:
                vals_before.append(float(v) * 100)
            else:
                vals_after.append(float(v) * 100)
        
        avg_before = (sum(vals_before) / len(vals_before)) if vals_before else 0
        avg_after = (sum(vals_after) / len(vals_after)) if vals_after else 0
        
        before_utils.append(avg_before)
        after_utils.append(avg_after)
        
        textstr += f"  M{mid}: {avg_before:.1f}% → {avg_after:.1f}%\n"
    
    # Calculate standard deviation and range for balance analysis
    import statistics
    std_before = statistics.stdev(before_utils) if len(before_utils) > 1 else 0
    std_after = statistics.stdev(after_utils) if len(after_utils) > 1 else 0
    
    # Calculate range (max - min) for spread analysis
    range_before = max(before_utils) - min(before_utils) if before_utils else 0
    range_after = max(after_utils) - min(after_utils) if after_utils else 0
    
    # First text box (top left) - Machine utilization values
    ax.text(0.02, 0.98, textstr.strip(), transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=9)
    
    # Second text box (bottom right) - Balance metrics
    balance_textstr = f"Std Dev: {std_before:.1f}% → {std_after:.1f}%\n"
    balance_textstr += f"Range (max-min): {range_before:.1f}% → {range_after:.1f}%"
    
    # Add balance interpretation (only if clear outcome)
    if std_after < std_before and range_after < range_before:
        balance_text = "\n(More Balanced ✓✓)"
    elif std_after > std_before and range_after > range_before:
        balance_text = "\n(Less Balanced ✗✗)"
    else:
        balance_text = ""  # No text for mixed results
    balance_textstr += balance_text
    
    ax.text(0.98, 0.02, balance_textstr.strip(), transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'),
            fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(history_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'machine_utilization.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Machine Utilization plot saved: {out_path}")
    print(f"\n📊 Summary:")
    print(f"   Total epochs: {len(evols)}")
    print(f"   Machines tracked: {len(machine_ids)}")
    print(f"   Machine IDs: {', '.join(machine_ids)}")


if __name__ == '__main__':
    plot_machine_utilization()
