#!/usr/bin/env python3
"""
Operator Utilization Plot - Per Operator Utilization Over Training Epochs
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Operator capabilities from utils/workcenter.py (qualified machines)
OPERATOR_CAPABILITIES = {
    "O1": 3,  # Qualified for M0, M3, M4
    "O2": 3,  # Qualified for M1, M2, M4
    "O3": 3,  # Qualified for M0, M3, M4
    "O4": 3,  # Qualified for M1, M2, M4
}

def plot_operator_utilization(history_dir='my_data_and_graph/historydata'):
    """Generate per-operator utilization plot from run_summary.json"""
    
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

    # Extract operator utilization data
    x = list(range(len(evols)))
    operator_ids = []
    
    # Collect all operator IDs
    for e in evols:
        po = e.get('per_operator_utilization', {}) or {}
        for k in po.keys():
            ks = str(k)
            if ks not in operator_ids:
                operator_ids.append(ks)
    
    if not operator_ids:
        print("❌ No operator utilization data found!")
        return
    
    # Sort operator IDs for consistent ordering
    operator_ids = sorted(operator_ids, key=lambda x: int(x.replace('O', '')) if 'O' in x else x)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    
    # Color map
    cmap = plt.get_cmap('tab10')
    
    # Plot each operator with moving average
    import pandas as pd
    window = min(20, len(evols) // 5) if len(evols) > 10 else 1
    
    # Use colors 5-8 from tab10 (brown, pink, gray, lime) to avoid machine colors (0-4)
    color_offset = 5
    
    for i, oid in enumerate(operator_ids):
        vals = []
        for e in evols:
            po = e.get('per_operator_utilization', {}) or {}
            v = po.get(oid, None)
            if v is None and oid.replace('O', '').isdigit():
                v = po.get(int(oid.replace('O', '')), None)
            if v is None:
                v = 0.0
            vals.append(float(v) * 100)  # Convert to percentage
        
        # Get capability count for this operator
        capability = OPERATOR_CAPABILITIES.get(oid, 0)
        
        # Calculate and plot moving average only
        if len(vals) > 10:
            vals_series = pd.Series(vals)
            vals_ma = vals_series.rolling(window=window, center=True, min_periods=1).mean()
            ax.plot(x, vals_ma, label=f"Operator {oid} ({capability} machines)", 
                    color=cmap((i + color_offset) % 10), linewidth=2.0, alpha=0.9)
        else:
            ax.plot(x, vals, label=f"Operator {oid} ({capability} machines)", 
                    color=cmap((i + color_offset) % 10), linewidth=1.5, alpha=0.8)
    
    # Add exploitation start marker
    exploitation_start = 118
    ax.axvline(x=exploitation_start, color='orange', linestyle='--', linewidth=2, 
               alpha=0.8, label=f'Exploitation Start (epoch {exploitation_start})')
    
    # Add note about moving average in title or as separate legend entry
    if len(evols) > 10:
        ax.plot([], [], ' ', label=f'({window}-epoch Moving Average)')
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Operator Utilization (%)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(evols))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, framealpha=0.95, ncol=2)
    
    # Calculate before/after exploitation utilization
    exploitation_start = 118
    textstr = "Avg Utilization: Before and After Exploitation\n"
    
    before_utils = []
    after_utils = []
    
    for oid in operator_ids:
        vals_before = []
        vals_after = []
        
        for idx, e in enumerate(evols):
            po = e.get('per_operator_utilization', {}) or {}
            v = po.get(oid, None)
            if v is None and oid.replace('O', '').isdigit():
                v = po.get(int(oid.replace('O', '')), None)
            if v is None:
                v = 0.0
            
            if idx <= exploitation_start:
                vals_before.append(float(v) * 100)
            else:
                vals_after.append(float(v) * 100)
        
        avg_before = (sum(vals_before) / len(vals_before)) if vals_before else 0
        avg_after = (sum(vals_after) / len(vals_after)) if vals_after else 0
        
        before_utils.append(avg_before)
        after_utils.append(avg_after)
        
        textstr += f"  {oid}: {avg_before:.1f}% → {avg_after:.1f}%\n"
    
    # Calculate standard deviation and range for balance analysis
    import statistics
    std_before = statistics.stdev(before_utils) if len(before_utils) > 1 else 0
    std_after = statistics.stdev(after_utils) if len(after_utils) > 1 else 0
    
    # Calculate range (max - min) for spread analysis
    range_before = max(before_utils) - min(before_utils) if before_utils else 0
    range_after = max(after_utils) - min(after_utils) if after_utils else 0
    
    # First text box (top left) - Operator utilization values
    ax.text(0.02, 0.98, textstr.strip(), transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
            fontsize=9)
    
    # Second text box (bottom right) - Balance metrics
    balance_textstr = f"Std Dev: {std_before:.1f}% → {std_after:.1f}%\n"
    balance_textstr += f"Range (max-min): {range_before:.1f}% → {range_after:.1f}%"
    
    # Add balance interpretation (only if more balanced)
    if std_after < std_before and range_after < range_before:
        balance_text = "\n(More Balanced ✓✓)"
    else:
        balance_text = ""  # No text otherwise
    balance_textstr += balance_text
    
    ax.text(0.98, 0.02, balance_textstr.strip(), transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'),
            fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(history_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, 'operator_utilization.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Operator Utilization plot saved: {out_path}")
    print(f"\n📊 Summary:")
    print(f"   Total epochs: {len(evols)}")
    print(f"   Operators tracked: {len(operator_ids)}")
    print(f"   Operator IDs: {', '.join(operator_ids)}")


if __name__ == '__main__':
    plot_operator_utilization()
