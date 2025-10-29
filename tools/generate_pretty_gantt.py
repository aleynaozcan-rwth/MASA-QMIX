#!/usr/bin/env python3
import csv,os,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np

p='my_data_and_graph/historydata/gantt_epoch1.csv'
out='my_data_and_graph/historydata/gantt_epoch1_pretty.png'
if not os.path.exists(p):
    print('CSV missing', p); sys.exit(1)
rows=[]
with open(p) as f:
    r=csv.reader(f)
    try:
        hdr=next(r)
    except StopIteration:
        print('CSV empty'); sys.exit(1)
    for row in r:
        if not row: continue
        try:
            start = float(row[0])
            end = float(row[1])
            op_idx = int(float(row[2])) if row[2] else row[2]
            # header in file is start,end,op_idx,op_name,wc,job_id,operator_grp,arrival,duration
            wc = int(row[4]) if len(row)>4 and row[4] else None
            job_id = int(row[5]) if len(row)>5 and row[5] else None
            op_grp = int(row[6]) if len(row)>6 and row[6] else None
            arrival = float(row[7]) if len(row)>7 and row[7] else None
            duration = float(row[8]) if len(row)>8 and row[8] else None
            rows.append((start,end,op_idx,wc,job_id,op_grp,arrival,duration))
        except Exception:
            continue
if not rows:
    print('No rows parsed'); sys.exit(2)
# collect ops and jobs
op_vals = sorted(list({r[2] for r in rows if r[2] is not None}))
job_ids = sorted(list({int(r[4]) for r in rows if r[4] is not None}))
palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
op_color_map = {op: palette[i % len(palette)] for i,op in enumerate(op_vals)}
# plot
fig,ax = plt.subplots(figsize=(14,6))
max_end = 0
arrival_by_job = {}
for start,end,op_idx,wc,job_id,op_grp,arrival,duration in rows:
    jid = int(job_id) if job_id is not None else 0
    op_key = op_idx
    color = op_color_map.get(op_key,'gray')
    ax.barh(jid, float(end)-float(start), left=float(start), color=color, edgecolor='black')
    try:
        width = float(end)-float(start)
    except Exception:
        width = 0.0
    if width >= 1.0:
        fontsize = min(12, max(8, int(width*3)))
        if op_grp is not None:
            op_label = "O{}".format(op_grp)
        else:
            op_label = "O?"
        ax.text((float(start)+float(end))/2, jid, "WC{} | {}".format(wc, op_label), va='center', ha='center', fontsize=fontsize, color='white')
    if arrival is not None:
        arrival_by_job.setdefault(jid, arrival)
    max_end = max(max_end, float(end))
# legend for ops
op_handles = [plt.Rectangle((0,0),1,1,color=op_color_map[op]) for op in op_vals]
op_labels = [ 'Op{}'.format(int(op)+1) if isinstance(op,(int,float)) else str(op) for op in op_vals]
if op_handles:
    ax.legend(op_handles, op_labels, title='Operation Types', bbox_to_anchor=(1.05,1), loc='upper left', fontsize=9)
# arrival markers
if arrival_by_job:
    xs = [arrival_by_job[j] for j in sorted(arrival_by_job.keys())]
    ys = [j for j in sorted(arrival_by_job.keys())]
    sc = ax.scatter(xs, ys, marker='^', color='green', edgecolor='black', s=70, zorder=6)
    try:
        arrival_legend = Line2D([0],[0], marker='^', color='w', markerfacecolor='green', markeredgecolor='black', markersize=8, linestyle='None', label='Arrival Times')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(arrival_legend)
        labels.append('Arrival Times')
        ax.legend(handles, labels, title='Operation Types', bbox_to_anchor=(1.05,1), loc='upper left', fontsize=9)
    except Exception:
        pass
# y ticks
if job_ids:
    min_j = min(job_ids); max_j = max(job_ids)
    ax.set_yticks(list(range(min_j, max_j+1)))
    ax.set_yticklabels([str(j) for j in range(min_j, max_j+1)])
    ax.set_ylim(min_j-0.5, max_j+0.5)
# x ticks and grid
spacing = 0.5 if max_end <= 40 else 1.0
ax.set_xticks(np.arange(0, max_end+spacing, spacing))
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlabel('Simulation Time (SimPy clock)')
ax.set_ylabel('JobAgent (ID)')
plt.tight_layout()
plt.savefig(out, dpi=300)
plt.close()
print('Saved pretty PNG', out)
