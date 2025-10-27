#!/usr/bin/env python3
import re
import json
from pathlib import Path

history = Path("my_data_and_graph/historydata")
init = history / "initial_jobs.txt"
if not init.exists():
    print("initial_jobs.txt not found at", init)
    raise SystemExit(1)

text = init.read_text()
# Parse the human-readable runner output. The format lists Jobs, then Ops, and
# for each Op it prints a `base_per_wc_durations:` block with lines like
#   WC0 -> dur=3.327 | OpGroups=0
# The original parser expected a single Dur on the Op line; update it to
# collect the per-WC durations and use their mean as the op's base duration.
jobs = {}
lines = text.splitlines()
current = None
current_op = None
i = 0
while i < len(lines):
    raw = lines[i]
    line = raw.strip()
    i += 1
    if not line:
        continue
    m_job = re.match(r"Job (\d+):", line)
    if m_job:
        current = int(m_job.group(1))
        jobs[current] = {"ops": []}
        current_op = None
        continue
    # Op header line: Op 0 | Type 8 | WCs [...] | Groups [...] | base_per_wc_durations:
    m_op = re.match(r"Op (\d+) \| Type (\S+) \| WCs \[(.*?)\] \| Groups \[(.*?)\] \| base_per_wc_durations:", line)
    if m_op and current is not None:
        # finalize previous op if any
        if current_op is not None:
            jobs[current]["ops"].append(current_op)
        op_idx = int(m_op.group(1))
        op_type = m_op.group(2)
        wcs = [int(x.strip()) for x in m_op.group(3).split(",") if x.strip()]
        groups = [int(x.strip()) for x in m_op.group(4).split(",") if x.strip()]
        current_op = {"op_idx": op_idx, "op_type": op_type, "wcs": wcs, "groups": groups, "per_wc": []}
        # read following WC lines
        while i < len(lines):
            nxt = lines[i].strip()
            m_wc = re.match(r"WC\s*(\d+)\s*->\s*dur=([0-9]+\.?[0-9]*)\s*\|\s*OpGroups=(\d+)", nxt)
            if m_wc:
                wc = int(m_wc.group(1))
                dur = float(m_wc.group(2))
                opgrp = int(m_wc.group(3))
                current_op["per_wc"].append({"wc": wc, "dur": dur, "op_group": opgrp})
                i += 1
                continue
            break
        # compute a base duration (mean of per-wc durations) so downstream tools
        # that expect a single numeric duration still work.
        if current_op["per_wc"]:
            avg = sum(p["dur"] for p in current_op["per_wc"]) / len(current_op["per_wc"])
            current_op["dur"] = float(avg)
        else:
            current_op["dur"] = 0.0
        continue
    # If we reach here and we have an active op but hit another non-per-wc line,
    # finalize the current op (this handles files where the Op block ends).
    if current_op is not None and line.startswith("Op "):
        jobs[current]["ops"].append(current_op)
        current_op = None

# finalize last op if open
if current is not None and current_op is not None:
    jobs[current]["ops"].append(current_op)

# compute job totals
for jid, info in jobs.items():
    total = sum(op['dur'] for op in info['ops'])
    info['total_dur'] = total

out_json = history / 'job_ops_summary.json'
out_txt = history / 'job_ops_summary.txt'
out_json.write_text(json.dumps(jobs, indent=2))

with out_txt.open('w') as f:
    f.write('Job operations summary\n')
    for jid in sorted(jobs):
        info = jobs[jid]
        f.write(f"Job {jid}: ops={len(info['ops'])} total_dur={info['total_dur']:.3f}\n")
        for op in info['ops']:
            f.write(f"  Op {op['op_idx']} | Type {op['op_type']} | WCs {op['wcs']} | Groups {op['groups']} | Dur {op['dur']:.3f}\n")
        f.write('\n')

print('Wrote', out_json, out_txt)
print(out_txt.read_text())
