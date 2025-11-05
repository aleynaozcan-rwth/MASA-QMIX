#!/usr/bin/env python3
import re
from collections import defaultdict
import sys

PATH = '/home/aleynaozcan/projects/MASA-QMIX/my_data_and_graph/historydata/scheduling_timeline.txt'

start_patterns = [
    re.compile(r"^\[t=(?P<t>[0-9]+\.?[0-9]*)\].* started on .* by Operator (?P<op>O?\d+) \(duration=(?P<dur>[0-9]+\.?[0-9]*)\)"),
    re.compile(r"^\[t=(?P<t>[0-9]+\.?[0-9]*)\].* started on .* \(duration=(?P<dur>[0-9]+\.?[0-9]*)\) \| operator=(?P<opnum>\d+)")
]
job_re = re.compile(r"(Job_[^\.\s]+\.[^\s]+)")

intervals = defaultdict(list)

with open(PATH, 'r') as f:
    for ln in f:
        ln = ln.strip()
        if 'started on' not in ln:
            continue
        m = None
        for pat in start_patterns:
            m = pat.search(ln)
            if m:
                break
        if not m:
            # couldn't parse this start line; skip
            continue
        t = float(m.group('t'))
        dur = float(m.group('dur'))
        end = t + dur
        job = job_re.search(ln)
        jobtxt = job.group(1) if job else ''
        if 'op' in m.groupdict() and m.group('op') is not None:
            op = m.group('op')
        else:
            # from operator number group
            op = 'group:' + m.group('opnum')
        # normalize: if op is digits, keep as 'group:N'
        if re.fullmatch(r"\d+", op):
            op = 'group:' + op
        intervals[op].append((t, end, jobtxt, ln))

# report summary
print('Parsed operators and interval counts:')
for op, ivs in sorted(intervals.items(), key=lambda x: (-len(x[1]), x[0])):
    print(f"  {op}: {len(ivs)} intervals")

# detect overlaps per operator
print('\nChecking overlaps (same operator):')
any_overlaps = False
for op, ivs in intervals.items():
    # sort by start
    ivs_sorted = sorted(ivs, key=lambda x: x[0])
    for i in range(len(ivs_sorted)):
        s1,e1,job1,line1 = ivs_sorted[i]
        for j in range(i+1, len(ivs_sorted)):
            s2,e2,job2,line2 = ivs_sorted[j]
            if s2 < e1 and s1 < e2:
                any_overlaps = True
                print(f"Overlap for {op}: [{s1:.2f}, {e1:.2f}) {job1}  overlaps  [{s2:.2f}, {e2:.2f}) {job2}")
                # also print the lines
                print('  L1:', line1)
                print('  L2:', line2)

if not any_overlaps:
    print('No overlaps detected for any operator based on parsed start+duration intervals.')

# exit status
if any_overlaps:
    sys.exit(2)
else:
    sys.exit(0)
