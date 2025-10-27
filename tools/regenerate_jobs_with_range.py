#!/usr/bin/env python3
"""
Regenerate initial jobs with a configurable op count range and export mappings.
Usage: PYTHONPATH=. python3 tools/regenerate_jobs_with_range.py --min 1 --max 9 --jobs 10
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--min', type=int, default=1)
parser.add_argument('--max', type=int, default=9)
parser.add_argument('--jobs', type=int, default=10)
args = parser.parse_args()

# import environment and helpers
from environment import MASAEnv
from utils.operator import Operators
from utils.site import Sites

history = Path('my_data_and_graph/historydata')
history.mkdir(parents=True, exist_ok=True)

# create env
env = MASAEnv(num_jobs=args.jobs)
# override job ops range
env.job_min_ops = args.min
env.job_max_ops = args.max
# regenerate jobs
env._generate_initial_jobs()

# build job sequences
job_op_sequences = {}
for job in env.jobs:
    ops = []
    for op in job.operations:
        # op = (op_type, allowed_wcs, per_wc_dict)
        if len(op) == 3:
            op_type, allowed_wcs, per_wc = op
        elif len(op) == 2:
            # legacy
            allowed_wcs, base = op
            op_type = None
            per_wc = {wc: base for wc in allowed_wcs}
        else:
            op_type = None
            allowed_wcs = []
            per_wc = {}
        ops.append({
            'op_type': op_type,
            'allowed_wcs': allowed_wcs,
            'per_wc': {str(k): float(v) for k, v in per_wc.items()}
        })
    job_op_sequences[str(job.id)] = {'ops': ops}

# build op catalog
op_catalog = {}
for jid, info in job_op_sequences.items():
    for op in info['ops']:
        op_type = str(op['op_type']) if op['op_type'] is not None else 'None'
        rec = op_catalog.setdefault(op_type, {})
        for wc_str, est in op['per_wc'].items():
            wc = int(wc_str)
            wcd = rec.setdefault(wc, {'durations': [], 'operator_groups': set(), 'machine_ids': set()})
            wcd['durations'].append(est)

sites = Sites()
for op_type, wcs in op_catalog.items():
    for wc, info in list(wcs.items()):
        # operator groups from sites
        mids = [m for m, md in sites.machine_registry.items() if int(md['workcenter']) == wc]
        info['machine_ids'] = mids
        groups = sites.eligible_operator_groups_by_site.get(wc, [])
        info['operator_groups'] = groups
        durs = info.pop('durations')
        info['min_duration'] = min(durs) if durs else 0.0
        info['max_duration'] = max(durs) if durs else 0.0
        info['avg_duration'] = sum(durs)/len(durs) if durs else 0.0

# operator mappings
ops_obj = Operators(sites)
operator_mappings = {}
for op in ops_obj.operators_object_list:
    operator_mappings[op.operator_id] = {
        'qualified_wcs': op.qualified_workcenters,
        'can_do_op_types': []
    }
for op_type, wcs in op_catalog.items():
    allowed_wcs = set(wcs.keys())
    for op in ops_obj.operators_object_list:
        if allowed_wcs.intersection(set(op.qualified_workcenters)):
            operator_mappings[op.operator_id]['can_do_op_types'].append(op_type)

# write outputs
(history / 'job_op_sequences.json').write_text(json.dumps(job_op_sequences, indent=2))
(history / 'op_catalog.json').write_text(json.dumps(op_catalog, indent=2))
(history / 'operator_mappings.json').write_text(json.dumps(operator_mappings, indent=2))

# quick stats
num_jobs = len(job_op_sequences)
ops_counts = [len(v['ops']) for v in job_op_sequences.values()]
import statistics
print('Regenerated jobs:')
print(' num_jobs=', num_jobs)
print(' ops_per_job: min=', min(ops_counts), ' max=', max(ops_counts), ' avg=', round(statistics.mean(ops_counts),2))

# op_type frequency
op_counts = {}
for jid, info in job_op_sequences.items():
    for op in info['ops']:
        t = str(op['op_type'])
        op_counts[t] = op_counts.get(t, 0) + 1
print(' op_type_counts=', op_counts)
print('\nWrote JSONs to', str(history))
