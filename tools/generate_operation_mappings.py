#!/usr/bin/env python3
"""
Generate consolidated mappings:
 - op_catalog.json: for each operation id (op_type) list allowed WCs, per-WC min/avg/max estimated durations, operator_groups
 - operator_mappings.json: operator_id -> qualified_wcs and which op_types they can perform
 - job_op_sequences.json: concise per-job sequences (seq_idx, op_type, allowed_wcs, per_wc estimates)

Usage: PYTHONPATH=. python3 tools/generate_operation_mappings.py
"""
import json
from pathlib import Path
from collections import defaultdict

history = Path('my_data_and_graph/historydata')
js = history / 'job_ops_detailed.json'
if not js.exists():
    print('job_ops_detailed.json not found. Run parse/expand tools first.')
    raise SystemExit(1)

jobs = json.loads(js.read_text())

# Build op_catalog: op_type -> wc -> list of est durations + operator_groups
op_catalog = {}
for jid, info in jobs.items():
    for op in info.get('ops', []):
        op_type = str(op['op_type'])
        rec = op_catalog.setdefault(op_type, {})
        for p in op.get('per_wc', []):
            wc = int(p['wc'])
            d = float(p.get('estimated_duration', 0.0))
            grp = p.get('operator_groups', [])
            wcd = rec.setdefault(wc, {'durations': [], 'operator_groups': set(), 'machine_ids': set()})
            wcd['durations'].append(d)
            for g in grp:
                wcd['operator_groups'].add(int(g))
            if 'machine_id' in p and p['machine_id']:
                wcd['machine_ids'].add(p['machine_id'])

# summarize per wc
for op_type, wcs in op_catalog.items():
    for wc, info in wcs.items():
        durs = info['durations']
        info['min_duration'] = min(durs) if durs else 0.0
        info['max_duration'] = max(durs) if durs else 0.0
        info['avg_duration'] = sum(durs)/len(durs) if durs else 0.0
        info['operator_groups'] = sorted(list(info['operator_groups']))
        info['machine_ids'] = sorted(list(info['machine_ids']))
        # remove raw list to keep file compact
        del info['durations']

# operator mappings via utils.operator
try:
    from utils.operator import Operators
    from utils.site import Sites
    sites = Sites()
    ops_obj = Operators(sites)
    operator_mappings = {}
    for op in ops_obj.operators_object_list:
        operator_mappings[op.operator_id] = {
            'qualified_wcs': op.qualified_workcenters,
            'can_do_op_types': []
        }
    # infer can_do_op_types by checking which op_types have allowed WC in operator's qualified list
    for op_type, wcs in op_catalog.items():
        allowed_wcs = set(wcs.keys())
        for op in ops_obj.operators_object_list:
            if allowed_wcs.intersection(set(op.qualified_workcenters)):
                operator_mappings[op.operator_id]['can_do_op_types'].append(op_type)

except Exception as e:
    print('Could not import Operators or Sites:', e)
    operator_mappings = {}

# job_op_sequences simplified
job_op_sequences = {}
for jid, info in jobs.items():
    seq = []
    for op in info.get('ops', []):
        simplified = {
            'seq_idx': op['seq_idx'],
            'op_type': str(op['op_type']),
            'allowed_wcs': op.get('allowed_wcs', []),
            'base_duration': op.get('base_duration', 0.0),
            'per_wc': {str(p['wc']): p.get('estimated_duration') for p in op.get('per_wc', [])}
        }
        seq.append(simplified)
    job_op_sequences[jid] = {'ops': seq, 'total_dur_base': info.get('total_dur_base', 0.0)}

# Write files
out_op_catalog = history / 'op_catalog.json'
out_operator_map = history / 'operator_mappings.json'
out_job_seq = history / 'job_op_sequences.json'

out_op_catalog.write_text(json.dumps(op_catalog, indent=2))
out_operator_map.write_text(json.dumps(operator_mappings, indent=2))
out_job_seq.write_text(json.dumps(job_op_sequences, indent=2))

print('Wrote:', out_op_catalog, out_operator_map, out_job_seq)
