#!/usr/bin/env python3
import json
from pathlib import Path
from utils.workcenter import WorkCenters

history = Path("my_data_and_graph/historydata")
js = history / 'job_ops_summary.json'
if not js.exists():
    print('job_ops_summary.json not found; run parse_initial_jobs first')
    raise SystemExit(1)

jobs = json.loads(js.read_text())
workcenters = WorkCenters()

# build machine registry mapping: wc -> speed_factor and eligible operator groups
machine_info = {}
for mid, mdata in workcenters.machine_registry.items():
    wc = int(mdata['workcenter'])
    speed = float(mdata.get('speed_factor', 1.0))
    # groups eligible for this WorkCenter
    groups = workcenters.eligible_operator_groups_by_wc.get(wc, [])
    machine_info[wc] = {'machine_id': mid, 'speed_factor': speed, 'operator_groups': groups}

# Build detailed per-job structure
detailed = {}
op_type_map = {}
for jid, info in jobs.items():
    detailed[jid] = {'ops': [], 'total_dur_base': info.get('total_dur', 0.0)}
    for op in info['ops']:
        op_idx = op['op_idx']
        op_type = op['op_type']
        base_dur = op['dur']
        allowed = op['wcs']
        per_wc = []
        for wc in allowed:
            wcinfo = machine_info.get(wc, {'speed_factor': 1.0, 'operator_groups': []})
            est_dur = base_dur * wcinfo['speed_factor']
            per_wc.append({
                'wc': wc,
                'machine_id': wcinfo.get('machine_id'),
                'speed_factor': wcinfo.get('speed_factor', 1.0),
                'operator_groups': wcinfo.get('operator_groups', []),
                'estimated_duration': round(float(est_dur), 3),
            })
        detailed[jid]['ops'].append({
            'seq_idx': op_idx,                     # sequence index inside job (0-based)
            'op_type': op_type,                    # operation type/class (0..8)
            'base_duration': base_dur,
            'allowed_wcs': allowed,
            'per_wc': per_wc,
        })
        # accumulate op_type map
        op_type_entry = op_type_map.setdefault(str(op_type), {'allowed_wcs': set(), 'operator_groups': set()})
        for wc in allowed:
            op_type_entry['allowed_wcs'].add(wc)
            for g in machine_info.get(wc, {}).get('operator_groups', []):
                op_type_entry['operator_groups'].add(g)

# convert sets to sorted lists
for k, v in op_type_map.items():
    v['allowed_wcs'] = sorted(list(v['allowed_wcs']))
    v['operator_groups'] = sorted(list(v['operator_groups']))

# Save detailed files
out_json = history / 'job_ops_detailed.json'
out_txt = history / 'job_ops_detailed.txt'
out_map = history / 'op_type_map.json'

out_json.write_text(json.dumps(detailed, indent=2))

with out_txt.open('w') as f:
    f.write('Job operations detailed (per-workcenter estimated durations)\n')
    for jid in sorted(detailed.keys(), key=lambda x: int(x)):
        info = detailed[jid]
        f.write(f"Job {jid}: total_base_dur={info['total_dur_base']:.3f}\n")
        for op in info['ops']:
            f.write(f"  Seq {op['seq_idx']} | Type {op['op_type']} | base={op['base_duration']:.3f}\n")
            for p in op['per_wc']:
                f.write(f"    WC{p['wc']} (machine {p['machine_id']}) | groups={p['operator_groups']} | speed={p['speed_factor']} | est_dur={p['estimated_duration']:.3f}\n")
        f.write('\n')

out_map.write_text(json.dumps(op_type_map, indent=2))
print('Wrote:', out_json, out_txt, out_map)
print(out_txt.read_text()[:2000])
