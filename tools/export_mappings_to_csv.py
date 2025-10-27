#!/usr/bin/env python3
import json
from pathlib import Path

history = Path('my_data_and_graph/historydata')
files = {
    'op_catalog': history / 'op_catalog.json',
    'operator_mappings': history / 'operator_mappings.json',
    'job_op_sequences': history / 'job_op_sequences.json'
}

if not files['op_catalog'].exists():
    print('Missing op_catalog.json — run tools first')
    raise SystemExit(1)

op_catalog = json.loads(files['op_catalog'].read_text())
operator_mappings = json.loads(files['operator_mappings'].read_text()) if files['operator_mappings'].exists() else {}
job_op_sequences = json.loads(files['job_op_sequences'].read_text()) if files['job_op_sequences'].exists() else {}

# Write op_catalog.csv
with (history / 'op_catalog.csv').open('w') as f:
    f.write('op_type,wc,machine_ids,operator_groups,min_duration,avg_duration,max_duration\n')
    for op_type, wcs in sorted(op_catalog.items(), key=lambda x: int(x[0] if x[0].isdigit() else -1)):
        for wc, info in sorted(wcs.items(), key=lambda x: int(x[0])):
            mids = ';'.join(info.get('machine_ids', []))
            grps = ';'.join(str(g) for g in info.get('operator_groups', []))
            f.write(f"{op_type},{wc},\"{mids}\",\"{grps}\",{info.get('min_duration',0)},{info.get('avg_duration',0)},{info.get('max_duration',0)}\n")

# Write operator_mappings.csv
with (history / 'operator_mappings.csv').open('w') as f:
    f.write('operator_id,qualified_wcs,can_do_op_types\n')
    for op_id, info in sorted(operator_mappings.items(), key=lambda x: int(x[0])):
        q = ';'.join(str(w) for w in info.get('qualified_wcs', []))
        types = ';'.join(info.get('can_do_op_types', []))
        f.write(f"{op_id},\"{q}\",\"{types}\"\n")

# Write job_op_sequences.csv
with (history / 'job_op_sequences.csv').open('w') as f:
    f.write('job_id,seq_idx,op_type,allowed_wcs,base_duration,per_wc_estimates\n')
    for job_id, info in sorted(job_op_sequences.items(), key=lambda x: int(x[0])):
        for op in info.get('ops', []):
            allowed = ';'.join(str(w) for w in op.get('allowed_wcs', []))
            perwc = ';'.join(f"{k}:{v}" for k, v in op.get('per_wc', {}).items())
            base = op.get('base_duration', '')
            f.write(f"{job_id},{op.get('seq_idx')},{op.get('op_type')},\"{allowed}\",{base},\"{perwc}\"\n")

print('Wrote CSVs to', str(history))
