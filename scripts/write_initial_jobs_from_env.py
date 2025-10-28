from environment import MASAEnv
import os

env = MASAEnv(num_jobs=10, num_operators=2, num_wcs=3)
history_dir = './my_data_and_graph/historydata/'
os.makedirs(history_dir, exist_ok=True)
init_path = os.path.join(history_dir, 'initial_jobs_regenerated.txt')
with open(init_path, 'w') as hf:
    hf.write('Initial Job -> Operation mapping\n')
    hf.write('Format: JobID | OpIdx | OpType | Allowed_WCs | OpGroups | BaseDur\n\n')
    for job in getattr(env, 'jobs', []):
        hf.write(f'Job {int(job.id)}:\n')
        for idx, op in enumerate(getattr(job, 'operations', [])):
            try:
                if isinstance(op, (list, tuple)) and len(op) == 2:
                    allowed_wcs, dur = op
                    op_type = 'legacy'
                    hf.write(f'  Op {idx} | Type {op_type} | WCs {allowed_wcs} | Dur {float(dur):.3f}\n')
                else:
                    op_type = op[0]
                    allowed_wcs = op[1]
                    third = op[2]
                    if isinstance(third, dict):
                        per_wc = third
                        groups_by_wc = []
                        for wc in allowed_wcs:
                            try:
                                eligible = env.workcenters_meta.eligible_operator_groups_by_wc.get(int(wc), [])
                            except Exception:
                                eligible = []
                            groups_by_wc.append({'wc': int(wc), 'eligible_ops': eligible})
                        hf.write(f'  Op {idx} | Type {op_type} | WCs {allowed_wcs} | Groups {groups_by_wc} | base_per_wc_durations:\n')
                        for wc in allowed_wcs:
                            try:
                                dur_wc = float(per_wc.get(int(wc), 0.0))
                            except Exception:
                                dur_wc = 0.0
                            try:
                                eligible = env.workcenters_meta.eligible_operator_groups_by_wc.get(int(wc), [])
                            except Exception:
                                eligible = []
                            hf.write(f'    WC{wc} -> dur={dur_wc:.3f} | eligible_ops={eligible}\n')
                    else:
                        base_dur = float(third)
                        groups_info = []
                        for wc in allowed_wcs:
                            try:
                                eligible = env.workcenters_meta.eligible_operator_groups_by_wc.get(int(wc), [])
                            except Exception:
                                eligible = []
                            groups_info.append({'wc': int(wc), 'eligible_ops': eligible})
                        hf.write(f'  Op {idx} | Type {op_type} | WCs {allowed_wcs} | Groups {groups_info} | base_dur {base_dur:.3f}\n')
            except Exception:
                hf.write(f'  Op {idx} | malformed: {op}\n')
        hf.write('\n')

print('Wrote', init_path)
