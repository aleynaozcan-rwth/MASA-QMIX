#!/usr/bin/env python3
"""Test concurrency: create 2 WCs, 4 machines, 2 operators and run 20 sim time.
Print any overlapping operations in the same WorkCenter (indicating per-machine concurrency).
"""
import json
from environment import MASAEnv

def find_concurrency(gantt):
    # gantt_records: list of (start, end, op_tag, wc, job_id, operator_group)
    examples = []
    for i in range(len(gantt)):
        s1,e1,op1,wc1,j1,grp1 = gantt[i]
        for j in range(i+1,len(gantt)):
            s2,e2,op2,wc2,j2,grp2 = gantt[j]
            if wc1 != wc2:
                continue
            # overlap?
            if not (e1 <= s2 or e2 <= s1):
                examples.append({
                    'pair': (i,j),
                    'wc': wc1,
                    'rec1': (s1,e1,op1,j1,grp1),
                    'rec2': (s2,e2,op2,j2,grp2)
                })
    return examples


def main():
    env = MASAEnv(num_jobs=0, episode_limit=20)
    # define two WCs (0,1) with two machines each
    env.workcenters_meta.machine_list = ['M1','M2','M3','M4']
    env.workcenters_meta.machine_index = {m:i for i,m in enumerate(env.workcenters_meta.machine_list)}
    env.workcenters_meta.machine_registry = {
        'M1': {'workcenter': 0, 'capabilities': [0], 'speed_factor': 1.0},
        'M2': {'workcenter': 0, 'capabilities': [0], 'speed_factor': 1.0},
        'M3': {'workcenter': 1, 'capabilities': [0], 'speed_factor': 1.0},
        'M4': {'workcenter': 1, 'capabilities': [0], 'speed_factor': 1.0},
    }
    # eligible operator groups per WC
    env.workcenters_meta.eligible_operator_groups_by_wc = {0:[0], 1:[0]}
    env.num_wcs = 2
    env.num_ops = 2
    # create fresh sim environment resources matching the new metadata
    env.env = env.env.__class__()
    env.machine_resources = [env.env.process if False else None for _ in env.workcenters_meta.machine_list]
    # proper create simpy.Resources
    import simpy
    env.machine_resources = [simpy.Resource(env.env, capacity=1) for _ in env.workcenters_meta.machine_list]
    env.wc_resources = [simpy.Resource(env.env, capacity=1) for _ in range(env.num_wcs)]
    # give operator group capacity=2 so two operators can be busy concurrently
    env.operator_groups = [simpy.Resource(env.env, capacity=2) for _ in range(env.num_ops)]

    # Add two jobs targeting WC 0 but leave machines free so both can be processed concurrently
    # Each job will have a single op: op_type 0, allowed_wcs [0], per_wc duration 5
    op = (0, [0], {0:5.0})
    env.add_job([op])
    env.add_job([op])
    # drive the decision loop so job processes can resume and run
    # loop until env finishes or we hit the time limit
    while not env.done and env.env.now < env.episode_limit:
        batch, sim_t = env.wait_for_decisions()
        if not batch:
            break
        # debug: show decision items
        try:
            import pprint
            print('[DEBUG] batch items:')
            pprint.pprint(batch)
        except Exception:
            pass
        # for each pending decision, pick the first allowed machine index
        for d in batch:
            # prefer explicit allowed_machine_indices, but fallback to avail_row
            allowed = d.get('allowed_machine_indices') or []
            if not allowed:
                # use avail_row to infer available machine indices
                try:
                    import numpy as _np
                    ar = d.get('avail_row')
                    if ar is not None:
                        allowed = [int(i) for i, v in enumerate(_np.asarray(ar).tolist()) if int(v) == 1]
                except Exception:
                    allowed = []
            if allowed:
                # pick different machines for different jobs (round-robin by job id)
                jid = int(d.get('job_id', 0))
                choice = allowed[jid % len(allowed)]
            else:
                # fallback: pick 0
                choice = 0
            # call resume (this will succeed the job's resume event)
            try:
                d.get('resume')(choice)
            except Exception:
                pass

    # After run, inspect gantt_records
    gantt = getattr(env, 'gantt_records', [])
    examples = find_concurrency(gantt)

    out = {
        'gantt_len': len(gantt),
        'gantt_records': gantt,
        'concurrency_same_wc_examples': examples,
    }
    print(json.dumps(out, indent=2, default=float))
    try:
        from utils.io_control import allow_history_writes
    except Exception:
        def allow_history_writes():
            return False

    if allow_history_writes():
        try:
            import os
            os.makedirs('artifacts', exist_ok=True)
            with open('artifacts/test_concurrency_out.json','w') as f:
                json.dump(out, f, indent=2, default=float)
        except Exception as e:
            print('[WARN] Could not write artifacts:', e)
    else:
        print('[INFO] history writes disabled; skipping artifacts/test_concurrency_out.json')

if __name__ == '__main__':
    main()
