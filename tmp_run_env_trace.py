from MARL.common.arguments import get_mutable_args
from environment import MASAEnv


if __name__ == "__main__":
    args = get_mutable_args()
    setattr(args, 'seed', 123)
    setattr(args, 'initial_jobs', 4)
    setattr(args, 'n_agents', 10)

    env = MASAEnv(args=args, auto_build=True)

    print('\nStarting long run to allow ops to complete (until t=200)')
    try:
        env.env.run(until=200.0)
    except Exception as e:
        print('Run exception:', e)

    print('\nGantt records count:', len(env.gantt_records))
    # Print first record with decision_trace if present
    for rec in env.gantt_records:
        if isinstance(rec, dict) and 'decision_trace' in rec:
            print('\nSample DecisionTrace (from gantt_records):')
            import json
            print(json.dumps(rec['decision_trace'], indent=2))
            break
    else:
        print('\nNo decision_trace found in gantt_records; printing first few gantt records:')
        import json
        for rec in env.gantt_records[:5]:
            print(json.dumps(rec, default=str))

    print('\nTotal jobs:', len(env.jobs))
    print('n_actions from get_env_info:', env.get_env_info()['n_actions'])

    # Build a sample DecisionTrace for the first job by querying machine/operator status
    if env.jobs:
        job = env.jobs[0]
        mlist = getattr(env.workcenters_meta, 'machine_list', []) or []
        elig_entries = []
        now_t = float(env.env.now)
        for midx in range(len(mlist)):
            mname = mlist[midx]
            # machine busy
            m_busy = False
            try:
                res = env.machine_resources[midx]
                m_busy = len(getattr(res, 'users', [])) > 0
            except Exception:
                m_busy = False
            # operator candidates
            op_cands = []
            try:
                for op_obj in getattr(env.operators, 'operators_object_list', []) or []:
                    try:
                        opid = getattr(op_obj, 'operator_id', None)
                        busy = bool(getattr(op_obj, 'is_busy', False))
                        # simple qualification by machine name membership
                        is_qualified = mname in getattr(op_obj, 'qualified_machines', [])
                        op_cands.append({'operator_id': str(opid), 'qualified': bool(is_qualified), 'busy': busy, 'next_free': now_t if not busy else now_t + 1.0, 'load': int(len(getattr(op_obj, 'history', []) or []))})
                    except Exception:
                        continue
            except Exception:
                op_cands = []
            elig_entries.append({'machine_id': mname, 'machine_busy': m_busy, 'machine_available_at': now_t, 'operator_candidates': op_cands})

        sample_dt = {'chosen_machine': mlist[0] if mlist else 'M0', 'chosen_operator': None, 'policy_reason': 'sample_trace', 'at_time': now_t, 'eligibilities': elig_entries}
        import json
        print('\nSample DecisionTrace (synthetic):')
        print(json.dumps(sample_dt, indent=2))
