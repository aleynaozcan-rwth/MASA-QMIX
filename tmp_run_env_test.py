from MARL.common.arguments import get_mutable_args
from environment import MASAEnv


if __name__ == "__main__":
    # Build args and ensure deterministic seed
    args = get_mutable_args()
    setattr(args, 'seed', 123)
    setattr(args, 'initial_jobs', 4)
    # configure reasonable capacity
    setattr(args, 'n_agents', 10)

    # Create environment with auto_build enabled so initial jobs + generator start
    env = MASAEnv(args=args, auto_build=True, auto_start_arrivals=False)

    print('\n=== Initial jobs summary ===')
    env.print_initial_jobs_summary()

    print('\n=== get_env_info ===')
    print(env.get_env_info())

    # Run simulation for 20 simulated seconds to collect arrivals
    print('\n=== Running sim for 20s to capture dynamic arrivals (if any) ===')
    try:
        env.env.run(until=20.0)
    except Exception as e:
        print('Run exception:', e)

    print('\n=== First 20s Gantt records (showing decision_trace if present) ===')
    for rec in env.gantt_records[:20]:
        print(rec.get('decision_trace', rec))

    print('\n=== Total jobs generated ===', len(env.jobs))
    print('\n=== Jobs list brief ===')
    for j in env.jobs[:10]:
        print(f'Job {j.id} arrival={j.arrival_time:.2f} ops={len(j.operations)}')
