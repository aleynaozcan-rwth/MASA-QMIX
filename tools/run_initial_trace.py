"""Run a controlled initial-only MASAEnv with dynamic arrivals disabled and
print the first decision batch plus a resume->assignment->next-observation trace.
"""
import sys
sys.path.insert(0, '')
from environment import MASAEnv
from utils.gantt import format_gantt_records
import numpy as np

# instantiate with the no-arrival config
env = MASAEnv(num_jobs=4, num_operators=2, num_wcs=3, seed=123, obs_dim_agent=6, config_path='configs/env_no_arrival.yaml')

print('\n=== MACHINE REGISTRY ===')
mlist = getattr(env.workcenters_meta, 'machine_list', [])
mreg = getattr(env.workcenters_meta, 'machine_registry', {})
print('machine_list (index:name):')
for i, m in enumerate(mlist):
    print(f'  {i}: {m}')
print('\nmachine_registry entries:')
for mname, md in mreg.items():
    print(f"  {mname}: workcenter={md.get('workcenter')}, capabilities={md.get('capabilities')}, speed_factor={md.get('speed_factor')}")

print('\n=== INITIAL JOBS (human readable) ===')
if hasattr(env, 'print_jobs_human_readable'):
    env.print_jobs_human_readable()
else:
    for job in env.jobs:
        print(f'Job {job.id}: arrival={getattr(job, "arrival_time", 0.0)} ops={len(job.operations)}')
        for i, op in enumerate(job.operations):
            print('  ', i, op)

# Spawn job processes and compute initial obs
env.reset()

batch, sim_time = env.wait_for_decisions()
print(f"\n=== FIRST DECISION BATCH @ sim_time={sim_time} (len={len(batch)}) ===")
for item in batch:
    jid = item.get('job_id')
    print('\n--- Decision for Job', jid, '---')
    obs = item.get('obs')
    # print observation with dynamic length (no hard-coded '11')
    print(f'obs (len={len(obs)}):', [float(round(x,6)) for x in obs])
    ar = item.get('avail_row')
    print('avail_row:', [int(x) for x in ar])
    print('allowed_machine_indices:', item.get('allowed_machine_indices'))
    print('allowed_machines:', item.get('allowed_machines'))
    print('allowed_machine_indices:', item.get('allowed_machine_indices'))
    print('per_machine_durations:', {int(k): float(v) for k, v in (item.get('per_machine_durations') or {}).items()})
    print('base_duration:', float(item.get('base_duration')))
    print('eligible_ops_by_wc:', item.get('eligible_ops_by_wc'))
    print('resume callable present:', callable(item.get('resume')))

# Show global state and avail matrix
print('\n=== GLOBAL STATE VECTOR (first 10 entries) ===')
state = env._build_state_vector()
print([float(round(x,6)) for x in state[:10]])

avail_mat = env._build_avail_actions()
print('\n=== AVAIL ACTIONS MATRIX (jobs x machines) ===')
print('shape=', avail_mat.shape)
for i, row in enumerate(avail_mat):
    print(f' Job {i} avail row:', [int(x) for x in row])

# Simulate a naive policy: pick the first available machine per job and resume
print('\n=== APPLY NAIVE POLICY (first available machine) ===')
choices = []
for item in batch:
    ar = item.get('avail_row')
    allowed_inds = item.get('allowed_machine_indices') or []
    chosen = None
    # choose first allowed and available index
    for mi in allowed_inds:
        if int(ar[mi]) == 1:
            chosen = int(mi)
            break
    if chosen is None and allowed_inds:
        chosen = int(allowed_inds[0])
    choices.append((item.get('job_id'), chosen))
    print(f"Job {item.get('job_id')} -> chosen machine idx {chosen}")

# Resume choices
for item, (_, choice) in zip(batch, choices):
    resume = item.get('resume')
    print(f"Resuming Job {item.get('job_id')} with choice {choice}")
    resume(choice)

# advance sim until next decision(s)
env.env.run(until=env.env.now + 5.0)
print('\nAfter running: sim.now=', env.env.now)
print('Gantt records sample:', format_gantt_records(env.gantt_records[:5]))

# print next decision batch if any
next_batch, next_time = env.wait_for_decisions()
print(f"\n=== NEXT DECISION BATCH @ sim_time={next_time} (len={len(next_batch)}) ===")
for item in next_batch:
    print('Job', item.get('job_id'), 'obs:', [float(round(x,6)) for x in item.get('obs')])

print('\nDone.')
