#!/usr/bin/env python3
"""Test arrivals: run 200 sim time with a dummy TaskGenerator to ensure dynamic_adds > 0."""
import json
from environment import MASAEnv

class DummyJob:
    def __init__(self, index_id=0, time_span=1.0, name='Op1'):
        self.index_id = index_id
        self.time_span = time_span
        self.name = name

class DummyTG:
    def generate_constrained_task(self, jobagent_id=None):
        # return a single op-like object with index_id/time_span
        return [DummyJob(index_id=0, time_span=2.0, name='Op1')]


def main():
    env = MASAEnv(num_jobs=0, episode_limit=200)
    # minimal machines/operators so arrival process can map wcs
    env.workcenters_meta.machine_list = ['M1']
    env.workcenters_meta.machine_index = { 'M1': 0 }
    env.workcenters_meta.machine_registry = {'M1': {'workcenter': 0, 'capabilities': [0], 'speed_factor':1.0}}
    env.workcenters_meta.eligible_operator_groups_by_wc = {0:[0]}
    env.num_wcs = 1
    env.num_ops = 1

    import simpy
    env.env = simpy.Environment()
    env.machine_resources = [simpy.Resource(env.env, capacity=1)]
    env.wc_resources = [simpy.Resource(env.env, capacity=1)]
    env.operator_groups = [simpy.Resource(env.env, capacity=1)]

    # inject a dummy TaskGenerator and start the dynamic loop with lambda=0.05
    env._task_generator = DummyTG()
    env.env.process(env._dynamic_arrival_loop(0.05))

    # run for 200 simulated time
    env.env.run(until=200)

    # count added jobs (jobs list length)
    added = len(env.jobs)
    out = {
        'num_jobs_total': added,
        'dynamic_adds_count': added, # since initially 0 jobs
    }
    print(json.dumps(out, indent=2))
    try:
        import os
        os.makedirs('artifacts', exist_ok=True)
        with open('artifacts/test_arrivals_out.json','w') as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        print('[WARN] write artifacts failed:', e)

if __name__ == '__main__':
    main()
