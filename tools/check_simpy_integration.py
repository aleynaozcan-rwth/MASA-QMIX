#!/usr/bin/env python3
"""Sanity check: verify MASAEnv is using SimPy resources and environment.
Prints simpy version and verifies types for env and resources.
"""
import simpy
from environment import MASAEnv

def main():
    print('simpy version:', getattr(simpy, '__version__', 'unknown'))
    env = MASAEnv(num_jobs=0)
    print('env.env type:', type(env.env))
    print('is simpy.Environment:', isinstance(env.env, simpy.Environment))
    # ensure resources exist after reset
    env.reset()
    print('machine_resources count:', len(getattr(env, 'machine_resources', [])))
    mr0 = None
    if getattr(env, 'machine_resources', None):
        mr0 = env.machine_resources[0]
        print('machine resource type:', type(mr0))
        print('machine resource capacity:', getattr(mr0, 'capacity', 'N/A'))
    print('operator_groups count:', len(getattr(env, 'operator_groups', [])))
    og0 = None
    if getattr(env, 'operator_groups', None):
        og0 = env.operator_groups[0]
        print('operator group type:', type(og0))
        print('operator group capacity:', getattr(og0, 'capacity', 'N/A'))
    # ensure env.process exists and small run works
    try:
        def _proc(e):
            yield e.timeout(0)
        p = env.env.process(_proc(env.env))
        env.env.run(until=0)
        print('env.run succeeded')
    except Exception as e:
        print('env.run failed:', e)

if __name__ == '__main__':
    main()
