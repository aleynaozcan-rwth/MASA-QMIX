#!/usr/bin/env python3
"""Check that MASAEnv picks up EXP_* env variables for reward weights."""
import os
import json
from environment import MASAEnv

def main():
    os.environ['EXP_ALPHA'] = '1.0'
    os.environ['EXP_BETA'] = '0.3'
    os.environ['EXP_GAMMA'] = '0.1'
    os.environ['EXP_DELTA'] = '0.1'

    env = MASAEnv(num_jobs=0)
    # call reset to trigger env variable reads during setup
    obs, info = env.reset()
    avail = info.get('avail_actions')
    try:
        avail_shape = str(avail.shape) if hasattr(avail, 'shape') else None
    except Exception:
        avail_shape = None
    out = {
        'alpha': env.alpha,
        'beta': env.beta,
        'gamma': env.gamma,
        'delta': env.delta,
        'reward_weights_runtime': {
            'alpha': env.alpha,
            'beta': env.beta,
            'gamma': env.gamma,
            'delta': env.delta
        },
        'info_avail_shape': avail_shape
    }
    print(json.dumps(out, indent=2))
    try:
        from utils.io_control import allow_history_writes
    except Exception:
        def allow_history_writes():
            return False

    if allow_history_writes():
        try:
            os.makedirs('artifacts', exist_ok=True)
            with open('artifacts/reward_envcheck.json','w') as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            print('[WARN] write artifacts failed:', e)
    else:
        print('[INFO] history writes disabled; skipping artifacts/reward_envcheck.json')

if __name__ == '__main__':
    main()
