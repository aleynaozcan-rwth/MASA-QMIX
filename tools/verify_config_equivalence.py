#!/usr/bin/env python3
"""Verify that MASAEnv with config and without config produce equivalent parameters.

Run:
  PYTHONPATH=. python3 tools/verify_config_equivalence.py

Compares:
 - env.workcenters.machine_registry
 - env.config['reward_params']
 - env.config['task_generator']

Prints differences and a final summary.
"""
from pathlib import Path
import json
import pprint

try:
    from environment import MASAEnv, DEFAULT_ENV_PARAMS
except Exception as e:
    raise


def pretty(v):
    try:
        return json.dumps(v, indent=2, sort_keys=True)
    except Exception:
        return pprint.pformat(v)


def compare_dicts(path, a, b, diffs):
    # Compare dict-like or scalar values
    if isinstance(a, dict) and isinstance(b, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        for k in sorted(a_keys - b_keys):
            diffs.append((f"{path}.{k}", a[k], None))
        for k in sorted(b_keys - a_keys):
            diffs.append((f"{path}.{k}", None, b[k]))
        for k in sorted(a_keys & b_keys):
            av = a.get(k)
            bv = b.get(k)
            if isinstance(av, dict) and isinstance(bv, dict):
                compare_dicts(f"{path}.{k}", av, bv, diffs)
            else:
                if av != bv:
                    diffs.append((f"{path}.{k}", av, bv))
    else:
        if a != b:
            diffs.append((path, a, b))


def main():
    print("Initializing MASAEnv with config (auto_load_config=True)")
    env_cfg = MASAEnv(config_path=str(Path('configs') / 'env_config_enabled.yaml'), auto_load_config=True, auto_build=False, auto_start_arrivals=False)

    print("Initializing MASAEnv without config (auto_load_config=False)")
    env_nocfg = MASAEnv(auto_load_config=False, auto_build=False, auto_start_arrivals=False)

    # Workcenters machine_registry
    wc_a = getattr(getattr(env_cfg, 'workcenters', None), 'machine_registry', {}) or {}
    wc_b = getattr(getattr(env_nocfg, 'workcenters', None), 'machine_registry', {}) or {}

    diffs = []
    # Compare machine_registry keys
    a_keys = set(wc_a.keys())
    b_keys = set(wc_b.keys())
    for k in sorted(a_keys - b_keys):
        diffs.append((f"workcenters.machine_registry.{k}", wc_a[k], None))
    for k in sorted(b_keys - a_keys):
        diffs.append((f"workcenters.machine_registry.{k}", None, wc_b[k]))
    for k in sorted(a_keys & b_keys):
        compare_dicts(f"workcenters.machine_registry.{k}", wc_a.get(k), wc_b.get(k), diffs)

    # Reward params
    cfg_a = env_cfg.config if isinstance(env_cfg.config, dict) else {}
    cfg_b = env_nocfg.config if isinstance(env_nocfg.config, dict) else None
    if cfg_b is None:
        # fall back to module defaults
        cfg_b = DEFAULT_ENV_PARAMS

    rp_a = cfg_a.get('reward_params', {})
    rp_b = cfg_b.get('reward_params', {})
    compare_dicts('reward_params', rp_a or {}, rp_b or {}, diffs)

    # Task generator
    tg_a = cfg_a.get('task_generator', {})
    tg_b = cfg_b.get('task_generator', {})
    compare_dicts('task_generator', tg_a or {}, tg_b or {}, diffs)

    # Print results
    if not diffs:
        print('\n✅ Config-free run matches enabled config.')
        return

    print('\n⚠️ Differences found between config-enabled and config-free runs:')
    for path, av, bv in diffs:
        print(f"- {path}")
        print(f"    - with config:  {pretty(av)}")
        print(f"    - no config:    {pretty(bv)}")

    print(f"\nSummary: {len(diffs)} differing paths")
    print('\n⚠️ Differences found between config-enabled and config-free runs.')


if __name__ == '__main__':
    main()
