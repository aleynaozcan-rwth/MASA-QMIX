#!/usr/bin/env python3
"""Compare YAML configs under configs/ with in-module defaults.

Usage:
  PYTHONPATH=. python3 tools/compare_config_to_modules.py [--config configs/env_config_enabled.yaml]

Prints a human-readable report of keys only-in-config, only-in-defaults, and differing values.
"""
from pathlib import Path
import argparse
import yaml
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs'
DEFAULT_FILES = ['env_config_enabled.yaml', 'env_config.yaml', 'env_no_arrival.yaml']

# Import module defaults
try:
    from utils.workcenter import DEFAULT_WORKCENTERS
except Exception:
    DEFAULT_WORKCENTERS = {}

try:
    from environment import DEFAULT_ENV_PARAMS
except Exception:
    DEFAULT_ENV_PARAMS = {}

try:
    from utils.operator import DEFAULT_OPERATORS
except Exception:
    DEFAULT_OPERATORS = None

try:
    from utils.task_generator import DEFAULT_TASKGEN_PARAMS
except Exception:
    DEFAULT_TASKGEN_PARAMS = None

MODULE_DEFAULTS = {
    'workcenters': DEFAULT_WORKCENTERS,
    'env': DEFAULT_ENV_PARAMS,
}
if DEFAULT_OPERATORS is not None:
    MODULE_DEFAULTS['operators'] = {'operators': DEFAULT_OPERATORS}
if DEFAULT_TASKGEN_PARAMS is not None:
    MODULE_DEFAULTS['task_generator'] = DEFAULT_TASKGEN_PARAMS


def load_yaml(path: Path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None


def iter_paths(obj, prefix=''):
    """Yield (path, value) for every leaf in obj where path is dot-separated."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            newp = f"{prefix}.{k}" if prefix else str(k)
            yield from iter_paths(v, newp)
    elif isinstance(obj, (list, tuple)):
        # treat whole list as a value (do not expand indexed paths)
        yield prefix, obj
    else:
        yield prefix, obj


def compare_dicts(cfg: dict, defaults: dict, base_path=''):
    only_in_cfg = []
    only_in_def = []
    different = []  # list of tuples (path, cfg_value, def_value)

    cfg_keys = set(cfg.keys() if isinstance(cfg, dict) else [])
    def_keys = set(defaults.keys() if isinstance(defaults, dict) else [])

    for k in sorted(cfg_keys - def_keys):
        path = f"{base_path}.{k}" if base_path else str(k)
        only_in_cfg.append(path)

    for k in sorted(def_keys - cfg_keys):
        path = f"{base_path}.{k}" if base_path else str(k)
        only_in_def.append(path)

    for k in sorted(cfg_keys & def_keys):
        path = f"{base_path}.{k}" if base_path else str(k)
        cv = cfg.get(k)
        dv = defaults.get(k)
        # If both are dicts, recurse
        if isinstance(cv, dict) and isinstance(dv, dict):
            a, b, c = compare_dicts(cv, dv, path)
            only_in_cfg.extend(a)
            only_in_def.extend(b)
            different.extend(c)
        else:
            # For lists and scalars, compare values directly
            if cv != dv:
                different.append((path, cv, dv))
    return only_in_cfg, only_in_def, different


def compare_configs_to_module_defaults(cfg_dict):
    results = []
    for module_key, defaults in MODULE_DEFAULTS.items():
        if not defaults:
            continue
        cfg_section = None
        # map module_key to expected config section
        if module_key == 'workcenters':
            cfg_section = cfg_dict.get('work_centers') or cfg_dict.get('workcenters') or cfg_dict.get('machines') or {}
            # If config provides machines keyed by M1.. names, compare against DEFAULT_WORKCENTERS['machines'] if available
            if not cfg_section and 'machines' in cfg_dict:
                cfg_section = cfg_dict.get('machines')
            defaults_section = defaults.get('machines') if isinstance(defaults, dict) and 'machines' in defaults else defaults
        elif module_key == 'env':
            cfg_section = cfg_dict
            defaults_section = defaults
        elif module_key == 'operators':
            cfg_section = {'operators': cfg_dict.get('operators')} if 'operators' in cfg_dict else {}
            defaults_section = defaults
        elif module_key == 'task_generator':
            cfg_section = {'task_generator': cfg_dict.get('task_generator')} if 'task_generator' in cfg_dict else {}
            defaults_section = defaults
        elif module_key == 'processing_time_means':
            cfg_section = {'processing_time_means': cfg_dict.get('processing_time_means')} if 'processing_time_means' in cfg_dict else {}
            defaults_section = defaults
        else:
            cfg_section = cfg_dict.get(module_key, {})
            defaults_section = defaults

        if cfg_section is None:
            cfg_section = {}

        only_cfg, only_def, diff = compare_dicts(cfg_section if isinstance(cfg_section, dict) else {}, defaults_section if isinstance(defaults_section, dict) else {}, base_path=module_key)
        results.append((module_key, only_cfg, only_def, diff))
    return results


def _pretty(v):
    try:
        return json.dumps(v, indent=2, sort_keys=True)
    except Exception:
        return repr(v)


def print_report(file_label, only_cfg, only_def, diff):
    print(f"\nReport for {file_label}:\n")
    print("Only in config:")
    for p in only_cfg:
        print("  ", p)
    print("Only in defaults:")
    for p in only_def:
        print("  ", p)
    print("Different values (path \t | config value | default value):")
    for path, cv, dv in diff:
        print(f"  {path}")
        print(f"    - config:  {_pretty(cv)}")
        print(f"    - default: {_pretty(dv)}")
    print("\nSummary: config-only=%d  default-only=%d  differing=%d\n" % (len(only_cfg), len(only_def), len(diff)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Specific config file under configs/ to compare')
    parser.add_argument('--export', type=str, default=None, help='Path to write JSON diff report (e.g. tools/config_diff_report.json)')
    args = parser.parse_args()

    # Default to focusing on the enabled config when none provided
    files = [args.config] if args.config is not None else ['env_config_enabled.yaml']

    for fname in files:
        path = CONFIG_DIR / fname
        cfg = load_yaml(path)
        if cfg is None:
            print(f"Config file not found: {path} — skipping")
            continue
        results = compare_configs_to_module_defaults(cfg)
        # aggregate
        all_only_cfg = []
        all_only_def = []
        all_diff = []
        for module_key, only_cfg, only_def, diff in results:
            all_only_cfg.extend(only_cfg)
            all_only_def.extend(only_def)
            all_diff.extend(diff)
        print_report(fname, all_only_cfg, all_only_def, all_diff)

        if args.export:
            out = {
                'config_file': str(path),
                'only_in_config': all_only_cfg,
                'only_in_defaults': all_only_def,
                'differing': [
                    {'path': p, 'config_value': cv, 'default_value': dv} for (p, cv, dv) in all_diff
                ]
            }
            export_path = Path(args.export)
            try:
                export_path.parent.mkdir(parents=True, exist_ok=True)
                with open(export_path, 'w') as fh:
                    json.dump(out, fh, indent=2, sort_keys=True)
                print(f"Wrote JSON diff report to {export_path}")
            except Exception as e:
                print(f"Failed to write export file {export_path}: {e}")


if __name__ == '__main__':
    main()
