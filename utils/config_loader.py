"""Simple YAML config loader for MASA-QMIX.

Usage:
  from utils.config_loader import load_config
  cfg = load_config('configs/env_config.yaml')

This helper requires PyYAML. If PyYAML is not installed the loader will raise
an explicit error with a hint on how to install it.
"""
from pathlib import Path
import json
import copy
import logging

def _ensure_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception:
        raise RuntimeError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")

def load_config(path: str = "configs/env_config.yaml"):
    """Load and return configuration as a Python dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    yaml = _ensure_yaml()
    with p.open('r') as f:
        return yaml.safe_load(f)


def merge_config(defaults: dict, cfg: dict | None) -> dict:
    """Deep-merge a loaded YAML config (if present) into in-module defaults.

    - If cfg is None, return a deep copy of defaults.
    - Otherwise, recursively merge cfg into a copy of defaults and return it.
    - Do not mutate either input.
    - Log overridden keys at debug level.
    """
    logger = logging.getLogger(__name__)

    def _merge(a, b, path=""):
        # a is the base (copied defaults), b is the overriding cfg
        for k, v in (b or {}).items():
            cur_path = f"{path}.{k}" if path else k
            if k in a and isinstance(a[k], dict) and isinstance(v, dict):
                _merge(a[k], v, cur_path)
            else:
                if k in a:
                    try:
                        logger.debug("Overriding config key %s: %r -> %r", cur_path, a.get(k), v)
                    except Exception:
                        logger.debug("Overriding config key %s", cur_path)
                a[k] = copy.deepcopy(v)

    base = copy.deepcopy(defaults) if defaults is not None else {}
    if cfg is None:
        return base
    try:
        _merge(base, cfg)
    except Exception:
        logger.exception("merge_config failed; returning copy of defaults")
        return base
    return base

# TODO(Phase3C.2): Once YAML decoupling is complete, merge_config() will be
# used as the sole source of defaults for in-module configuration mirrors.

def print_config(path: str = "configs/env_config.yaml"):
    cfg = load_config(path)
    print(json.dumps(cfg, indent=2))

if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else 'configs/env_config.yaml'
    print_config(p)
