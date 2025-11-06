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
    """Load and return configuration as a Python dict.

    NOTE: YAML-based config loading has been deprecated for config-free
    operation. This helper now returns an empty dict and logs a debug
    message to preserve compatibility with callers that import it.
    """
    logger = logging.getLogger(__name__)
    try:
        logger.debug("load_config called for %s but YAML loading is disabled in config-free mode; returning {}", path)
    except Exception:
        pass
    return {}


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

    # In config-free mode we avoid applying external YAML overrides.
    # For compatibility, return a deep copy of defaults regardless of `cfg`.
    try:
        return copy.deepcopy(defaults) if defaults is not None else {}
    except Exception:
        logger.exception("merge_config fallback: returning shallow copy of defaults")
        return dict(defaults) if isinstance(defaults, dict) else defaults

# TODO(Phase3C.2): Once YAML decoupling is complete, merge_config() will be
# used as the sole source of defaults for in-module configuration mirrors.

def print_config(path: str = "configs/env_config.yaml"):
    cfg = load_config(path)
    logger = logging.getLogger(__name__)
    logger.info("Config at %s: %s", path, json.dumps(cfg, indent=2))

if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else 'configs/env_config.yaml'
    # When executed as a script, print the (debug) config via logging.
    print_config(p)
