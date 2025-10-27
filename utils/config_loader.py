"""Simple YAML config loader for MASA-QMIX.

Usage:
  from utils.config_loader import load_config
  cfg = load_config('configs/env_config.yaml')

This helper requires PyYAML. If PyYAML is not installed the loader will raise
an explicit error with a hint on how to install it.
"""
from pathlib import Path
import json

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

def print_config(path: str = "configs/env_config.yaml"):
    cfg = load_config(path)
    print(json.dumps(cfg, indent=2))

if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else 'configs/env_config.yaml'
    print_config(p)
