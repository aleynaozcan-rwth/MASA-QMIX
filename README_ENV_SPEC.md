# MASA-QMIX — Environment configuration & how to run (short)

This file contains an example environment config and quick instructions for running a smoke test.

Files added:
- `configs/env_config.yaml` — example YAML describing WCs, machines, operators, processing-time means, task-generator and reward/training defaults.
- `utils/config_loader.py` — tiny helper to load and print the YAML config (requires `pyyaml`).

Quick usage:

1. Install PyYAML in your conda env if needed:
   ```bash
   conda run -n masa-qmix --no-capture-output pip install pyyaml
   ```

2. Print the configuration:
   ```bash
   conda run -n masa-qmix --no-capture-output python -m utils.config_loader configs/env_config.yaml
   ```

3. Run a short smoke test:
   ```bash
   conda run -n masa-qmix --no-capture-output python run_short.py
   ```

Notes:
- Edit `configs/env_config.yaml` to adapt machine/operator/op type mappings and processing-time means.
- Later steps will wire this config into `environment.py` and TaskGenerator to ensure a single source of truth.
