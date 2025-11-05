import yaml
from utils.workcenter import DEFAULT_PROCESSING_TIMES


def transpose_machine_to_op(machine_map):
    op_map = {}
    for m, ops in machine_map.items():
        for op, val in (ops or {}).items():
            op_map.setdefault(op, {})[m] = float(val)
    return op_map


def test_yaml_and_fallback_are_identical():
    with open("configs/env_config_enabled.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    yaml_proc = cfg.get("processing_time_means", {}) or {}
    fallback_proc = transpose_machine_to_op(DEFAULT_PROCESSING_TIMES)
    assert yaml_proc == fallback_proc, "WorkCenter DEFAULT_PROCESSING_TIMES must match YAML"
