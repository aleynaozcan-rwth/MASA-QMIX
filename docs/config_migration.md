| Concept | Old Source | New Source | Migration Note |
|---------|------------|------------|----------------|
| work_centers | YAML | `workcenter.py:DEFAULT_WORKCENTERS` | fully internal |
| jobs | YAML | `job.py:DEFAULT_JOBS` | fully internal |
| task_generator | YAML | `task_generator.py:DEFAULT_TASKGEN_PARAMS` | fully internal |
| processing_time_means | YAML | `configs/*.yaml` (YAML-only) | No longer provided as an in-module default; supply via YAML/config |
| operators | YAML | `operator.py:DEFAULT_OPERATORS` | fully internal |
| reward_params, training_defaults | YAML | `environment.py:DEFAULT_ENV_PARAMS` | fully internal |

## Notes

- This document records the Phase 3 migration where several configuration concepts were made available as module-level defaults and a safe merge layer (`utils/config_loader.merge_config`) ensures YAML overrides remain supported.
- During Phase 3C the YAML keys were progressively detached and the code validated against those defaults. Phase 3D validates runtime behavior and metrics when using module defaults vs YAML.

Committed as part of: "Phase 3D – YAML Detachment Verification & Migration Summary"
