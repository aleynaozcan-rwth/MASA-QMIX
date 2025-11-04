# Configuration Diff Report

This file was generated from `logs/config_diff_report.txt`.

| Key Path | Enabled (A) | Detached (B) | Δ Type |
|---|---|---|---|
| `machines.M1.capable_ops` | `['Op1', 'Op2', 'Op3', 'Op5', 'Op9']` | `<MISSING>` | Missing in Detached |
| `machines.M1.speed_factor` | `1.0` | `<MISSING>` | Missing in Detached |
| `machines.M1.wc` | `WC1` | `<MISSING>` | Missing in Detached |
| `machines.M2.capable_ops` | `['Op4', 'Op5', 'Op8']` | `<MISSING>` | Missing in Detached |
| `machines.M2.speed_factor` | `1.0` | `<MISSING>` | Missing in Detached |
| `machines.M2.wc` | `WC1` | `<MISSING>` | Missing in Detached |
| `machines.M3.capable_ops` | `['Op1', 'Op2', 'Op4', 'Op6', 'Op9']` | `<MISSING>` | Missing in Detached |
| `machines.M3.speed_factor` | `1.0` | `<MISSING>` | Missing in Detached |
| `machines.M3.wc` | `WC2` | `<MISSING>` | Missing in Detached |
| `machines.M4.capable_ops` | `['Op1', 'Op3', 'Op7', 'Op8']` | `<MISSING>` | Missing in Detached |
| `machines.M4.speed_factor` | `1.0` | `<MISSING>` | Missing in Detached |
| `machines.M4.wc` | `WC2` | `<MISSING>` | Missing in Detached |
| `machines.M5.capable_ops` | `['Op1', 'Op3', 'Op4', 'Op6', 'Op8', 'Op9']` | `<MISSING>` | Missing in Detached |
| `machines.M5.speed_factor` | `1.0` | `<MISSING>` | Missing in Detached |
| `machines.M5.wc` | `WC3` | `<MISSING>` | Missing in Detached |
| `op_type_distribution.probs` | `[0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.12]` | `<MISSING>` | Missing in Detached |
| `operators` | `[{'id': 'O1', 'qualified_machines': ['M1', 'M4', 'M5']}, {'id': 'O2', 'qualif...` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op1.M1` | `1.225` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op1.M3` | `1.575` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op1.M4` | `1.4` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op1.M5` | `1.05` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op2.M1` | `1.05` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op2.M3` | `1.75` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op3.M1` | `1.575` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op3.M4` | `1.75` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op3.M5` | `1.925` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op4.M2` | `1.575` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op4.M3` | `1.68` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op4.M5` | `2.1` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op5.M1` | `2.275` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op5.M2` | `1.75` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op6.M3` | `2.1` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op6.M5` | `2.8` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op7.M4` | `1.82` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op8.M2` | `2.975` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op8.M4` | `2.625` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op8.M5` | `1.575` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op9.M1` | `2.975` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op9.M3` | `3.15` | `<MISSING>` | Missing in Detached |
| `processing_time_means.Op9.M5` | `1.925` | `<MISSING>` | Missing in Detached |
| `reward_params.alpha` | `0.01` | `1.0` | Different |
| `reward_params.beta` | `0.005` | `0.5` | Different |
| `reward_params.c_time` | `0.001` | `0.0` | Different |
| `reward_params.delta` | `1.0` | `0.1` | Different |
| `reward_params.gamma` | `0.02` | `0.2` | Different |
| `task_generator.arrival_lambda` | `0.05` | `0.0` | Different |
| `training_defaults.batch_size` | `32` | `<MISSING>` | Missing in Detached |
| `training_defaults.discount` | `0.99` | `<MISSING>` | Missing in Detached |
| `training_defaults.epsilon_decay_steps` | `100000` | `<MISSING>` | Missing in Detached |
| `training_defaults.epsilon_end` | `0.05` | `<MISSING>` | Missing in Detached |
| `training_defaults.epsilon_start` | `1.0` | `<MISSING>` | Missing in Detached |
| `training_defaults.lr` | `1e-3` | `<MISSING>` | Missing in Detached |
| `training_defaults.rnn_hidden` | `64` | `<MISSING>` | Missing in Detached |
| `training_defaults.target_update` | `200` | `<MISSING>` | Missing in Detached |
| `work_centers.WC1.machines` | `['M1', 'M2']` | `<MISSING>` | Missing in Detached |
| `work_centers.WC2.machines` | `['M3', 'M4']` | `<MISSING>` | Missing in Detached |
| `work_centers.WC3.machines` | `['M5']` | `<MISSING>` | Missing in Detached |
