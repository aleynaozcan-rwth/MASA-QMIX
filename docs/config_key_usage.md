# Configuration Key Usage Mapping

Configuration Key Usage Mapping  
Below is a concise mapping of top-level keys from env_config_enabled.yaml to where they are used in the codebase.  
You can paste this into developer docs.  

| 1️⃣ Config Key | 2️⃣ Python File(s) | 3️⃣ Variable or Function Using It | 4️⃣ Short Description of Its Role |
|----------------|--------------------|----------------------------------|-----------------------------------|
| work_centers | workcenter.py<br>environment.py<br>compare_config_to_modules.py | DEFAULT_WORKCENTERS['work_centers']<br>WorkCenters.from_config(config)<br>MASAEnv.__init__ (inference) | Named workcenter definitions and grouping of machines. Used to build canonical WorkCenters metadata and to infer num_wcs when loading config. |
| machines | workcenter.py<br>environment.py<br>task_generator.py | DEFAULT_WORKCENTERS['machines']<br>WorkCenters.from_config(config) → machine_registry<br>TaskGenerator.__init__ (self.config.get('machines')) | Per-machine metadata (workcenter membership, capable_ops). Drives machine_registry, capability mapping and per-machine durations. |
| operators | operator.py<br>workcenter.py | DEFAULT_OPERATORS<br>Operators.__init__ (merge/construct Operator objects)<br>WorkCenters.from_config (ops_cfg) | Operator definitions (id, qualified_machines). Used to construct runtime Operator objects and determine eligible operator groups per WC. |
| op_type_distribution | (configs/docs/artifacts) | — (no active code reference) | Intended op-type sampling probabilities (probs). Present in config and docs but not consumed by current TaskGenerator code. |
| processing_time_means | task_generator.py<br>workcenter.py | YAML `processing_time_means` → self.proc_time_means in TaskGenerator<br>WorkCenters.create_decision_item (reads env.config['processing_time_means']) | Mean processing time per operation (Op1..OpN) per machine. Used to determine per-machine job durations and to fill per_machine_durations. |
| task_generator | task_generator.py<br>environment.py | DEFAULT_TASKGEN_PARAMS → merged into TaskGenerator.config<br>MASAEnv.__init__ (reads task_generator to set job_min_ops/job_max_ops and arrival_lambda) | Controls job arrival process and sequence length (arrival_lambda, seq_length). Drives TaskGenerator behavior and MASAEnv arrival start. |
| reward_params | environment.py | DEFAULT_ENV_PARAMS['reward_params'] → MASAEnv.__init__ sets self.alpha, self.beta, self.gamma, self.delta, self.c_time | Reward shaping hyper-parameters used by MASAEnv.pop_decision_reward() to compute shaped rewards. |
| training_defaults | environment.py (mirror)<br>MARL training modules (indirect) | DEFAULT_ENV_PARAMS['training_defaults'] (obs/state shape + hyperparams) → used in MASAEnv.__init__ to infer obs_dim_agent, state_dim | Observation/state shape defaults and training hyperparameters (rnn_hidden, lr, batch_size, discount, target_update, epsilon_*). Mirrored in environment defaults; MARL training code often reads training hyperparams from runner/args instead of env.config. |


Machine-level fields inside machines (e.g., wc, capable_ops) are parsed and normalized in WorkCenters.from_config() and used across environment and TaskGenerator logic.

Some config keys appear only in docs/artifacts (e.g., op_type_distribution) and are not yet used by runtime code.

The codebase contains module-level mirrors for many config sections:

utils/workcenter.py → DEFAULT_WORKCENTERS

utils/operator.py → DEFAULT_OPERATORS

utils/task_generator.py → DEFAULT_TASKGEN_PARAMS

environment.py → DEFAULT_ENV_PARAMS

These mirrors enable the environment to run even when no external YAML config is provided.

Note: `processing_time_means` is authoritative in YAML/config and is no longer included as an in-module default; provide it via YAML or `config` when strict mode is required.

To enable config loading, set auto_load_config=True when initializing MASAEnv.

To run purely with in-code defaults for non-duration keys, set auto_load_config=False or omit config_path. Duration mappings must still be provided via YAML/config.
