# Module dependency audit

=== Active Call Graph ===
MARL/agent/agent.py -> imports:
  - MARL/policy/vdn.py
  - MARL/policy/qmix.py
  - MARL/policy/coma.py
  - MARL/policy/reinforce.py
  - MARL/policy/central_v.py
  - MARL/policy/qtran_alt.py
  - MARL/policy/qtran_base.py
  - MARL/policy/maven.py
  defines:
    - class Agents (used elsewhere: yes)
    - class CommAgents (used elsewhere: yes)

MARL/common/arguments.py -> imports:
  defines:
    - func get_mutable_args  (used elsewhere: no)
    - func get_common_args  (used elsewhere: yes)
    - func _ensure_mutable  (used elsewhere: no)
    - func get_mixer_args  (used elsewhere: yes)
    - func get_smoke_args  (used elsewhere: yes)
    - func get_coma_args  (used elsewhere: yes)
    - func get_centralv_args  (used elsewhere: yes)
    - func get_reinforce_args  (used elsewhere: yes)
    - func get_commnet_args  (used elsewhere: yes)
    - func get_g2anet_args  (used elsewhere: yes)
    - class ReadOnlyArgs (used elsewhere: no)

MARL/common/mask_utils.py -> imports:
  defines:
    - func build_index_map  (used elsewhere: yes)
    - func build_mask_for_job  (used elsewhere: yes)

MARL/common/replay_buffer.py -> imports:
  defines:
    - func _as_agents_obs  (used elsewhere: no)
    - func _as_agents_mask  (used elsewhere: no)
    - func _infer_time_len_from_dict  (used elsewhere: no)
    - func _take_step  (used elsewhere: no)
    - func _infer_agents_obs  (used elsewhere: no)
    - class ReplayBuffer (used elsewhere: yes)

MARL/common/rollout.py -> imports:
  - MARL/common/mask_utils.py
  - MARL/common/mask_utils.py
  defines:
    - class RolloutWorker (used elsewhere: yes)

MARL/common/terms.py -> imports:
  defines:
    - func t  (used elsewhere: yes)

MARL/common/utils.py -> imports:
  defines:
    - func store_args  (used elsewhere: no)
    - func td_lambda_target  (used elsewhere: yes)

MARL/network/base_net.py -> imports:
  defines:
    - class RNNAgent (used elsewhere: yes)
    - class BasicCritic (used elsewhere: no)

MARL/network/coma_critic.py -> imports:
  defines:
    - class ComaCritic (used elsewhere: yes)

MARL/network/commnet.py -> imports:
  defines:
    - class CommNet (used elsewhere: yes)

MARL/network/g2anet.py -> imports:
  defines:
    - class G2ANet (used elsewhere: yes)

MARL/network/maven_net.py -> imports:
  defines:
    - class HierarchicalPolicy (used elsewhere: yes)
    - class BootstrappedRNN (used elsewhere: yes)
    - class VarDistribution (used elsewhere: yes)

MARL/network/qmix_net.py -> imports:
  defines:
    - class QMixNet (used elsewhere: yes)

MARL/network/qtran_net.py -> imports:
  defines:
    - class QtranQAlt (used elsewhere: yes)
    - class QtranQBase (used elsewhere: yes)
    - class QtranV (used elsewhere: yes)

MARL/network/vdn_net.py -> imports:
  defines:
    - class VDNNet (used elsewhere: yes)

MARL/policy/central_v.py -> imports:
  - MARL/network/base_net.py
  - MARL/network/commnet.py
  - MARL/network/g2anet.py
  defines:
    - class CentralV (used elsewhere: yes)

MARL/policy/coma.py -> imports:
  - MARL/network/base_net.py
  - MARL/network/commnet.py
  - MARL/network/g2anet.py
  - MARL/network/coma_critic.py
  - MARL/common/utils.py
  defines:
    - class COMA (used elsewhere: yes)

MARL/policy/maven.py -> imports:
  - MARL/network/maven_net.py
  - MARL/network/qmix_net.py
  defines:
    - class MAVEN (used elsewhere: yes)

MARL/policy/qmix.py -> imports:
  - MARL/network/base_net.py
  - MARL/network/qmix_net.py
  defines:
    - class QMIX (used elsewhere: yes)

MARL/policy/qtran_alt.py -> imports:
  - MARL/network/base_net.py
  - MARL/network/qtran_net.py
  defines:
    - class QtranAlt (used elsewhere: yes)

MARL/policy/qtran_base.py -> imports:
  - MARL/network/base_net.py
  - MARL/network/qtran_net.py
  defines:
    - class QtranBase (used elsewhere: yes)

MARL/policy/reinforce.py -> imports:
  - MARL/network/base_net.py
  - MARL/network/commnet.py
  - MARL/network/g2anet.py
  defines:
    - class Reinforce (used elsewhere: yes)

MARL/policy/vdn.py -> imports:
  - MARL/network/base_net.py
  - MARL/network/vdn_net.py
  defines:
    - class VDN (used elsewhere: yes)

MARL/runner.py -> imports:
  - MARL/common/rollout.py
  - MARL/agent/agent.py
  - MARL/common/replay_buffer.py
  - MARL/common/terms.py
  - MARL/common/rollout.py
  defines:
    - func plot_gantt  (used elsewhere: yes)
    - class Runner (used elsewhere: yes)

environment.py -> imports:
  - utils/workcenter.py
  - utils/jobagent.py
  - utils/env_obs.py
  - utils/env_obs.py
  - utils/env_obs.py
  - utils/config_loader.py
  - utils/task_generator.py
  defines:
    - class MASAEnv (used elsewhere: yes)

main.py -> imports:
  - environment.py
  - MARL/runner.py
  - MARL/common/arguments.py
  defines:
    - func marl_agent_wrapper  (used elsewhere: yes)
    - func random_agent_wrapper  (used elsewhere: no)

scripts/run_train_qmix.py -> imports:
  - MARL/common/arguments.py
  - environment.py
  - MARL/runner.py
  defines:
    - func main  (used elsewhere: yes)

tools/check_simpy_integration.py -> imports:
  - environment.py
  defines:
    - func main  (used elsewhere: yes)

tools/generate_module_audit.py -> imports:
  defines:
    - func iter_py_files  (used elsewhere: yes)
    - func parse_imports  (used elsewhere: no)
    - func defs_in_file  (used elsewhere: no)
    - func build_module_map  (used elsewhere: no)
    - func resolve_module  (used elsewhere: no)
    - func gather_graph  (used elsewhere: no)
    - func find_references  (used elsewhere: no)
    - func main  (used elsewhere: yes)

tools/generate_pretty_gantt.py -> imports:

tools/run_debug_smoke.py -> imports:
  - MARL/common/arguments.py
  - main.py

tools/run_debug_smoke_enabled.py -> imports:
  - MARL/common/arguments.py
  - main.py

tools/run_initial_trace.py -> imports:
  - environment.py
  - utils/gantt.py

tools/run_smoke_gantt.py -> imports:
  - MARL/common/arguments.py
  - environment.py
  - MARL/runner.py

tools/smoke_generate_gantt.py -> imports:
  - environment.py
  - utils/gantt.py
  defines:
    - func plot_gantt_local  (used elsewhere: no)

tools/test_arrivals.py -> imports:
  - environment.py
  defines:
    - func main  (used elsewhere: yes)
    - class DummyJob (used elsewhere: no)
    - class DummyTG (used elsewhere: no)

tools/test_concurrency.py -> imports:
  - environment.py
  defines:
    - func find_concurrency  (used elsewhere: no)
    - func main  (used elsewhere: yes)

tools/test_reward_envvars.py -> imports:
  - environment.py
  defines:
    - func main  (used elsewhere: yes)

utils/config_loader.py -> imports:
  defines:
    - func _ensure_yaml  (used elsewhere: no)
    - func load_config  (used elsewhere: yes)
    - func merge_config  (used elsewhere: yes)
    - func print_config  (used elsewhere: no)

utils/env_obs.py -> imports:
  defines:
    - func build_agent_obs  (used elsewhere: yes)
    - func build_state_vector  (used elsewhere: yes)

utils/gantt.py -> imports:
  defines:
    - func format_gantt_records  (used elsewhere: yes)
    - func gantt_records_to_csv  (used elsewhere: yes)
    - func write_gantt_csv  (used elsewhere: yes)
    - func write_scheduling_trace  (used elsewhere: yes)
    - func write_job_timeline  (used elsewhere: yes)
    - func append_selection_log  (used elsewhere: yes)

utils/job.py -> imports:
  - utils/config_loader.py
  defines:
    - class Jobs (used elsewhere: yes)
    - class Job (used elsewhere: yes)

utils/jobagent.py -> imports:
  defines:
    - class JobAgents (used elsewhere: yes)
    - class JobAgent (used elsewhere: yes)

utils/task_generator.py -> imports:
  - utils/workcenter.py
  - utils/job.py
  - utils/config_loader.py
  - utils/config_loader.py
  defines:
    - class TaskGenerator (used elsewhere: yes)

utils/workcenter.py -> imports:
  - utils/config_loader.py
  defines:
    - class WorkCenter (used elsewhere: yes)
    - class WorkCenters (used elsewhere: yes)


=== Inactive / Unused Files ===
MARL/__init__.py -> 🔴 Obsolete

MARL/common/analyse.py -> 🟢 Candidate for reintegration
  defines:
    - func plt_win_rate_mean

tests/test_args_hygiene_strict.py -> 🟡 Duplicate functionality
  defines:
    - func iter_py_files
    - func test_no_args_field_assignments_outside_arguments_and_tests
    - func test_no_mixed_args_and_self_args_in_same_file

tests/test_args_no_ad_hoc_assignments.py -> 🟡 Duplicate functionality
  defines:
    - func iter_py_files
    - func test_no_ad_hoc_args_assignments

tests/test_decision_item.py -> 🟢 Candidate for reintegration
  defines:
    - func test_create_decision_item_basic

tests/test_decision_item_extended.py -> 🔴 Obsolete
  defines:
    - func test_resume_with_machine_name
    - func test_resume_with_workcenter_index
    - func test_resume_with_invalid_choice_returns_none

tests/test_env_obs.py -> 🟢 Candidate for reintegration
  defines:
    - func test_build_agent_obs_and_state_shape
    - func test_build_state_vector_shape

tests/test_gantt_helpers.py -> 🟢 Candidate for reintegration
  defines:
    - func test_format_gantt_records_basic
    - func test_gantt_csv_roundtrip

tests/test_gantt_utils.py -> 🟢 Candidate for reintegration
  defines:
    - func test_write_scheduling_trace_happy_and_empty
    - func test_write_job_timeline_and_counts
    - func test_append_selection_log_appends

tests/test_job_generation.py -> 🟢 Candidate for reintegration
  defines:
    - func test_allowed_wcs_subset_of_registry

tests/test_mask_utils.py -> 🟢 Candidate for reintegration
  defines:
    - func run_checks

tests/test_metrics_writer.py -> 🟢 Candidate for reintegration
  defines:
    - func cleanup_metrics
    - func test_metrics_writer_tempdir

tests/test_plot_gantt.py -> 🔴 Obsolete
  defines:
    - func make_record
    - func test_plot_gantt_accepts_various_shapes
    - func test_csv_export_normalizes_columns

tests/test_rollout_transitions.py -> 🟢 Candidate for reintegration
  defines:
    - func make_batch
    - func test_build_transitions_basic_shapes
    - func test_build_transitions_granular_fallback
    - class DummyEnv
    - class Args

tests/test_simpy_integration.py -> 🟢 Candidate for reintegration
  defines:
    - func test_simpy_integration_runner_and_worker
    - class FakeJob
    - class MinimalSimPyEnv

tests/test_taskgenerator_start_on_reset.py -> 🟢 Candidate for reintegration
  defines:
    - func test_taskgenerator_start_called_on_reset

utils/PDRs/shortestDistence.py -> 🔴 Obsolete
  defines:
    - class SDrules

utils/machine_registry.py -> 🟢 Candidate for reintegration
  defines:
    - func build_machine_registry
utils/machine_registry.py -> � Archived (merged into `utils/workcenter.py`)
  notes:
    - Deprecated and archived. Functionality merged into `WorkCenters.from_config` in `utils/workcenter.py` (Phase 4A.1).
    - Original implementation preserved under `archive/legacy/machine_registry.py`.

utils/operator.py -> 🟢 Candidate for reintegration
  defines:
    - class Operator
    - class Operators

utils/util.py -> 🔴 Obsolete
  defines:
    - func left_planes
    - func min_but_zero
    - func advance_by_min_time
    - func count_path_on_road
