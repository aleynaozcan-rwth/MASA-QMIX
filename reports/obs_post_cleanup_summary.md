# Post-cleanup observation & normalization scan summary

Date: 2025-11-09T17:32:40.847192

## Quick totals

- Total hits: 70
- HIGH: 51
- MEDIUM: 17
- LOW: 2
- Files affected: 19

## Acceptance criteria

- HIGH=0 before merging this cleanup commit.
- No references to 11D or legacy helpers remain in production code.
- Observations must use only the 6 canonical features; any deviation listed below must be remediated.
- max_jobs must be derived from args.n_agents (no aliasing).

## HIGH severity matches (code/breaking)

### ./MARL/common/replay_buffer.py — 1 match(es)

- Line 382: tag=ELEVEN_DIM match=`obs_dim = 11`
  Suggestion: Replace 11D obs usage: ensure obs_shape==6 and update builders to utils.env_obs.build_agent_obs. Remove obs_dim=11 references.

### ./MARL/policy/qmix.py — 1 match(es)

- Line 5: tag=ELEVEN_DIM match=`# - Compatible with MASAEnv (11D obs, 64D state)`
  Suggestion: Replace 11D obs usage: ensure obs_shape==6 and update builders to utils.env_obs.build_agent_obs. Remove obs_dim=11 references.

### ./MARL/runner.py — 9 match(es)

- Line 795: tag=ELEVEN_DIM match=`- 11D observations (progress_ratio)`
  Suggestion: Replace 11D obs usage: ensure obs_shape==6 and update builders to utils.env_obs.build_agent_obs. Remove obs_dim=11 references.
- Line 1193: tag=LEGACY_FEATURE_API match=`# Compute finished job count on-demand; legacy env.completed_jobs removed`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1245: tag=LEGACY_FEATURE_API match=`# fallback: try env.total_wait_time / completed_jobs`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1247: tag=LEGACY_FEATURE_API match=`# Replace legacy completed_jobs usage with computed count`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1541: tag=LEGACY_FEATURE_API match=`util_m = self.env._util_machines()`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1542: tag=LEGACY_FEATURE_API match=`util_o = self.env._util_ops()`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1582: tag=LEGACY_FEATURE_API match=`'completed_jobs': int(delta_completed),`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1888: tag=LEGACY_FEATURE_API match=`util_m = float(self.env._util_machines())`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1889: tag=LEGACY_FEATURE_API match=`util_o = float(self.env._util_ops())`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()

### ./environment.py — 26 match(es)

- Line 60: tag=OBS_BUILDER match=`from utils.env_obs import build_agent_obs, build_state_vector  # type: ignore`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 107: tag=LEGACY_FEATURE_API match=`self.remaining_time = 0.0`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 117: tag=LEGACY_FEATURE_API match=`def progress_ratio(self):`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 329: tag=OBS_BUILDER match=`# These attributes are used by observation builders to produce`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 402: tag=OBS_BUILDER match=`# observation builders that rely on them. Fail fast to surface`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 770: tag=LEGACY_FEATURE_API match=`# legacy `completed_jobs` counter removed in favour of computed metrics`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1104: tag=LEGACY_FEATURE_API match=`# legacy `completed_jobs` removed; compute finished counts from jobs when needed`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1338: tag=LEGACY_FEATURE_API match=`# legacy `_wip()` helper.`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1349: tag=LEGACY_FEATURE_API match=`# NOTE: removed appending to internal `_recent_rewards` deque. Consumers`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1363: tag=LEGACY_FEATURE_API match=`# legacy `completed_jobs` counter removed; compute finished`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1404: tag=LEGACY_FEATURE_API match=`completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1407: tag=LEGACY_FEATURE_API match=`tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 1463: tag=OBS_BUILDER match=`'obs': self._build_agent_obs(job),`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 1873: tag=LEGACY_FEATURE_API match=`job.remaining_time = dur`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 2028: tag=LEGACY_FEATURE_API match=`job.remaining_time = dur`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 2101: tag=LEGACY_FEATURE_API match=`job.remaining_time = 0.0`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 2107: tag=LEGACY_FEATURE_API match=`# legacy `completed_jobs` counter removed; compute finished`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 2188: tag=OBS_BUILDER match=`# Strict delegation to utils.env_obs.build_agent_obs. Missing helper`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 2190: tag=OBS_BUILDER match=`return [build_agent_obs(self, j) for j in self.jobs]`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 2192: tag=OBS_BUILDER match=`def _build_agent_obs(self, job: JobAgent):`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 2195: tag=OBS_BUILDER match=`return build_agent_obs(self, job)`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 2553: tag=LEGACY_FEATURE_API match=`completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 2556: tag=LEGACY_FEATURE_API match=`tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 2599: tag=LEGACY_FEATURE_API match=`completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 2602: tag=LEGACY_FEATURE_API match=`tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 2730: tag=LEGACY_FEATURE_API match=`- average_wait_time: total_wait_time / completed_jobs (seconds)`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()

### ./my_data_and_graph/metrics.py — 2 match(es)

- Line 199: tag=LEGACY_FEATURE_API match=`# preserve legacy fallback when item contains a completed_jobs field`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 200: tag=LEGACY_FEATURE_API match=`completed_jobs_count = int(item.get('completed_jobs', 0))`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()

### ./utils/env_obs.py — 7 match(es)

- Line 5: tag=OBS_BUILDER match=`compatible with the MASAEnv canonical attributes. The observation builder`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 6: tag=ELEVEN_DIM match=`does NOT contain any legacy 11-D features or silent fallbacks — if a`
  Suggestion: Replace 11D obs usage: ensure obs_shape==6 and update builders to utils.env_obs.build_agent_obs. Remove obs_dim=11 references.
- Line 15: tag=LEGACY_NORM match=`5. wait_time_norm            -> job.wait_time / env.max_wait_time`
  Suggestion: Remove hard-coded divisors; use env.max_wait_time, env.max_operations_per_job, env.n_operation_types, or args-derived max_jobs.
- Line 39: tag=OBS_BUILDER match=`def build_agent_obs(env: Any, job: Any) -> np.ndarray:`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 121: tag=LEGACY_NORM match=`wait_time_norm = np.clip(wait_time_val / float(max_wait), 0.0, 1.0)`
  Suggestion: Remove hard-coded divisors; use env.max_wait_time, env.max_operations_per_job, env.n_operation_types, or args-derived max_jobs.
- Line 129: tag=LEGACY_NORM match=`wait_time_norm,`
  Suggestion: Remove hard-coded divisors; use env.max_wait_time, env.max_operations_per_job, env.n_operation_types, or args-derived max_jobs.
- Line 142: tag=LEGACY_FEATURE_API match=`legacy helpers such as _wip, _recent_rewards or completed_jobs.`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()

### ./utils/gantt.py — 1 match(es)

- Line 644: tag=LEGACY_FEATURE_API match=`avg_wait = float(getattr(env, 'total_wait_time', 0.0)) / max(1.0, float(getattr(env, 'completed_jobs', 0) or 1))`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()

### ./utils/jobagent.py — 3 match(es)

- Line 118: tag=LEGACY_FEATURE_API match=`self.remaining_time = 0.0`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 148: tag=LEGACY_FEATURE_API match=`def progress_ratio(self):`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()
- Line 263: tag=LEGACY_FEATURE_API match=`self.remaining_time = 0.0`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()

### ./utils/workcenter.py — 1 match(es)

- Line 339: tag=OBS_BUILDER match=`"obs": env._build_agent_obs(job),`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.

## MEDIUM severity matches (tests/docs/examples)

### ./CODE_HYGIENE_REPORT.md — 1 match(es)

- Line 158: tag=OBS_BUILDER match=`Purpose: Pure helper functions to build agent-level observations and global state vector.`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.

### ./SYSTEM_STATUS.md — 1 match(es)

- Line 72: tag=ELEVEN_DIM match=`|State/obs shape consistency|✅|state_dim=64, obs_dim=11 as reported in runtime snapshot|`
  Suggestion: Replace 11D obs usage: ensure obs_shape==6 and update builders to utils.env_obs.build_agent_obs. Remove obs_dim=11 references.

### ./docs/module_dependency_audit.md — 2 match(es)

- Line 254: tag=OBS_BUILDER match=`- func build_agent_obs  (used elsewhere: yes)`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 326: tag=OBS_BUILDER match=`- func test_build_agent_obs_and_state_shape`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.

### ./tests/test_env_normalization.py — 3 match(es)

- Line 4: tag=OBS_BUILDER match=`from utils.env_obs import build_agent_obs`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 29: tag=OBS_BUILDER match=`# attributes expected by build_agent_obs():`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 68: tag=OBS_BUILDER match=`obs = build_agent_obs(env, job)`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.

### ./tests/test_env_obs.py — 3 match(es)

- Line 6: tag=OBS_BUILDER match=`def test_build_agent_obs_and_state_shape():`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 11: tag=OBS_BUILDER match=`aobs = env._build_agent_obs(job)`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 12: tag=OBS_BUILDER match=`aobs2 = env_obs.build_agent_obs(env, job)`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.

### ./tests/test_observation_equivalence.py — 4 match(es)

- Line 18: tag=OBS_BUILDER match=`from utils.env_obs import build_agent_obs`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 21: tag=OBS_BUILDER match=`# Add a minimal job so build_agent_obs has something to read`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 29: tag=OBS_BUILDER match=`a_env = env._build_agent_obs(job)`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 30: tag=OBS_BUILDER match=`a_helper = build_agent_obs(env, job)`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.

### ./tests/test_observation_shapes.py — 1 match(es)

- Line 10: tag=OBS_BUILDER match=`obs = env._build_agent_obs(env.jobs[0])`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.

### ./tmp_smoke_test.py — 1 match(es)

- Line 14: tag=LEGACY_FEATURE_API match=`# compute completed count on-demand (legacy env.completed_jobs removed)`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()

### ./tmp_test_gantt_decision_trace.py — 1 match(es)

- Line 35: tag=LEGACY_FEATURE_API match=`self.completed_jobs = 0`
  Suggestion: Replace legacy feature with computed canonical: e.g. completed_jobs -> len([j for j in getattr(env,'jobs',[]) if getattr(j,'finished',False)]) or use env.metrics.completed_jobs()

## LOW severity matches (others)

### ./smoke_test_obs_6d.py — 2 match(es)

- Line 9: tag=OBS_BUILDER match=`from utils.env_obs import build_agent_obs`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.
- Line 188: tag=OBS_BUILDER match=`obs = build_agent_obs(env, job)`
  Suggestion: Use canonical builder utils.env_obs.build_agent_obs(env, job) or central state builder.

## File-level notes and recommended next steps

Priority files (highest first):

- ./utils/workcenter.py (severity=HIGH)
- ./utils/jobagent.py (severity=HIGH)
- ./utils/gantt.py (severity=HIGH)
- ./utils/env_obs.py (severity=HIGH)
- ./my_data_and_graph/metrics.py (severity=HIGH)
- ./environment.py (severity=HIGH)
- ./MARL/runner.py (severity=HIGH)
- ./MARL/policy/qmix.py (severity=HIGH)
- ./MARL/common/replay_buffer.py (severity=HIGH)
- ./tmp_test_gantt_decision_trace.py (severity=MEDIUM)
- ./tmp_smoke_test.py (severity=MEDIUM)
- ./tests/test_observation_shapes.py (severity=MEDIUM)
- ./tests/test_observation_equivalence.py (severity=MEDIUM)
- ./tests/test_env_obs.py (severity=MEDIUM)
- ./tests/test_env_normalization.py (severity=MEDIUM)
- ./docs/module_dependency_audit.md (severity=MEDIUM)
- ./SYSTEM_STATUS.md (severity=MEDIUM)
- ./CODE_HYGIENE_REPORT.md (severity=MEDIUM)
- ./smoke_test_obs_6d.py (severity=LOW)

Critical nodes to review: utils/env_obs.py, environment.py, MARL/runner.py, my_data_and_graph/metrics.py, utils/gantt.py

## Observed deviations from 6D canonical features

The following legacy or non-6D tokens were found (sample):
- `# - Compatible with MASAEnv (11D obs, 64D state)`
- `# Compute finished job count on-demand; legacy env.completed_jobs removed`
- `# NOTE: removed appending to internal `_recent_rewards` deque. Consumers`
- `# Replace legacy completed_jobs usage with computed count`
- `# compute completed count on-demand (legacy env.completed_jobs removed)`
- `# fallback: try env.total_wait_time / completed_jobs`
- `# legacy `_wip()` helper.`
- `# legacy `completed_jobs` counter removed in favour of computed metrics`
- `# legacy `completed_jobs` counter removed; compute finished`
- `# legacy `completed_jobs` removed; compute finished counts from jobs when needed`
- `# preserve legacy fallback when item contains a completed_jobs field`
- `'completed_jobs': int(delta_completed),`
- `- 11D observations (progress_ratio)`
- `- average_wait_time: total_wait_time / completed_jobs (seconds)`
- `5. wait_time_norm            -> job.wait_time / env.max_wait_time`
- `avg_wait = float(getattr(env, 'total_wait_time', 0.0)) / max(1.0, float(getattr(env, 'completed_jobs', 0) or 1))`
- `completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]`
- `completed_jobs_count = int(item.get('completed_jobs', 0))`
- `def progress_ratio(self):`
- `does NOT contain any legacy 11-D features or silent fallbacks — if a`
- `job.remaining_time = 0.0`
- `job.remaining_time = dur`
- `legacy helpers such as _wip, _recent_rewards or completed_jobs.`
- `obs_dim = 11`
- `self.completed_jobs = 0`
- `self.remaining_time = 0.0`
- `tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")`
- `tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")`
- `tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")`
- `util_m = float(self.env._util_machines())`
- `util_m = self.env._util_machines()`
- `util_o = float(self.env._util_ops())`
- `util_o = self.env._util_ops()`
- `wait_time_norm = np.clip(wait_time_val / float(max_wait), 0.0, 1.0)`
- `wait_time_norm,`
- `|State/obs shape consistency|✅|state_dim=64, obs_dim=11 as reported in runtime snapshot|`

## Acceptance criteria reminder

- Do not merge until HIGH==0.
- After remediation, re-run this scan to confirm zero HIGH hits.
