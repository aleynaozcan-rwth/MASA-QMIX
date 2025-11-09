# High-risk Legacy Observation Replacement Plan

Date: 2025-11-09T15:52:47.142249Z

- Total HIGH-risk lines: 56
- SAFE changes suggested: 1
- NEEDS_TEST changes suggested: 50
- REQUIRES_REF changes: 5

## MARL/runner.py

- Line 795: token `progress_ratio` — classification: read — status: NEEDS_TEST

  Context:
  ```
      - 11D observations (progress_ratio)
                  before_completed = int(getattr(self.env, 'completed_jobs', 0))
                      # fallback: try env.total_wait_time / completed_jobs
                          self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1193: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
      - 11D observations (progress_ratio)
                  before_completed = int(getattr(self.env, 'completed_jobs', 0))
                      # fallback: try env.total_wait_time / completed_jobs
                          self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                  metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1246: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
      - 11D observations (progress_ratio)
                  before_completed = int(getattr(self.env, 'completed_jobs', 0))
                      # fallback: try env.total_wait_time / completed_jobs
                          self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                  metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                      avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                      util_m = self.env._util_machines()
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1493: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                  before_completed = int(getattr(self.env, 'completed_jobs', 0))
                      # fallback: try env.total_wait_time / completed_jobs
                          self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                  metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                      avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                      util_m = self.env._util_machines()
                      util_o = self.env._util_ops()
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1533: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                      # fallback: try env.total_wait_time / completed_jobs
                          self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                  metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                      avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                      util_m = self.env._util_machines()
                      util_o = self.env._util_ops()
                          after_completed = int(getattr(self.env, 'completed_jobs', 0))
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1534: token `_util_machines` — classification: write — status: NEEDS_TEST

  Context:
  ```
                          self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                  metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                      avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                      util_m = self.env._util_machines()
                      util_o = self.env._util_ops()
                          after_completed = int(getattr(self.env, 'completed_jobs', 0))
                          'completed_jobs': int(delta_completed),
  ```

  Replacement suggestion:
  ```
  Replace with a canonical metrics call, e.g. ``env.metrics.machine_utilization()`` or compute as ``sum(1 for m in env.machines if m.is_busy())/len(env.machines)``; for operator util similarly over operations or use gantt aggregator.
  ```

  Note: This is often used in dashboards/metrics; validate numeric range and unit tests.

- Line 1535: token `_util_ops` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                                  metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                      avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                      util_m = self.env._util_machines()
                      util_o = self.env._util_ops()
                          after_completed = int(getattr(self.env, 'completed_jobs', 0))
                          'completed_jobs': int(delta_completed),
                          util_m = float(self.env._util_machines())
  ```

  Replacement suggestion:
  ```
  Replace with a canonical metrics call, e.g. ``env.metrics.machine_utilization()`` or compute as ``sum(1 for m in env.machines if m.is_busy())/len(env.machines)``; for operator util similarly over operations or use gantt aggregator.
  ```

  Note: This is often used in dashboards/metrics; validate numeric range and unit tests.

- Line 1549: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                      avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                      util_m = self.env._util_machines()
                      util_o = self.env._util_ops()
                          after_completed = int(getattr(self.env, 'completed_jobs', 0))
                          'completed_jobs': int(delta_completed),
                          util_m = float(self.env._util_machines())
                          util_o = float(self.env._util_ops())
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1574: token `completed_jobs` — classification: read — status: NEEDS_TEST

  Context:
  ```
                      util_m = self.env._util_machines()
                      util_o = self.env._util_ops()
                          after_completed = int(getattr(self.env, 'completed_jobs', 0))
                          'completed_jobs': int(delta_completed),
                          util_m = float(self.env._util_machines())
                          util_o = float(self.env._util_ops())
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1880: token `_util_machines` — classification: write — status: NEEDS_TEST

  Context:
  ```
                      util_o = self.env._util_ops()
                          after_completed = int(getattr(self.env, 'completed_jobs', 0))
                          'completed_jobs': int(delta_completed),
                          util_m = float(self.env._util_machines())
                          util_o = float(self.env._util_ops())
  ```

  Replacement suggestion:
  ```
  Replace with a canonical metrics call, e.g. ``env.metrics.machine_utilization()`` or compute as ``sum(1 for m in env.machines if m.is_busy())/len(env.machines)``; for operator util similarly over operations or use gantt aggregator.
  ```

  Note: This is often used in dashboards/metrics; validate numeric range and unit tests.

- Line 1881: token `_util_ops` — classification: write — status: NEEDS_TEST

  Context:
  ```
                          after_completed = int(getattr(self.env, 'completed_jobs', 0))
                          'completed_jobs': int(delta_completed),
                          util_m = float(self.env._util_machines())
                          util_o = float(self.env._util_ops())
  ```

  Replacement suggestion:
  ```
  Replace with a canonical metrics call, e.g. ``env.metrics.machine_utilization()`` or compute as ``sum(1 for m in env.machines if m.is_busy())/len(env.machines)``; for operator util similarly over operations or use gantt aggregator.
  ```

  Note: This is often used in dashboards/metrics; validate numeric range and unit tests.

## environment.py

- Line 107: token `remaining_time` — classification: write — status: NEEDS_TEST

  Context:
  ```
              self.remaining_time = 0.0
          def progress_ratio(self):
          self.completed_jobs = 0
          self._recent_rewards = deque(maxlen=20)
  ```

  Replacement suggestion:
  ```
  Use job-level attributes or compute from job.operations: e.g. ``job.remaining_time`` or derive progress as ``(total_ops - remaining_ops)/total_ops``. For agent obs use ``build_agent_obs(env, job)`` and read the 6-D normalized value.
  ```

- Line 117: token `progress_ratio` — classification: compute — status: NEEDS_TEST

  Context:
  ```
              self.remaining_time = 0.0
          def progress_ratio(self):
          self.completed_jobs = 0
          self._recent_rewards = deque(maxlen=20)
          self.completed_jobs = 0
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 770: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
              self.remaining_time = 0.0
          def progress_ratio(self):
          self.completed_jobs = 0
          self._recent_rewards = deque(maxlen=20)
          self.completed_jobs = 0
              self._recent_rewards.clear()
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 773: token `_recent_rewards` — classification: write — status: REQUIRES_REF

  Context:
  ```
              self.remaining_time = 0.0
          def progress_ratio(self):
          self.completed_jobs = 0
          self._recent_rewards = deque(maxlen=20)
          self.completed_jobs = 0
              self._recent_rewards.clear()
                  self._recent_rewards = deque(maxlen=20)
  ```

  Replacement suggestion:
  ```
  Remove dependency on internal deque; use the canonical reward stream or environment episode history exposed via ``env.metrics`` or event logs. Consumers should request recent rewards via metrics APIs.
  ```

  Note: Consumers must be refactored to accept metrics API instead of internal deque.

- Line 1100: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
          def progress_ratio(self):
          self.completed_jobs = 0
          self._recent_rewards = deque(maxlen=20)
          self.completed_jobs = 0
              self._recent_rewards.clear()
                  self._recent_rewards = deque(maxlen=20)
                  self._recent_rewards = []
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1104: token `_recent_rewards` — classification: write — status: REQUIRES_REF

  Context:
  ```
          self.completed_jobs = 0
          self._recent_rewards = deque(maxlen=20)
          self.completed_jobs = 0
              self._recent_rewards.clear()
                  self._recent_rewards = deque(maxlen=20)
                  self._recent_rewards = []
          wip = float(self._wip())
  ```

  Replacement suggestion:
  ```
  Remove dependency on internal deque; use the canonical reward stream or environment episode history exposed via ``env.metrics`` or event logs. Consumers should request recent rewards via metrics APIs.
  ```

  Note: Consumers must be refactored to accept metrics API instead of internal deque.

- Line 1108: token `_recent_rewards` — classification: write — status: REQUIRES_REF

  Context:
  ```
          self._recent_rewards = deque(maxlen=20)
          self.completed_jobs = 0
              self._recent_rewards.clear()
                  self._recent_rewards = deque(maxlen=20)
                  self._recent_rewards = []
          wip = float(self._wip())
              self._recent_rewards.append(reward)
  ```

  Replacement suggestion:
  ```
  Remove dependency on internal deque; use the canonical reward stream or environment episode history exposed via ``env.metrics`` or event logs. Consumers should request recent rewards via metrics APIs.
  ```

  Note: Consumers must be refactored to accept metrics API instead of internal deque.

- Line 1110: token `_recent_rewards` — classification: write — status: REQUIRES_REF

  Context:
  ```
          self.completed_jobs = 0
              self._recent_rewards.clear()
                  self._recent_rewards = deque(maxlen=20)
                  self._recent_rewards = []
          wip = float(self._wip())
              self._recent_rewards.append(reward)
                          self.completed_jobs += 1
  ```

  Replacement suggestion:
  ```
  Remove dependency on internal deque; use the canonical reward stream or environment episode history exposed via ``env.metrics`` or event logs. Consumers should request recent rewards via metrics APIs.
  ```

  Note: Consumers must be refactored to accept metrics API instead of internal deque.

- Line 1338: token `_wip` — classification: write — status: NEEDS_TEST

  Context:
  ```
              self._recent_rewards.clear()
                  self._recent_rewards = deque(maxlen=20)
                  self._recent_rewards = []
          wip = float(self._wip())
              self._recent_rewards.append(reward)
                          self.completed_jobs += 1
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
  ```

  Replacement suggestion:
  ```
  Replace with computed WIP: ``sum(1 for j in getattr(env,'active_agents',[]) if not getattr(j,'finished',False))`` or ``int(env.current_wip)`` if a canonical counter exists. Prefer exposing via ``env.metrics.wip()``.
  ```

- Line 1346: token `_recent_rewards` — classification: write — status: REQUIRES_REF

  Context:
  ```
                  self._recent_rewards = deque(maxlen=20)
                  self._recent_rewards = []
          wip = float(self._wip())
              self._recent_rewards.append(reward)
                          self.completed_jobs += 1
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
  ```

  Replacement suggestion:
  ```
  Remove dependency on internal deque; use the canonical reward stream or environment episode history exposed via ``env.metrics`` or event logs. Consumers should request recent rewards via metrics APIs.
  ```

  Note: Consumers must be refactored to accept metrics API instead of internal deque.

- Line 1361: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                  self._recent_rewards = []
          wip = float(self._wip())
              self._recent_rewards.append(reward)
                          self.completed_jobs += 1
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1387: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
          wip = float(self._wip())
              self._recent_rewards.append(reward)
                          self.completed_jobs += 1
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1398: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
              self._recent_rewards.append(reward)
                          self.completed_jobs += 1
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                              job.remaining_time = dur
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1401: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                          self.completed_jobs += 1
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                              job.remaining_time = dur
                                          job.remaining_time = dur
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1415: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                              job.remaining_time = dur
                                          job.remaining_time = dur
              job.remaining_time = 0.0
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 1866: token `remaining_time` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                              job.remaining_time = dur
                                          job.remaining_time = dur
              job.remaining_time = 0.0
                          self.completed_jobs += 1
  ```

  Replacement suggestion:
  ```
  Use job-level attributes or compute from job.operations: e.g. ``job.remaining_time`` or derive progress as ``(total_ops - remaining_ops)/total_ops``. For agent obs use ``build_agent_obs(env, job)`` and read the 6-D normalized value.
  ```

- Line 2021: token `remaining_time` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                  tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                              job.remaining_time = dur
                                          job.remaining_time = dur
              job.remaining_time = 0.0
                          self.completed_jobs += 1
                                              tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
  ```

  Replacement suggestion:
  ```
  Use job-level attributes or compute from job.operations: e.g. ``job.remaining_time`` or derive progress as ``(total_ops - remaining_ops)/total_ops``. For agent obs use ``build_agent_obs(env, job)`` and read the 6-D normalized value.
  ```

- Line 2094: token `remaining_time` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                      tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                              job.remaining_time = dur
                                          job.remaining_time = dur
              job.remaining_time = 0.0
                          self.completed_jobs += 1
                                              tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
      def _wip(self):
  ```

  Replacement suggestion:
  ```
  Use job-level attributes or compute from job.operations: e.g. ``job.remaining_time`` or derive progress as ``(total_ops - remaining_ops)/total_ops``. For agent obs use ``build_agent_obs(env, job)`` and read the 6-D normalized value.
  ```

- Line 2100: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                              job.remaining_time = dur
                                          job.remaining_time = dur
              job.remaining_time = 0.0
                          self.completed_jobs += 1
                                              tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
      def _wip(self):
          to the internal work-in-progress count `_wip()` which is already used
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 2154: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                          job.remaining_time = dur
              job.remaining_time = 0.0
                          self.completed_jobs += 1
                                              tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
      def _wip(self):
          to the internal work-in-progress count `_wip()` which is already used
              return int(self._wip())
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 2368: token `_wip` — classification: compute — status: NEEDS_TEST

  Context:
  ```
              job.remaining_time = 0.0
                          self.completed_jobs += 1
                                              tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
      def _wip(self):
          to the internal work-in-progress count `_wip()` which is already used
              return int(self._wip())
      def _util_machines(self) -> float:
  ```

  Replacement suggestion:
  ```
  Replace with computed WIP: ``sum(1 for j in getattr(env,'active_agents',[]) if not getattr(j,'finished',False))`` or ``int(env.current_wip)`` if a canonical counter exists. Prefer exposing via ``env.metrics.wip()``.
  ```

- Line 2382: token `_wip` — classification: read — status: NEEDS_TEST

  Context:
  ```
                          self.completed_jobs += 1
                                              tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
      def _wip(self):
          to the internal work-in-progress count `_wip()` which is already used
              return int(self._wip())
      def _util_machines(self) -> float:
      def _util_ops(self) -> float:
  ```

  Replacement suggestion:
  ```
  Replace with computed WIP: ``sum(1 for j in getattr(env,'active_agents',[]) if not getattr(j,'finished',False))`` or ``int(env.current_wip)`` if a canonical counter exists. Prefer exposing via ``env.metrics.wip()``.
  ```

- Line 2391: token `_wip` — classification: read — status: NEEDS_TEST

  Context:
  ```
                                              tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
      def _wip(self):
          to the internal work-in-progress count `_wip()` which is already used
              return int(self._wip())
      def _util_machines(self) -> float:
      def _util_ops(self) -> float:
                                      tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
  ```

  Replacement suggestion:
  ```
  Replace with computed WIP: ``sum(1 for j in getattr(env,'active_agents',[]) if not getattr(j,'finished',False))`` or ``int(env.current_wip)`` if a canonical counter exists. Prefer exposing via ``env.metrics.wip()``.
  ```

- Line 2396: token `_util_machines` — classification: compute — status: NEEDS_TEST

  Context:
  ```
      def _wip(self):
          to the internal work-in-progress count `_wip()` which is already used
              return int(self._wip())
      def _util_machines(self) -> float:
      def _util_ops(self) -> float:
                                      tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                      tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
  ```

  Replacement suggestion:
  ```
  Replace with a canonical metrics call, e.g. ``env.metrics.machine_utilization()`` or compute as ``sum(1 for m in env.machines if m.is_busy())/len(env.machines)``; for operator util similarly over operations or use gantt aggregator.
  ```

  Note: This is often used in dashboards/metrics; validate numeric range and unit tests.

- Line 2415: token `_util_ops` — classification: compute — status: NEEDS_TEST

  Context:
  ```
          to the internal work-in-progress count `_wip()` which is already used
              return int(self._wip())
      def _util_machines(self) -> float:
      def _util_ops(self) -> float:
                                      tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                      tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
  ```

  Replacement suggestion:
  ```
  Replace with a canonical metrics call, e.g. ``env.metrics.machine_utilization()`` or compute as ``sum(1 for m in env.machines if m.is_busy())/len(env.machines)``; for operator util similarly over operations or use gantt aggregator.
  ```

  Note: This is often used in dashboards/metrics; validate numeric range and unit tests.

- Line 2569: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
              return int(self._wip())
      def _util_machines(self) -> float:
      def _util_ops(self) -> float:
                                      tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                      tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 2570: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
      def _util_machines(self) -> float:
      def _util_ops(self) -> float:
                                      tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                      tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                          tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 2581: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
      def _util_ops(self) -> float:
                                      tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                      tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                          tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                          completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 2584: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                      tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                      tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                          tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                          completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                          tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 2615: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                      tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                          tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                          completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                          tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
            - average_wait_time: total_wait_time / completed_jobs (seconds)
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 2626: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                  completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                  tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                          tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                          completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                          tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
            - average_wait_time: total_wait_time / completed_jobs (seconds)
                  completed = int(getattr(self, 'completed_jobs', 0))
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 2629: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                  tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                          tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                          completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                          tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
            - average_wait_time: total_wait_time / completed_jobs (seconds)
                  completed = int(getattr(self, 'completed_jobs', 0))
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 2757: token `completed_jobs` — classification: read — status: NEEDS_TEST

  Context:
  ```
                                          tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                          completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                          tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
            - average_wait_time: total_wait_time / completed_jobs (seconds)
                  completed = int(getattr(self, 'completed_jobs', 0))
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 2903: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
                                          completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                          tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
            - average_wait_time: total_wait_time / completed_jobs (seconds)
                  completed = int(getattr(self, 'completed_jobs', 0))
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

## my_data_and_graph/metrics.py

- Line 108: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
      def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
              avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                      completed_jobs = int(item.get('completed_jobs', 0))
                      util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 170: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
      def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
              avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                      completed_jobs = int(item.get('completed_jobs', 0))
                      util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 198: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
      def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
              avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                      completed_jobs = int(item.get('completed_jobs', 0))
                      util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 199: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
      def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
              avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                      completed_jobs = int(item.get('completed_jobs', 0))
                      util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

## tmp_smoke_test.py

- Line 14: token `completed_jobs` — classification: read — status: NEEDS_TEST

  Context:
  ```
  print("Done", "Completed:", env.completed_jobs, "Active:", len(env.active_agents), "Pending:", len(env.pending_jobs))
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

## tmp_test_gantt_decision_trace.py

- Line 35: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
          self.completed_jobs = 0
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

## utils/env_obs.py

- Line 121: token `time_norm` — classification: write — status: SAFE

  Context:
  ```
   5. wait_time_norm            -> job.wait_time / env.max_wait_time
      wait_time_norm = np.clip(wait_time_val / float(max_wait), 0.0, 1.0)
          wait_time_norm,
      legacy helpers such as _wip, _recent_rewards or completed_jobs.
  ```

  Replacement suggestion:
  ```
  Use canonical env normalization attributes like ``env.max_wait_time``, ``env.max_operations_per_job``, ``env.max_jobs`` and compute normalized values in-place. The observation builder already reads these; update consumers to use builder output.
  ```

## utils/gantt.py

- Line 644: token `completed_jobs` — classification: write — status: NEEDS_TEST

  Context:
  ```
          avg_wait = float(getattr(env, 'total_wait_time', 0.0)) / max(1.0, float(getattr(env, 'completed_jobs', 0) or 1))
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

## utils/jobagent.py

- Line 118: token `remaining_time` — classification: write — status: NEEDS_TEST

  Context:
  ```
          self.remaining_time = 0.0
      def progress_ratio(self):
          self.remaining_time = 0.0
  ```

  Replacement suggestion:
  ```
  Use job-level attributes or compute from job.operations: e.g. ``job.remaining_time`` or derive progress as ``(total_ops - remaining_ops)/total_ops``. For agent obs use ``build_agent_obs(env, job)`` and read the 6-D normalized value.
  ```

- Line 148: token `progress_ratio` — classification: compute — status: NEEDS_TEST

  Context:
  ```
          self.remaining_time = 0.0
      def progress_ratio(self):
          self.remaining_time = 0.0
  ```

  Replacement suggestion:
  ```
  Replace with canonical count of finished jobs, e.g. ``completed = len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])`` or use ``env.metrics.completed_jobs()`` if available. For per-agent observations prefer using ``utils.env_obs.build_agent_obs(env, job)`` for agent-level fields.
  ```

  Note: Ensure metrics consumers (training/plots) still match expected semantics; update tests.

- Line 263: token `remaining_time` — classification: write — status: NEEDS_TEST

  Context:
  ```
          self.remaining_time = 0.0
      def progress_ratio(self):
          self.remaining_time = 0.0
  ```

  Replacement suggestion:
  ```
  Use job-level attributes or compute from job.operations: e.g. ``job.remaining_time`` or derive progress as ``(total_ops - remaining_ops)/total_ops``. For agent obs use ``build_agent_obs(env, job)`` and read the 6-D normalized value.
  ```


# Next PR checklist

- Priority order: `environment.py`, `utils/env_obs.py`, `MARL/runner.py`, `my_data_and_graph/metrics.py`, `utils/gantt.py`, `utils/jobagent.py`, then tmp files.
- Group SAFE changes together where possible.
- For NEEDS_TEST items, add/modify unit tests prior to merging.
- For REQUIRES_REF, open PR and notify owners of dependent modules.
