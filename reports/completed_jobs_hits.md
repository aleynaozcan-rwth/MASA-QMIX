# completed_jobs Usage Audit

Date: 2025-11-09T17:12:23.952146

Total matches: 26

## MARL/runner.py — 7 match(es)

- Line 1193: classification=read
  ```
                before_completed = int(getattr(self.env, 'completed_jobs', 0))
                    # fallback: try env.total_wait_time / completed_jobs
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

- Line 1244: classification=comment
  ```
                before_completed = int(getattr(self.env, 'completed_jobs', 0))
                    # fallback: try env.total_wait_time / completed_jobs
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

- Line 1246: classification=read
  ```
                before_completed = int(getattr(self.env, 'completed_jobs', 0))
                    # fallback: try env.total_wait_time / completed_jobs
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

- Line 1493: classification=read
  ```
                before_completed = int(getattr(self.env, 'completed_jobs', 0))
                    # fallback: try env.total_wait_time / completed_jobs
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
                        'completed_jobs': int(delta_completed),
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

- Line 1533: classification=read
  ```
                    # fallback: try env.total_wait_time / completed_jobs
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
                        'completed_jobs': int(delta_completed),
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

- Line 1549: classification=read
  ```
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
                        'completed_jobs': int(delta_completed),
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

- Line 1574: classification=read
  ```
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
                        'completed_jobs': int(delta_completed),
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

## environment.py — 11 match(es)

- Line 770: classification=comment
  ```
        # legacy `completed_jobs` counter removed in favour of computed metrics
        # legacy `completed_jobs` removed; compute finished counts from jobs when needed
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

- Line 1104: classification=comment
  ```
        # legacy `completed_jobs` counter removed in favour of computed metrics
        # legacy `completed_jobs` removed; compute finished counts from jobs when needed
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

- Line 1363: classification=comment
  ```
        # legacy `completed_jobs` counter removed in favour of computed metrics
        # legacy `completed_jobs` removed; compute finished counts from jobs when needed
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                        # legacy `completed_jobs` counter removed; compute finished
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

- Line 1404: classification=assignment
  ```
        # legacy `completed_jobs` counter removed in favour of computed metrics
        # legacy `completed_jobs` removed; compute finished counts from jobs when needed
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

- Line 1407: classification=print/logging
  ```
        # legacy `completed_jobs` removed; compute finished counts from jobs when needed
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

- Line 2107: classification=comment
  ```
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

- Line 2553: classification=assignment
  ```
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

- Line 2556: classification=read
  ```
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
          - average_wait_time: total_wait_time / completed_jobs (seconds)
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

- Line 2599: classification=assignment
  ```
                        # legacy `completed_jobs` counter removed; compute finished
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
          - average_wait_time: total_wait_time / completed_jobs (seconds)
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

- Line 2602: classification=read
  ```
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
          - average_wait_time: total_wait_time / completed_jobs (seconds)
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

- Line 2730: classification=read
  ```
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
          - average_wait_time: total_wait_time / completed_jobs (seconds)
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

## my_data_and_graph/metrics.py — 4 match(es)

- Line 108: classification=read
  ```
    def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
            avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                    completed_jobs = int(item.get('completed_jobs', 0))
                    util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

- Line 170: classification=read
  ```
    def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
            avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                    completed_jobs = int(item.get('completed_jobs', 0))
                    util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

- Line 198: classification=assignment
  ```
    def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
            avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                    completed_jobs = int(item.get('completed_jobs', 0))
                    util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

- Line 199: classification=assignment
  ```
    def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
            avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                    completed_jobs = int(item.get('completed_jobs', 0))
                    util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

## tmp_smoke_test.py — 1 match(es)

- Line 14: classification=print/logging
  ```
print("Done", "Completed:", env.completed_jobs, "Active:", len(env.active_agents), "Pending:", len(env.pending_jobs))
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

## tmp_test_gantt_decision_trace.py — 1 match(es)

- Line 35: classification=assignment
  ```
        self.completed_jobs = 0
  ```
  Suggested replacement:
  ```
Remove assignment to legacy counter; compute finished counts on-demand from env.jobs or expose a metrics API.
  ```

## utils/env_obs.py — 1 match(es)

- Line 142: classification=read
  ```
    legacy helpers such as _wip, _recent_rewards or completed_jobs.
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

## utils/gantt.py — 1 match(es)

- Line 644: classification=read
  ```
        avg_wait = float(getattr(env, 'total_wait_time', 0.0)) / max(1.0, float(getattr(env, 'completed_jobs', 0) or 1))
  ```
  Suggested replacement:
  ```
Replace with computed finished count: len([j for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False)])  (or use env.metrics.completed_jobs() if a metrics API is provided)
  ```

