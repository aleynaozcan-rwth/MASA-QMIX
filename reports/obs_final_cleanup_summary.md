# Post-Cleanup Repo Scan Summary

**Total hits:** 24

## HIGH (0)

## MEDIUM (0)

## LOW (24)
- `./environment.py:107` → `self.remaining_time = 0.0`
- `./environment.py:117` → `def progress_ratio(self):`
- `./environment.py:770` → `# legacy `completed_jobs` counter removed in favour of computed metrics`
- `./environment.py:1104` → `# legacy `completed_jobs` removed; compute finished counts from jobs when needed`
- `./environment.py:1338` → `# legacy `_wip()` helper.`
- `./environment.py:1363` → `# legacy `completed_jobs` counter removed; compute finished`
- `./environment.py:1404` → `completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]`
- `./environment.py:1407` → `tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")`
- `./environment.py:1873` → `job.remaining_time = dur`
- `./environment.py:2028` → `job.remaining_time = dur`
- `./environment.py:2101` → `job.remaining_time = 0.0`
- `./environment.py:2107` → `# legacy `completed_jobs` counter removed; compute finished`
- `./environment.py:2553` → `completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]`
- `./environment.py:2556` → `tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")`
- `./environment.py:2599` → `completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]`
- `./environment.py:2602` → `tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")`
- `./environment.py:2730` → `- average_wait_time: total_wait_time / completed_jobs (seconds)`
- `./tmp_test_gantt_decision_trace.py:35` → `self.completed_jobs = 0`
- `./tmp_smoke_test.py:14` → `# compute completed count on-demand (legacy env.completed_jobs removed)`
- `./utils/jobagent.py:118` → `self.remaining_time = 0.0`
- `./utils/jobagent.py:148` → `def progress_ratio(self):`
- `./utils/jobagent.py:263` → `self.remaining_time = 0.0`
- `./utils/env_obs.py:6` → `does NOT contain any legacy 11-D features or silent fallbacks — if a`
- `./utils/env_obs.py:142` → `legacy helpers such as _wip, _recent_rewards or completed_jobs.`

### Acceptance Criteria
- HIGH == 0 → fully clean for commit.
- MEDIUM == 0 → optional before push.
- LOW == 0 → full repo documentation clean.
