# Observation Legacy Full Scan Summary

Date: 2025-11-09T15:50:04.780719Z

- Total hits: 60
- HIGH: 56
- MEDIUM: 4
- LOW: 0

## HIGH risk (called in active code)

- `MARL/runner.py`:795 — match `progress_ratio`

``
    - 11D observations (progress_ratio)
                before_completed = int(getattr(self.env, 'completed_jobs', 0))
                    # fallback: try env.total_wait_time / completed_jobs
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
``

- `MARL/runner.py`:1193 — match `completed_jobs`

``
    - 11D observations (progress_ratio)
                before_completed = int(getattr(self.env, 'completed_jobs', 0))
                    # fallback: try env.total_wait_time / completed_jobs
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
``

- `MARL/runner.py`:1246 — match `completed_jobs`

``
    - 11D observations (progress_ratio)
                before_completed = int(getattr(self.env, 'completed_jobs', 0))
                    # fallback: try env.total_wait_time / completed_jobs
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                    util_m = self.env._util_machines()
``

- `MARL/runner.py`:1493 — match `completed_jobs`

``
                before_completed = int(getattr(self.env, 'completed_jobs', 0))
                    # fallback: try env.total_wait_time / completed_jobs
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                    util_m = self.env._util_machines()
                    util_o = self.env._util_ops()
``

- `MARL/runner.py`:1533 — match `completed_jobs`

``
                    # fallback: try env.total_wait_time / completed_jobs
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                    util_m = self.env._util_machines()
                    util_o = self.env._util_ops()
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
``

- `MARL/runner.py`:1534 — match `_util_machines`

``
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                    util_m = self.env._util_machines()
                    util_o = self.env._util_ops()
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
                        'completed_jobs': int(delta_completed),
``

- `MARL/runner.py`:1535 — match `_util_ops`

``
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                    util_m = self.env._util_machines()
                    util_o = self.env._util_ops()
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
                        'completed_jobs': int(delta_completed),
                        util_m = float(self.env._util_machines())
``

- `MARL/runner.py`:1549 — match `completed_jobs`

``
                    avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                    util_m = self.env._util_machines()
                    util_o = self.env._util_ops()
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
                        'completed_jobs': int(delta_completed),
                        util_m = float(self.env._util_machines())
                        util_o = float(self.env._util_ops())
``

- `MARL/runner.py`:1574 — match `completed_jobs`

``
                    util_m = self.env._util_machines()
                    util_o = self.env._util_ops()
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
                        'completed_jobs': int(delta_completed),
                        util_m = float(self.env._util_machines())
                        util_o = float(self.env._util_ops())
``

- `MARL/runner.py`:1880 — match `_util_machines`

``
                    util_o = self.env._util_ops()
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
                        'completed_jobs': int(delta_completed),
                        util_m = float(self.env._util_machines())
                        util_o = float(self.env._util_ops())
``

- `MARL/runner.py`:1881 — match `_util_ops`

``
                        after_completed = int(getattr(self.env, 'completed_jobs', 0))
                        'completed_jobs': int(delta_completed),
                        util_m = float(self.env._util_machines())
                        util_o = float(self.env._util_ops())
``

- `environment.py`:107 — match `remaining_time`

``
            self.remaining_time = 0.0
        def progress_ratio(self):
        self.completed_jobs = 0
        self._recent_rewards = deque(maxlen=20)
``

- `environment.py`:117 — match `progress_ratio`

``
            self.remaining_time = 0.0
        def progress_ratio(self):
        self.completed_jobs = 0
        self._recent_rewards = deque(maxlen=20)
        self.completed_jobs = 0
``

- `environment.py`:770 — match `completed_jobs`

``
            self.remaining_time = 0.0
        def progress_ratio(self):
        self.completed_jobs = 0
        self._recent_rewards = deque(maxlen=20)
        self.completed_jobs = 0
            self._recent_rewards.clear()
``

- `environment.py`:773 — match `_recent_rewards`

``
            self.remaining_time = 0.0
        def progress_ratio(self):
        self.completed_jobs = 0
        self._recent_rewards = deque(maxlen=20)
        self.completed_jobs = 0
            self._recent_rewards.clear()
                self._recent_rewards = deque(maxlen=20)
``

- `environment.py`:1100 — match `completed_jobs`

``
        def progress_ratio(self):
        self.completed_jobs = 0
        self._recent_rewards = deque(maxlen=20)
        self.completed_jobs = 0
            self._recent_rewards.clear()
                self._recent_rewards = deque(maxlen=20)
                self._recent_rewards = []
``

- `environment.py`:1104 — match `_recent_rewards`

``
        self.completed_jobs = 0
        self._recent_rewards = deque(maxlen=20)
        self.completed_jobs = 0
            self._recent_rewards.clear()
                self._recent_rewards = deque(maxlen=20)
                self._recent_rewards = []
        wip = float(self._wip())
``

- `environment.py`:1108 — match `_recent_rewards`

``
        self._recent_rewards = deque(maxlen=20)
        self.completed_jobs = 0
            self._recent_rewards.clear()
                self._recent_rewards = deque(maxlen=20)
                self._recent_rewards = []
        wip = float(self._wip())
            self._recent_rewards.append(reward)
``

- `environment.py`:1110 — match `_recent_rewards`

``
        self.completed_jobs = 0
            self._recent_rewards.clear()
                self._recent_rewards = deque(maxlen=20)
                self._recent_rewards = []
        wip = float(self._wip())
            self._recent_rewards.append(reward)
                        self.completed_jobs += 1
``

- `environment.py`:1338 — match `_wip`

``
            self._recent_rewards.clear()
                self._recent_rewards = deque(maxlen=20)
                self._recent_rewards = []
        wip = float(self._wip())
            self._recent_rewards.append(reward)
                        self.completed_jobs += 1
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
``

- `environment.py`:1346 — match `_recent_rewards`

``
                self._recent_rewards = deque(maxlen=20)
                self._recent_rewards = []
        wip = float(self._wip())
            self._recent_rewards.append(reward)
                        self.completed_jobs += 1
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
``

- `environment.py`:1361 — match `completed_jobs`

``
                self._recent_rewards = []
        wip = float(self._wip())
            self._recent_rewards.append(reward)
                        self.completed_jobs += 1
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
``

- `environment.py`:1387 — match `completed_jobs`

``
        wip = float(self._wip())
            self._recent_rewards.append(reward)
                        self.completed_jobs += 1
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
``

- `environment.py`:1398 — match `completed_jobs`

``
            self._recent_rewards.append(reward)
                        self.completed_jobs += 1
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                            job.remaining_time = dur
``

- `environment.py`:1401 — match `completed_jobs`

``
                        self.completed_jobs += 1
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                            job.remaining_time = dur
                                        job.remaining_time = dur
``

- `environment.py`:1415 — match `completed_jobs`

``
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                            job.remaining_time = dur
                                        job.remaining_time = dur
            job.remaining_time = 0.0
``

- `environment.py`:1866 — match `remaining_time`

``
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                            job.remaining_time = dur
                                        job.remaining_time = dur
            job.remaining_time = 0.0
                        self.completed_jobs += 1
``

- `environment.py`:2021 — match `remaining_time`

``
                                tfs.write(f"[Lifecycle] t={float(now_t):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                            job.remaining_time = dur
                                        job.remaining_time = dur
            job.remaining_time = 0.0
                        self.completed_jobs += 1
                                            tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
``

- `environment.py`:2094 — match `remaining_time`

``
                                    tf.write(f"[t={float(now_t):.2f}] Job {getattr(job, 'id', None)} completed -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                            job.remaining_time = dur
                                        job.remaining_time = dur
            job.remaining_time = 0.0
                        self.completed_jobs += 1
                                            tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
    def _wip(self):
``

- `environment.py`:2100 — match `completed_jobs`

``
                            job.remaining_time = dur
                                        job.remaining_time = dur
            job.remaining_time = 0.0
                        self.completed_jobs += 1
                                            tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
    def _wip(self):
        to the internal work-in-progress count `_wip()` which is already used
``

- `environment.py`:2154 — match `completed_jobs`

``
                                        job.remaining_time = dur
            job.remaining_time = 0.0
                        self.completed_jobs += 1
                                            tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
    def _wip(self):
        to the internal work-in-progress count `_wip()` which is already used
            return int(self._wip())
``

- `environment.py`:2368 — match `_wip`

``
            job.remaining_time = 0.0
                        self.completed_jobs += 1
                                            tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
    def _wip(self):
        to the internal work-in-progress count `_wip()` which is already used
            return int(self._wip())
    def _util_machines(self) -> float:
``

- `environment.py`:2382 — match `_wip`

``
                        self.completed_jobs += 1
                                            tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
    def _wip(self):
        to the internal work-in-progress count `_wip()` which is already used
            return int(self._wip())
    def _util_machines(self) -> float:
    def _util_ops(self) -> float:
``

- `environment.py`:2391 — match `_wip`

``
                                            tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(next_job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
    def _wip(self):
        to the internal work-in-progress count `_wip()` which is already used
            return int(self._wip())
    def _util_machines(self) -> float:
    def _util_ops(self) -> float:
                                    tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
``

- `environment.py`:2396 — match `_util_machines`

``
    def _wip(self):
        to the internal work-in-progress count `_wip()` which is already used
            return int(self._wip())
    def _util_machines(self) -> float:
    def _util_ops(self) -> float:
                                    tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                    tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
``

- `environment.py`:2415 — match `_util_ops`

``
        to the internal work-in-progress count `_wip()` which is already used
            return int(self._wip())
    def _util_machines(self) -> float:
    def _util_ops(self) -> float:
                                    tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                    tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
``

- `environment.py`:2569 — match `completed_jobs`

``
            return int(self._wip())
    def _util_machines(self) -> float:
    def _util_ops(self) -> float:
                                    tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                    tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
``

- `environment.py`:2570 — match `completed_jobs`

``
    def _util_machines(self) -> float:
    def _util_ops(self) -> float:
                                    tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                    tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
``

- `environment.py`:2581 — match `completed_jobs`

``
    def _util_ops(self) -> float:
                                    tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                    tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
``

- `environment.py`:2584 — match `completed_jobs`

``
                                    tf.write(f"[t={float(getattr(job, 'arrival_time', 0.0)):.2f}] New job {getattr(job, 'id', None)} arrived with {len(getattr(job, 'operations', []) or [])} ops -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                    tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
``

- `environment.py`:2615 — match `completed_jobs`

``
                                    tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} became active agent → Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
          - average_wait_time: total_wait_time / completed_jobs (seconds)
``

- `environment.py`:2626 — match `completed_jobs`

``
                                completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
          - average_wait_time: total_wait_time / completed_jobs (seconds)
                completed = int(getattr(self, 'completed_jobs', 0))
``

- `environment.py`:2629 — match `completed_jobs`

``
                                tf2.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
                                        tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
          - average_wait_time: total_wait_time / completed_jobs (seconds)
                completed = int(getattr(self, 'completed_jobs', 0))
``

- `environment.py`:2757 — match `completed_jobs`

``
                                        tf.write(f"[t={float(getattr(self.env, 'now', 0.0)):.2f}] Job {getattr(job, 'id', None)} queued (pending) -> Active:{len(getattr(self, 'active_agents', []) or [])} | Pending:{len(getattr(self, 'pending_jobs', []) or [])} | Completed:{int(getattr(self, 'completed_jobs', 0))}\n")
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
          - average_wait_time: total_wait_time / completed_jobs (seconds)
                completed = int(getattr(self, 'completed_jobs', 0))
``

- `environment.py`:2903 — match `completed_jobs`

``
                                        completed_jobs = [j for j in (getattr(self, 'jobs', []) or []) if getattr(j, 'finished', False)]
                                        tfq.write(f"[Lifecycle] t={float(getattr(self, 't', getattr(self.env, 'now', 0.0))):.2f} | Active={len(active_jobs)} Pending={len(pending_jobs)} Completed={len(completed_jobs)} / Total={total_jobs}\n")
          - average_wait_time: total_wait_time / completed_jobs (seconds)
                completed = int(getattr(self, 'completed_jobs', 0))
``

- `my_data_and_graph/metrics.py`:108 — match `completed_jobs`

``
    def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
            avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                    completed_jobs = int(item.get('completed_jobs', 0))
                    util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
``

- `my_data_and_graph/metrics.py`:170 — match `completed_jobs`

``
    def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
            avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                    completed_jobs = int(item.get('completed_jobs', 0))
                    util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
``

- `my_data_and_graph/metrics.py`:198 — match `completed_jobs`

``
    def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
            avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                    completed_jobs = int(item.get('completed_jobs', 0))
                    util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
``

- `my_data_and_graph/metrics.py`:199 — match `completed_jobs`

``
    def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_jobs: int = 0):
            avg_wait = float(total_wait_time) / max(1.0, float(completed_jobs))
                    completed_jobs = int(item.get('completed_jobs', 0))
                    util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_jobs=completed_jobs)
``

- `tmp_smoke_test.py`:14 — match `completed_jobs`

``
print("Done", "Completed:", env.completed_jobs, "Active:", len(env.active_agents), "Pending:", len(env.pending_jobs))
``

- `tmp_test_gantt_decision_trace.py`:35 — match `completed_jobs`

``
        self.completed_jobs = 0
``

- `utils/env_obs.py`:121 — match `time_norm`

``
 5. wait_time_norm            -> job.wait_time / env.max_wait_time
    wait_time_norm = np.clip(wait_time_val / float(max_wait), 0.0, 1.0)
        wait_time_norm,
    legacy helpers such as _wip, _recent_rewards or completed_jobs.
``

- `utils/gantt.py`:644 — match `completed_jobs`

``
        avg_wait = float(getattr(env, 'total_wait_time', 0.0)) / max(1.0, float(getattr(env, 'completed_jobs', 0) or 1))
``

- `utils/jobagent.py`:118 — match `remaining_time`

``
        self.remaining_time = 0.0
    def progress_ratio(self):
        self.remaining_time = 0.0
``

- `utils/jobagent.py`:148 — match `progress_ratio`

``
        self.remaining_time = 0.0
    def progress_ratio(self):
        self.remaining_time = 0.0
``

- `utils/jobagent.py`:263 — match `remaining_time`

``
        self.remaining_time = 0.0
    def progress_ratio(self):
        self.remaining_time = 0.0
``

## MEDIUM risk (tests or imports)

- `MARL/runner.py`:1244 — `completed_jobs`
- `utils/env_obs.py`:15 — `time_norm`
- `utils/env_obs.py`:129 — `time_norm`
- `utils/env_obs.py`:142 — `_wip`

## LOW risk (comments, docs, reports)

None


## Recommended actions
- Remove or migrate HIGH-risk references first; modify consumers to use canonical attributes or the new metrics APIs.
- Review MEDIUM hits: ensure tests are updated to reflect new API or explicitly whitelist references.
- LOW hits may be informational; clean docs over time.

## Clean State Criteria
All HIGH-risk references must be removed from active code.
