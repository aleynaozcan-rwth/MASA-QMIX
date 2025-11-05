"""
utils/jobagent.py
Step 8A.5.3 — Machine-level Decision Integration
------------------------------------------------
Enhancements vs Step 8A.3:
- JobAgent can now choose and record actions at the Machine level.
- execute_task() accepts (machine_id, workcenter_id) and expects an
    env-style op tuple with explicit per-machine durations. Legacy
    `time_span` semantics have been removed.
"""

from typing import Optional


# ======================================================================
# Container for all JobAgents
# ======================================================================
class JobAgents:
    """Manages all JobAgent objects and simple bookkeeping utilities."""
    def __init__(self, numbers: Optional[int] = None, dynamic: bool = False, task_provider: Optional[object] = None):
        """
        Create JobAgents container.

        Args:
            numbers: optional number of JobAgent instances to create. If None,
                     will try to infer from `task_provider` (length or provider.num_jobagents).
            task_provider: optional Task-like provider or a list of job-objects.
                           If None, no agents are precreated (caller should create them).
        """
        self.dynamic = dynamic
        self.agents_object_list = []

        # Resolve job list and default count from task_provider
        job_list = []
        inferred_numbers = None
        if task_provider is not None:
            # Task instance with attribute .simple_task_object and optional .num_jobagents
            if hasattr(task_provider, 'simple_task_object'):
                job_list = list(getattr(task_provider, 'simple_task_object') or [])
                try:
                    inferred_numbers = int(getattr(task_provider, 'num_jobagents'))
                except Exception:
                    inferred_numbers = len(job_list) or None
            # direct list/iterable of job-objects
            elif isinstance(task_provider, (list, tuple)):
                job_list = list(task_provider)
                inferred_numbers = len(job_list)
            else:
                # try to coerce arbitrary iterables
                try:
                    job_list = list(task_provider)
                    inferred_numbers = len(job_list)
                except Exception:
                    job_list = []

        # Finalize number of agents to create
        n_create = int(numbers) if numbers is not None else (int(inferred_numbers) if inferred_numbers is not None else 0)

        for i in range(n_create):
            temp_object = JobAgent(i, job_list, arrival_time=0.0)
            self.agents_object_list.append(temp_object)

    # ------------------------------------------------------------------
    def count_jobs(self):
        """Return (remaining_jobs, total_jobs) across all JobAgents."""
        left_jobs, all_jobs = 0, 0
        for a in self.agents_object_list:
            left_jobs += len(a.left_job)
            all_jobs += len(a.static_job_list)
        return left_jobs, all_jobs

    def add_new_agent(self, agent):
        """Add a new JobAgent object (for dynamic arrivals)."""
        self.agents_object_list.append(agent)
        print(f"[JobAgents] Added new JobAgent {agent.agent_id} at t={agent.arrival_time}, "
              f"{len(agent.left_job)} jobs.")

    def active_agent_ids(self):
        """Return IDs of currently active JobAgents."""
        return [a.agent_id for a in self.agents_object_list if a.is_active]

    def reset_all(self):
        """Reset all JobAgents (for environment re-init)."""
        for a in self.agents_object_list:
            a.reset()


# ======================================================================
# Individual JobAgent
# ======================================================================
class JobAgent:
    """
    Represents one JobAgent executing a predefined sequence of jobs.
    Step 8A.5.3 adds machine-level decision tracking and explainable logs.
    """

    def __init__(self, agent_id, job_object_list, arrival_time: float = 0.0):
        # --- Core state ---
        # `agent_id` historically named but environment expects `.id`
        self.agent_id = agent_id
        self.id = int(agent_id)

    # support two input shapes:
    #  - a list of Task-like objects (existing utils usage)
    #  - a list of env-style operation tuples (op_type, allowed_machine_indices, per_machine_durations)
        self.static_job_list = list(job_object_list)
        self.operations = list(job_object_list)
        # left_job mirrors previous behaviour: remaining work items (can be tasks or op-tuples)
        self.left_job = [j for j in job_object_list]
        self.finished_job = []
        self.time_spent = 0.0

        # env-compatible fields
        self.current_op_idx = 0
        self.remaining_time = 0.0
        self.wait_time = 0.0
        self.finished = False
        # --- Decision tracking ---
        self.last_chosen_machine = None
        self.machine_history = []     # Stores chosen machine IDs
        self.workcenter_history = []  # Still stored for compatibility
    # legacy speed tracking removed

        # --- Arrival / activity control ---
        self.arrival_time = float(arrival_time)
        self.is_active = False

        # --- Replay/log fields ---
        self.completed_at = None  # SimPy time when completed

    # Environment-compatible accessors expected by environment.py
    def current_op(self):
        """Return the current operation tuple or Task-like object for the env._job_process."""
        try:
            if self.finished or int(self.current_op_idx) >= len(self.operations):
                return None
            return self.operations[int(self.current_op_idx)]
        except Exception:
            # fallback to left_job items if operations not suitable
            try:
                return self.left_job[0]
            except Exception:
                return None

    def progress_ratio(self):
        try:
            return float(self.current_op_idx) / max(1.0, float(len(self.operations)))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------
    def execute_task(self, job_object, machine_id, workcenter_id=None):
        """
        Execute next job and update internal state.
        Machine-level integration (Step 8A.5.3):
          - Records selected machine and speed factor.
          - Returns actual processing time after speed adjustment.
        """
        # Strict mode: expect env-style op tuple with explicit per-machine
        # durations as the third element (dict mapping machine_index -> duration).
        # Legacy job_object.time_span support has been removed.
        base_time = None
        if isinstance(job_object, (list, tuple)) and len(job_object) >= 3:
            third = job_object[2]
            if isinstance(third, dict):
                # machine_id is an integer index into per-machine durations
                if int(machine_id) in third:
                    base_time = float(third.get(int(machine_id)))
                else:
                    raise ValueError(f"Missing explicit duration for machine {machine_id} in op tuple")
            else:
                # third element is a scalar duration
                try:
                    base_time = float(third)
                except Exception:
                    raise ValueError("Operation tuple third element must be a duration or a dict of per-machine durations")
        else:
            raise ValueError("JobAgent.execute_task expects op tuples with explicit per-machine durations.")

        # In strict mode, adjusted_time is the explicit duration (no speed scaling)
        adjusted_time = float(base_time)

        # Update internal metrics
        self.time_spent += adjusted_time
        # consume one left_job if present
        if self.left_job:
            try:
                self.finished_job.append(self.left_job.pop(0))
            except Exception:
                pass
        # update env-like pointers
        try:
            self.current_op_idx = int(self.current_op_idx) + 1
        except Exception:
            self.current_op_idx = getattr(self, 'current_op_idx', 0) + 1
        self.last_chosen_machine = machine_id
        self.machine_history.append(machine_id)
        if workcenter_id is not None:
            self.workcenter_history.append(workcenter_id)
        print(f"[JobAgent {self.agent_id}] Executed task on Machine {machine_id} (WC {workcenter_id}) | duration={adjusted_time:.2f}")

        # mark finished flag if no left jobs remain
        if not self.left_job:
            self.finished = True

        return adjusted_time

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def mark_completed(self, sim_time: Optional[float] = None):
        """Mark JobAgent as completed (for replay/log credit)."""
        self.is_active = False
        self.completed_at = sim_time
        print(f"[JobAgent] JobAgent {self.agent_id} completed all jobs at t={sim_time}")

    def reset(self):
        """Reset internal state (used on environment reset)."""
        self.left_job = list(self.static_job_list)
        self.finished_job.clear()
        self.time_spent = 0.0
        self.machine_history.clear()
        self.workcenter_history.clear()
    # legacy speed history cleared implicitly by removing list
        self.last_chosen_machine = None
        self.is_active = False
        self.completed_at = None
        # env-compatible resets
        self.current_op_idx = 0
        self.remaining_time = 0.0
        self.wait_time = 0.0
        self.finished = False

    # ------------------------------------------------------------------
    # Debug representation
    # ------------------------------------------------------------------
    def __repr__(self):
        status = "active" if self.is_active else "waiting"
        last_m = self.last_chosen_machine if self.last_chosen_machine else "-"
        return (f"<JobAgent id={self.agent_id}, status={status}, "
                f"last_machine={last_m}, left_jobs={len(self.left_job)}, "
                f"current_op_idx={getattr(self,'current_op_idx',0)}, "
                f"arrival={self.arrival_time}, completed_at={self.completed_at}>")
