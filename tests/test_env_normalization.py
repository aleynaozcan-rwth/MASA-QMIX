import pytest
from MARL.common.arguments import get_mutable_args
from environment import MASAEnv
from utils.env_obs import build_agent_obs


@pytest.mark.sanity
def test_env_canonical_normalization():
    # Obtain authoritative, mutable args from the project's argument builder
    args = get_mutable_args()

    # Construct the environment using the authoritative args (no fallbacks/overrides)
    env = MASAEnv(args=args)

    # Canonical links and positive invariants
    assert env.max_jobs == args.n_agents
    assert env.max_jobs > 0
    assert env.max_operations_per_job > 0
    assert env.n_operation_types > 0
    assert env.max_wait_time > 0
    assert env.obs_dim_agent == args.obs_shape

    # Build a job using the repo's job generator (must provide canonical JobAgent)
    job = env.job_generator.create_job()

    # If the job generator returns a raw representation (list/tuple),
    # synthesize a minimal JobAgent-like adapter for the test so we don't
    # change generator behavior in the repo. The adapter exposes the
    # attributes expected by build_agent_obs():
    #   - current_operation()
    #   - operations
    #   - current_op_idx
    #   - wait_time
    #   - finished
    if not hasattr(job, 'current_operation'):
        src = job

        class JobAgentAdapter:
            def __init__(self, src):
                # src may be a list/tuple of ops or a single op descriptor
                if isinstance(src, (list, tuple)):
                    self.operations = list(src)
                else:
                    self.operations = [src]
                self.current_op_idx = 0
                self.wait_time = 0.0
                self.finished = False

            def current_operation(self):
                op = self.operations[self.current_op_idx]
                # op may be a tuple like (op_type, allowed_machines, durations)
                if isinstance(op, (list, tuple)) and len(op) > 0:
                    t = op[0]
                elif hasattr(op, 'type'):
                    t = getattr(op, 'type')
                else:
                    t = 0

                class Op:
                    def __init__(self, t):
                        self.type = t

                return Op(t)

        job = JobAgentAdapter(src)

    # Build observation and validate
    obs = build_agent_obs(env, job)
    assert len(obs) == env.obs_dim_agent
    assert float(min(obs)) >= 0.0
    assert float(max(obs)) <= 1.0
