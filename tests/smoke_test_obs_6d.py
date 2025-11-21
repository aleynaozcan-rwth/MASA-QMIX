# Smoke test for 6-D agent observation
import sys
import os
# Ensure project root is on sys.path for imports
sys.path.append(os.getcwd())

import numpy as np
from environment import MASAEnv
from utils.env_obs import build_agent_obs


# Obtain authoritative args from the project's parser
try:
    from MARL.common.arguments import get_mutable_args
    args = get_mutable_args()
except Exception:
    # Fall back to common args if mutable builder not available
    try:
        from MARL.common.arguments import get_common_args
        args = get_common_args().as_mutable()
    except Exception:
        raise

print("[Smoke Test] Using authoritative args from MARL.common.arguments")
print(f"  n_agents = {getattr(args, 'n_agents', None)}")
print(f"  job_max_ops = {getattr(args, 'job_max_ops', None)}")
print(f"  n_operation_types = {getattr(args, 'n_operation_types', None)}")
print(f"  max_wait_time = {getattr(args, 'max_wait_time', None)}")
print(f"  obs_shape = {getattr(args, 'obs_shape', None)}")

# Construct environment using repo-provided args (no ad-hoc overrides)
env = MASAEnv(args=args)

print("\nEnvironment normalization references (post-construction):")
print(f"  n_operation_types: {env.n_operation_types}")
print(f"  max_operations_per_job: {env.max_operations_per_job}")
print(f"  max_wait_time: {env.max_wait_time}")
print(f"  max_jobs: {env.max_jobs}")
print(f"  obs_dim_agent: {env.obs_dim_agent}")

# Determine source mapping for each canonical attribute
source_map = {}
# max_jobs: prefer args.max_jobs, then args.n_agents, then env.initial_jobs
if getattr(args, 'max_jobs', None) is not None and env.max_jobs == int(getattr(args, 'max_jobs')):
    source_map['max_jobs'] = f"args.max_jobs:{args.max_jobs}"
elif getattr(args, 'n_agents', None) is not None and env.max_jobs == int(getattr(args, 'n_agents')):
    source_map['max_jobs'] = f"args.n_agents:{args.n_agents}"
elif getattr(env, 'initial_jobs', None) is not None and env.max_jobs == int(getattr(env, 'initial_jobs')):
    source_map['max_jobs'] = f"env.initial_jobs={env.initial_jobs}"
else:
    source_map['max_jobs'] = 'inferred/default'

# max_operations_per_job: args.job_max_ops or job_generator.default_max_ops
if getattr(args, 'job_max_ops', None) is not None and env.max_operations_per_job == int(getattr(args, 'job_max_ops')):
    source_map['max_operations_per_job'] = f"args.job_max_ops:{args.job_max_ops}"
elif getattr(env, 'job_generator', None) is not None and hasattr(env.job_generator, 'default_max_ops') and env.max_operations_per_job == int(getattr(env.job_generator, 'default_max_ops')):
    source_map['max_operations_per_job'] = 'job_generator.default_max_ops'
else:
    source_map['max_operations_per_job'] = 'inferred/default'

# n_operation_types: args.n_operation_types or inferred from workcenters_meta capabilities
if getattr(args, 'n_operation_types', None) is not None:
    source_map['n_operation_types'] = f"args.n_operation_types:{args.n_operation_types}"
else:
    try:
        wc = getattr(env, 'workcenters_meta', None)
        caps = []
        for mname, md in (getattr(wc, 'machine_registry', {}) or {}).items():
            try:
                caps.extend(list(md.get('capabilities', []) or []))
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                continue
        if caps:
            inferred = int(max(caps) + 1)
            if env.n_operation_types == inferred:
                source_map['n_operation_types'] = 'workcenters_meta.capabilities (max+1)'
            else:
                source_map['n_operation_types'] = 'inferred/default'
        else:
            source_map['n_operation_types'] = 'inferred/default'
    except Exception:
        source_map['n_operation_types'] = 'inferred/default'

# max_wait_time: args.max_wait_time or env default
if getattr(args, 'max_wait_time', None) is not None and float(env.max_wait_time) == float(getattr(args, 'max_wait_time')):
    source_map['max_wait_time'] = f"args.max_wait_time:{args.max_wait_time}"
else:
    source_map['max_wait_time'] = 'env default or inferred'

# obs_dim_agent source
if getattr(args, 'obs_shape', None) is not None and env.obs_dim_agent == int(getattr(args, 'obs_shape')):
    source_map['obs_dim_agent'] = f"args.obs_shape:{args.obs_shape}"
else:
    source_map['obs_dim_agent'] = 'env default or inferred'

print("\nSource map:")
for k, v in source_map.items():
    print(f"  [OK] {k} = {getattr(env, k)} (source={v})")

# Build one job using the repo's generator if available, else use env.jobs[0]
job = None
if getattr(env, 'job_generator', None) is not None and hasattr(env.job_generator, 'create_job'):
    try:
        candidate = env.job_generator.create_job()
        # ensure the created object is JobAgent-like (provides current_operation)
        if hasattr(candidate, 'current_operation'):
            job = candidate
            print('\n[Smoke Test] Created job via job_generator.create_job()')
        else:
            # ignore non-JobAgent return types (e.g., a raw operations list)
            job = None
    except Exception:
        job = None

if job is None:
    try:
        if getattr(env, 'jobs', None):
            job = env.jobs[0]
            print('\n[Smoke Test] Using env.jobs[0] (initial job)')
    except Exception:
        job = None

if job is None:
    # Last resort: synthesize a JobAgent-like object using env defaults (keep minimal)
    class _J:
        def __init__(self, ops, cur=0, wt=0.0, fin=False):
            self.operations = ops
            self.current_op_idx = cur
            self.wait_time = wt
            self.finished = fin
        def current_operation(self):
            class Op:
                def __init__(self, t):
                    self.type = t
            # provide a type within range of n_operation_types
            return Op(t=0)
    job = _J(list(range(min(3, int(env.max_operations_per_job or 1)))), cur=0, wt=0.0, fin=False)
    print('\n[Smoke Test] Synthesized a minimal job object for observation build')

# Adapt job to JobAgent-like API if necessary
if not hasattr(job, 'current_operation'):
    # Create a small adapter that provides current_operation() and required fields
    class JobAdapter:
        def __init__(self, src):
            # src may be a JobAgent-like object or a plain operations list
            self._src = src
            # operations
            if hasattr(src, 'operations'):
                self.operations = src.operations
            elif isinstance(src, (list, tuple)):
                self.operations = list(src)
            else:
                # fallback: single op
                self.operations = [0]
            # current op idx
            self.current_op_idx = int(getattr(src, 'current_op_idx', getattr(src, 'current_op', 0) or 0))
            # wait_time
            self.wait_time = float(getattr(src, 'wait_time', 0.0))
            # finished flag
            self.finished = bool(getattr(src, 'finished', False))

        def current_operation(self):
            # prefer src.current_operation, then src.current_op, else synthesize
            if hasattr(self._src, 'current_operation'):
                return self._src.current_operation()
            if hasattr(self._src, 'current_op'):
                return self._src.current_op()
            # synthesize a minimal op object with a type=0
            class Op:
                def __init__(self, t=0):
                    self.type = t
            return Op(t=0)

    job = JobAdapter(job)

# Sanity check divisors are positive (will raise ValueError if not)
divisors = {
    'max_jobs': getattr(env, 'max_jobs', None),
    'max_operations_per_job': getattr(env, 'max_operations_per_job', None),
    'n_operation_types': getattr(env, 'n_operation_types', None),
    'max_wait_time': getattr(env, 'max_wait_time', None),
}
for name, val in divisors.items():
    if val is None or float(val) <= 0.0:
        raise ValueError(f"Smoke test precondition failed: divisor {name} is not > 0 (value={val})")

# Build observation
obs = build_agent_obs(env, job)

print('\nObservation vector:', np.round(np.array(obs), 6).tolist())
print('Length:', len(obs))
print('Min:', float(np.min(obs)), 'Max:', float(np.max(obs)))

# Assertions
assert len(obs) == env.obs_dim_agent == 7, f"Observation length mismatch: {len(obs)} vs {env.obs_dim_agent}"
arr = np.array(obs)
assert np.all(arr >= 0.0), "Observation values should be non-negative"

print('\n✅ Smoke test assertions passed: obs length and value ranges are valid.')

print('\nFinal source map summary:')
for k, v in source_map.items():
    print(f"  {k} ← {v}")
