#!/usr/bin/env python3
"""Run a short controlled episode to validate the 6D observation pipeline.
Saves numeric logs and plots to reports/observation_validation/.
"""
import os
import sys
import math
import json
from types import SimpleNamespace
import numpy as np

OUTDIR = os.path.join(os.path.dirname(__file__), "")
os.makedirs(OUTDIR, exist_ok=True)
# Ensure project root is on sys.path so imports like `environment` and `utils` work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from environment import MASAEnv
    from utils.env_obs import build_agent_obs
except Exception as e:
    print("Failed to import MASAEnv or env_obs:", e)
    raise

# Minimal args namespace to satisfy MASAEnv requirements
args = SimpleNamespace(
    n_agents=4,
    n_operation_types=5,
    job_max_ops=5,
    max_wait_time=50.0,
)

# Create env
env = MASAEnv(args=args, auto_build=False, auto_start_arrivals=False)
# Ensure at least one machine in workcenters_meta for action space
try:
    if not getattr(env.workcenters_meta, 'machine_list', None):
        env.workcenters_meta.machine_list = ['M0']
        env.workcenters_meta.machine_index = {'M0': 0}
        env.num_wcs = 1
        env.n_actions = 1
except Exception:
    pass

# Create a few deterministic jobs with simple ops
ops_template = [
    (0, [0], {0: 1.0}),
    (1, [0], {0: 1.5}),
]

NUM_JOBS = 3
for i in range(NUM_JOBS):
    env.add_job(list(ops_template), start_immediately=True, set_arrival_zero=True)

# Run one episode (step-based) and collect data
obs_samples = []  # list of arrays (num_agents x 6) per step
reward_trace = []
time_trace = []
job_completion_ratio = []

max_steps = 500
step = 0
print("Starting validation episode...")
while step < max_steps:
    obs, reward, done, info = env.step()
    # obs is a list of per-agent numpy arrays
    if obs is None:
        break
    # convert to stacked array (agents x dim)
    try:
        stacked = np.vstack([o.reshape(1, -1) if hasattr(o, 'reshape') else np.asarray(o).reshape(1, -1) for o in obs])
    except Exception:
        # fallback: try to coerce
        stacked = np.asarray(obs)
    obs_samples.append(stacked)
    reward_trace.append(float(reward))
    time_trace.append(float(env.t))
    try:
        completed = len([j for j in (getattr(env, 'jobs', []) or []) if getattr(j, 'finished', False)])
        total = len(getattr(env, 'jobs', []) or [])
        ratio = float(completed) / max(1.0, float(total))
    except Exception:
        ratio = 0.0
    job_completion_ratio.append(ratio)
    step += 1
    if done:
        print(f"Episode done at step={step}, t={env.t}")
        break

# Convert observations into array: steps x agents x dim (pad if variable agents)
if len(obs_samples) == 0:
    print("No observations captured; aborting")
    sys.exit(2)

# Determine max agents and dim
max_agents = max(a.shape[0] for a in obs_samples)
dim = obs_samples[0].shape[1]

# Pad samples to rectangular shape
arr = np.zeros((len(obs_samples), max_agents, dim), dtype=np.float32)
for i, s in enumerate(obs_samples):
    arr[i, : s.shape[0], : s.shape[1]] = s

# Stats per dimension across all agent observations
all_obs = arr.reshape(-1, dim)
obs_min = np.min(all_obs, axis=0)
obs_max = np.max(all_obs, axis=0)
obs_mean = np.mean(all_obs, axis=0)

print("Observation shape per step: agents x dim =>", max_agents, "x", dim)
for d in range(dim):
    print(f"Dim {d}: min={obs_min[d]:.4f}, max={obs_max[d]:.4f}, mean={obs_mean[d]:.4f}")

# Save numeric logs
np.save(os.path.join(OUTDIR, 'obs_samples.npy'), arr)
np.save(os.path.join(OUTDIR, 'reward_trace.npy'), np.asarray(reward_trace))
np.save(os.path.join(OUTDIR, 'time_trace.npy'), np.asarray(time_trace))
np.save(os.path.join(OUTDIR, 'job_completion_ratio.npy'), np.asarray(job_completion_ratio))

# Try to plot if matplotlib available
plots_saved = []
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Per-dimension min/max plots over steps (agents collapsed)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # plot mean per-dim over time (mean across agents)
    mean_per_step = np.mean(arr, axis=1)  # steps x dim
    for d in range(dim):
        ax.plot(mean_per_step[:, d], label=f'dim{d}')
    ax.set_title('Per-dimension mean observation over steps')
    ax.set_xlabel('step')
    ax.set_ylabel('value')
    ax.legend()
    p1 = os.path.join(OUTDIR, 'obs_dim_mean_over_steps.png')
    fig.tight_layout()
    fig.savefig(p1)
    plots_saved.append(p1)

    # Reward curve
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    ax2.plot(reward_trace, marker='o')
    ax2.set_title('Reward trace per step')
    ax2.set_xlabel('step')
    ax2.set_ylabel('reward')
    p2 = os.path.join(OUTDIR, 'reward_trace.png')
    fig2.tight_layout()
    fig2.savefig(p2)
    plots_saved.append(p2)

    # Job completion ratio
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 4))
    ax3.plot(job_completion_ratio, marker='o')
    ax3.set_title('Job completion ratio over steps')
    ax3.set_xlabel('step')
    ax3.set_ylabel('finished fraction')
    p3 = os.path.join(OUTDIR, 'job_completion_ratio.png')
    fig3.tight_layout()
    fig3.savefig(p3)
    plots_saved.append(p3)

except Exception as e:
    print("Matplotlib not available or plot failed:", e)

# Simple reward trend check (linear slope)
try:
    y = np.asarray(reward_trace)
    x = np.arange(len(y))
    if len(y) >= 2:
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        slope = 0.0
except Exception:
    slope = 0.0

stability = 'stable' if slope >= 0.0 else 'unstable'

summary = {
    'steps': len(reward_trace),
    'episode_time': float(env.t),
    'obs_shape_per_step': [int(max_agents), int(dim)],
    'obs_min': obs_min.tolist(),
    'obs_max': obs_max.tolist(),
    'obs_mean': obs_mean.tolist(),
    'reward_slope': float(slope),
    'stability': stability,
    'plots_saved': plots_saved,
}

with open(os.path.join(OUTDIR, 'summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=2)

print(json.dumps(summary, indent=2))
print('Saved numeric logs to', OUTDIR)
if plots_saved:
    print('Saved plots:')
    for p in plots_saved:
        print('  -', p)

# Check for legacy fields being accessed during run: quick runtime scan of env and job attributes
legacy_accesses = []
for j in env.jobs:
    if hasattr(j, 'remaining_time'):
        legacy_accesses.append(('job', j.id, 'remaining_time'))
    if hasattr(j, 'progress_ratio'):
        # don't call it, just note presence
        legacy_accesses.append(('job', j.id, 'progress_ratio'))
# env-level
if hasattr(env, 'completed_jobs'):
    legacy_accesses.append(('env', None, 'completed_jobs'))

with open(os.path.join(OUTDIR, 'legacy_accesses.json'), 'w') as fh:
    json.dump(legacy_accesses, fh, indent=2)

print('Legacy runtime attributes present (not necessarily used by obs):', legacy_accesses)

# Exit 0
print('Validation complete')
