import logging
import os
import sys
import datetime
# ensure project root is on sys.path when running from scripts/
sys.path.insert(0, os.getcwd())
from environment import MASAEnv


# --------------------
# Dual logging setup
# --------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"run_{timestamp}.log")
metrics_csv = os.path.join(log_dir, "learning_metrics.csv")

# Create metrics CSV if missing with header
if not os.path.exists(metrics_csv):
    try:
        with open(metrics_csv, "w") as mf:
            mf.write("episode,episode_reward,td_loss,epsilon,q_value_mean,avg_wait,avg_util\n")
    except Exception:
        pass


class Tee:
    """Write to both a stream (terminal) and a file-like object.

    This is used to capture plain print() output into the same logfile used
    by the logging FileHandler while still sending prints to the terminal.
    """
    def __init__(self, stream, fileobj):
        self.stream = stream
        self.file = fileobj

    def write(self, data):
        try:
            self.stream.write(data)
        except Exception:
            pass
        try:
            self.file.write(data)
            self.file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass


# configure logging: FileHandler to log_path, StreamHandler to original stdout
file_handler = logging.FileHandler(log_path, mode="w")
stream_handler = logging.StreamHandler(sys.__stdout__)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[file_handler, stream_handler],
)

# Replace stdout/stderr with a tee so plain prints from other modules are
# duplicated into the same logfile. Use a separate file handle so we don't
# accidentally create recursive logging via logging handlers.
try:
    log_file_for_tee = open(log_path, "a")
    sys.stdout = Tee(sys.__stdout__, log_file_for_tee)
    sys.stderr = Tee(sys.__stderr__, log_file_for_tee)
except Exception:
    # if tee setup fails, continue with logging only
    pass
import traceback

logging.basicConfig(level=logging.INFO)
print("Starting short dynamic simulation (strict_mode=True).\n")
try:
    env = MASAEnv(num_jobs=0, num_operators=2)
    print("env created: strict_mode=", env.strict_mode)
    print("config_path=", getattr(env, 'config_path', None))
    cfg = getattr(env, 'config', {}) or {}
    print("arrival_lambda (from config):", cfg.get('task_generator', {}).get('arrival_lambda'))

    arrival_times = []
    orig_add = env.add_job
    def wrapped_add(ops_sequence):
        arrival_times.append(env.env.now)
        print(f"[DynamicArrival] t={env.env.now:.4f} - adding job id={len(env.jobs)} ops={len(ops_sequence)}")
        return orig_add(ops_sequence)
    env.add_job = wrapped_add

    run_until = 200
    print(f"Running simulation until t={run_until} ...")

    # Simple auto-runner: resume each decision by picking the first allowed WC.
    # This simulates a naive policy to exercise the matching and resource usage.
    try:
        while env.env.now < run_until and not getattr(env, 'done', False):
            batch, sim_t = env.wait_for_decisions()
            # if no decisions and env advanced, loop will call wait_for_decisions again
            for item in batch:
                allowed = item.get('allowed_wcs', []) or []
                # Deterministic policy: pick the WorkCenter with the shortest
                # per-WC duration for this operation when available.
                choice = None
                try:
                    jid = int(item.get('job_id'))
                    job = env.jobs[jid]
                    op = job.current_op()
                    per_wc = None
                    if isinstance(op, (list, tuple)) and len(op) >= 3 and isinstance(op[2], dict):
                        # normalize keys to int
                        per_wc = {int(k): float(v) for k, v in op[2].items()}

                    if per_wc:
                        # choose allowed WC with minimum per_wc duration
                        candidates = []
                        for wc in allowed:
                            dur = per_wc.get(int(wc), float('inf'))
                            candidates.append((float(dur), int(wc)))
                        if candidates:
                            choice = min(candidates)[1]
                    else:
                        # fallback: estimate using base_duration and speed factor
                        base = float(item.get('base_duration', 0.0))
                        best_wc = None
                        best_est = float('inf')
                        for wc in allowed:
                            try:
                                speed = env._speed_factor_for_wc(int(wc))
                                est = base / max(1e-6, float(speed)) if base > 0 else 0.0
                            except Exception:
                                est = base
                            if est < best_est:
                                best_est = est
                                best_wc = int(wc)
                        choice = best_wc if best_wc is not None else (allowed[0] if allowed else 0)
                except Exception:
                    choice = allowed[0] if allowed else 0

                # call resume callback with chosen wc
                try:
                    item.get('resume', lambda x: None)(int(choice))
                except Exception:
                    pass
        # ensure we advance sim to run_until to finish outstanding ops
        env.env.run(until=run_until)
    except Exception:
        # allow observation of partial progress if something goes wrong
        raise

    print("Run finished. sim.now=", env.env.now)

    print("\nArrival times (count=", len(arrival_times), "):")
    for i,t in enumerate(arrival_times):
        print(f"  #{i}: t={t:.4f}")
    if len(arrival_times) >= 2:
        diffs = [arrival_times[i+1]-arrival_times[i] for i in range(len(arrival_times)-1)]
        print("Inter-arrival intervals:", [round(d,3) for d in diffs])

    print("\nMachine registry:")
    for m,md in sorted(env.workcenters_meta.machine_registry.items()):
        print(f"  {m}: wc={md.get('workcenter')} caps={md.get('capabilities')} speed={md.get('speed_factor')}")

    print("\nEligible operator groups by WC:")
    for wc,ops in sorted(env.workcenters_meta.eligible_operator_groups_by_wc.items()):
        print(f"  WC{wc}: eligible_ops={ops}")

    print("\nGantt records (sample up to 50):")
    for rec in env.gantt_records[:50]:
        s,e,op,wc,jid,grp = rec
        print(f"  job={jid} op={op} wc={wc} grp={grp} start={s:.2f} end={e:.2f} dur={e-s:.2f}")

    used_wcs = sorted(set([int(r[3]) for r in env.gantt_records]))
    print("\nWorkCenters used in run:", used_wcs)

    ops_by_wc = {}
    for s,e,op,wc,jid,grp in env.gantt_records:
        ops_by_wc.setdefault(wc, set()).add(grp)
    print("\nOperator groups used per WC:")
    for wc,grps in sorted(ops_by_wc.items()):
        print(f" WC{wc}: groups={sorted(list(grps))}")

    print("\nFinal job list (human readable):")
    env.print_jobs_human_readable()

except Exception:
    print("Simulation raised an exception:")
    traceback.print_exc()
