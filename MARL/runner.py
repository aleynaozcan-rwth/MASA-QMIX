# ...existing code...
import os
# Ensure headless Qt / matplotlib backend before any pyplot import (avoid Wayland/Qt plugin errors)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")

import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Rollout
from MARL.common.rollout import RolloutWorker
try:
    from MARL.common.rollout import CommRolloutWorker  # type: ignore
except Exception:
    CommRolloutWorker = RolloutWorker

# Agents / Buffer
from MARL.agent.agent import Agents, CommAgents
from MARL.common.replay_buffer import ReplayBuffer
from MARL.common.terms import t  # unified terminology helper


def plot_gantt(for_gantt_data, filename="gantt.png"):
    """Step 8A.6.6 Gantt – Machine-level layout with WC + Operator labels."""
    if not for_gantt_data or not isinstance(for_gantt_data, (list, tuple)):
        return False
    if len(for_gantt_data) == 0 or len(for_gantt_data[0]) not in [5, 6]:
        return False

    fig, ax = plt.subplots(figsize=(13, 6))
    operation_types = sorted(set([rec[2] for rec in for_gantt_data]))
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_map = {op: colors[i % len(colors)] for i, op in enumerate(operation_types)}

    max_end = 0
    for rec in for_gantt_data:
        if len(rec) == 6:
            start, end, operation, workcenter, jobagent, operator = rec
        else:
            start, end, operation, workcenter, jobagent = rec
            operator = "?"
        ax.barh(jobagent, end - start, left=start,
                color=color_map.get(operation, "gray"), edgecolor="black")
        ax.text((start + end) / 2, jobagent,
                f"M{operation} | WC{workcenter} | O{operator}",
                va="center", ha="center", fontsize=7, color="black")
        max_end = max(max_end, end)

    jobagent_ids = sorted(set([rec[4] for rec in for_gantt_data]))
    ax.set_yticks(jobagent_ids)
    ax.set_yticklabels([str(j) for j in jobagent_ids])
    ax.set_xlabel("Simulation Time (SimPy clock)")
    ax.set_ylabel(f"{t('JobAgent')} (ID)")
    ax.set_title("Step 8A.6.6 – Machine-level Schedule (WC + Operator + Dynamic Arrivals)")
    ax.set_xlim(0, max_end + 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[op]) for op in operation_types]
    labels = [f"Operation {op}" for op in operation_types]
    ax.legend(handles, labels, title="Operation Types", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return True


class Runner:
    """
    Step 8A.6.6 Runner – Replay-based QMIX + KPI Logging + Learning Stability
    -------------------------------------------------------------------------
    - 11D observations (progress_ratio)
    - Extended KPI metrics (avg wait, utilization, makespan)
    - Moving-average loss/TD tracking
    """

    def __init__(self, env, args):
        self.env = env
        self.args = args

        # Log environment config provenance and sizes for visibility (helps
        # diagnose legacy vs current WorkCenter topology issues).
        try:
            cfg_path = getattr(self.env, 'config_path', None)
            num_wcs = getattr(self.env, 'num_wcs', None)
            # fallback: infer from workcenters_meta if not set
            if num_wcs is None and hasattr(self.env, 'workcenters_meta'):
                try:
                    num_wcs = len(getattr(self.env.workcenters_meta, 'workcenters_list', []))
                except Exception:
                    num_wcs = None
            num_ops = getattr(self.env, 'num_ops', None)
            logging.getLogger(__name__).info("[Runner] Env config_path=%s num_wcs=%s num_ops=%s", cfg_path, num_wcs, num_ops)
        except Exception:
            pass

        # Ensure n_actions exists for Agents constructors. If granular actions
        # are requested we'll overwrite n_actions below; otherwise default to
        # number of WorkCenters if available, or 1 as a safe fallback.
        try:
            if not hasattr(self.args, 'n_actions'):
                if bool(getattr(self.args, 'use_machine_actions', False)):
                    # use global machine list as action space
                    try:
                        mlist = getattr(self.env.workcenters_meta, 'machine_list', None)
                        if mlist is not None and len(mlist) > 0:
                            self.args.n_actions = int(len(mlist))
                        elif hasattr(self.env, 'num_wcs'):
                            self.args.n_actions = int(getattr(self.env, 'num_wcs'))
                        else:
                            self.args.n_actions = 1
                    except Exception:
                        self.args.n_actions = 1
                else:
                    if hasattr(self.env, 'num_wcs'):
                        self.args.n_actions = int(getattr(self.env, 'num_wcs'))
                    else:
                        self.args.n_actions = 1
        except Exception:
            try:
                self.args.n_actions = 1
            except Exception:
                pass

        # propagate quiet flag to environment to suppress verbose SimPy debug prints
        try:
            setattr(self.env, "quiet_env", bool(getattr(self.args, "quiet_env", False)))
        except Exception:
            pass

        # If the user requested operator-granular actions, compute and set the
        # effective number of actions (num_wcs * num_ops) early so policy
        # constructors (which read args.n_actions) see the correct value.
        try:
            if bool(getattr(self.args, 'use_granular_actions', False)):
                if hasattr(self.env, 'num_wcs') and hasattr(self.env, 'num_ops'):
                    self.args.n_actions = int(self.env.num_wcs) * int(self.env.num_ops)
                    print(f"[Runner] Using granular actions: n_actions={self.args.n_actions}")
        except Exception:
            pass

        # Auto-set number of agents for policies/agents if environment exposes it.
        # Prefer env.get_env_info() as the authoritative source, then fall back
        # to len(env.jobs), and finally to env.num_jobs. This reduces surprises
        # when environments expose a richer info API or generate jobs dynamically.
        try:
            # Treat the CLI default (10) as 'unset' for convenience so Runner
            # can adopt the environment's actual agent count. If the user has
            # explicitly provided a different value, we keep it.
            current_n_agents = getattr(self.args, 'n_agents', None)
            if current_n_agents is None or int(current_n_agents) == 10:
                info = None
                try:
                    if hasattr(self.env, 'get_env_info') and callable(getattr(self.env, 'get_env_info')):
                        info = self.env.get_env_info()
                except Exception:
                    info = None

                if info and isinstance(info, dict) and ('n_agents' in info or 'n_agents' in info.keys()):
                    try:
                        self.args.n_agents = int(info.get('n_agents'))
                        print(f"[Runner] Auto-set args.n_agents = {self.args.n_agents} from env.get_env_info()['n_agents']")
                    except Exception:
                        pass
                elif hasattr(self.env, 'jobs'):
                    try:
                        self.args.n_agents = int(len(getattr(self.env, 'jobs')))
                        print(f"[Runner] Auto-set args.n_agents = {self.args.n_agents} from len(env.jobs)")
                    except Exception:
                        pass
                elif hasattr(self.env, 'num_jobs'):
                    try:
                        self.args.n_agents = int(getattr(self.env, 'num_jobs'))
                        print(f"[Runner] Auto-set args.n_agents = {self.args.n_agents} from env.num_jobs")
                    except Exception:
                        pass
        except Exception:
            # keep silent on failures — this is a best-effort convenience
            pass

        # Agents & rollout setup
        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:
            self.agents = CommAgents(args)
            self.buffer = ReplayBuffer(episode_capacity=args.buffer_size, seed=args.seed) \
                if getattr(args, "learn", True) else None
            self.rolloutWorker = CommRolloutWorker(env=env, agents=self.agents, buffer=self.buffer, args=args)
        else:
            self.agents = Agents(args)
            self.buffer = ReplayBuffer(episode_capacity=args.buffer_size, seed=args.seed) \
                if getattr(args, "learn", True) else None
            self.rolloutWorker = RolloutWorker(env=env, agents=self.agents, buffer=self.buffer, args=args)

        # If the user requested operator-granular actions, compute and set the
        # effective number of actions (num_wcs * num_ops) so the replay buffer
        # and policies can operate on the flattened action space.
        try:
            if bool(getattr(self.args, 'use_granular_actions', False)):
                if hasattr(self.env, 'num_wcs') and hasattr(self.env, 'num_ops'):
                    self.args.n_actions = int(self.env.num_wcs) * int(self.env.num_ops)
                    print(f"[Runner] Using granular actions: n_actions={self.args.n_actions}")
        except Exception:
            pass

        self.win_rates = []
        self.episode_rewards = []
        self.episode_durations = []
        self.wait_time_records = []

        self.save_path = os.path.join(self.args.result_dir, args.alg, args.map)
        os.makedirs(self.save_path, exist_ok=True)
        self.history_dir = "./my_data_and_graph/historydata/"
        os.makedirs(self.history_dir, exist_ok=True)

        # snapshot_on_eval: produce gantt snapshots only on evaluation by default
        # if the user explicitly sets --gantt_snapshot_every, training-step snapshots
        # are enabled only when snapshot_on_eval is False.
        self.snapshot_on_eval = bool(getattr(self.args, 'snapshot_on_eval', True))

        # Optionally clean previous history artifacts to avoid mixing runs
        try:
            if bool(getattr(self.args, 'clean_history', False)):
                print(f"[Runner] Cleaning history folder: {self.history_dir}")
                # remove common artifact types
                exts = ['*.png', '*.csv', '*.txt', '*.json', '*.npy', '*.pkl']
                import glob
                for pat in exts:
                    for f in glob.glob(os.path.join(self.history_dir, pat)):
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                print("[Runner] History cleaned.")
        except Exception:
            pass

        print(f"[Runner 8A.6.6] Initialized | alg={self.args.alg} | buffer={getattr(self.args,'buffer_size','-')} | batch={getattr(self.args,'batch_size','-')}")
        # Write initial job -> operations mapping for easy inspection
        try:
            init_path = os.path.join(self.history_dir, "initial_jobs.txt")
            with open(init_path, 'w') as hf:
                hf.write("Initial Job -> Operation mapping\n")
                hf.write("Format: JobID | OpIdx | OpType | Allowed_WCs | OpGroups | BaseDur\n\n")
                for job in getattr(self.env, 'jobs', []):
                    hf.write(f"Job {int(job.id)}:\n")
                    for idx, op in enumerate(getattr(job, 'operations', [])):
                        try:
                            # canonical formats:
                            # legacy: (allowed_wcs, dur)
                            # old: (op_type, allowed_wcs, base_dur)
                            # new canonical: (op_type, allowed_wcs, per_wc_durations_dict)
                            if isinstance(op, (list, tuple)) and len(op) == 2:
                                allowed_wcs, dur = op
                                op_type = 'legacy'
                                hf.write(f"  Op {idx} | Type {op_type} | WCs {allowed_wcs} | Dur {float(dur):.3f}\n")
                            else:
                                op_type = op[0]
                                allowed_wcs = op[1]
                                third = op[2]
                                # if third is dict, print per-WC durations and both coarse/eligible operator info
                                if isinstance(third, dict):
                                    per_wc = third
                                    groups_by_wc = []
                                    for wc in allowed_wcs:
                                        try:
                                            eligible = self.env.workcenters_meta.eligible_operator_groups_by_wc.get(int(wc), [])
                                        except Exception:
                                            eligible = []
                                        groups_by_wc.append({'wc': int(wc), 'eligible_ops': eligible})
                                    hf.write(f"  Op {idx} | Type {op_type} | WCs {allowed_wcs} | Groups {groups_by_wc} | base_per_wc_durations:\n")
                                    for wc in allowed_wcs:
                                        try:
                                            dur_wc = float(per_wc.get(int(wc), 0.0))
                                        except Exception:
                                            dur_wc = 0.0
                                        try:
                                            eligible = self.env.workcenters_meta.eligible_operator_groups_by_wc.get(int(wc), [])
                                        except Exception:
                                            eligible = []
                                        hf.write(f"    WC{wc} -> dur={dur_wc:.3f} | eligible_ops={eligible}\n")
                                else:
                                    # legacy-ish third numeric
                                    base_dur = float(third)
                                    groups_info = []
                                    for wc in allowed_wcs:
                                        try:
                                            eligible = self.env.workcenters_meta.eligible_operator_groups_by_wc.get(int(wc), [])
                                        except Exception:
                                            eligible = []
                                        groups_info.append({'wc': int(wc), 'eligible_ops': eligible})
                                    hf.write(f"  Op {idx} | Type {op_type} | WCs {allowed_wcs} | Groups {groups_info} | base_dur {base_dur:.3f}\n")
                        except Exception:
                            hf.write(f"  Op {idx} | malformed: {op}\n")
                    hf.write("\n")
            if getattr(self.args, 'enable_logs', True):
                try:
                    with open(init_path, 'r') as hf_read:
                        # log initial mapping at INFO level
                        logging.getLogger(__name__).info(hf_read.read())
                except Exception:
                    pass
            # Append human-readable job list produced by env helper if available
            try:
                if hasattr(self.env, 'print_jobs_human_readable') and callable(getattr(self.env, 'print_jobs_human_readable')):
                    with open(init_path, 'a') as hf:
                        hf.write('\nHuman-readable job list:\n')
                        # capture print output by redirecting stdout temporarily
                        try:
                            import io, sys as _sys
                            buf = io.StringIO()
                            old = _sys.stdout
                            _sys.stdout = buf
                            try:
                                self.env.print_jobs_human_readable()
                            finally:
                                _sys.stdout = old
                            hf.write(buf.getvalue())
                        except Exception:
                            # fallback: call without capture
                            try:
                                self.env.print_jobs_human_readable()
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] Could not write initial job mapping: {e}")

    def _append_learning_metrics(self, ep_idx: int, epoch: int, ep_r: float):
        """Append one line to learning_metrics.csv in history_dir.

        This centralizes metric writes so there's a single writer location.
        The write is idempotent for consecutive duplicate attempts (it will
        not append the exact same line twice).
        """
        try:
            os.makedirs(self.history_dir, exist_ok=True)
            metrics_path = os.path.join(self.history_dir, 'learning_metrics.csv')

            # read last loss / td from loss files if present
            last_loss = ''
            last_td = ''
            try:
                lpath = os.path.join(self.history_dir, 'loss.txt')
                if os.path.exists(lpath):
                    with open(lpath, 'r') as lf:
                        lines = [ln.strip() for ln in lf.readlines() if ln.strip()]
                        if lines:
                            last_loss = lines[-1]
            except Exception:
                last_loss = ''
            try:
                tpath = os.path.join(self.history_dir, 'td_error.txt')
                if os.path.exists(tpath):
                    with open(tpath, 'r') as tf:
                        lines = [ln.strip() for ln in tf.readlines() if ln.strip()]
                        if lines:
                            last_td = lines[-1]
            except Exception:
                last_td = ''

            # Build the line we'd like to append.
            new_line = f"{ep_idx},{epoch},{float(ep_r):.4f},{last_loss},{last_td}"

            header_needed = not os.path.exists(metrics_path)

            # Read last non-empty line to avoid duplicate consecutive writes
            last_line = None
            try:
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as mf_read:
                        prev = [ln.strip() for ln in mf_read.readlines() if ln.strip()]
                        if prev:
                            last_line = prev[-1]
            except Exception:
                last_line = None

            if last_line != new_line:
                try:
                    with open(metrics_path, 'a') as mf:
                        if header_needed:
                            mf.write('episode,epoch,episode_reward,last_loss,last_td\n')
                        mf.write(new_line + '\n')
                except Exception:
                    pass
        except Exception:
            pass

    def run(self, num):
        train_steps = 0
        all_gantt_data = []
        avg_rewards = [0]
        global_ep_idx = 0

        print("[Runner] === Training loop started ===")

        for epoch in range(self.args.n_epoch):
            sys.stdout.write(f"\rRun {num}, epoch {epoch}, avg rewards {np.mean(avg_rewards):.2f}")
            sys.stdout.flush()

            # === Periodic evaluation ===
            if epoch % self.args.evaluate_cycle == 0 and epoch != 0:
                win_rate, ep_reward, global_ep_idx, gantt_eval = self.evaluate(all_gantt_data, global_ep_idx)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(ep_reward)
                print(f"\n[Eval] Epoch {epoch} | Reward={ep_reward:.2f}")
                try:
                    # create a combined gantt for this evaluation (env records + collected gantt)
                    combined = list(all_gantt_data)
                    try:
                        if hasattr(self.env, 'gantt_records'):
                            combined.extend(list(self.env.gantt_records))
                    except Exception:
                        pass
                    png_path = os.path.join(self.history_dir, f"gantt_epoch{epoch}.png")
                    plot_gantt(combined, filename=png_path)
                    # optionally also write a CSV for detailed inspection
                    if getattr(self.args, 'gantt_csv', False):
                        csv_path = os.path.join(self.history_dir, f"gantt_epoch{epoch}.csv")
                        with open(csv_path, 'w') as cf:
                            cf.write('start,end,op_idx,wc,job_id,operator_grp\n')
                            for r in combined:
                                try:
                                    cf.write(','.join([str(x) for x in r]) + '\n')
                                except Exception:
                                    pass
                except Exception as e:
                    print("[WARN] Gantt plot failed:", e)

            episodes, avg_rewards = [], []

            # detect 9A event-driven SimPy API on the environment
            use_event_driven = hasattr(self.env, "wait_for_decisions") and callable(getattr(self.env, "wait_for_decisions"))

            for _ in range(self.args.n_episodes):
                if use_event_driven:
                    episode, ep_r, win_tag, gantt_data = self._run_event_driven_episode(global_ep_idx)
                else:
                    episode, _, _, gantt_data = self.rolloutWorker.generate_episode(global_ep_idx)
                    ep_r = float(np.sum(episode.get('r', 0)))
                    win_tag = False

                all_gantt_data.extend(gantt_data)
                avg_rewards.append(ep_r)
                episodes.append(episode)
                global_ep_idx += 1

                # Append per-episode metrics via single writer method
                try:
                    self._append_learning_metrics(global_ep_idx-1, epoch, ep_r)
                except Exception:
                    pass

                # Episode durations
                if hasattr(self.rolloutWorker, "episode_duration"):
                    self.episode_durations.append(self.rolloutWorker.episode_duration)
                else:
                    self.episode_durations.append(len(episode.get('r', [])))

                # Wait times
                if hasattr(self.env, "wait_time_dict"):
                    self.wait_time_records.append(dict(self.env.wait_time_dict))
                else:
                    # fallback: try env.total_wait_time / completed_jobs
                    try:
                        self.wait_time_records.append({0: (self.env.total_wait_time / max(1, max(1, getattr(self.env, "completed_jobs", 1))))})
                    except Exception:
                        self.wait_time_records.append({0: ep_r / 20.0})

            # === Training updates (Replay-based) ===
            if self.args.alg not in ['coma', 'central_v', 'reinforce'] and self.buffer is not None:
                # configurable warm-up threshold (use args.min_warmup_size if present,
                # fall back to args.batch_size or 32)
                min_warm = getattr(self.args, "min_warmup_size",
                                   getattr(self.args, "min_warmup",
                                           getattr(self.args, "batch_size", 32)))
                if len(self.buffer) < min_warm:
                    print(f"\n[DEBUG] Buffer warm-up ({len(self.buffer)}/{min_warm}) – skipping training")
                else:
                    print(f"\n[DEBUG] Training active (buffer={len(self.buffer)}) – gradient updates...")
                    for _ in range(self.args.train_steps):
                        mini_batch = self.buffer.sample(self.args.batch_size, n_actions=self.args.n_actions)
                        if mini_batch is None:
                            break
                        result = self.agents.train(mini_batch, train_steps)
                        train_steps += 1
                        if isinstance(result, dict):
                            loss, td = result.get("loss"), result.get("td_error")
                            if (train_steps % 20 == 0) and (loss is not None):
                                msg = f"[TRAIN] step={train_steps}, loss={loss:.4f}"
                                if td is not None:
                                    msg += f", td={td:.4f}"
                                msg += f", buffer={len(self.buffer)}"
                                print(msg)
                        # optional gantt snapshot on training progress
                        try:
                            gs = getattr(self.args, 'gantt_snapshot_every', 0)
                            # Only perform step-based snapshots if the user explicitly
                            # requested them (gs>0) AND snapshot_on_eval is False
                            # (snapshot_on_eval==True means produce snapshots only during evaluation).
                            if gs and (not self.snapshot_on_eval) and train_steps > 0 and (train_steps % gs == 0):
                                # snapshot using recent gantt (all_gantt_data) and env records
                                try:
                                    snap_name = f"gantt_snapshot_step{train_steps}.png"
                                    # combine latest gantt: prefer env.gantt_records if present
                                    snapshot_gantt = list(all_gantt_data)
                                    if hasattr(self.env, 'gantt_records'):
                                        snapshot_gantt.extend(list(self.env.gantt_records))
                                    plot_gantt(snapshot_gantt, filename=os.path.join(self.history_dir, snap_name))
                                    if getattr(self.args, 'gantt_csv', False):
                                        csv_path = os.path.join(self.history_dir, f"gantt_snapshot_step{train_steps}.csv")
                                        with open(csv_path, 'w') as cf:
                                            cf.write('start,end,op_idx,wc,job_id,operator_grp\n')
                                            for r in snapshot_gantt:
                                                try:
                                                    cf.write(','.join([str(x) for x in r]) + '\n')
                                                except Exception:
                                                    pass
                                        # also write a human-readable summary for easier inspection
                                        def _unpack_rec(rec):
                                            # robustly unpack records of length 6 or 5
                                            if not isinstance(rec, (list, tuple)):
                                                raise ValueError('invalid rec')
                                            if len(rec) >= 6:
                                                return rec[0], rec[1], rec[2], rec[3], rec[4], rec[5]
                                            if len(rec) == 5:
                                                return rec[0], rec[1], rec[2], rec[3], rec[4], None
                                            raise ValueError('unsupported rec len')

                                        try:
                                            txt_path = os.path.join(self.history_dir, f"gantt_snapshot_step{train_steps}_readable.txt")
                                            with open(txt_path, 'w') as tf:
                                                # group by job_id
                                                jobs_map = {}
                                                for rec in snapshot_gantt:
                                                    try:
                                                        s, e, op_idx, wc, job_id, op_grp = _unpack_rec(rec)
                                                        jobs_map.setdefault(int(job_id), []).append((s, e, int(op_idx), int(wc), op_grp))
                                                    except Exception:
                                                        continue
                                                for jid in sorted(jobs_map.keys()):
                                                    tf.write(f"Job {jid}:\n")
                                                    for (s, e, op_idx, wc, op_grp) in sorted(jobs_map[jid], key=lambda x: x[0]):
                                                        tf.write(f"  Op {op_idx} @ WC{wc} (OpGrp {op_grp}) — start={s:.3f}, end={e:.3f}\n")
                                                    tf.write('\n')
                                        except Exception:
                                            pass

                                        # timeline-style snapshot: list active ops per unique time boundary
                                        try:
                                            timeline_path = os.path.join(self.history_dir, f"gantt_snapshot_step{train_steps}_timeline.txt")
                                            # collect unique event times
                                            times = set()
                                            for rec in snapshot_gantt:
                                                try:
                                                    s, e, *_ = _unpack_rec(rec)
                                                    times.add(float(s)); times.add(float(e))
                                                except Exception:
                                                    continue
                                            times_list = sorted(times)
                                            with open(timeline_path, 'w') as tf2:
                                                tf2.write('Timeline snapshot (active operations at event times)\n')
                                                for tval in times_list:
                                                    tf2.write(f'-- time {tval:.3f} --\n')
                                                    for rec in snapshot_gantt:
                                                        try:
                                                            s, e, op_idx, wc, job_id, op_grp = _unpack_rec(rec)
                                                            if float(s) <= float(tval) < float(e):
                                                                tf2.write(f"  Job {int(job_id)} | OpType {int(op_idx)} @ WC{int(wc)} (Grp {op_grp}) | start={s:.3f} end={e:.3f}\n")
                                                        except Exception:
                                                            continue
                                                tf2.write('\n')
                                        except Exception:
                                            pass

                                        # metrics for snapshot (loss/td/avg_wait/avg_reward)
                                        try:
                                            metrics_path = os.path.join(self.history_dir, f"gantt_snapshot_step{train_steps}_metrics.json")
                                            import json
                                            metrics = {
                                                'step': train_steps,
                                                'buffer_len': len(self.buffer) if getattr(self, 'buffer', None) is not None else None,
                                                'avg_wait': None,
                                                'avg_reward': None,
                                                'loss': None,
                                                'td': None,
                                            }
                                            try:
                                                metrics['avg_wait'] = float(self.env.total_wait_time / max(1, self.env.completed_jobs))
                                            except Exception:
                                                metrics['avg_wait'] = None
                                            try:
                                                metrics['avg_reward'] = float(np.mean(self.episode_rewards)) if self.episode_rewards else None
                                            except Exception:
                                                metrics['avg_reward'] = None
                                            # read last loss/td from files if exist
                                            try:
                                                lpath = os.path.join(self.history_dir, 'loss.txt')
                                                tpath = os.path.join(self.history_dir, 'td_error.txt')
                                                if os.path.exists(lpath):
                                                    with open(lpath, 'r') as lf:
                                                        last = lf.read().strip().split('\n')[-1]
                                                        metrics['loss'] = float(last) if last else None
                                                if os.path.exists(tpath):
                                                    with open(tpath, 'r') as tfm:
                                                        lastt = tfm.read().strip().split('\n')[-1]
                                                        metrics['td'] = float(lastt) if lastt else None
                                            except Exception:
                                                pass
                                            with open(metrics_path, 'w') as mf:
                                                json.dump(metrics, mf, indent=2)
                                        except Exception:
                                            pass
                                    print(f"[Runner] Gantt snapshot saved @ step {train_steps}")
                                except Exception as e:
                                    print(f"[WARN] Could not save gantt snapshot: {e}")
                        except Exception:
                            pass

            # === KPI LOGGING (Step 8A.6.6) ===
            try:
                avg_wait = self.env.total_wait_time / max(1, self.env.completed_jobs)
                util_m = self.env._util_machines()
                util_o = self.env._util_ops()
                makespan = getattr(self.env, "t", getattr(self.env, "env", None) and getattr(self.env, "env").now or 0.0)
                makespan = getattr(self.env, "t", makespan)
                with open(os.path.join(self.history_dir, "kpi_log.txt"), "a") as f:
                    f.write(f"{epoch},{avg_wait:.4f},{util_m:.4f},{util_o:.4f},{makespan:.2f}\n")
            except Exception as e:
                print(f"[WARN] KPI logging failed: {e}")

        # === Save episode statistics ===
        try:
            with open(os.path.join(self.history_dir, "episode_rewards.txt"), "w") as f_r:
                for ep_idx, ep_r in enumerate(self.episode_rewards):
                    f_r.write(f"{ep_idx},{ep_r:.2f}\n")

            with open(os.path.join(self.history_dir, "times.txt"), "w") as f_t:
                for ep_idx, dur in enumerate(self.episode_durations):
                    f_t.write(f"{ep_idx},{dur:.2f}\n")

            with open(os.path.join(self.history_dir, "waittimes.txt"), "w") as f_w:
                for ep_idx, waits in enumerate(self.wait_time_records):
                    for job_id, wait_val in waits.items():
                        f_w.write(f"Episode {ep_idx} | JobAgent J{job_id} | Wait {wait_val:.2f}\n")
            print("[Runner] Reward/time/wait logs saved for analysis.")
        except Exception as e:
            print("[WARN] Could not save episode stats:", e)

        print("\n✅ [Runner] Training completed successfully.")

        # === Post-training moving average plot ===
        try:
            loss = np.loadtxt(f"{self.history_dir}/loss.txt")
            td = np.loadtxt(f"{self.history_dir}/td_error.txt")
            window = 50
            if len(loss) > window:
                loss_smooth = np.convolve(loss, np.ones(window)/window, mode='valid')
                td_smooth = np.convolve(td, np.ones(window)/window, mode='valid')
                plt.figure()
                plt.plot(loss_smooth, label="Loss (avg)")
                plt.plot(td_smooth, label="TD Error (avg)", alpha=0.7)
                plt.legend(); plt.xlabel("Training Step"); plt.ylabel("Value")
                plt.title("Step 8A.6.6 – Learning Stability Trends")
                plt.tight_layout()
                plt.savefig(f"{self.history_dir}/learning_stability.png", dpi=300)
                plt.close()
                print("[Runner] Learning stability plot saved.")
        except Exception as e:
            print("[WARN] Could not plot learning stability:", e)

    def evaluate(self, all_gantt_data, global_ep_idx):
        win_number, episode_rewards, gantt_eval = 0, 0, []
        for _ in range(self.args.evaluate_epoch):
            # respect event-driven env if available
            if hasattr(self.env, "wait_for_decisions") and callable(getattr(self.env, "wait_for_decisions")):
                _, ep_reward, win_tag, gantt = self._run_event_driven_episode(global_ep_idx, evaluate=True)
            else:
                _, ep_reward, win_tag, gantt = self.rolloutWorker.generate_episode(global_ep_idx, evaluate=True)
            all_gantt_data.extend(gantt)
            episode_rewards += ep_reward
            if win_tag:
                win_number += 1
            global_ep_idx += 1
        avg_reward = episode_rewards / self.args.evaluate_epoch
        win_rate = win_number / self.args.evaluate_epoch
        return win_rate, avg_reward, global_ep_idx, gantt_eval

    def _select_actions_from_agents(self, batch, evaluate=False):
        """
        Best-effort wrapper to ask the agent stack for actions for a batch of decision items.
        Tries common agent APIs; falls back to greedy/random allowed-wc pick.
        Returns: list of chosen wc indices (or None)
        """
        obs_batch = [item.get("obs") for item in batch]
        # Prefer granular per-action masks ('avail_mask') if present, otherwise
        # expand per-machine 'avail_row' into per-action mask by repeating per operator.
        avail_batch = []
        for item in batch:
            if item is None:
                avail_batch.append(None)
                continue

            # If granular operator-level actions are requested, prefer passing
            # the flattened per-(machine×operator) mask directly to the agent
            # so policies that understand operator granularity can select an
            # index in that flattened space.
            if bool(getattr(self.args, 'use_granular_actions', False)):
                if 'avail_mask' in item and item.get('avail_mask') is not None:
                    try:
                        mask_arr = np.asarray(item.get('avail_mask'), dtype=np.int32)
                        # validate shape: should be num_wcs * num_ops
                        if hasattr(self.env, 'num_ops') and hasattr(self.env, 'num_wcs'):
                            ops = int(self.env.num_ops)
                            mcnt = int(self.env.num_wcs)
                            if mask_arr.size >= mcnt * ops:
                                avail_batch.append(mask_arr.tolist())
                                continue
                        # fallback: pass raw mask
                        avail_batch.append(mask_arr.tolist())
                        continue
                    except Exception:
                        pass

            # Default behavior: provide per-machine availability (n_actions == num_wcs)
            # If a granular mask exists but user didn't request granular actions,
            # reduce it by OR-ing per-operator slots into a per-machine vector.
            if 'avail_mask' in item and item.get('avail_mask') is not None:
                try:
                    mask_arr = np.asarray(item.get('avail_mask'), dtype=np.int32)
                    if hasattr(self.env, 'num_ops') and hasattr(self.env, 'num_wcs'):
                        ops = int(self.env.num_ops)
                        mcnt = int(self.env.num_wcs)
                        if mask_arr.size >= mcnt * ops:
                            per_machine = []
                            for m in range(mcnt):
                                start = m * ops
                                end = start + ops
                                per_machine.append(int(bool(mask_arr[start:end].any())))
                            avail_batch.append(per_machine)
                            continue
                except Exception:
                    pass

            ar = item.get('avail_row')
            if ar is not None:
                try:
                    per_machine = [1 if int(bool(x)) else 0 for x in list(ar)]
                    avail_batch.append(per_machine)
                    continue
                except Exception:
                    pass

            avail_batch.append(item.get('avail_row'))

        # common agent APIs attempted (in order)
        try:
            if hasattr(self.agents, "select_actions"):
                return self.agents.select_actions(obs_batch, avail_batch, evaluate=evaluate)
            if hasattr(self.agents, "choose_actions"):
                return self.agents.choose_actions(obs_batch, avail_batch, evaluate=evaluate)
            if hasattr(self.agents, "act"):
                return self.agents.act(obs_batch, avail_batch, evaluate=evaluate)
        except Exception:
            pass

        # fallback to rolloutWorker if it provides a decision helper
        try:
            if hasattr(self.rolloutWorker, "decide_batch"):
                return self.rolloutWorker.decide_batch(batch, evaluate=evaluate)
        except Exception:
            pass

        # last resort: simple deterministic / random pick from allowed_wcs
        actions = []
        for item in batch:
            allowed = item.get("allowed_wcs", [])
            if not allowed:
                actions.append(None)
            else:
                # pick first available or random if evaluate==False
                if evaluate:
                    actions.append(int(allowed[0]))
                else:
                    try:
                        actions.append(int(self.rolloutWorker.rng.choice(allowed)))
                    except Exception:
                        actions.append(int(np.random.choice(allowed)))
        return actions

    def _run_event_driven_episode(self, global_ep_idx, evaluate=False):
        """
        Drives the SimPy env via wait_for_decisions / pop_decision_reward.
        Returns episode_dict, ep_reward, win_tag, gantt_list
        """
        # start/reset environment (env.reset returns initial obs/info)
        try:
            obs_init, info = self.env.reset()
        except TypeError:
            # some reset signatures may return only obs
            obs_init = self.env.reset()
            info = {}

        episode = {"r": []}
        gantt = []
        # collect per-decision transitions for replay
        ep_transitions = []

        # run until environment signals done
        done = False
        # ensure previous internal counters are set
        if not hasattr(self.env, "prev_total_remaining"):
            try:
                self.env.prev_total_remaining = self.env._total_remaining_work()
            except Exception:
                self.env.prev_total_remaining = 0.0

        while True:
            batch, sim_time = self.env.wait_for_decisions()
            # empty batch may mean done
            if not batch:
                break

            # capture global state before decision (if available)
            try:
                s_before = np.asarray(self.env._build_state_vector(), dtype=np.float32)
            except Exception:
                s_before = None

            # get actions for the batch
            actions = self._select_actions_from_agents(batch, evaluate=evaluate)

            # Pre-process actions: when using machine-level actions map/validate
            # agent output (action_idx) -> global machine index, resample/clip
            # if the selected action is incompatible with the decision's avail mask.
            processed_actions = []
            processed_machine_names = []
            for item, act in zip(batch, (actions or [])):
                chosen = act
                chosen_machine_name = None
                try:
                    if bool(getattr(self.args, 'use_machine_actions', False)):
                        mlist = getattr(self.env.workcenters_meta, 'machine_list', []) or []
                        ops = int(getattr(self.env, 'num_ops', 1))
                        allowed_m_inds = item.get('allowed_machine_indices') or []

                        # normalize incoming action to int if possible
                        try:
                            chosen_i = int(act) if act is not None else None
                        except Exception:
                            chosen_i = None

                        # if out-of-range or None, try to pick from allowed list
                        if chosen_i is None or chosen_i < 0 or chosen_i >= len(mlist):
                            if allowed_m_inds:
                                chosen_i = int(self.rolloutWorker.rng.choice(allowed_m_inds))
                            else:
                                # clip into range
                                if len(mlist) > 0:
                                    chosen_i = max(0, min(len(mlist) - 1, (chosen_i or 0)))
                                else:
                                    chosen_i = 0

                        # validate against granular avail_mask if present
                        mask = None
                        try:
                            mask = np.asarray(item.get('avail_mask')) if item.get('avail_mask') is not None else None
                        except Exception:
                            mask = None

                        if mask is not None:
                            start = chosen_i * ops
                            end = start + ops
                            # if the flattened mask indicates no operator slot free for this machine
                            if end <= mask.size and not bool(mask[start:end].any()):
                                # try to find an allowed machine index with at least one operator slot
                                found = None
                                for cand in (allowed_m_inds or list(range(len(mlist)))):
                                    s = int(cand) * ops
                                    if s + ops <= mask.size and bool(mask[s:s+ops].any()):
                                        found = int(cand)
                                        break
                                if found is not None:
                                    chosen_i = found
                                elif allowed_m_inds:
                                    chosen_i = int(allowed_m_inds[0])
                                else:
                                    # as last resort, leave chosen_i as-is
                                    pass

                        # ensure chosen_i within bounds
                        if not (0 <= chosen_i < len(mlist)) and len(mlist) > 0:
                            chosen_i = max(0, min(len(mlist) - 1, chosen_i if chosen_i is not None else 0))

                        chosen = int(chosen_i)
                        try:
                            chosen_machine_name = mlist[chosen]
                        except Exception:
                            chosen_machine_name = None
                    else:
                        # legacy workcenter-level action — ensure integer
                        chosen = int(act) if act is not None else None
                except Exception:
                    # best-effort fallback
                    try:
                        chosen = int(act) if act is not None else 0
                    except Exception:
                        chosen = 0

                # Logging: map chosen machine -> WC for human-readable logs
                try:
                    if chosen_machine_name is None and bool(getattr(self.args, 'use_machine_actions', False)):
                        mlist = getattr(self.env.workcenters_meta, 'machine_list', []) or []
                        if 0 <= chosen < len(mlist):
                            chosen_machine_name = mlist[chosen]
                except Exception:
                    pass

                try:
                    if bool(getattr(self.args, 'use_machine_actions', False)) and chosen_machine_name is not None:
                        wc_for_m = int(self.env.workcenters_meta.machine_registry.get(chosen_machine_name, {}).get('workcenter', -1))
                        msg = f"[JobAgent {item.get('job_id')}] selected action={act} → Machine={chosen_machine_name} (WC{wc_for_m})"
                    else:
                        msg = f"[JobAgent {item.get('job_id')}] selected action={chosen}"
                    print(msg)
                    logging.getLogger(__name__).info(msg)
                except Exception:
                    pass

                processed_actions.append(chosen)
                processed_machine_names.append(chosen_machine_name)

                # apply the (possibly remapped) action via resume
                try:
                    item.get("resume")(chosen)
                except Exception:
                    try:
                        item.get("resume")(int(chosen))
                    except Exception:
                        try:
                            item.get("resume")(chosen_machine_name)
                        except Exception:
                            pass

            # replace actions with processed ones for downstream bookkeeping
            actions = processed_actions

            # after resuming processes, collect reward accumulated since last decision boundary
            try:
                r = float(self.env.pop_decision_reward())
            except Exception:
                r = 0.0
            episode["r"].append(r)

            # capture global state after decision (if available)
            try:
                s_after = np.asarray(self.env._build_state_vector(), dtype=np.float32)
            except Exception:
                s_after = None

            # capture availabilities after decision (for avail_a_next)
            try:
                avail_after = None
                if hasattr(self.env, '_build_avail_actions'):
                    avail_after = self.env._build_avail_actions()
            except Exception:
                avail_after = None

            # --- build a replay transition for this decision boundary ---
            try:
                obs_batch = [item.get("obs") for item in batch]
            except Exception:
                obs_batch = None

            # Build avail_batch according to requested granularity. If the Runner
            # was configured to use operator-granular actions, prefer a flattened
            # per-(machine×operator) mask. Try to compute a precise mask using
            # MARL.common.mask_utils when available; otherwise fall back to
            # repeating per-machine rows.
            try:
                use_gran = bool(getattr(self.args, 'use_granular_actions', False))
            except Exception:
                use_gran = False

            avail_batch = []
            if use_gran:
                # attempt to import mask utilities
                try:
                    from MARL.common.mask_utils import build_index_map, build_mask_for_job
                except Exception:
                    build_index_map = None
                    build_mask_for_job = None

                # build operator->machines mapping
                op_to_m = {}
                try:
                    groups_map = getattr(self.env.workcenters_meta, 'eligible_operator_groups_by_wc', {})
                    # groups_map: wc_idx -> list(operator_idx)
                    # invert to op->machines
                    for wc_idx, ops in groups_map.items():
                        for p in ops:
                            op_to_m.setdefault(int(p), []).append(int(wc_idx))
                except Exception:
                    op_to_m = {}

                num_m = int(getattr(self.env, 'num_wcs', 0))
                num_p = int(getattr(self.env, 'num_ops', 0))

                # Precompute resource free states (machine/operator) at this time
                try:
                    machine_free = [self.env._resource_free(self.env.wc_resources[m]) for m in range(num_m)]
                except Exception:
                    machine_free = [True] * num_m
                try:
                    operator_free = [self.env._resource_free(self.env.operator_groups[p]) for p in range(num_p)]
                except Exception:
                    operator_free = [True] * num_p

                for item in batch:
                    try:
                        # prefer an existing granular mask if rollout attached one
                        if item.get('avail_mask') is not None:
                            mask = item.get('avail_mask')
                            # ensure list/numpy
                            avail_batch.append(list(mask))
                            continue

                        allowed = item.get('allowed_wcs', [])
                        if build_index_map is not None and build_mask_for_job is not None and num_m > 0 and num_p > 0:
                            idx_map = build_index_map(num_m, num_p)
                            try:
                                msk = build_mask_for_job(idx_map, allowed, op_to_m, machine_free, operator_free)
                                avail_batch.append(msk.tolist())
                                # also attach to item for downstream readers
                                item['avail_mask'] = msk.tolist()
                                continue
                            except Exception:
                                pass

                        # fallback: repeat per-machine avail_row into flattened mask
                        row = item.get('avail_row') or []
                        flat = []
                        for m in range(num_m):
                            v = 1 if (m < len(row) and int(bool(row[m]))) else 0
                            flat.extend([int(v)] * max(1, num_p))
                        avail_batch.append(flat)
                    except Exception:
                        avail_batch.append(None)
            else:
                # default per-machine availability
                try:
                    avail_batch = [item.get('avail_row') for item in batch]
                except Exception:
                    avail_batch = None

            # actions may be shorter than n_agents; create a per-agent action list
            try:
                u_list = []
                u_machine_list = []
                for i, a in enumerate(actions):
                    try:
                        u_list.append(int(a) if a is not None else 0)
                    except Exception:
                        u_list.append(0)
                    try:
                        # map to machine index if available
                        uname = processed_machine_names[i] if i < len(processed_machine_names) else None
                        if uname is not None:
                            u_machine_list.append(int(getattr(self.env.workcenters_meta, 'machine_index', {}).get(uname, 0)))
                        else:
                            u_machine_list.append(None)
                    except Exception:
                        u_machine_list.append(None)
            except Exception:
                u_list = [int(a) if a is not None else 0 for a in (actions or [])]

            # pad/truncate to args.n_agents
            n_agents = getattr(self.args, "n_agents", len(u_list) if u_list else 1)
            if len(u_list) < n_agents:
                u_list = u_list + [0] * (n_agents - len(u_list))
            elif len(u_list) > n_agents:
                u_list = u_list[:n_agents]

            # Prepare obs array with shape (n_agents, obs_dim)
            obs_dim = getattr(self.args, "obs_shape", None)
            if obs_batch is None:
                o_arr = np.zeros((n_agents, obs_dim if obs_dim is not None else 1), dtype=np.float32)
            else:
                try:
                    o_tmp = np.asarray(obs_batch, dtype=np.float32)
                    if o_tmp.ndim == 1:
                        # single flattened obs -> assume per-agent obs dim
                        o_tmp = o_tmp.reshape(1, -1)
                    # pad agents
                    if obs_dim is None:
                        obs_dim = o_tmp.shape[1]
                    if o_tmp.shape[0] < n_agents:
                        pad_rows = np.zeros((n_agents - o_tmp.shape[0], obs_dim), dtype=np.float32)
                        if o_tmp.shape[1] < obs_dim:
                            # pad columns
                            col_pad = np.zeros((o_tmp.shape[0], obs_dim - o_tmp.shape[1]), dtype=np.float32)
                            o_tmp = np.concatenate([o_tmp, col_pad], axis=1)
                        o_arr = np.concatenate([o_tmp, pad_rows], axis=0)
                    else:
                        # truncate agents and cols if necessary
                        o_arr = o_tmp[:n_agents, :obs_dim]
                except Exception:
                    o_arr = np.zeros((n_agents, obs_dim if obs_dim is not None else 1), dtype=np.float32)

            # Prepare avail array with shape (n_agents, n_actions)
            n_actions = getattr(self.args, "n_actions", None)
            if avail_batch is None or n_actions is None:
                avail_arr = None
            else:
                try:
                    a_tmp = np.asarray(avail_batch, dtype=np.float32)
                    if a_tmp.ndim == 1:
                        a_tmp = a_tmp.reshape(1, -1)
                    # pad rows
                    if a_tmp.shape[0] < n_agents:
                        pad_rows = np.zeros((n_agents - a_tmp.shape[0], n_actions), dtype=np.float32)
                        if a_tmp.shape[1] < n_actions:
                            col_pad = np.zeros((a_tmp.shape[0], n_actions - a_tmp.shape[1]), dtype=np.float32)
                            a_tmp = np.concatenate([a_tmp, col_pad], axis=1)
                        avail_arr = np.concatenate([a_tmp, pad_rows], axis=0)
                    else:
                        avail_arr = a_tmp[:n_agents, :n_actions]
                except Exception:
                    avail_arr = None

            tr = {}
            tr["o"] = o_arr
            tr["u"] = u_list
            # store machine-level info (action_idx -> global machine index)
            try:
                # u_machine: per-agent machine index (or -1 if unavailable)
                tr["u_machine"] = [(-1 if x is None else int(x)) for x in u_machine_list]
            except Exception:
                tr["u_machine"] = [(-1 if x is None else int(x)) for x in (u_machine_list if 'u_machine_list' in locals() else [None]*len(u_list))]
            try:
                tr["u_machine_name"] = [(None if x is None else str(x)) for x in processed_machine_names]
            except Exception:
                tr["u_machine_name"] = [None] * len(u_list)
            tr["r"] = r
            # Attach availability vectors for this transition. If granular
            # operator-level actions are enabled, prefer the flattened
            # per-(machine×operator) mask attached to each batch item. If not
            # available, expand the per-machine avail_arr into the flattened
            # space by repeating each machine slot `num_ops` times.
            try:
                if bool(getattr(self.args, 'use_granular_actions', False)):
                    # build per-agent flattened avail arrays
                    try:
                        ops = int(self.env.num_ops)
                    except Exception:
                        ops = None
                    n_agents_local = len(batch)
                    # decide n_actions if we can
                    if ops is not None and hasattr(self.env, 'num_wcs'):
                        n_actions_local = int(self.env.num_wcs) * ops
                    else:
                        n_actions_local = None

                    avail_flat = []
                    for i_item, item in enumerate(batch):
                        # prefer item-level granular mask
                        mask = None
                        try:
                            mask = item.get('avail_mask')
                        except Exception:
                            mask = None
                        if mask is not None:
                            try:
                                arr = np.asarray(mask, dtype=np.float32)
                                if n_actions_local is None or arr.size >= n_actions_local:
                                    # crop/pad to expected size if needed
                                    if n_actions_local is not None:
                                        s = arr.size
                                        if s < n_actions_local:
                                            pad = np.zeros((n_actions_local - s,), dtype=np.float32)
                                            arr = np.concatenate([arr, pad], axis=0)
                                        arr = arr[:n_actions_local]
                                    avail_flat.append(arr.astype(np.float32))
                                    continue
                            except Exception:
                                pass

                        # fallback: expand per-machine avail_arr
                        try:
                            if avail_arr is not None:
                                row = np.asarray(avail_arr[i_item], dtype=np.float32)
                                if ops is not None:
                                    expanded = np.repeat(row.astype(np.float32), ops)
                                    if n_actions_local is not None:
                                        expanded = expanded[:n_actions_local]
                                    avail_flat.append(expanded)
                                    continue
                                else:
                                    avail_flat.append(row)
                                    continue
                        except Exception:
                            pass

                        # last resort: zeros
                        if n_actions_local is not None:
                            avail_flat.append(np.zeros((n_actions_local,), dtype=np.float32))
                        else:
                            avail_flat.append(np.zeros((len(u_list),), dtype=np.float32))

                    tr["avail_a"] = np.asarray(avail_flat, dtype=np.float32)
                else:
                    if avail_arr is not None:
                        tr["avail_a"] = avail_arr
            except Exception:
                # if anything fails, don't block the episode; leave avail unset
                pass
            # include next-step availabilities per-job if we computed them
            try:
                if avail_after is not None and avail_arr is not None:
                    n_agents_local = avail_arr.shape[0]
                    n_actions_local = avail_arr.shape[1]
                    avail_next_arr = np.zeros((n_agents_local, n_actions_local), dtype=np.float32)
                    for i_item, item in enumerate(batch):
                        job_id = item.get('job_id')
                        if job_id is None:
                            continue
                        try:
                            row = avail_after[int(job_id)]
                            row = np.asarray(row, dtype=np.float32)
                            # row from env._build_avail_actions is per-machine; expand to per-action
                            if row.ndim == 1 and hasattr(self.env, 'num_ops'):
                                ops = int(self.env.num_ops)
                                expanded = np.repeat(row.astype(np.float32), ops)
                                avail_next_arr[i_item, :] = expanded[:n_actions_local]
                            elif row.ndim == 1:
                                avail_next_arr[i_item, :] = row[:n_actions_local]
                        except Exception:
                            # leave zeros if mapping fails
                            pass
                    tr["avail_a_next"] = avail_next_arr
            except Exception:
                pass
            # include global state and next-state for mixer networks
            if s_before is not None:
                tr["s"] = s_before
            if s_after is not None:
                tr["s_next"] = s_after
            tr["done"] = getattr(self.env, "done", False)

            ep_transitions.append(tr)

            # collect possible lightweight gantt info if present on env or items
            if hasattr(self.rolloutWorker, "collect_gantt_from_batch"):
                try:
                    gantt.extend(self.rolloutWorker.collect_gantt_from_batch(batch, sim_time))
                except Exception:
                    pass

            # stop if env signals done
            if getattr(self.env, "done", False):
                break
            # safety: break if time limit reached
            if getattr(self.env, "t", 0.0) >= getattr(self.env, "episode_limit", self.args.n_steps if hasattr(self.args, "n_steps") else 1e9):
                break

        ep_reward = float(np.sum(episode.get("r", [])))
        win_tag = all(j.finished for j in getattr(self.env, "jobs", []))

        # store episode into replay buffer if available
        try:
            if self.buffer is not None and len(ep_transitions) > 0:
                try:
                    self.buffer.store_episode(ep_transitions)
                    print(f"[DEBUG] Stored episode to buffer | transitions={len(ep_transitions)} | buffer_len={len(self.buffer)}")
                except Exception as e:
                    print(f"[WARN] Could not store episode to buffer: {e}")
        except Exception:
            pass

        # include environment-level gantt records if present
        try:
            if hasattr(self.env, "gantt_records") and isinstance(self.env.gantt_records, (list, tuple)):
                # each record is (start, end, op_idx, wc, job_id, operator_grp)
                gantt.extend(list(self.env.gantt_records))
        except Exception:
            pass

        return episode, ep_reward, bool(win_tag), gantt

    def plt(self, num):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel(f'epoch × {self.args.evaluate_cycle}')
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel(f'epoch × {self.args.evaluate_cycle}')
        plt.ylabel('episode_rewards')

        plt.tight_layout()
        outdir = self.save_path
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f'plt_{num}.png'), format='png')
        np.save(os.path.join(outdir, f'win_rates_{num}.npy'), self.win_rates)
        np.save(os.path.join(outdir, f'episode_rewards_{num}.npy'), self.episode_rewards)
        print(f"[Runner] Plots saved to {outdir}")
# ...existing code...