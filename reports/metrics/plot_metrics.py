import json
import os

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _compute_metrics_from_gantt(records):
    """Best-effort compute util/makespan/wait from gantt-like records.

    Supports records as list of dicts (keys: start,end,wc_idx,op_grp,job_id,arrival,duration)
    or list/tuple with positions (start,end,op_idx,wc,job_id,op_grp,arrival,duration).
    Returns dict with keys similar to env._compute_utilization_summary.
    """
    import numpy as _np
    if not records:
        return {}
    starts = []
    ends = []
    total_machine_busy = {}
    total_operator_busy = {}
    job_first_start = {}
    job_arrival = {}
    for r in (records or []):
        try:
            if isinstance(r, dict):
                s = float(r.get('start', r.get('s', 0.0)))
                e = float(r.get('end', r.get('e', s)))
                wc = r.get('wc_idx', r.get('wc', None))
                op = r.get('op_grp', r.get('op_id', None))
                jid = r.get('job_id', None)
                arrival = r.get('arrival', None)
            else:
                try:
                    s = float(r[0])
                except Exception as e:
                    s = 0.0
                try:
                    e = float(r[1])
                except Exception as e:
                    e = s
                wc = r[3] if len(r) > 3 else None
                op = r[5] if len(r) > 5 else None
                jid = r[4] if len(r) > 4 else None
                arrival = r[6] if len(r) > 6 else None

            dur = max(0.0, float(e) - float(s))
            if dur <= 0:
                continue
            starts.append(s); ends.append(e)
            # machine
            try:
                mid = int(wc) if wc is not None else None
            except Exception as e:
                try:
                    mid = int(str(wc))
                except Exception as e:
                    mid = None
            if mid is not None:
                total_machine_busy[mid] = total_machine_busy.get(mid, 0.0) + dur
            # operator
            if op is not None and str(op) != 'UNASSIGNED':
                oid = str(op)
                total_operator_busy[oid] = total_operator_busy.get(oid, 0.0) + dur
            # job timings
            try:
                if jid is not None:
                    jid_i = int(jid)
                    if jid_i not in job_first_start or job_first_start[jid_i] > s:
                        job_first_start[jid_i] = s
                    if arrival is not None:
                        try:
                            job_arrival[jid_i] = float(arrival)
                        except Exception as e:
                            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
            continue

    if starts and ends:
        episode_length = float(max(ends) - min(starts))
    else:
        episode_length = 1e-9
    if episode_length <= 0:
        episode_length = 1e-9

    # total counts
    total_machines = max(list(total_machine_busy.keys()) or [0]) + 1 if total_machine_busy else 1
    total_operators = max(1, len(total_operator_busy))

    mm_total = float(sum(total_machine_busy.values()))
    oo_total = float(sum(total_operator_busy.values()))

    try:
        avg_machine_util = float(mm_total) / (episode_length * max(1, int(total_machines)))
    except Exception as e:
        avg_machine_util = 0.0
    try:
        avg_operator_util = float(oo_total) / (episode_length * max(1, int(total_operators)))
    except Exception as e:
        avg_operator_util = 0.0

    per_machine_util = {}
    for mid, busy in total_machine_busy.items():
        try:
            per_machine_util[mid] = float(min(max(busy / float(episode_length), 0.0), 1.0))
        except Exception as e:
            per_machine_util[mid] = 0.0

    per_operator_util = {}
    for oid, busy in total_operator_busy.items():
        try:
            per_operator_util[oid] = float(min(max(busy / float(episode_length), 0.0), 1.0))
        except Exception as e:
            per_operator_util[oid] = 0.0

    # average wait time per job if arrival info present
    avg_wait = None
    try:
        waits = []
        for jid, start in job_first_start.items():
            if jid in job_arrival:
                waits.append(float(start) - float(job_arrival[jid]))
        if waits:
            avg_wait = float(_np.mean(waits))
    except Exception as e:
        avg_wait = None

    return {
        'avg_machine_utilization': float(min(max(avg_machine_util, 0.0), 1.0)),
        'avg_operator_utilization': float(min(max(avg_operator_util, 0.0), 1.0)),
        'average_makespan': float(episode_length),
        'avg_makespan': float(episode_length),
        'average_wait_time': float(avg_wait) if avg_wait is not None else None,
        'avg_wait_time': float(avg_wait) if avg_wait is not None else None,
        'per_machine_utilization': per_machine_util,
        'per_operator_utilization': per_operator_util,
    }


def plot_utilization_and_makespan(history_dir: str = "my_data_and_graph/historydata"):
    path = os.path.join(history_dir, "run_summary.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return

    evols = data.get("evolutions", []) if isinstance(data, dict) else []
    if not evols:
        return

    x = list(range(len(evols)))
    y_mach = [e.get("avg_machine_utilization", 0) for e in evols]
    y_oper = [e.get("avg_operator_utilization", 0) for e in evols]
    # Resolve makespan robustly: support both key names and fallback to gantt records
    y_makespan = []
    for e in evols:
        m = None
        try:
            m = e.get('avg_makespan', None)
        except Exception as e:
            m = None
        if m is None:
            try:
                m = e.get('average_makespan', None)
            except Exception as e:
                m = None
        # Fallback: compute from embedded gantt if present
        if m is None or (isinstance(m, (int, float)) and float(m) == 0.0):
            try:
                gantt = e.get('gantt', None)
                if gantt:
                    comp = _compute_metrics_from_gantt(gantt)
                    m = comp.get('avg_makespan', comp.get('average_makespan', m))
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
        y_makespan.append(float(m) if m is not None else 0.0)

    if plt is None:
        print("matplotlib not available: cannot produce plots")
        return

    os.makedirs(history_dir, exist_ok=True)

    plots = [
        ("Machine Utilization per Evolution", y_mach, "machine_utilization_per_evolution.png", "Utilization"),
        ("Operator Utilization per Evolution", y_oper, "operator_utilization_per_evolution.png", "Utilization"),
        ("Average Makespan per Evolution", y_makespan, "makespan_per_evolution.png", "Makespan"),
    ]

    for title, y, fname, ylabel in plots:
        try:
            plt.figure(figsize=(8, 4))
            plt.plot(x, y, marker="o")
            plt.title(title)
            plt.xlabel("Evolution")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(history_dir, fname))
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not save plot {fname}: {e}")


def plot_per_machine_utilization(history_dir: str = "my_data_and_graph/historydata"):
    path = os.path.join(history_dir, "run_summary.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return
    evols = data.get("evolutions", []) if isinstance(data, dict) else []
    if not evols:
        return
    # Produce a multi-line plot across evolutions: one colored line per machine id
    if plt is None:
        print("matplotlib not available: cannot produce per-machine plot")
        return
    os.makedirs(history_dir, exist_ok=True)
    try:
        # collect union of machine ids across evolutions (preserve string form)
        machine_ids = []
        for e in evols:
            pm = e.get('per_machine_utilization', {}) or {}
            for k in pm.keys():
                ks = str(k)
                if ks not in machine_ids:
                    machine_ids.append(ks)

        x = list(range(len(evols)))
        cmap = plt.get_cmap('tab10')
        plt.figure(figsize=(8, 4))
        for i, mid in enumerate(machine_ids):
            vals = []
            for e in evols:
                pm = e.get('per_machine_utilization', {}) or {}
                # try numeric key then string key
                v = None
                try:
                    if mid.isdigit():
                        v = pm.get(int(mid), None)
                    if v is None:
                        v = pm.get(mid, None)
                except Exception as e:
                    v = pm.get(mid, 0.0)
                # Fallback: if per-machine map missing, try computing from gantt
                if v is None:
                    try:
                        gantt = e.get('gantt', None)
                        if gantt:
                            comp = _compute_metrics_from_gantt(gantt)
                            pm_comp = comp.get('per_machine_utilization', {}) or {}
                            try:
                                if mid.isdigit():
                                    v = pm_comp.get(int(mid), None)
                                if v is None:
                                    v = pm_comp.get(mid, None)
                            except Exception as e:
                                v = pm_comp.get(mid, 0.0)
                    except Exception as e:
                        v = None
                vals.append(float(v) if v is not None else 0.0)
            plt.plot(x, vals, marker='o', label=f"M{mid}", color=cmap(i % 10))

        plt.title('Machine Utilization per Evolution')
        plt.xlabel('Evolution')
        plt.ylabel('Utilization')
        plt.ylim(0, 1)
        plt.grid(True)
        if machine_ids:
            plt.legend(loc='best', fontsize='x-small')
        plt.tight_layout()
        # Save under the requested filename (preserves historical naming)
        plt.savefig(os.path.join(history_dir, 'machine_utilization_per_evolution.png'))
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not save machine_utilization_per_evolution.png: {e}")


def plot_per_operator_utilization(history_dir: str = "my_data_and_graph/historydata"):
    path = os.path.join(history_dir, "run_summary.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return
    evols = data.get("evolutions", []) if isinstance(data, dict) else []
    if not evols:
        return
    # Produce a multi-line plot across evolutions: one colored line per operator id
    if plt is None:
        print("matplotlib not available: cannot produce per-operator plot")
        return
    os.makedirs(history_dir, exist_ok=True)
    try:
        op_ids = []
        for e in evols:
            po = e.get('per_operator_utilization', {}) or {}
            for k in po.keys():
                ks = str(k)
                if ks not in op_ids:
                    op_ids.append(ks)

        x = list(range(len(evols)))
        cmap = plt.get_cmap('tab10')
        plt.figure(figsize=(8, 4))
        for i, oid in enumerate(op_ids):
            vals = []
            for e in evols:
                po = e.get('per_operator_utilization', {}) or {}
                v = None
                try:
                    if oid.isdigit():
                        v = po.get(int(oid), None)
                    if v is None:
                        v = po.get(oid, None)
                except Exception as e:
                    v = po.get(oid, 0.0)
                # Fallback: compute from gantt records if missing
                if v is None:
                    try:
                        gantt = e.get('gantt', None)
                        if gantt:
                            comp = _compute_metrics_from_gantt(gantt)
                            po_comp = comp.get('per_operator_utilization', {}) or {}
                            v = po_comp.get(oid, None)
                            if v is None and oid.isdigit():
                                v = po_comp.get(int(oid), None)
                    except Exception as e:
                        v = None
                vals.append(float(v) if v is not None else 0.0)
            plt.plot(x, vals, marker='o', label=f"O{oid}", color=cmap(i % 10))

        plt.title('Operator Utilization per Evolution')
        plt.xlabel('Evolution')
        plt.ylabel('Utilization')
        plt.ylim(0, 1)
        plt.grid(True)
        if op_ids:
            plt.legend(loc='best', fontsize='x-small')
        plt.tight_layout()
        plt.savefig(os.path.join(history_dir, 'operator_utilization_per_evolution.png'))
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not save operator_utilization_per_evolution.png: {e}")


# NOTE: multi-evolution comparative plots were intentionally removed.
# The per-evolution bar charts and the combined `utilization_summary_grid`
# continue to provide the necessary views. Keeping these functions would
# duplicate information and produce extra artifacts, so they are omitted.


# NOTE: multi-evolution comparative operator plots were intentionally removed.
# Use `utilization_summary_grid()` for combined views that include reward,
# makespan, and per-id trends without creating separate multi-evolution PNGs.


def utilization_summary_grid(history_dir: str = "my_data_and_graph/historydata", combine_plots: bool = True):
    path = os.path.join(history_dir, "run_summary.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return
    evols = data.get("evolutions", []) if isinstance(data, dict) else []
    if not evols:
        return

    x = list(range(len(evols)))
    makespans = []
    avg_rewards = []
    for e in evols:
        # reward
        try:
            avg_r = e.get('avg_epoch_reward') if e.get('avg_epoch_reward') is not None else None
        except Exception as e:
            avg_r = None
        avg_rewards.append(avg_r)
        # makespan: prefer normalized keys then gantt fallback
        m = e.get('avg_makespan', None) if isinstance(e, dict) else None
        if m is None:
            m = e.get('average_makespan', None) if isinstance(e, dict) else None
        if m is None or (isinstance(m, (int, float)) and float(m) == 0.0):
            try:
                gantt = e.get('gantt', None)
                if gantt:
                    comp = _compute_metrics_from_gantt(gantt)
                    m = comp.get('avg_makespan', comp.get('average_makespan', m))
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
        makespans.append(float(m) if m is not None else 0.0)

    if plt is None:
        print("matplotlib not available: cannot produce grid plot")
        return
    os.makedirs(history_dir, exist_ok=True)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax0 = axes[0,0]
        # machine trends
        # reuse function logic to build machine_ids
        machine_ids = []
        for e in evols:
            pm = e.get('per_machine_utilization', {}) or {}
            for k in pm.keys():
                ks = str(k)
                if ks not in machine_ids:
                    machine_ids.append(ks)
        cmap = plt.get_cmap('tab10')
        for i, mid in enumerate(machine_ids):
            vals = []
            for e in evols:
                pm = e.get('per_machine_utilization', {}) or {}
                v = pm.get(int(mid), None) if mid.isdigit() else pm.get(mid, None)
                if v is None:
                    v = pm.get(str(mid), 0.0)
                vals.append(float(v))
            ax0.plot(x, vals, marker='o', label=f"M{mid}", color=cmap(i % 10))
        ax0.set_title('Per-Machine Utilization')
        ax0.set_ylim(0,1)
        ax0.grid(True)

        ax1 = axes[0,1]
        # operator trends
        op_ids = []
        for e in evols:
            po = e.get('per_operator_utilization', {}) or {}
            for k in po.keys():
                ks = str(k)
                if ks not in op_ids:
                    op_ids.append(ks)
        for i, oid in enumerate(op_ids):
            vals = []
            for e in evols:
                po = e.get('per_operator_utilization', {}) or {}
                v = po.get(oid, None)
                if v is None and oid.isdigit():
                    v = po.get(int(oid), 0.0)
                if v is None:
                    v = 0.0
                vals.append(float(v))
            ax1.plot(x, vals, marker='o', label=f"O{oid}", color=cmap(i % 10))
        ax1.set_title('Per-Operator Utilization')
        ax1.set_ylim(0,1)
        ax1.grid(True)

        ax2 = axes[1,0]
        ax2.plot(x, makespans, marker='o', color='tab:green')
        ax2.set_title('Average Makespan')
        ax2.grid(True)

        ax3 = axes[1,1]
        # plot avg rewards if present
        if any(v is not None for v in avg_rewards):
            rr = [v if v is not None else float('nan') for v in avg_rewards]
            ax3.plot(x, rr, marker='o', color='tab:purple')
            ax3.set_title('Average Reward')
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'No reward data', ha='center', va='center')
            ax3.set_title('Average Reward')

        # legends for top row
        ax0.legend(loc='best', fontsize='x-small')
        ax1.legend(loc='best', fontsize='x-small')

        plt.tight_layout()
        plt.savefig(os.path.join(history_dir, 'utilization_summary_grid.png'))
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not save utilization_summary_grid.png: {e}")


def check_timeline_consistency(history_dir: str = "my_data_and_graph/historydata", util_threshold: float = 0.6):
    """Heuristic consistency check between scheduling_timeline.txt and latest per-id utilizations.

    Logs warnings when a machine/operator shows high utilization but few mentions in timeline.
    """
    timeline_path = os.path.join(history_dir, 'scheduling_timeline.txt')
    summary_path = os.path.join(history_dir, 'run_summary.json')
    if not os.path.exists(summary_path):
        return
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return
    evols = data.get('evolutions', []) or []
    if not evols:
        return
    latest = evols[-1]
    pm = latest.get('per_machine_utilization', {}) or {}
    po = latest.get('per_operator_utilization', {}) or {}

    # count mentions in timeline heuristically
    machine_counts = {}
    operator_counts = {}
    if os.path.exists(timeline_path):
        try:
            import re
            with open(timeline_path, 'r', encoding='utf-8') as tf:
                for ln in tf:
                    # find M<digits>
                    for m in re.findall(r"M(\d+)", ln):
                        machine_counts[m] = machine_counts.get(m, 0) + 1
                    for o in re.findall(r"O(\d+)", ln):
                        operator_counts[o] = operator_counts.get(o, 0) + 1
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

    # warn heuristically
    try:
        for mid_str, util in ((str(k), v) for k, v in pm.items()):
            try:
                if float(util) >= util_threshold:
                    cnt = machine_counts.get(mid_str, 0)
                    if cnt < 2:
                        print(f"WARNING: Machine {mid_str} shows high utilization ({util:.2f}) but only {cnt} mentions in timeline.")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                continue
        for oid, util in po.items():
            try:
                if float(util) >= util_threshold:
                    cnt = operator_counts.get(str(oid), 0)
                    if cnt < 2:
                        print(f"WARNING: Operator {oid} shows high utilization ({util:.2f}) but only {cnt} mentions in timeline.")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                continue
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
