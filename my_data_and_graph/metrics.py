import json
import os
from datetime import datetime
from typing import List, Any

import numpy as np


def append_run_summary(summary: dict, history_dir: str = "my_data_and_graph/historydata"):
    """Append a per-evolution summary into run_summary.json.

    If the file exists and does not contain an "evolutions" list, the existing
    content is preserved under the "meta" key and a new "evolutions" list is
    created. This is best-effort and avoids clobbering unrelated fields.
    """
    os.makedirs(history_dir, exist_ok=True)
    path = os.path.join(history_dir, "run_summary.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
    except Exception:
        data = {}

    # normalize summary timestamp and index
    if 'timestamp' not in summary:
        summary['timestamp'] = datetime.now().isoformat()
    # Normalize common key name variants so downstream plotters/readers
    # can rely on a consistent set of keys. Support both env-style
    # ('average_makespan','average_wait_time') and metrics-style
    # ('avg_makespan','avg_wait_time'). Also mirror machine/operator util
    # naming if one or the other was provided.
    try:
        if 'average_makespan' in summary and 'avg_makespan' not in summary:
            summary['avg_makespan'] = summary.get('average_makespan')
        if 'avg_makespan' in summary and 'average_makespan' not in summary:
            summary['average_makespan'] = summary.get('avg_makespan')
    except Exception:
        pass
    try:
        if 'average_wait_time' in summary and 'avg_wait_time' not in summary:
            summary['avg_wait_time'] = summary.get('average_wait_time')
        if 'avg_wait_time' in summary and 'average_wait_time' not in summary:
            summary['average_wait_time'] = summary.get('avg_wait_time')
    except Exception:
        pass
    try:
        if 'avg_machine_utilization' in summary and 'average_machine_utilization' not in summary:
            summary['average_machine_utilization'] = summary.get('avg_machine_utilization')
        if 'average_machine_utilization' in summary and 'avg_machine_utilization' not in summary:
            summary['avg_machine_utilization'] = summary.get('average_machine_utilization')
    except Exception:
        pass
    try:
        if 'avg_operator_utilization' in summary and 'average_operator_utilization' not in summary:
            summary['average_operator_utilization'] = summary.get('avg_operator_utilization')
        if 'average_operator_utilization' in summary and 'avg_operator_utilization' not in summary:
            summary['avg_operator_utilization'] = summary.get('average_operator_utilization')
    except Exception:
        pass
    # If file already contains evolutions, append. Otherwise, preserve
    # existing top-level content under 'meta' and create 'evolutions'.
    if 'evolutions' in data and isinstance(data['evolutions'], list):
        data['evolutions'].append(summary)
    else:
        if data:
            data = {'meta': data, 'evolutions': [summary]}
        else:
            data = {'evolutions': [summary]}

    # Preserve incoming avg fields when present on the summary. Do not
    # overwrite with zero fallbacks when evolutions exist. If the caller
    # provided avg_machine_utilization, avg_operator_utilization or
    # avg_epoch_reward on the summary, copy them to the top-level so other
    # tools that expect a top-level summary can see the latest values.
    for k in ('avg_machine_utilization', 'avg_operator_utilization', 'avg_epoch_reward'):
        if k in summary:
            try:
                # only set when a real numeric value is provided
                v = summary.get(k)
                if v is not None:
                    data[k] = v
            except Exception:
                continue

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not write run_summary.json: {e}")


def collect_evolution_summary(envs: List[object], evol_index: int):
    """Collect averaged metrics across a list of completed environments for one evolution.

    Args:
      envs: list of completed MASAEnv instances
      evol_index: integer evolution index
    """
    total_makespan = []
    total_wait = []
    mach_utils = []
    oper_utils = []
    avg_rewards = []

    def _compute_from_gantt(gantt_records: Any, n_machines: int = 1, n_ops: int = 1, total_wait_time: float = 0.0, completed_count: int = None):
        """Best-effort compute util/makespan from raw gantt_records list.

        gantt_records: list of dict or tuple records
        n_machines, n_ops: fallback counts
        returns dict matching env._compute_utilization_summary output
        """
        # reuse logic similar to MASAEnv._compute_utilization_summary
        starts = []
        ends = []
        machine_busy = {}
        operator_busy = {}
        for r in (gantt_records or []):
            try:
                if isinstance(r, dict):
                    s = float(r.get('start', r.get('s', 0.0)))
                    e = float(r.get('end', r.get('e', s)))
                    wc_idx = r.get('wc_idx', r.get('wc', None))
                    op_grp = r.get('op_grp', r.get('op_grp', r.get('op_id', None)))
                else:
                    try:
                        s = float(r[0])
                    except Exception:
                        s = 0.0
                    try:
                        e = float(r[1])
                    except Exception:
                        e = s
                    wc_idx = r[3] if len(r) > 3 else None
                    op_grp = r[5] if len(r) > 5 else None
                dur = max(0.0, float(e) - float(s))
                starts.append(float(s)); ends.append(float(e))
                if wc_idx is not None:
                    try:
                        midx = int(wc_idx)
                    except Exception:
                        midx = None
                    if midx is not None:
                        machine_busy[midx] = machine_busy.get(midx, 0.0) + dur
                if op_grp is not None:
                    try:
                        opid = str(op_grp)
                    except Exception:
                        opid = None
                    if opid and opid != 'UNASSIGNED':
                        operator_busy[opid] = operator_busy.get(opid, 0.0) + dur
            except Exception:
                continue

        if starts and ends:
            min_start = min(starts); max_end = max(ends)
            episode_len = max(1e-9, float(max_end) - float(min_start))
        else:
            episode_len = 1.0

        total_machine_busy = sum(machine_busy.values())
        total_operator_busy = sum(operator_busy.values())

        avg_machine_util = float(total_machine_busy) / (episode_len * max(1, int(n_machines))) if episode_len > 0 else 0.0
        avg_operator_util = float(total_operator_busy) / (episode_len * max(1, int(n_ops))) if episode_len > 0 else 0.0
        avg_makespan = float(episode_len)
    # Use canonical completed_count passed by caller and enforce fail-fast behavior
        if completed_count is None:
            raise ValueError("Cannot compute average wait time: completed job count is required")
        if completed_count <= 0:
            raise ValueError("Cannot compute average wait time: no completed jobs available")
        avg_wait = float(total_wait_time) / float(completed_count)

        return {
            'avg_machine_utilization': float(avg_machine_util),
            'avg_operator_utilization': float(avg_operator_util),
            'average_makespan': float(avg_makespan),
            'average_wait_time': float(avg_wait),
        }

    for item in envs:
        util = None
        # item may be an env object (has _compute_utilization_summary)
        if hasattr(item, '_compute_utilization_summary') and callable(getattr(item, '_compute_utilization_summary')):
            util = item._compute_utilization_summary()

        # If item is a dict describing gantt and counters
        if util is None:
            if isinstance(item, dict) and 'gantt' in item:
                gantt = item.get('gantt', [])
                n_m = int(item.get('n_machines', 1)) if item.get('n_machines', None) is not None else 1
                n_o = int(item.get('n_ops', 1)) if item.get('n_ops', None) is not None else 1
                total_wait_time = float(item.get('total_wait_time', 0.0))
                # compute canonical completed count from provided jobs list; fail-fast when not available
                if 'jobs' in item and isinstance(item.get('jobs'), (list, tuple)):
                    # Support both object-like jobs (with attribute access) and
                    # dict-like job records (from Runner's serializable epoch_item).
                    def _is_finished(j):
                        try:
                            if isinstance(j, dict):
                                return bool(j.get('finished', False))
                            return bool(getattr(j, 'finished', False))
                        except Exception:
                            return False

                    completed_count = sum(1 for j in item.get('jobs') if _is_finished(j))
                else:
                    raise ValueError("Cannot compute metrics for dict item: missing 'jobs' list to derive completed job count")
                util = _compute_from_gantt(gantt, n_machines=n_m, n_ops=n_o, total_wait_time=total_wait_time, completed_count=completed_count)
            elif isinstance(item, (list, tuple)):
                # list/tuple gantt-only item: cannot compute avg_wait without completed job count
                raise ValueError("Cannot compute metrics from raw gantt list: completed job count required")

        if util:
            mach_utils.append(float(util.get("avg_machine_utilization", 0.0)))
            oper_utils.append(float(util.get("avg_operator_utilization", 0.0)))
            total_makespan.append(float(util.get("average_makespan", 0.0)))
            total_wait.append(float(util.get("average_wait_time", 0.0)))
            # capture per-evolution avg reward if present on the item
            try:
                if isinstance(item, dict) and item.get('avg_epoch_reward') is not None:
                    avg_rewards.append(float(item.get('avg_epoch_reward')))
                elif isinstance(util, dict) and util.get('avg_epoch_reward') is not None:
                    avg_rewards.append(float(util.get('avg_epoch_reward')))
            except Exception:
                pass
        else:
            mach_utils.append(0.0)
            oper_utils.append(0.0)
            total_makespan.append(0.0)
            total_wait.append(0.0)

    summary = {
        "evolution_index": int(evol_index),
        "avg_makespan": float(np.mean(total_makespan)) if total_makespan else 0.0,
        "avg_wait_time": float(np.mean(total_wait)) if total_wait else 0.0,
        "avg_machine_utilization": float(np.mean(mach_utils)) if mach_utils else 0.0,
        "avg_operator_utilization": float(np.mean(oper_utils)) if oper_utils else 0.0,
        # Ensure avg_epoch_reward is always present (fallback to 0.0)
        "avg_epoch_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
        "timestamp": datetime.now().isoformat(),
    }
    # Attempt to average per-machine and per-operator utilizations across items
    try:
        # collect per-machine dicts
        per_machine_dicts = [item.get('per_machine_utilization') for item in envs if isinstance(item, dict) and item.get('per_machine_utilization')]
        # also include util dicts returned from env objects
        for item in envs:
            try:
                if hasattr(item, '_compute_utilization_summary') and callable(getattr(item, '_compute_utilization_summary')):
                    u = item._compute_utilization_summary()
                    if isinstance(u, dict) and 'per_machine_utilization' in u:
                        per_machine_dicts.append(u.get('per_machine_utilization'))
            except Exception:
                continue
    except Exception:
        per_machine_dicts = []

    try:
        per_operator_dicts = [item.get('per_operator_utilization') for item in envs if isinstance(item, dict) and item.get('per_operator_utilization')]
        for item in envs:
            try:
                if hasattr(item, '_compute_utilization_summary') and callable(getattr(item, '_compute_utilization_summary')):
                    u = item._compute_utilization_summary()
                    if isinstance(u, dict) and 'per_operator_utilization' in u:
                        per_operator_dicts.append(u.get('per_operator_utilization'))
            except Exception:
                continue
    except Exception:
        per_operator_dicts = []

    # Average per-id dictionaries (mean across dicts); keys aligned by union
    def _avg_dicts(dicts_list):
        if not dicts_list:
            return {}
        keys = set()
        for d in dicts_list:
            try:
                keys.update(d.keys())
            except Exception:
                continue
        out = {}
        for k in sorted(keys, key=lambda x: str(x)):
            vals = []
            for d in dicts_list:
                try:
                    if k in d:
                        vals.append(float(d.get(k, 0.0)))
                except Exception:
                    continue
            out[k] = float(np.mean(vals)) if vals else 0.0
        return out

    try:
        summary['per_machine_utilization'] = _avg_dicts(per_machine_dicts)
    except Exception:
        summary['per_machine_utilization'] = {}
    try:
        summary['per_operator_utilization'] = _avg_dicts(per_operator_dicts)
    except Exception:
        summary['per_operator_utilization'] = {}

    append_run_summary(summary, history_dir=os.path.join('my_data_and_graph', 'historydata'))
    return summary
