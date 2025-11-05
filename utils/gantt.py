"""Gantt helpers: formatting and simple CSV writer for gantt_records.

Each gantt record is expected as a tuple:
  (start, end, operation, wc, job_id, operator_grp, arrival_time, duration)

This module provides lightweight utilities used by tools and the env to
produce human-readable output and CSV exports for analysis.
"""
from typing import Iterable, List, Tuple
from pathlib import Path
try:
    from utils.io_control import allow_history_writes
except Exception:
    # defensive: if import fails, default to denying writes
    def allow_history_writes():
        return False

Record = Tuple[float, float, int, int, int, int, float, float]


def format_gantt_records(records: Iterable[Record], max_items: int = 10) -> str:
    """Return a short human-readable summary string for a list of gantt records."""
    lines: List[str] = []
    cnt = 0
    for r in records:
        if cnt >= max_items:
            break
        try:
            start, end, op, wc, job_id, op_grp, arrival, dur = r
            lines.append(f"({start:.2f},{end:.2f},op={op},wc={wc},job={job_id},grp={op_grp},arr={arrival:.2f},dur={dur:.2f})")
        except Exception:
            lines.append(str(r))
        cnt += 1
    if cnt < len(list(records)):
        lines.append("...")
    return "[GANTT] " + " ".join(lines)


def gantt_records_to_csv(records: Iterable[Record]) -> str:
    """Return CSV string for gantt records with header.

    Columns: start,end,operation,wc,job_id,operator_grp,arrival,duration
    """
    out_lines = ["start,end,operation,wc,job_id,operator_grp,arrival,duration"]
    for r in records:
        try:
            start, end, op, wc, job_id, op_grp, arrival, dur = r
            out_lines.append(f"{start},{end},{op},{wc},{job_id},{op_grp},{arrival},{dur}")
        except Exception:
            out_lines.append(",".join([str(x) for x in r]))
    return "\n".join(out_lines)


def write_gantt_csv(path: str, records: Iterable[Record]) -> None:
    """Write gantt records to a CSV file at `path`. Overwrites existing file."""
    if not allow_history_writes():
        # history writes are disabled; skip writing
        return
    csv = gantt_records_to_csv(records)
    with open(path, "w", encoding="utf-8") as f:
        f.write(csv)


def write_scheduling_trace(path: str, records: Iterable[Record]) -> None:
    """Write a scheduling_trace CSV compatible with Runner's historical output.

    Columns: start,end,op_idx,op_name,wc,job_id,operator_grp,arrival,duration
    This function overwrites any existing file at `path`.
    """
    out_lines = ["start,end,op_idx,op_name,wc,job_id,operator_grp,arrival,duration"]
    for r in records:
        try:
            if not isinstance(r, (list, tuple)):
                continue
            if len(r) >= 8:
                start, end, op_idx, wc, job_id, op_grp, arrival, duration = r[:8]
            elif len(r) == 6:
                start, end, op_idx, wc, job_id, op_grp = r
                arrival = ''
                duration = ''
            elif len(r) == 5:
                start, end, op_idx, wc, job_id = r
                op_grp = ''
                arrival = ''
                duration = ''
            else:
                continue
            try:
                if isinstance(op_idx, (int, float)) and float(op_idx).is_integer():
                    op_name = f"Op{int(op_idx) + 1}"
                else:
                    op_name = str(op_idx)
            except Exception:
                op_name = str(op_idx)
            vals = [start, end, op_idx, op_name, wc, job_id, op_grp, arrival, duration]
            out_lines.append(','.join([str(x) for x in vals]))
        except Exception:
            continue
    if not allow_history_writes():
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))


def write_job_timeline(path: str, records: Iterable[Record]) -> None:
    """Build and write per-job timeline CSV from gantt-like records.

    Output columns: job_id,arrival_time,operations_count,operations_json
    """
    try:
        jobs_map = {}
        for rec in records:
            try:
                if not isinstance(rec, (list, tuple)):
                    continue
                if len(rec) >= 8:
                    s, e, op_idx, wc, job_id, op_grp, arrival, duration = rec[:8]
                elif len(rec) == 6:
                    s, e, op_idx, wc, job_id, op_grp = rec
                    arrival = None
                    duration = None
                elif len(rec) == 5:
                    s, e, op_idx, wc, job_id = rec
                    op_grp = None
                    arrival = None
                    duration = None
                else:
                    continue
                jid = int(job_id)
                jobs_map.setdefault(jid, []).append({'start': float(s), 'end': float(e), 'op_idx': int(op_idx) if isinstance(op_idx, (int, float)) and float(op_idx).is_integer() else op_idx, 'wc': wc, 'op_grp': op_grp, 'arrival': arrival, 'duration': duration})
            except Exception:
                continue

        import json
        if not allow_history_writes():
            return
        with open(path, 'w', encoding='utf-8') as jf:
            jf.write('job_id,arrival_time,operations_count,operations_json\n')
            for jid in sorted(jobs_map.keys()):
                ops = sorted(jobs_map[jid], key=lambda x: x.get('start', 0.0))
                arrival = ops[0].get('arrival') if ops and ops[0].get('arrival') is not None else ''
                jf.write(f"{jid},{arrival},{len(ops)},{json.dumps(ops)}\n")
    except Exception:
        # best-effort: do not raise from utils writer
        return


def append_selection_log(path: str, sim_time, job_id, allowed_wcs, avail_mask, chosen_idx, chosen_name, reason):
    """Append a single selection event line to a scheduling_trace-like CSV.

    Columns: time,job_id,allowed_wcs,avail_mask,chosen_machine_idx,chosen_machine_name,reason
    Creates file with header if it does not exist.
    """
    try:
        header_needed = not Path(path).exists()
        if not allow_history_writes():
            return
        with open(path, 'a', encoding='utf-8') as sf:
            if header_needed:
                sf.write('time,job_id,allowed_wcs,avail_mask,chosen_machine_idx,chosen_machine_name,reason\n')
            sf.write(f"{sim_time},{job_id},{allowed_wcs},{avail_mask},{chosen_idx},{repr(chosen_name)},{reason}\n")
    except Exception:
        pass


def generate_scheduling_timeline(env, episode_id=None, episode_reward=None, write_if_allowed: bool = True, out_dir: str = None) -> str:
    """Generate a human-readable scheduling timeline from an in-memory MASAEnv.

    The output is plain text (no CSV/JSON) and contains:
      - INITIAL JOBS: a summary of jobs present at the earliest arrival time
      - TIMELINE: chronological events (job arrivals, operation starts and finishes)
      - SUMMARY: basic aggregated stats

    If `write_if_allowed` is True and the central IO gate allows history
    writes (`allow_history_writes()`), the report is also written to
    `my_data_and_graph/historydata/scheduling_timeline.txt` (creating the
    directory if necessary). The function always returns the string report.
    """
    try:
        from utils.io_control import allow_history_writes
    except Exception:
        def allow_history_writes():
            return False

    # Defensive: support both env object and dict-like surfaces
    try:
        jobs = list(getattr(env, 'jobs', []) or [])
        records = list(getattr(env, 'gantt_records', []) or [])
        sim_now = float(getattr(getattr(env, 'env', None), 'now', getattr(env, 't', 0.0)))
    except Exception:
        jobs = []
        records = []
        sim_now = 0.0

    # detect if records include an episode stamp as a trailing element
    has_ep_stamp = any(isinstance(r, (list, tuple)) and len(r) >= 9 and r[8] is not None for r in records)

    # If caller provided an explicit episode_id and records carry an
    # episode stamp, filter records to that episode for focused reporting.
    if episode_id is not None and has_ep_stamp:
        try:
            records = [r for r in records if isinstance(r, (list, tuple)) and len(r) >= 9 and int(r[8]) == int(episode_id)]
        except Exception:
            # fallback: leave records unchanged
            pass

    # Determine earliest arrival among jobs (initial set)
    arrivals = {}
    for j in jobs:
        try:
            at = float(getattr(j, 'arrival_time', 0.0))
        except Exception:
            at = 0.0
        arrivals.setdefault(at, []).append(j)
    earliest = min(arrivals.keys()) if arrivals else 0.0

    lines = []
    lines.append('=== INITIAL JOBS ===')
    initial_jobs = arrivals.get(earliest, [])
    if not initial_jobs:
        lines.append('(none)')
    else:
        for job in initial_jobs:
            try:
                jid = int(getattr(job, 'id', getattr(job, 'job_id', -1)))
                ops = getattr(job, 'operations', []) or []
                op_names = []
                eligible = {}
                for i, op in enumerate(ops):
                    try:
                        if isinstance(op, (list, tuple)) and len(op) >= 2:
                            op_type = int(op[0]) if isinstance(op[0], (int, float)) else None
                            allowed_wcs = list(op[1])
                        else:
                            op_type = None
                            allowed_wcs = []
                    except Exception:
                        op_type = None
                        allowed_wcs = []
                    opname = f"Op{(op_type + 1) if op_type is not None else i+1}"
                    op_names.append(opname)
                    eligible[opname] = list(allowed_wcs)
                lines.append(f"Job_{jid} → {len(ops)} ops: [{', '.join(op_names)}] | Eligible: {{{', '.join([f'{k}:{v}' for k,v in eligible.items()])}}}")
            except Exception:
                lines.append(str(job))

    # Build timeline events: job arrivals + op start/finish events from gantt records
    # Events: tuple (time, priority, text) where priority orders same-time events
    events = []

    # Job arrivals (priority 0)
    for atime, js in arrivals.items():
        for j in js:
            try:
                jid = int(getattr(j, 'id', getattr(j, 'job_id', -1)))
                ops = getattr(j, 'operations', []) or []
                op_names = [f"Op{(int(o[0]) + 1) if isinstance(o, (list, tuple)) and isinstance(o[0], (int, float)) else idx+1}" for idx, o in enumerate(ops)]
                text = f"[t={atime:.2f}] New job Job_{jid} arrived with {len(ops)} ops: [{', '.join(op_names)}]"
            except Exception:
                text = f"[t={atime:.2f}] New job arrived"
            events.append((float(atime), 0, text))

    # Map machine indices to names when possible
    mlist = getattr(getattr(env, 'workcenters_meta', None), 'machine_list', []) or []

    # Build a mapping of job -> op_name -> eligible machine indices (from job definitions)
    jobs_allowed = {}
    for j in jobs:
        try:
            jid = int(getattr(j, 'id', getattr(j, 'job_id', -1)))
            ops = getattr(j, 'operations', []) or []
            amap = {}
            for idx, op in enumerate(ops):
                try:
                    if isinstance(op, (list, tuple)) and len(op) >= 2:
                        allowed_wcs = list(op[1])
                    else:
                        allowed_wcs = []
                except Exception:
                    allowed_wcs = []
                opname = f"Op{(int(op[0]) + 1) if isinstance(op, (list, tuple)) and isinstance(op[0], (int, float)) else idx+1}"
                amap[opname] = list(allowed_wcs)
            jobs_allowed[jid] = amap
        except Exception:
            continue

    # Operation start and finish events from gantt records
    # For job completion detection, track counts per job
    job_op_seen = {}
    for rec in records:
        try:
            # rec layout in this module: (start, end, op, wc, job_id, op_grp, arrival, dur)
            start, end, op_idx, wc_idx, job_id, op_grp, arrival, dur = rec[:8]
        except Exception:
            # try other fallbacks
            try:
                start, end, op_idx, wc_idx, job_id = rec[:5]
                arrival = None; dur = None; op_grp = None
            except Exception:
                continue
        try:
            op_name = f"Op{int(op_idx) + 1}" if isinstance(op_idx, (int, float)) and float(op_idx).is_integer() else str(op_idx)
        except Exception:
            op_name = str(op_idx)
        try:
            machine_name = mlist[int(wc_idx)] if mlist and 0 <= int(wc_idx) < len(mlist) else f"M{int(wc_idx)}"
        except Exception:
            machine_name = str(wc_idx)

        # Prepare eligible machine names if available
        try:
            eligible_idxs = jobs_allowed.get(int(job_id), {}).get(op_name, [])
            eligible_names = [mlist[i] if mlist and 0 <= int(i) < len(mlist) else f"M{int(i)}" for i in eligible_idxs]
        except Exception:
            eligible_idxs = []
            eligible_names = []

        # start event (priority 1) — include operator id (op_grp), eligible machines
        # and a reason that reflects both machine and operator availability.
        try:
            if op_grp is None or op_grp == '':
                op_label = 'UNASSIGNED'
            else:
                op_label = str(op_grp)
        except Exception:
            op_label = 'UNASSIGNED'

        start_text = f"[t={start:.2f}] Job_{int(job_id)}.{op_name} started on {machine_name} by {op_label} (duration={float(dur) if dur is not None else (end - start):.2f}) | eligible={eligible_names}"

        # Determine machine/operator busy state at start time by scanning other records
        machine_busy = False
        operator_busy = False
        operator_unknown = op_label in (None, '', 'UNASSIGNED')
        try:
            for other in records:
                try:
                    os_ = float(other[0]); oe_ = float(other[1]); owc = other[3]
                except Exception:
                    continue
                # skip self-record equality checks by identity if possible
                try:
                    same = (other is rec)
                except Exception:
                    same = False
                if same:
                    continue
                # machine busy if another record overlaps start on same machine
                try:
                    if int(owc) == int(wc_idx) and os_ < float(start) and oe_ > float(start):
                        machine_busy = True
                except Exception:
                    pass
                # operator busy if operator identifier matches and overlaps start
                try:
                    other_op = other[5]
                    if (not operator_unknown) and other_op is not None and str(other_op) == op_label and float(other[0]) < float(start) and float(other[1]) > float(start):
                        operator_busy = True
                except Exception:
                    pass
        except Exception:
            pass

        if not machine_busy and not operator_busy:
            if operator_unknown:
                reason = "selected machine available (operator unknown)"
            else:
                reason = "selected machine and operator available"
        elif machine_busy and not operator_busy:
            reason = "waiting for available machine"
        elif not machine_busy and operator_busy:
            reason = "waiting for available operator"
        else:
            reason = "waiting for available machine and operator"

        start_text = start_text + f"\n  Reason: {reason}"
        events.append((float(start), 1, start_text))

        # finish event (priority 2) — always show verbatim operator label
        finish_label = op_label
        events.append((float(end), 2, f"[t={end:.2f}] Job_{int(job_id)}.{op_name} finished → next queued by {finish_label}"))

    # Sort events by time then priority
    events.sort(key=lambda x: (float(x[0]), int(x[1])))

    lines.append('\n=== TIMELINE ===')
    # Iterate events and detect job completions when we've seen all ops for a job
    # Build map of job total ops
    job_total_ops = {}
    for j in jobs:
        try:
            jid = int(getattr(j, 'id', getattr(j, 'job_id', -1)))
            job_total_ops[jid] = len(getattr(j, 'operations', []) or [])
        except Exception:
            continue

    # For completion detection, count finish events per job
    job_finished_count = {jid: 0 for jid in job_total_ops.keys()}

    for t, pri, text in events:
        lines.append(text)
        # if this is a finish event, try detect completion
        if ' finished ' in text or text.strip().endswith('finished → next queued'):
            # extract job id
            try:
                part = text.split(']')[-1].strip()
                # example: Job_0.Op1 finished → next queued
                if part.startswith('Job_'):
                    jid_str = part.split()[0].split('_')[1].split('.')[0]
                    jid = int(jid_str)
                else:
                    jid = None
            except Exception:
                jid = None
            if jid is not None and jid in job_finished_count:
                job_finished_count[jid] += 1
                if job_finished_count[jid] >= job_total_ops.get(jid, 0):
                    lines.append(f"[t={t:.2f}] Job_{jid} completed all operations")

    # SUMMARY
    total_jobs_generated = len(jobs)
    total_operations_executed = len(records)
    # use sim_now or max end time
    max_end = 0.0
    for rec in records:
        try:
            end = float(rec[1])
            if end > max_end:
                max_end = end
        except Exception:
            continue
    total_simpy_time = float(sim_now if sim_now and sim_now > 0 else max_end)
    try:
        avg_wait = float(getattr(env, 'total_wait_time', 0.0)) / max(1.0, float(getattr(env, 'completed_jobs', 0) or 1))
    except Exception:
        avg_wait = 0.0
    # average makespan: average of job last end times
    job_last_end = {}
    for rec in records:
        try:
            s, e, op_idx, wc_idx, job_id = rec[:5]
            jid = int(job_id)
            job_last_end[jid] = max(job_last_end.get(jid, 0.0), float(e))
        except Exception:
            continue
    avg_makespan = 0.0
    if job_last_end:
        avg_makespan = sum(job_last_end.values()) / max(1, len(job_last_end))

    lines.append('\n=== SUMMARY ===')
    lines.append(f"total_jobs_generated = {total_jobs_generated}")
    lines.append(f"total_operations_executed = {total_operations_executed}")
    lines.append(f"total_simpy_time = {total_simpy_time:.2f}")
    lines.append(f"average_wait_time = {avg_wait:.2f}")
    lines.append(f"average_makespan = {avg_makespan:.2f}")

    report = "\n".join(lines)

    # Optionally write to disk when the caller explicitly asked for it.
    # The authoritative `scheduling_timeline.txt` is still gated by the
    # central IO control (allow_history_writes()). However, when the
    # caller supplied `write_if_allowed=True` we make a best-effort to
    # produce a deterministic always-overwrite copy
    # `scheduling_timeline.latest.txt` so debug/inspection tools can
    # rely on a stable filename even when the central gate denies full
    # history writes (e.g., in CI/tests).
    if write_if_allowed:
        try:
            import os
            # prefer provided out_dir, otherwise default to project historydata
            out_dir = out_dir or os.path.join('my_data_and_graph', 'historydata')
            os.makedirs(out_dir, exist_ok=True)

            # Write the authoritative timeline only when history writes are allowed.
            if allow_history_writes():
                try:
                    out_path = os.path.join(out_dir, 'scheduling_timeline.txt')
                    # If records include per-record episode stamps and caller
                    # did not provide an episode_id, write grouped per-episode
                    # sections by invoking this generator for each episode.
                    if episode_id is None and has_ep_stamp:
                        try:
                            ep_ids = sorted({int(r[8]) for r in records if isinstance(r, (list, tuple)) and len(r) >= 9 and r[8] is not None})
                        except Exception:
                            ep_ids = []
                        combined_latest_parts = []
                        for i, ep in enumerate(ep_ids):
                            try:
                                # produce per-episode text without triggering writes
                                ep_report = generate_scheduling_timeline(env, episode_id=ep, episode_reward=None, write_if_allowed=False, out_dir=out_dir)
                            except Exception:
                                ep_report = ''
                            # Decide mode: overwrite if first episode and file missing or ep==0 requested as fresh
                            mode = 'a'
                            if (i == 0 and (episode_id == 0 or not os.path.exists(out_path))) or (ep == 0 and not os.path.exists(out_path)):
                                mode = 'w'
                            try:
                                with open(out_path, mode, encoding='utf-8') as f:
                                    f.write(f"=== EPISODE {ep} ===\n")
                                    f.write(ep_report)
                                    f.write("\n")
                                    # Append some conservative episode-level metadata
                                    try:
                                        f.write("--- EPISODE METADATA ---\n")
                                        f.write(f"episode_reward = {None}\n")
                                    except Exception:
                                        pass
                                    f.write(f"total_jobs_generated = {total_jobs_generated}\n")
                                    f.write(f"total_operations_executed = {total_operations_executed}\n")
                                    f.write(f"average_wait_time = {avg_wait:.2f}\n")
                                    f.write(f"average_makespan = {avg_makespan:.2f}\n")
                                    f.write("\n")
                            except Exception:
                                # best-effort: continue with other episodes
                                pass
                            # collect for the deterministic latest copy
                            try:
                                combined_latest_parts.append(f"=== EPISODE {ep} ===\n" + ep_report + "\n" + "--- EPISODE METADATA ---\n" + f"episode_reward = {None}\n" + f"total_jobs_generated = {total_jobs_generated}\n" + f"total_operations_executed = {total_operations_executed}\n" + f"average_wait_time = {avg_wait:.2f}\n" + f"average_makespan = {avg_makespan:.2f}\n")
                            except Exception:
                                pass
                        # replace report with combined for latest file write below
                        try:
                            report_for_latest = "\n".join(combined_latest_parts)
                        except Exception:
                            report_for_latest = report
                    else:
                        # legacy behavior: single-block write
                        mode = 'a'
                        if episode_id == 0 or not os.path.exists(out_path):
                            mode = 'w'
                        try:
                            with open(out_path, mode, encoding='utf-8') as f:
                                # Episode header
                                if episode_id is not None:
                                    f.write(f"=== EPISODE {episode_id} ===\n")
                                else:
                                    f.write("=== EPISODE (unknown) ===\n")
                                # Write the report (initial jobs, timeline, summary)
                                f.write(report)
                                f.write("\n")
                                # Append episode-level metadata
                                try:
                                    f.write("--- EPISODE METADATA ---\n")
                                    f.write(f"episode_reward = {episode_reward}\n")
                                except Exception:
                                    pass
                                f.write(f"total_jobs_generated = {total_jobs_generated}\n")
                                f.write(f"total_operations_executed = {total_operations_executed}\n")
                                f.write(f"average_wait_time = {avg_wait:.2f}\n")
                                f.write(f"average_makespan = {avg_makespan:.2f}\n")
                                f.write("\n")
                        except Exception:
                            # don't fail the outer writer when the authoritative write fails
                            pass
                except Exception:
                    # don't fail the outer writer when the authoritative write fails
                    pass

            # Always attempt to write the deterministic latest copy when the
            # caller requested writing. This is best-effort and will not
            # raise on failure.
            try:
                latest_path = os.path.join(out_dir, 'scheduling_timeline.latest.txt')
                with open(latest_path, 'w', encoding='utf-8') as lf:
                    if episode_id is not None:
                        lf.write(f"=== EPISODE {episode_id} ===\n")
                    else:
                        lf.write("=== EPISODE (unknown) ===\n")
                    lf.write(report)
                    lf.write("\n")
                    try:
                        lf.write("--- EPISODE METADATA ---\n")
                        lf.write(f"episode_reward = {episode_reward}\n")
                    except Exception:
                        pass
                    lf.write(f"total_jobs_generated = {total_jobs_generated}\n")
                    lf.write(f"total_operations_executed = {total_operations_executed}\n")
                    lf.write(f"average_wait_time = {avg_wait:.2f}\n")
                    lf.write(f"average_makespan = {avg_makespan:.2f}\n")
                    lf.write("\n")
            except Exception:
                # best-effort: do not fail main writer on errors writing latest copy
                pass
        except Exception:
            # do not fail on write errors; return report anyway
            pass

    return report
