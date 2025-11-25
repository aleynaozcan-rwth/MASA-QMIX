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
except Exception as e:
    # defensive: if import fails, default to denying writes
    def allow_history_writes():
        return False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
except Exception as e:
    plt = None
    mpatches = None
    cm = None

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
        except Exception as e:
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
        except Exception as e:
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
            # Handle dict records (from environment.py gantt_records)
            if isinstance(r, dict):
                start = r.get('start', '')
                end = r.get('end', '')
                op_idx = r.get('op_idx', '')
                wc = r.get('wc_idx', r.get('wc', ''))
                job_id = r.get('job_id', '')
                op_grp = r.get('op_grp', '')
                arrival = r.get('arrival', '')
                duration = r.get('duration', '')
            # Handle tuple/list records (legacy format)
            elif isinstance(r, (list, tuple)):
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
            else:
                continue
            try:
                if isinstance(op_idx, (int, float)) and float(op_idx).is_integer():
                    op_name = f"Op{int(op_idx) + 1}"
                else:
                    op_name = str(op_idx)
            except Exception as e:
                op_name = str(op_idx)
            vals = [start, end, op_idx, op_name, wc, job_id, op_grp, arrival, duration]
            out_lines.append(','.join([str(x) for x in vals]))
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
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
                # Handle dict records (from environment.py gantt_records)
                if isinstance(rec, dict):
                    s = rec.get('start', 0)
                    e = rec.get('end', 0)
                    op_idx = rec.get('op_idx', 0)
                    wc = rec.get('wc_idx', rec.get('wc', 0))
                    job_id = rec.get('job_id', 0)
                    op_grp = rec.get('op_grp', None)
                    arrival = rec.get('arrival', None)
                    duration = rec.get('duration', None)
                # Handle tuple/list records (legacy format)
                elif isinstance(rec, (list, tuple)):
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
                else:
                    continue
                jid = int(job_id)
                jobs_map.setdefault(jid, []).append({'start': float(s), 'end': float(e), 'op_idx': int(op_idx) if isinstance(op_idx, (int, float)) and float(op_idx).is_integer() else op_idx, 'wc': wc, 'op_grp': op_grp, 'arrival': arrival, 'duration': duration})
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
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
    except Exception as e:
        # best-effort: do not raise from utils writer
        return


def append_selection_log(path: str, sim_time, job_id, allowed_machine_indices, avail_actions, chosen_idx, chosen_name, reason):
    """Append a single selection event line to a scheduling_trace-like CSV.

    Columns: time,job_id,allowed_machine_indices,avail_actions,chosen_machine_idx,chosen_machine_name,reason
    Creates file with header if it does not exist.
    Note: `avail_actions` is the machine-major availability row (list-like).
    """
    try:
        header_needed = not Path(path).exists()
        if not allow_history_writes():
            return
        with open(path, 'a', encoding='utf-8') as sf:
            if header_needed:
                sf.write('time,job_id,allowed_machine_indices,avail_actions,chosen_machine_idx,chosen_machine_name,reason\n')
            sf.write(f"{sim_time},{job_id},{allowed_machine_indices},{avail_actions},{chosen_idx},{repr(chosen_name)},{reason}\n")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")


def generate_scheduling_timeline(env, episode_id=None, episode_reward=None, write_if_allowed: bool = True, out_dir: str = None, records=None, skip_header: bool = False) -> str:
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
    except Exception as e:
        def allow_history_writes():
            return False

    # Defensive: support both env object and dict-like surfaces
    try:
        jobs = list(getattr(env, 'jobs', []) or [])
        # Use caller-provided records when available; otherwise read from env
        if records is None:
            records = list(getattr(env, 'gantt_records', []) or [])
        else:
            # ensure we have a list copy so downstream filtering is safe
            try:
                records = list(records)
            except Exception as e:
                records = []
        # Normalize runtime wrapper-shaped records: some appenders wrap the
        # canonical per-op dict under {'record': {...}, 'episode': X, ...}.
        # Unwrap those so downstream logic can find top-level 'start'/'end'.
        try:
            normalized = []
            for r in records:
                try:
                    if isinstance(r, dict) and 'record' in r and isinstance(r.get('record'), dict):
                        inner = dict(r.get('record') or {})
                        # preserve episode stamp from wrapper if inner lacks it
                        try:
                            if inner.get('episode') is None and r.get('episode') is not None:
                                inner['episode'] = r.get('episode')
                        except Exception as e:
                            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                        normalized.append(inner)
                    else:
                        normalized.append(r)
                except Exception as e:
                    normalized.append(r)
            records = normalized
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
        sim_now = float(getattr(getattr(env, 'env', None), 'now', getattr(env, 't', 0.0)))
    except Exception as e:
        jobs = []
        records = []
        sim_now = 0.0

    # detect if records include an episode stamp as a trailing element
    has_ep_stamp = any(
        (isinstance(r, (list, tuple)) and len(r) >= 9 and r[8] is not None)
        or (isinstance(r, dict) and r.get('episode') is not None)
        for r in records
    )

    # If caller provided an explicit episode_id and records carry an
    # episode stamp, filter records to that episode for focused reporting.
    if episode_id is not None and has_ep_stamp:
        try:
            def _rec_ep(r):
                try:
                    if isinstance(r, (list, tuple)) and len(r) >= 9:
                        return int(r[8])
                    if isinstance(r, dict) and r.get('episode') is not None:
                        return int(r.get('episode'))
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                    return None
                return None

            records = [r for r in records if _rec_ep(r) is not None and int(_rec_ep(r)) == int(episode_id)]
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

    # Determine earliest arrival among jobs (initial set)
    arrivals = {}
    for j in jobs:
        try:
            at = float(getattr(j, 'arrival_time', 0.0))
        except Exception as e:
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
                for i, op in enumerate(ops):
                    try:
                        if isinstance(op, (list, tuple)) and len(op) >= 2:
                            op_type = int(op[0]) if isinstance(op[0], (int, float)) else None
                        else:
                            op_type = None
                    except Exception as e:
                        op_type = None
                    
                    if op_type is None:
                        op_type = i
                    
                    opname = f"Op{op_type + 1}"
                    op_names.append(opname)
                
                # Just show job ops without capability details (shown when operation actually starts)
                lines.append(f"Job_{jid} → {len(ops)} ops: [{', '.join(op_names)}]")
            except Exception as e:
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
            except Exception as e:
                text = f"[t={atime:.2f}] New job arrived"
            events.append((float(atime), 0, text))

    # Map machine indices to names when possible
    mlist = getattr(getattr(env, 'workcenters_meta', None), 'machine_list', []) or []
    machine_registry = getattr(getattr(env, 'workcenters_meta', None), 'machine_registry', {}) or {}

    # Build a mapping of job -> op_name -> capable machine NAMES (not WC IDs)
    # This uses machine_registry capabilities to get actual capable machines
    jobs_allowed = {}
    for j in jobs:
        try:
            jid = int(getattr(j, 'id', getattr(j, 'job_id', -1)))
            ops = getattr(j, 'operations', []) or []
            amap = {}
            for idx, op in enumerate(ops):
                try:
                    if isinstance(op, (list, tuple)) and len(op) >= 2:
                        op_type = int(op[0]) if isinstance(op[0], (int, float)) else idx
                    else:
                        op_type = idx
                except Exception as e:
                    op_type = idx
                
                opname = f"Op{op_type + 1}"
                
                # Find machines that can do this operation type (from capabilities)
                capable_machines = []
                for mname, mdata in machine_registry.items():
                    caps = mdata.get('capabilities', [])
                    if op_type in caps:
                        capable_machines.append(mname)
                
                amap[opname] = list(capable_machines)
            jobs_allowed[jid] = amap
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
            continue

    # Operation start and finish events from gantt records
    # For job completion detection, track counts per job
    job_op_seen = {}
    for rec in records:
        # Support both dict-style records (new) and tuple/list legacy records
        decision_trace = None
        try:
            if isinstance(rec, dict):
                start = float(rec.get('start', rec.get('s', 0.0)))
                end = float(rec.get('end', rec.get('e', 0.0)))
                op_idx = rec.get('op_idx', rec.get('op', None))
                wc_idx = rec.get('wc_idx', rec.get('wc', None))
                job_id = rec.get('job_id', rec.get('job_id', None))
                op_grp = rec.get('op_grp', rec.get('operator_grp', None))
                arrival = rec.get('arrival', None)
                dur = rec.get('duration', None)
                decision_trace = rec.get('decision_trace', None)
            else:
                try:
                    # rec layout in this module: (start, end, op, wc, job_id, op_grp, arrival, dur)
                    start, end, op_idx, wc_idx, job_id, op_grp, arrival, dur = rec[:8]
                except Exception as e:
                    # try other fallbacks
                    try:
                        start, end, op_idx, wc_idx, job_id = rec[:5]
                        arrival = None; dur = None; op_grp = None
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                        continue
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
            continue
        try:
            op_name = f"Op{int(op_idx) + 1}" if isinstance(op_idx, (int, float)) and float(op_idx).is_integer() else str(op_idx)
        except Exception as e:
            op_name = str(op_idx)
        try:
            machine_name = mlist[int(wc_idx)] if mlist and 0 <= int(wc_idx) < len(mlist) else f"M{int(wc_idx)}"
        except Exception as e:
            machine_name = str(wc_idx)

        # Prepare capable machine names from jobs_allowed (now contains machine names, not indices)
        try:
            capable_machines = jobs_allowed.get(int(job_id), {}).get(op_name, [])
            eligible_names = list(capable_machines)  # Already machine names like ['M0', 'M2', 'M3', 'M4']
        except Exception as e:
            eligible_names = []

        # start event (priority 1) — include operator id (op_grp), eligible machines
        # and a reason that reflects both machine and operator availability.
        try:
            if op_grp is None or op_grp == '':
                op_label = 'UNASSIGNED'
            else:
                op_label = str(op_grp)
        except Exception as e:
            op_label = 'UNASSIGNED'

        start_text = f"[t={start:.2f}] Job_{int(job_id)}.{op_name} started on {machine_name} by {op_label} (duration={float(dur) if dur is not None else (end - start):.2f}) | capable_machines={eligible_names}"


        # If the record contains a decision_trace created at decision time,
        # prefer that structured explanation. Otherwise fall back to the
        # historical heuristic computed from records.
        if decision_trace:
            try:
                # decision_trace is expected as a dict (from asdict)
                chosen_m = decision_trace.get('chosen_machine')
                chosen_o = decision_trace.get('chosen_operator')
                pol = decision_trace.get('policy_reason', '')
                elig = decision_trace.get('eligibilities', []) or []

                # build top-level phrase
                if chosen_o:
                    reason = f"selected {chosen_m} & {chosen_o} ({pol})"
                else:
                    reason = f"selected {chosen_m} ({pol})"

                # build other-machine explanations
                other_parts = []
                for e in elig:
                    try:
                        mid = str(e.get('machine_id'))
                        if mid == str(chosen_m):
                            continue
                        mstate = 'busy' if bool(e.get('machine_busy')) else 'free'
                        opid = e.get('operator_id')
                        qual = e.get('qualified')
                        opbusy = e.get('operator_busy')
                        op_av = e.get('operator_available_at')
                        if qual is False:
                            note = f"{mid} no qualified operator"
                        else:
                            if opid is None:
                                note = f"{mid} {mstate} / operator unknown"
                            else:
                                if opbusy is True:
                                    if op_av is not None:
                                        note = f"{mid} {mstate} / {opid} busy (avail@ t={float(op_av):.2f})"
                                    else:
                                        note = f"{mid} {mstate} / {opid} busy"
                                elif opbusy is False:
                                    note = f"{mid} {mstate} / {opid} free but not chosen"
                                else:
                                    note = f"{mid} {mstate} / {opid} unknown"
                        other_parts.append(note)
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                        continue
                if other_parts:
                    reason = reason + " | Other: " + "; ".join(other_parts)
                start_text = start_text + f"\n  Reason: {reason}"
            except Exception as e:
                # if anything goes wrong while decoding trace, fall back
                start_text = start_text + "\n  Reason: (decision_trace present but failed to render)"
        else:
            # Determine machine/operator busy state at start time by scanning other records
            machine_busy = False
            operator_busy = False
            operator_unknown = op_label in (None, '', 'UNASSIGNED')
            try:
                for other in records:
                    try:
                        os_ = float(other[0]); oe_ = float(other[1]); owc = other[3]
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                        continue
                    # skip self-record equality checks by identity if possible
                    try:
                        same = (other is rec)
                    except Exception as e:
                        same = False
                    if same:
                        continue
                    # machine busy if another record overlaps start on same machine
                    try:
                        if int(owc) == int(wc_idx) and os_ < float(start) and oe_ > float(start):
                            machine_busy = True
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                    # operator busy if operator identifier matches and overlaps start
                    try:
                        other_op = other[5]
                        if (not operator_unknown) and other_op is not None and str(other_op) == op_label and float(other[0]) < float(start) and float(other[1]) > float(start):
                            operator_busy = True
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

            # Build per-eligible-machine reasons to explain why other eligible
            # machines were or were not chosen. For each eligible machine, indicate
            # whether the machine was busy at the op start time and whether any
            # qualified operator appeared busy (when operator metadata is present).
            others_reasons = []
            try:
                env_obj = env
                # iterate over indices and friendly names together
                for midx, mname in zip(eligible_idxs, eligible_names):
                    try:
                        # determine if this machine was busy at `start`
                        m_busy = False
                        for other in records:
                            try:
                                os_ = float(other[0]); oe_ = float(other[1]); owc = other[3]
                            except Exception as e:
                                logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                                continue
                            try:
                                if int(owc) == int(midx) and os_ < float(start) and oe_ > float(start):
                                    m_busy = True
                                    break
                            except Exception as e:
                                logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                                continue

                        # determine operator busy state for operators qualified for this machine
                        op_busy_for_machine = False
                        try:
                            ops_mgr = getattr(env_obj, 'operators', None)
                            if ops_mgr is not None:
                                # inspect known operator objects and their historical records
                                for op_obj in getattr(ops_mgr, 'operators_object_list', []) or []:
                                    try:
                                        # check whether this operator is qualified for the machine
                                        qual_machines = getattr(op_obj, 'qualified_machines', []) or []
                                        # fall back to qualified_workcenters mapping if present
                                        if mname in qual_machines:
                                            # look for overlapping gantt records that used this operator
                                            for other in records:
                                                try:
                                                    other_op = other[5]
                                                    if str(other_op) == str(getattr(op_obj, 'operator_id', '')) and float(other[0]) < float(start) and float(other[1]) > float(start):
                                                        op_busy_for_machine = True
                                                        break
                                                except Exception as e:
                                                    logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                                                    continue
                                        if op_busy_for_machine:
                                            break
                                    except Exception as e:
                                        logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                                        continue
                        except Exception as e:
                            op_busy_for_machine = False

                        # compose short reason
                        try:
                            if m_busy:
                                txt = f"{mname} busy"
                            else:
                                txt = f"{mname} available"
                                if op_busy_for_machine:
                                    txt += " (operator busy)"
                        except Exception as e:
                            txt = str(mname)
                        others_reasons.append(txt)
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                        continue
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

            # top-level reason for the chosen machine/operator (preserve existing phrasing)
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

            # append per-eligible-machine details when we were able to compute them
            try:
                if others_reasons:
                    # avoid repeating the chosen machine in the 'others' list
                    filtered = [r for r in others_reasons if not r.startswith(machine_name)]
                    if filtered:
                        reason = reason + " | Other eligibilities: " + "; ".join(filtered)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

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
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
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
            except Exception as e:
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
            if isinstance(rec, dict):
                end = float(rec.get('end', rec.get('e', 0.0)))
            else:
                end = float(rec[1])
            if end > max_end:
                max_end = end
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
            continue
    total_simpy_time = float(sim_now if sim_now and sim_now > 0 else max_end)
    try:
        # Use canonical completed count; fail-fast if no completed jobs present
        total_wait = float(getattr(env, 'total_wait_time', 0.0))
        completed_count = sum(1 for j in getattr(env, 'jobs', []) if getattr(j, 'finished', False))
        if completed_count <= 0:
            raise ValueError("Cannot compute average wait time: no completed jobs available")
        avg_wait = total_wait / float(completed_count)
    except Exception as e:
        # Do not silently fallback; re-raise so callers become aware of missing data
        raise
    # average makespan: average of job last end times
    job_last_end = {}
    for rec in records:
        try:
            if isinstance(rec, dict):
                s = float(rec.get('start', rec.get('s', 0.0)))
                e = float(rec.get('end', rec.get('e', 0.0)))
                job_id = rec.get('job_id', rec.get('job_id', None))
            else:
                s, e, op_idx, wc_idx, job_id = rec[:5]
            jid = int(job_id)
            job_last_end[jid] = max(job_last_end.get(jid, 0.0), float(e))
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
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
    # The authoritative `scheduling_timeline.txt` is gated by the central
    # IO control (`allow_history_writes()`). To ensure the timeline file is
    # meaningful and only created when runtime data exists, we only write
    # the authoritative timeline when there are gantt records present for
    # the requested episode(s). The legacy deterministic `.latest` copy has
    # been removed — we no longer write or reference any `scheduling_timeline.latest.txt`.
    if write_if_allowed:
        try:
            import os
            # prefer provided out_dir, otherwise default to project historydata
            out_dir = out_dir or os.path.join('my_data_and_graph', 'historydata')
            os.makedirs(out_dir, exist_ok=True)
            # Ensure any legacy `.latest` snapshot is removed so only the
            # authoritative `scheduling_timeline.txt` remains. We no longer
            # produce `.latest` files; remove any old copies left on disk.
            try:
                latest_path = os.path.join(out_dir, 'scheduling_timeline.latest.txt')
                if os.path.exists(latest_path):
                    try:
                        os.remove(latest_path)
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

            # Only write the authoritative timeline when history writes are allowed
            # and when there are gantt records to report. This prevents creating
            # placeholder timeline files before simulation data exists.
            if allow_history_writes():
                try:
                    out_path = os.path.join(out_dir, 'scheduling_timeline.txt')

                    # If records include per-record episode stamps and caller
                    # did not provide an episode_id, write grouped per-episode
                    # sections by invoking this generator for each episode. Only
                    # write an episode block if there are records for that episode.
                    if episode_id is None and has_ep_stamp:
                        try:
                            # Robustly collect episode ids from both legacy tuple/list
                            # shaped records (episode at index 8) and dict-shaped
                            # records (episode under 'episode' key).
                            ep_set = set()
                            for r in records:
                                try:
                                    if isinstance(r, (list, tuple)) and len(r) >= 9 and r[8] is not None:
                                        ep_set.add(int(r[8]))
                                    elif isinstance(r, dict) and r.get('episode') is not None:
                                        ep_set.add(int(r.get('episode')))
                                except Exception as e:
                                    logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                                    continue
                            ep_ids = sorted(ep_set)
                        except Exception as e:
                            ep_ids = []
                        for i, ep in enumerate(ep_ids):
                            try:
                                # produce per-episode text without triggering writes
                                ep_report = generate_scheduling_timeline(env, episode_id=ep, episode_reward=None, write_if_allowed=False, out_dir=out_dir)
                            except Exception as e:
                                ep_report = ''
                            # Skip writing empty episode reports
                            if not ep_report or ep_report.strip() == '':
                                continue
                            # Only create (w) the file if it does not yet exist.
                            # Avoid overwriting existing files — doing so can
                            # remove lifecycle START blocks that may have been
                            # written by the environment prior to timeline
                            # generation. Append to existing files instead.
                            mode = 'a'
                            if not os.path.exists(out_path):
                                mode = 'w'
                            try:
                                # Instrumentation: log a compact debug record indicating
                                # we're about to write an episode block so we can
                                # correlate ordering against env lifecycle writes.
                                try:
                                    from datetime import datetime
                                    dbg_path = os.path.join(out_dir, 'lifecycle_debug_log.txt')
                                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    dbg_line = f"[DEBUG] {ts} - WRITE episode block ep={ep}\n"
                                    with open(dbg_path, 'a', encoding='utf-8') as df:
                                        df.write(dbg_line)
                                    try:
                                        print(dbg_line.strip())
                                    except Exception as e:
                                        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                                except Exception as e:
                                    logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

                                # Write an EPISODE header immediately before the
                                # per-episode report unless the caller asked to
                                # skip it. Some callers (for example when the
                                # lifecycle END footer has already been written
                                # immediately prior) want to avoid duplicating the
                                # episode header.
                                header = f"=== EPISODE {ep} ===\n"
                                with open(out_path, mode, encoding='utf-8') as f:
                                    if not skip_header:
                                        f.write(header)
                                    f.write(ep_report)
                                    f.write("\n")
                                    # Append conservative episode-level metadata
                                    try:
                                        f.write("--- EPISODE METADATA ---\n")
                                        f.write(f"episode_reward = {None}\n")
                                    except Exception as e:
                                        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                                    f.write(f"total_jobs_generated = {total_jobs_generated}\n")
                                    f.write(f"total_operations_executed = {total_operations_executed}\n")
                                    f.write(f"average_wait_time = {avg_wait:.2f}\n")
                                    f.write(f"average_makespan = {avg_makespan:.2f}\n")
                                    f.write("\n")
                            except Exception as e:
                                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                    else:
                        # Single-block write for the provided episode_id or unknown
                        # episode – only write if there are records present.
                        if not records:
                            # no runtime records -> skip writing to avoid placeholders
                            pass
                        else:
                            try:
                                # Instrumentation: record we're writing the single-block episode header
                                try:
                                    from datetime import datetime
                                    dbg_path = os.path.join(out_dir, 'lifecycle_debug_log.txt')
                                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    dbg_ep = episode_id if episode_id is not None else 'unknown'
                                    dbg_line = f"[DEBUG] {ts} - WRITE episode block ep={dbg_ep}\n"
                                    with open(dbg_path, 'a', encoding='utf-8') as df:
                                        df.write(dbg_line)
                                    try:
                                        print(dbg_line.strip())
                                    except Exception as e:
                                        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                                except Exception as e:
                                    logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

                                # Only open in write mode if the authoritative
                                # timeline file does not yet exist. If it exists,
                                # append to preserve any lifecycle START/END
                                # blocks that were written earlier.
                                mode = 'a'
                                if not os.path.exists(out_path):
                                    mode = 'w'

                                # Write an EPISODE header immediately before the
                                # report unless the caller requested it to be
                                # skipped (skip_header=True). This avoids
                                # redundant headers when the lifecycle writer has
                                # already placed the episode marker.
                                header = f"=== EPISODE {episode_id} ===\n" if episode_id is not None else "=== EPISODE (unknown) ===\n"
                                with open(out_path, mode, encoding='utf-8') as f:
                                    if not skip_header:
                                        f.write(header)
                                    # Write the report (initial jobs, timeline, summary)
                                    f.write(report)
                                    f.write("\n")
                                    # Append episode-level metadata
                                    try:
                                        f.write("--- EPISODE METADATA ---\n")
                                        f.write(f"episode_reward = {episode_reward}\n")
                                    except Exception as e:
                                        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                                    f.write(f"total_jobs_generated = {total_jobs_generated}\n")
                                    f.write(f"total_operations_executed = {total_operations_executed}\n")
                                    f.write(f"average_wait_time = {avg_wait:.2f}\n")
                                    f.write(f"average_makespan = {avg_makespan:.2f}\n")
                                    f.write("\n")
                            except Exception as e:
                                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

    return report


def plot_gantt_image(records: Iterable[Record], path: str, by: str = 'machine', title: str = None, save_if_allowed: bool = True) -> bool:
    """Render a wide, readable Gantt chart from `records` and save to `path`.

    Parameters
    - records: iterable of gantt Record tuples or dicts (see module Record).
    - path: filesystem path to save the PNG image.
    - by: 'machine' (default) to layout y-axis by machine/workcenter index,
          or 'job' to layout by job id.
    - title: optional title string for the chart.
    - save_if_allowed: when True, only write the file when
          `allow_history_writes()` permits it (defensive default).

    Returns True on successful save, False otherwise.
    """
    # Defensive: require matplotlib available
    if plt is None:
        return False

    try:
        recs = list(records or [])
    except Exception as e:
        recs = []

    # Respect central IO gate if requested
    if save_if_allowed and not allow_history_writes():
        # do not create files when history writes are disabled
        return False

    # Normalize records to tuples: (start,end,op,wc,job_id,op_grp,...)
    normalized = []
    for r in recs:
        try:
            if isinstance(r, dict):
                s = float(r.get('start', r.get('s', 0.0)))
                e = float(r.get('end', r.get('e', 0.0)))
                op = r.get('op_idx', r.get('op', None))
                wc = r.get('wc_idx', r.get('wc', None))
                job = r.get('job_id', r.get('job_id', None))
            else:
                s, e, op, wc, job = (r[0], r[1], r[2] if len(r) > 2 else None, r[3] if len(r) > 3 else None, r[4] if len(r) > 4 else None)
            normalized.append((float(s), float(e), op, wc, job))
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
            continue

    # Group keys and map to y positions
    if by == 'job':
        keys = sorted({int(k[4]) for k in normalized if k[4] is not None})
        key_name = lambda k: f"Job_{k}"
    else:
        # default: machine / workcenter index
        keys = sorted({int(k[3]) for k in normalized if k[3] is not None})
        key_name = lambda k: f"M{k}"

    if not keys:
        # ensure at least a single row so labels render
        keys = [0]

    # map key -> y index (top-down visually)
    keys = list(keys)
    keys.sort()
    y_map = {k: i for i, k in enumerate(keys[::-1])}  # reversed so first key appears at top

    # Color map for operations
    unique_ops = sorted({str(int(r[2])) if isinstance(r[2], (int, float)) and float(r[2]).is_integer() else str(r[2]) for r in normalized})
    color_map = {}
    cmap = cm.get_cmap('tab20') if cm is not None else None
    for i, op in enumerate(unique_ops):
        color_map[op] = cmap(i % 20) if cmap is not None else None

    # Build the figure with a wide/tall panoramic aspect for readability
    fig, ax = plt.subplots(figsize=(32, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot each record as a horizontal bar and ensure legend handles exist
    try:
        plotted_ops = set()
        for s, e, op, wc, job in normalized:
            try:
                key = int(job) if by == 'job' and job is not None else int(wc) if wc is not None else 0
            except Exception as e:
                key = 0
            y = y_map.get(key, 0)
            # make bar height adapt to number of rows so bars stay readable
            height = 0.8
            start = float(s)
            width = max(0.0, float(e) - float(s))
            op_label = str(int(op)) if isinstance(op, (int, float)) and float(op).is_integer() else str(op)
            color = color_map.get(op_label, None) or 'tab:blue'
            # Add a label only for the first drawn bar of each operation type so
            # legend handles are created without duplicating entries.
            lab = None
            if op_label not in plotted_ops:
                lab = f"Op {op_label}"
                plotted_ops.add(op_label)
            try:
                ax.broken_barh([(start, width)], (y - height/2.0, height), facecolors=color, edgecolor='black', linewidth=0.4, label=lab)
            except Exception as e:
                # Some backends/versions may not accept label on broken_barh; fall back
                col = color
                rect = ax.broken_barh([(start, width)], (y - height/2.0, height), facecolors=col, edgecolor='black', linewidth=0.4)
                try:
                    if lab is not None:
                        rect.set_label(lab)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

    # Y ticks and labels
    y_positions = [y_map[k] for k in keys]
    y_labels = [key_name(k) for k in keys]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12)

    # X label and Y label with slightly larger fonts
    ax.set_xlabel('Simulation Time (s)', fontsize=14)
    ax.set_ylabel('Machine / Job ID', fontsize=14)

    # Title
    if title:
        ax.set_title(title, fontsize=16)

    # Build and show legend: prefer handles created during plotting; if none,
    # create a fallback legend from the op-type color map so the legend is
    # always visible and centered vertically to the right of the chart.
    try:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
                      frameon=True, fontsize=10, title='Operation Types')
        else:
            try:
                print("⚠️ No legend handles found for this chart (creating fallback).")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
            # create fallback patches
            try:
                patches = [mpatches.Patch(color=color_map.get(op, 'tab:blue'), label=f"Op {op}") for op in unique_ops]
                if patches:
                    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
                              frameon=True, fontsize=10, title='Operation Types')
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

    # Expand x-axis slightly beyond the last operation so labels aren't clipped
    try:
        max_end_time = 0.0
        for _, e, _, _, _ in normalized:
            try:
                if float(e) > max_end_time:
                    max_end_time = float(e)
            except Exception as e:
                logging.getLogger(__name__).warning(f"[C1] Exception in gantt: {e}")
                continue
        if max_end_time and max_end_time > 0:
            ax.set_xlim(0, max_end_time * 1.05)
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")

    # Adjust figure proportions to a panoramic layout and reserve room for the
    # vertical legend on the right. This makes the plot fill the figure width
    # while keeping the legend visible and centered.
    try:
        fig.set_size_inches(40, 10)
    except Exception as e:
        logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
    plt.subplots_adjust(left=0.05, right=0.80, top=0.93, bottom=0.12)

    # Ensure saved figure has tight bounding box and a small padding so the
    # chart fills the image but the legend remains visible.
    try:
        # create parent dir if missing
        pdir = Path(path).parent
        pdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches='tight', pad_inches=0.3, dpi=300)
        plt.close(fig)
        return True
    except Exception as e:
        try:
            plt.close(fig)
        except Exception as e:
            logging.getLogger(__name__).warning(f"[C1] Exception: {e}")
        return False
