"""Gantt helpers: formatting and simple CSV writer for gantt_records.

Each gantt record is expected as a tuple:
  (start, end, operation, wc, job_id, operator_grp, arrival_time, duration)

This module provides lightweight utilities used by tools and the env to
produce human-readable output and CSV exports for analysis.
"""
from typing import Iterable, List, Tuple
from pathlib import Path

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
        with open(path, 'a', encoding='utf-8') as sf:
            if header_needed:
                sf.write('time,job_id,allowed_wcs,avail_mask,chosen_machine_idx,chosen_machine_name,reason\n')
            sf.write(f"{sim_time},{job_id},{allowed_wcs},{avail_mask},{chosen_idx},{repr(chosen_name)},{reason}\n")
    except Exception:
        pass
