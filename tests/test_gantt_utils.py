import json
from pathlib import Path

from utils.gantt import write_scheduling_trace, write_job_timeline, append_selection_log


def test_write_scheduling_trace_happy_and_empty(tmp_path):
    csv_path = tmp_path / "sched.csv"
    # three records with varying shapes: 8-tuple, 6-tuple, 5-tuple
    records = [
        (1.0, 2.0, 0, 0, 101, 0, 0.0, 1.0),
        (2.0, 3.0, 1, 1, 102, 1),
        (3.0, 4.0, 2, 2, 103),
    ]

    write_scheduling_trace(str(csv_path), records)
    text = csv_path.read_text(encoding='utf-8')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # header + 3 data lines
    assert lines[0].strip() == 'start,end,op_idx,op_name,wc,job_id,operator_grp,arrival,duration'
    assert len(lines) == 1 + 3

    # edge case: empty records -> file contains only header
    csv_empty = tmp_path / "sched_empty.csv"
    write_scheduling_trace(str(csv_empty), [])
    e_text = csv_empty.read_text(encoding='utf-8')
    e_lines = [ln for ln in e_text.splitlines() if ln.strip()]
    assert e_lines[0].startswith('start,end,op_idx')
    assert len(e_lines) == 1


def test_write_job_timeline_and_counts(tmp_path):
    csv_path = tmp_path / "jt.csv"
    # build sample records: job 1 has two ops, job 2 has one op
    records = [
        (0.0, 1.0, 0, 0, 1, 0, 0.0, 1.0),
        (1.0, 2.5, 1, 1, 1, 1, 0.0, 1.5),
        (0.5, 3.0, 0, 2, 2, 0, 0.0, 2.5),
    ]

    write_job_timeline(str(csv_path), records)
    text = csv_path.read_text(encoding='utf-8')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0].strip() == 'job_id,arrival_time,operations_count,operations_json'
    # two jobs -> two data lines
    assert len(lines) == 1 + 2

    # verify operations_count matches expected counts
    # split only into 4 parts to preserve the JSON payload
    parts_job1 = lines[1].split(',', 3)
    assert parts_job1[0] == '1'
    assert parts_job1[2] == '2'
    ops_json = json.loads(parts_job1[3])
    assert isinstance(ops_json, list) and len(ops_json) == 2


def test_append_selection_log_appends(tmp_path):
    log_path = tmp_path / 'sel.csv'
    # first append creates file and header
    append_selection_log(str(log_path), 0.1, 10, [0, 1], [1, 0], 0, 'M0', 'ok')
    assert log_path.exists()
    lines = [ln for ln in log_path.read_text(encoding='utf-8').splitlines() if ln.strip()]
    assert lines[0].startswith('time,job_id,allowed_wcs')
    assert len(lines) == 2

    # second append adds another data line
    append_selection_log(str(log_path), 0.2, 11, [0], [0, 1], 1, 'M1', 'fallback')
    lines2 = [ln for ln in log_path.read_text(encoding='utf-8').splitlines() if ln.strip()]
    assert len(lines2) == 3
    # quick sanity check that the second data line contains job_id 11
    assert ',11,' in lines2[2] or lines2[2].split(',')[1] == '11'
