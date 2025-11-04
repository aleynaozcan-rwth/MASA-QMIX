from utils.gantt import format_gantt_records, gantt_records_to_csv


def test_format_gantt_records_basic():
    records = [
        (1.0, 2.0, 0, 0, 0, 0, 0.0, 1.0),
        (2.0, 3.5, 1, 0, 1, 0, 0.0, 1.5),
    ]
    s = format_gantt_records(records)
    assert '[GANTT]' in s
    assert 'op=0' in s


def test_gantt_csv_roundtrip():
    records = [
        (1.0, 2.0, 0, 0, 0, 0, 0.0, 1.0),
    ]
    csv = gantt_records_to_csv(records)
    assert csv.startswith('start,end,operation')
    assert '\n' in csv
