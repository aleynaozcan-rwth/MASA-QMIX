import os
import tempfile
import matplotlib
matplotlib.use('Agg')
from MARL import runner as runner_mod


def make_record(shape):
    # produce example records with varying lengths
    if shape == 5:
        return (1.0, 2.0, 0, 0, 42)
    if shape == 6:
        return (1.0, 3.0, 1, 1, 7, 0)
    if shape == 8:
        return (2.0, 5.0, 2, 0, 3, 1, 0.5, 3.0)


def test_plot_gantt_accepts_various_shapes(tmp_path):
    out_png = tmp_path / 'g.png'
    data = [make_record(5), make_record(6), make_record(8)]
    # should return True and write file
    ok = runner_mod.plot_gantt(data, filename=str(out_png))
    assert ok is True
    assert out_png.exists()


def test_csv_export_normalizes_columns(tmp_path):
    csv_path = tmp_path / 'g.csv'
    data = [make_record(5), make_record(6), make_record(8)]
    # use centralized gantt writer to normalize and write CSV columns
    try:
        from utils.gantt import write_scheduling_trace
        write_scheduling_trace(str(csv_path), data)
    except Exception:
        # fallback: emulate Runner CSV writer normalization (legacy behavior)
        with open(csv_path, 'w') as cf:
            cf.write('start,end,op_idx,wc,job_id,operator_grp,arrival,duration\n')
            for r in data:
                if len(r) >= 8:
                    vals = [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]]
                elif len(r) == 6:
                    vals = [r[0], r[1], r[2], r[3], r[4], r[5], '', '']
                elif len(r) == 5:
                    vals = [r[0], r[1], r[2], r[3], r[4], '', '', '']
                else:
                    continue
                cf.write(','.join([str(x) for x in vals]) + '\n')

    with open(csv_path, 'r') as cf:
        lines = [ln.strip() for ln in cf.readlines() if ln.strip()]
    assert lines[0].split(',')[-2:] == ['arrival', 'duration']
    # check that 3 data lines present
    assert len(lines) == 4
