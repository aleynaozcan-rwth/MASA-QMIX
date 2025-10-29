# Runner & Gantt Format Notes

This document describes the updated Gantt record format and backward compatibility.

Gantt record shapes produced by `environment.gantt_records`:

- Legacy (5-tuple): (start, end, operation, wc, job_id)
- Legacy+opgrp (6-tuple): (start, end, operation, wc, job_id, operator_grp)
- Extended (8-tuple): (start, end, operation, wc, job_id, operator_grp, arrival_time, duration)

Runner and plotting compatibility:

- `MARL.runner.plot_gantt()` accepts 5-, 6- and 8-tuple records and will:
  - Color bars by `job_id` (consistent across snapshots).
  - Annotate each bar with `WCx | Oy` where `WCx` is the workcenter and `Oy` is the operator-group when available.
  - Legacy records are normalized on CSV export: missing `operator_grp`, `arrival`, or `duration` are emitted as empty cells.

CSV export format (header):

    start,end,op_idx,wc,job_id,operator_grp,arrival,duration

Notes:
- Initial jobs default to `arrival_time = 0.0` when created at reset().
- Dynamically added jobs via `MASAEnv.add_job()` set `job.arrival_time = env.now`.
- The Runner writers handle both legacy and extended shapes transparently.

If you want different coloring (by op type) or additional annotations (arrival markers), update `MARL/runner.py::plot_gantt()` accordingly.
