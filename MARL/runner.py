# ...existing code...
import os
# Ensure headless Qt / matplotlib backend before any pyplot import (avoid Wayland/Qt plugin errors)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib
matplotlib.use("Agg")

import sys
import time
import logging
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import Patch
import re
# Line2D was previously used for arrival legend; no longer needed

# Rollout
from MARL.common.rollout import RolloutWorker
try:
    from MARL.common.rollout import CommRolloutWorker  # type: ignore
except Exception as e:
    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
    CommRolloutWorker = RolloutWorker

try:
    from MARL.common.mask_utils import build_machine_major_mask
except Exception:
    build_machine_major_mask = None

# Agents / Buffer
from MARL.agent.agent import Agents, CommAgents
from MARL.common.replay_buffer import ReplayBuffer
from MARL.common.terms import t  # unified terminology helper
# gantt helpers
try:
    from utils import gantt as gantt_utils
except Exception as e:
    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
    gantt_utils = None


def plot_gantt(for_gantt_data, filename="gantt.png"):
    """Step 8A.6.6 Gantt – Machine-level layout with WC + Operator labels."""
    if not for_gantt_data or not isinstance(for_gantt_data, (list, tuple)):
        return False
    # allow empty; handle gracefully
    if len(for_gantt_data) == 0:
        return False

    # allow legacy 5/6-tuple records and the new 8-tuple format
    # default size; may be increased vertically for many jobs below
    # Use a wider/taller default so the gantt fills the page better
    fig, ax = plt.subplots(figsize=(18, 9))
    try:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    except Exception:
        pass
    # choose color palette keyed by operation type for readability
    # collect unique operation identifiers (int or str) and job ids
    op_vals = []
    job_ids = []
    for rec in for_gantt_data:
        if not isinstance(rec, (list, tuple)):
            continue
        if len(rec) >= 5:
            try:
                op_vals.append(rec[2])
                job_ids.append(int(rec[4]))
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                continue
    op_vals = sorted(list({int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v for v in op_vals}))
    job_ids = sorted(list(set(job_ids)))
    # colors per operation type
    palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    op_color_map = {}
    for i, op in enumerate(op_vals):
        op_color_map[op] = palette[i % len(palette)]

    max_end = 0
    plotted_ops = []
    for rec in for_gantt_data:
        # robust unpack: support (s,e,op,wc,job[,op_grp]) and extended (s,e,op,wc,job,op_grp,arrival,duration)
        try:
            if not isinstance(rec, (list, tuple)):
                continue
            if len(rec) >= 8:
                start, end, operation, workcenter, jobagent, operator, arrival, duration = rec[:8]
            elif len(rec) == 6:
                start, end, operation, workcenter, jobagent, operator = rec
                arrival = None
                duration = None
            elif len(rec) == 5:
                start, end, operation, workcenter, jobagent = rec
                operator = None
                arrival = None
                duration = None
            else:
                continue
            jid = int(jobagent)
            # color by operation type
            op_key = operation
            try:
                # if numeric operation index, convert to integer for mapping
                if isinstance(operation, (float, int)) and float(operation).is_integer():
                    op_key = int(operation)
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass
            color = op_color_map.get(op_key, "gray")
            # place bars as full-unit rows with bottom at job id so center is job_id+0.5
            width = max(0.0, float(end) - float(start))
            # draw vertical start/end guide lines (subtle)
            try:
                ax.axvline(float(start), linestyle=':', alpha=0.35, color='gray', zorder=1)
                ax.axvline(float(end), linestyle=':', alpha=0.25, color='gray', zorder=1)
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass
            # choose bar style: default linewidth/alpha and edge; narrow bars get slightly thicker edge
            bar_kwargs = dict(left=float(start), color=color, edgecolor='black', linewidth=0.5, alpha=0.95)
            if width < 0.6:
                bar_kwargs.update(dict(linewidth=0.6))
            # draw bar with align='edge' so it occupies approximately the job row
            # reduce height so bars are slightly smaller than the full ID interval
            ax.barh(jid, width, align='edge', height=0.7, zorder=3, **bar_kwargs)
            # label text simplified: M<machine_id> | O<operator_id>
            label_text = f"M{workcenter} | O{operator if operator is not None else '?'}"
            # Adaptive font size: skip or minimal label for very narrow bars
            try:
                # width computed above
                pass
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                width = 0.0
            # choose fontsize by thresholds
            # label visibility & size rules per user request
            if width < 0.5:
                draw_label = False
            else:
                draw_label = True
                fontsize = 7 if width < 0.8 else 9

            if draw_label:
                ax.text((float(start) + float(end)) / 2, jid + 0.5, label_text,
                        va='center', ha='center', fontsize=fontsize, color='black', zorder=4)
            max_end = max(max_end, float(end))
            plotted_ops.append(op_key)
            # arrival timestamps are kept for CSV only; no on-plot markers
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            continue

    jobagent_ids = job_ids
    # set y-ticks to job ids (centered at job_id + 0.5) and increase spacing if many jobs
    if job_ids:
        try:
            min_j = min(job_ids)
            max_j = max(job_ids)
            y_centers = [j + 0.5 for j in range(min_j, max_j + 1)]
            ax.set_yticks(y_centers)
            ax.set_yticklabels([str(j) for j in range(min_j, max_j + 1)])
            # add a small vertical margin so bars don't touch plot edges
            ax.set_ylim(min_j - 0.1, max_j + 1 + 0.1)
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            y_centers = [j + 0.5 for j in jobagent_ids]
            ax.set_yticks(y_centers)
            ax.set_yticklabels([str(j) for j in jobagent_ids])
    ax.set_xlabel("Simulation Time (SimPy clock)")
    ax.set_ylabel(f"{t('JobAgent')} (ID)")
    ax.set_title("Step 8A.6.6 – Machine-level Schedule (WC + Operator + Dynamic Arrivals)")
    ax.set_xlim(0, max_end + 1)
    # build legend keyed by operation types (Op1..OpN) — use Patch handles
    op_handles = []
    op_labels = []
    for op in op_vals:
        lab = f"Op{int(op)+1}" if isinstance(op, (int, float)) and float(op).is_integer() else str(op)
        # ensure legend patch matches bar edge/linewidth
        op_handles.append(Patch(facecolor=op_color_map.get(op, 'gray'), edgecolor='black', linewidth=0.5, label=lab))
        op_labels.append(lab)
    if op_handles:
        # place legend slightly further right (still inside figure) and center vertically
        legend_ops = ax.legend(op_handles, op_labels, title="Operation Types", bbox_to_anchor=(1.25, 0.5), loc="center left", borderaxespad=0, frameon=True, fontsize=10, title_fontsize=10)
        ax.add_artist(legend_ops)

    # arrival markers removed by user request; arrivals remain in CSV only

    # x-axis ticks: major=1.0, minor=0.5 with grid styling
    try:
        major = MultipleLocator(1.0)
        minor = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.tick_params(axis='x', which='major', labelsize=9)
        ax.grid(True, which='major', linestyle='--', alpha=0.35)
        ax.grid(True, which='minor', linestyle='--', alpha=0.15)
    except Exception as e:
        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
        pass
    # adjust margins: reserve space on the right for the legend and remove excess borders
    try:
        # let tight_layout compute internal spacing then reserve a wider right
        # margin so the legend (at x=1.25 in axes coords) remains inside the
        # final figure. This reduces large empty borders while ensuring the
        # legend is visible.
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.72, top=0.95, bottom=0.08)
    except Exception:
        try:
            plt.tight_layout()
        except Exception:
            pass
    # Save with high resolution and trim extra whitespace
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    except Exception:
        try:
            plt.savefig(filename, dpi=150)
        except Exception:
            pass
    plt.close()
    return True


def plot_gantt_last_evolution(for_gantt_data, filename="gantt_last_evolution.png"):
    """Create a clean job-level Gantt for the last episode of an evolution.

    - Each job appears once on the Y axis as `Job_<id>`.
    - Each bar is a single operation of that job (deduplicated by (job,op)).
    - If multiple records exist for the same (job,op), keep the one with
      the largest duration.
    - Bars are colored by operation type with a legend on the right.
    - Saves PNG to `filename`. Returns True on success, False if no data.
    """
    logger = logging.getLogger(__name__)

    # prepare debug path (only two files allowed: png and debug txt)
    try:
        debug_dir = os.path.dirname(filename) or '.'
        debug_path = os.path.join(debug_dir, 'gantt_last_evolution.debug.txt')
    except Exception:
        debug_path = 'gantt_last_evolution.debug.txt'

    def _append_debug(**kwargs):
        try:
            with open(debug_path, 'a') as df:
                for k, v in kwargs.items():
                    df.write(f"{k}={v}\n")
        except Exception:
            pass

    # incoming count
    try:
        incoming_count = len(for_gantt_data) if isinstance(for_gantt_data, (list, tuple)) else 0
    except Exception:
        incoming_count = 0
    _append_debug(incoming_count=incoming_count)

    if not for_gantt_data or not isinstance(for_gantt_data, (list, tuple)) or incoming_count == 0:
        logger.info("No gantt records for last evolution; skipping gantt_last_evolution plot.")
        _append_debug(status='no_records_for_gantt')
        return False

    # unpack helper (support dict or tuple/list records)
    def _unpack(rec):
        if isinstance(rec, dict):
            s = rec.get('start') if 'start' in rec else rec.get('s') if 's' in rec else rec.get(0)
            e = rec.get('end') if 'end' in rec else rec.get('e') if 'e' in rec else rec.get(1)
            op_idx = rec.get('op_idx') if 'op_idx' in rec else rec.get('op') if 'op' in rec else rec.get(2)
            # support multiple possible keys for workcenter/machine index
            wc = None
            if 'wc' in rec:
                wc = rec.get('wc')
            elif 'wc_idx' in rec:
                wc = rec.get('wc_idx')
            elif 'workcenter' in rec:
                wc = rec.get('workcenter')
            elif 'machine' in rec:
                wc = rec.get('machine')
            else:
                wc = rec.get(3)
            job_id = rec.get('job_id') if 'job_id' in rec else rec.get('job') if 'job' in rec else rec.get(4)
            # operator/group may be named differently across envs; support several keys
            operator = None
            if 'operator' in rec:
                operator = rec.get('operator')
            elif 'op_agent' in rec:
                operator = rec.get('op_agent')
            elif 'op_grp' in rec:
                operator = rec.get('op_grp')
            else:
                operator = rec.get(5)
            arrival = rec.get('arrival') if 'arrival' in rec else None
            duration = rec.get('duration') if 'duration' in rec else None
            return s, e, op_idx, wc, job_id, operator, arrival, duration
        if not isinstance(rec, (list, tuple)):
            raise ValueError('invalid rec')
        if len(rec) >= 8:
            return rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7]
        if len(rec) == 6:
            return rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], None, None
        if len(rec) == 5:
            return rec[0], rec[1], rec[2], rec[3], rec[4], None, None, None
        raise ValueError('unsupported rec len')

    # Deduplicate by (job_id, op_idx) keeping longest duration, and only keep records with assigned operator
    best = {}
    min_start = float('inf')
    max_end = 0.0
    machines = set()
    operators = set()
    op_types_seen = []

    for rec in for_gantt_data:
        try:
            s, e, op_idx, wc, job_id, operator, arrival, duration = _unpack(rec)
        except Exception:
            continue
        # If operator not provided at top-level, try to extract from nested
        # decision trace (many records include chosen_operator inside
        # `decision_trace`). If still missing, mark as UNASSIGNED and
        # continue — we prefer to include operations even when operator
        # wasn't assigned so the Gantt reflects actual scheduling.
        if operator is None:
            # try nested decision_trace when record is a dict
            try:
                if isinstance(rec, dict):
                    dt = rec.get('decision_trace') or rec.get('decision') or {}
                    if isinstance(dt, dict):
                        operator = dt.get('chosen_operator') or dt.get('chosen_op') or operator
            except Exception:
                operator = operator
        # Accept explicit 'UNASSIGNED' marker as valid label; if still None,
        # set to a readable placeholder so the record is not dropped.
        if operator is None:
            operator = 'UNASSIGNED'
        # normalize machine id
        try:
            mid = int(wc)
        except Exception:
            try:
                m = re.search(r"(\d+)", str(wc))
                mid = int(m.group(1)) if m else None
            except Exception:
                mid = None
        if mid is None:
            continue
        machines.add(mid)
        # normalize operator id
        try:
            op_id = str(operator)
        except Exception:
            op_id = str(operator)
        operators.add(op_id)

        # normalize job id
        try:
            jid = int(job_id)
        except Exception:
            try:
                m = re.search(r"(\d+)", str(job_id))
                jid = int(m.group(1)) if m else None
            except Exception:
                jid = None
        if jid is None:
            continue

        try:
            start = float(s)
            end = float(e)
        except Exception:
            continue
        if end < start:
            # skip invalid intervals
            continue
        if duration is not None:
            try:
                dur = float(duration)
            except Exception:
                dur = end - start
        else:
            dur = end - start

        key = (jid, op_idx)
        prev = best.get(key)
        if prev is None or dur > prev['duration']:
            best[key] = {'start': start, 'end': end, 'op': op_idx, 'job': jid, 'duration': dur, 'machine': mid, 'operator': op_id}
            min_start = min(min_start, start)
            max_end = max(max_end, end)
            if op_idx not in op_types_seen:
                op_types_seen.append(op_idx)

    records = list(best.values())
    dedup_count = len(records)
    _append_debug(dedup_count=dedup_count)

    if not records:
        logger.info("No valid gantt records after deduplication; skipping gantt_last_evolution plot.")
        _append_debug(status='no_valid_records_after_dedup')
        return False

    # map op types to colors using a clear qualitative palette
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10') if len(op_types_seen) <= 10 else cm.get_cmap('tab20')
    except Exception:
        cmap = None
    op_unique = []
    for o in op_types_seen:
        if o not in op_unique:
            op_unique.append(o)
    op_color_map = {}
    for i, op in enumerate(op_unique):
        try:
            if cmap is not None:
                rgba = cmap(i % (cmap.N if hasattr(cmap, 'N') else 10))
                # convert to hex
                color = mcolors.to_hex(rgba)
            else:
                palette = list(mcolors.TABLEAU_COLORS.values())
                color = palette[i % len(palette)]
        except Exception:
            color = 'gray'
        op_color_map[op] = color

    # plotting: machine rows on Y axis
    machine_list = sorted(list(machines))
    machine_to_y = {m: idx for idx, m in enumerate(machine_list)}
    n_machines = len(machine_list)
    # autoscale figure size based on machines and makespan
    makespan = max_end if max_end > 0 else 0.0
    # scale_factor compresses long timelines visually while preserving relative durations
    scale_factor = max(1.0, float(makespan) / 60.0)

    fig_w = max(12, min(40, float(makespan) / 3.0 + 6.0))
    # keep a consistent aspect ratio (approx 2:1 width:height) while ensuring
    # a reasonable minimum height for readability
    fig_h = max(6, fig_w / 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # bar height and font scale with number of rows to remain readable
    bar_height = max(0.35, min(0.9, 0.6 if n_machines <= 8 else 0.6 * (8.0 / max(8.0, n_machines))))
    # draw bars grouped by machine; scale horizontal coordinates by scale_factor
    for r in records:
        m = r['machine']
        y = machine_to_y.get(m)
        if y is None:
            continue
        start = r['start']; end = r['end']; width = max(0.0, end - start)
        op = r['op']
        color = op_color_map.get(op, 'gray')
        # draw bar using scaled coordinates so even narrow durations stay visible
        try:
            left = float(start) / scale_factor
            wscaled = float(width) / scale_factor
            ax.barh(y, wscaled, left=left, height=bar_height, color=color, edgecolor='black', linewidth=0.6, zorder=3)
        except Exception:
            continue
        # label inside bar: "M{machine} | O{operator} | Op {op}"
        label_text = f"M{m} | O{r['operator']} | Op {op}"
        # choose contrasting text color
        try:
            rgb = mcolors.to_rgb(op_color_map.get(op, '#777777'))
            luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            text_color = 'black' if luminance > 0.6 else 'white'
        except Exception:
            text_color = 'white'
        # center label; compute fontsize relative to scaled width and fig size
        try:
            fontsize = int(max(8, min(14, 10 * bar_height)))
        except Exception:
            fontsize = 9
        if width >= 0.02:
            try:
                ax.text(left + wscaled / 2.0, y, label_text, va='center', ha='center', fontsize=fontsize, color=text_color, zorder=4, weight='bold', clip_on=True)
            except Exception:
                pass

    # show original time on x-axis by formatting scaled ticks back to original units
    try:
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{(x * scale_factor):.1f}"))
    except Exception:
        pass

    # Y axis: machines
    y_positions = [machine_to_y[m] for m in machine_list]
    y_labels = [f"M{m}" for m in machine_list]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_ylim(-0.5, max(0, n_machines - 1) + 0.5)

    # X axis ticks: adaptive spacing
    makespan = max_end if max_end > 0 else 0.0
    if makespan <= 100:
        major_step = 20 if makespan >= 40 else max(1, int(round(makespan / 5)))
    else:
        # try ~10 major ticks
        major_step = max(10, int(round(makespan / 10)))
    try:
        ax.xaxis.set_major_locator(MultipleLocator(major_step))
    except Exception:
        pass
    ax.set_xlabel('Simulation Time (s)', fontsize=12)
    ax.set_title(f"Gantt Chart — Last Evolution (Total Time: {makespan:.1f}s)", fontsize=14)

    # grid and background
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_facecolor('#f7f7f7')

    # Legend on the right with OpType -> Color mapping and explicit labels
    handles = []
    labels = []
    for op in op_unique:
        lab = f"OpType {op}"
        handles.append(Patch(facecolor=op_color_map.get(op, 'gray'), edgecolor='black', label=lab))
        labels.append(lab)
    if handles:
        # reserve a larger right margin and place legend further to the right
        # but still inside the figure. Use centered-left anchor so the legend
        # appears vertically centered alongside the plot.
        legend = ax.legend(handles, labels, title='Operation Type → Color', bbox_to_anchor=(1.25, 0.5), loc='center left', fontsize=10)
        ax.add_artist(legend)
        # let tight_layout compute internals then explicitly reserve right margin
        try:
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.1)
        except Exception:
            pass
    else:
        plt.tight_layout()

    # write debug summary before attempting save
    try:
        _append_debug(min_start=min_start if min_start != float('inf') else 0.0,
                      max_end=max_end,
                      total_machines=n_machines,
                      total_operators=len(operators))
        # also write op->color mapping
        try:
            with open(debug_path, 'a') as df:
                df.write('op_type_to_color_mapping:\n')
                for k, v in op_color_map.items():
                    df.write(f"  {k} -> {v}\n")
        except Exception:
            pass
    except Exception:
        pass

    # Prepare output paths
    try:
        history_dir = os.path.dirname(filename) or '.'
    except Exception:
        history_dir = '.'
    machine_path = os.path.join(history_dir, 'gantt_machine_specific.png')
    job_path = os.path.join(history_dir, 'gantt_job_specific.png')

    save_results = {
        'gantt_last_evolution': False,
        'gantt_machine_specific': False,
        'gantt_job_specific': False,
    }

    # --- Machine-Specific Gantt Chart ---
    try:
        n_bars = max(1, len(machine_list))
        # autosize using makespan and number of bars; ensure a generous minimum
        fig_w = max(22, min(60, float(makespan) / 3.0 + 6.0))
        # Maintain consistent aspect ratio across machine/job charts
        fig_h = max(8, fig_w / 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        # ensure white background (avoid large dark margins when saving)
        try:
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        except Exception:
            pass

        # dynamic bar height and font size (scaled to number of bars)
        bar_height = max(0.25, min(0.9, 0.6 if n_bars <= 8 else 0.6 * (8.0 / max(8.0, n_bars))))
        font_size = int(max(8, min(14, 10 * bar_height)))

        for r in records:
            m = r['machine']
            y = machine_to_y.get(m)
            if y is None:
                continue
            start = r['start']; end = r['end']; width = max(0.0, end - start)
            op = r['op']
            color = op_color_map.get(op, 'gray')
            # apply horizontal scaling for visibility
            left = float(start) / scale_factor
            wscaled = float(width) / scale_factor
            ax.barh(y, wscaled, left=left, height=bar_height, color=color, edgecolor='black', linewidth=0.6, zorder=3)
            # label: Job {job_id} | O{operator_id}
            label_text = f"Job {r['job']} | O{r['operator']}"
            try:
                rgb = mcolors.to_rgb(color)
                luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                text_color = 'black' if luminance > 0.6 else 'white'
            except Exception:
                text_color = 'white'
            if width >= 0.02:
                ax.text(left + wscaled / 2.0, y, label_text, va='center', ha='center', fontsize=font_size, color=text_color, weight='bold', clip_on=True, zorder=4)

        # Y axis: machines
        y_positions = [machine_to_y[m] for m in machine_list]
        y_labels = [f"M{m}" for m in machine_list]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=max(8, font_size-1))
        ax.set_ylim(-0.5, max(0, n_bars - 1) + 0.5)
        ax.set_xlabel('Simulation Time (s)', fontsize=12)
        ax.set_title(f"Machine-Specific Gantt Chart — Last Evolution (Total Time: {max_end:.1f}s)", fontsize=14)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

        # Legend: OpType -> Color (stable mapping) placed outside plot
        handles = []
        labels = []
        for op in op_unique:
            lab = f"Op{int(op)+1}" if isinstance(op, (int, float)) and float(op).is_integer() else str(op)
            handles.append(Patch(facecolor=op_color_map.get(op, 'gray'), edgecolor='black', label=lab))
            labels.append(lab)
        if handles:
            # place legend slightly further right and center it vertically
            legend = ax.legend(handles, labels, title='Operation Types', bbox_to_anchor=(1.25, 0.5), loc='center left', frameon=True, fontsize=10, title_fontsize=10)
            ax.add_artist(legend)
            # ensure tight layout then reserve right margin so legend stays inside
            try:
                plt.tight_layout()
                plt.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.1)
            except Exception:
                pass
        else:
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.1)

        # adaptive x ticks and restore original time values in labels
        try:
            step = max(1, int(max(1, round(max_end / 10))))
            ax.xaxis.set_major_locator(MultipleLocator(max(1, step / float(scale_factor))))
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{(x * scale_factor):.1f}"))
        except Exception:
            pass

        # Save machine-specific both as dedicated file and as legacy filename for backward compatibility
        try:
            plt.savefig(machine_path, dpi=300, bbox_inches='tight')
            save_results['gantt_machine_specific'] = True
        except Exception:
            logger.exception('Failed to save machine-specific gantt', exc_info=True)
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            save_results['gantt_last_evolution'] = True
        except Exception:
            logger.exception('Failed to save gantt_last_evolution image', exc_info=True)
        try:
            plt.close()
        except Exception:
            pass
    except Exception:
        logger.exception('Failed to generate machine-specific gantt', exc_info=True)

    # --- Job-Specific Gantt Chart ---
    try:
        job_list = sorted(list({r['job'] for r in records}))
        job_to_y = {j: idx for idx, j in enumerate(job_list)}
        n_bars = max(1, len(job_list))
        # autosize using makespan and number of jobs; ensure a generous minimum
        fig_w = max(22, min(60, float(makespan) / 3.0 + 6.0))
        # Maintain consistent aspect ratio across machine/job charts
        fig_h = max(8, fig_w / 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        try:
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
        except Exception:
            pass

        bar_height = max(0.25, min(0.9, 0.6 if n_bars <= 8 else 0.6 * (8.0 / max(8.0, n_bars))))
        font_size = int(max(8, min(14, 10 * bar_height)))

        for r in records:
            jid = r['job']
            y = job_to_y.get(jid)
            if y is None:
                continue
            start = r['start']; end = r['end']; width = max(0.0, end - start)
            op = r['op']
            color = op_color_map.get(op, 'gray')
            left = float(start) / scale_factor
            wscaled = float(width) / scale_factor
            ax.barh(y, wscaled, left=left, height=bar_height, color=color, edgecolor='black', linewidth=0.6, zorder=3)
            # label: M{machine_id} | O{operator_id}
            label_text = f"M{r['machine']} | O{r['operator']}"
            try:
                rgb = mcolors.to_rgb(color)
                luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                text_color = 'black' if luminance > 0.6 else 'white'
            except Exception:
                text_color = 'white'
            if width >= 0.02:
                ax.text(left + wscaled / 2.0, y, label_text, va='center', ha='center', fontsize=font_size, color=text_color, weight='bold', clip_on=True, zorder=4)

        # Y axis: jobs
        y_positions = [job_to_y[j] for j in job_list]
        y_labels = [f"J{j}" for j in job_list]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=max(8, font_size-1))
        ax.set_ylim(-0.5, max(0, n_bars - 1) + 0.5)
        ax.set_xlabel('Simulation Time (s)', fontsize=12)
        ax.set_title(f"Job-Specific Gantt Chart — Last Evolution (Total Time: {max_end:.1f}s)", fontsize=14)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)

        # Legend (outside)
        handles = []
        labels = []
        for op in op_unique:
            lab = f"Op{int(op)+1}" if isinstance(op, (int, float)) and float(op).is_integer() else str(op)
            handles.append(Patch(facecolor=op_color_map.get(op, 'gray'), edgecolor='black', label=lab))
            labels.append(lab)
        if handles:
            # place legend slightly further right and center it vertically
            legend = ax.legend(handles, labels, title='Operation Types', bbox_to_anchor=(1.25, 0.5), loc='center left', frameon=True, fontsize=10, title_fontsize=10)
            ax.add_artist(legend)
            try:
                plt.tight_layout()
                plt.subplots_adjust(left=0.05, right=0.72, top=0.92, bottom=0.1)
            except Exception:
                pass
        else:
            plt.tight_layout()
            plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.1)

        try:
            step = max(1, int(max(1, round(max_end / 10))))
            ax.xaxis.set_major_locator(MultipleLocator(max(1, step / float(scale_factor))))
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{(x * scale_factor):.1f}"))
        except Exception:
            pass

        try:
            plt.savefig(job_path, dpi=300, bbox_inches='tight')
            save_results['gantt_job_specific'] = True
        except Exception:
            logger.exception('Failed to save job-specific gantt', exc_info=True)
        try:
            plt.close()
        except Exception:
            pass
    except Exception:
        logger.exception('Failed to generate job-specific gantt', exc_info=True)

    # Write debug entries for save results and op->color mapping
    try:
        _append_debug(min_start=min_start if min_start != float('inf') else 0.0,
                      max_end=max_end,
                      total_machines=len(machine_list),
                      total_operators=len(operators))
        try:
            with open(debug_path, 'a') as df:
                df.write('op_type_to_color_mapping:\n')
                for k, v in op_color_map.items():
                    df.write(f"  {k} -> {v}\n")
                df.write(f"saved_files: {save_results}\n")
        except Exception:
            pass
    except Exception:
        pass

    # Also append separate save flags for easy machine parsing
    try:
        _append_debug(**{f"save_{k}": bool(v) for k, v in save_results.items()})
    except Exception:
        pass

    # Return True if at least the legacy filename was saved
    return bool(save_results.get('gantt_last_evolution'))


class Runner:
    """
    Step 8A.6.6 Runner – Replay-based QMIX + KPI Logging + Learning Stability
    -------------------------------------------------------------------------
    - 6D observations
    - Extended KPI metrics (avg wait, utilization, makespan)
    - Moving-average loss/TD tracking
    """

    def __init__(self, env, args):
        self.env = env
        self.args = args
        # run_args is the run-local copy used by many helper methods; ensure
        # it's available immediately to avoid AttributeError when accessed
        # earlier in the constructor.
        self.run_args = args
        # Configure whether history/artifact writes are allowed for this run.
        # Prefer the centralized decision helper so environment variables
        # and programmatic overrides are respected.
        try:
            from utils.io_control import allow_history_writes as _allow_fn
            self.allow_history_writes = bool(_allow_fn())
        except Exception:
            # Conservatively enable history writes if the helper isn't present.
            self.allow_history_writes = True
        # Snapshot the canonical args at construction time for audit/comparison.
        # This helps detect ad-hoc mutations to the global args object outside
        # of `MARL/common/arguments.py` or explicit test fixtures.
        try:
            self._args_snapshot = dict(copy.deepcopy(vars(self.args)))
        except Exception:
            # fallback: shallow copy of reprs
            try:
                self._args_snapshot = {k: getattr(self.args, k, None) for k in dir(self.args) if not k.startswith('__')}
            except Exception:
                self._args_snapshot = {}
        # Create a run-local copy of args for runtime-derived values so
        # we don't mutate the globally-shared args object. This enforces
        # the repository-wide rule that modules must not write into args.
        try:
            run_args = copy.deepcopy(self.args)
        except Exception:
            # fallback: use original args object if deepcopy fails
            run_args = self.args

        # propagate quiet flag to environment to suppress verbose SimPy debug prints
        try:
            setattr(self.env, "quiet_env", bool(getattr(self.args, "quiet_env", False)))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass

        # If the user requested operator-granular actions, compute and set the
        # effective number of actions (num_wcs * num_ops) early so policy
        # constructors (which read args.n_actions) see the correct value.
        try:
            if bool(getattr(run_args, 'use_granular_actions', False)):
                if hasattr(self.env, 'num_wcs') and hasattr(self.env, 'num_ops'):
                    run_args.n_actions = int(self.env.num_wcs) * int(self.env.num_ops)
                    print(f"[Runner] Using granular actions: n_actions={run_args.n_actions}")
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    info = None

                if info and isinstance(info, dict) and ('n_agents' in info or 'n_agents' in info.keys()):
                    try:
                        run_args.n_agents = int(info.get('n_agents'))
                        print(f"[Runner] Auto-set run_args.n_agents = {run_args.n_agents} from env.get_env_info()['n_agents']")
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        pass
                elif hasattr(self.env, 'jobs'):
                    try:
                        run_args.n_agents = int(len(getattr(self.env, 'jobs')))
                        print(f"[Runner] Auto-set run_args.n_agents = {run_args.n_agents} from len(env.jobs)")
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        pass
                elif hasattr(self.env, 'num_jobs'):
                    try:
                        run_args.n_agents = int(getattr(self.env, 'num_jobs'))
                        print(f"[Runner] Auto-set run_args.n_agents = {run_args.n_agents} from env.num_jobs")
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        pass
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            # keep silent on failures — this is a best-effort convenience
            pass

        # Agents & rollout setup
        # Instantiate Agents, ReplayBuffer and RolloutWorker using centralized shapes
        if self.args.alg.find('commnet') > -1 or self.args.alg.find('g2anet') > -1:
            self.agents = CommAgents(run_args)
            # Create buffer without assuming constructor accepts extra runtime kwargs.
            self.buffer = ReplayBuffer(episode_capacity=run_args.buffer_size, seed=run_args.seed) \
                if getattr(run_args, "learn", True) else None
            # Backwards-compatible: if Runner has runtime shapes, stash them on the
            # buffer instance so older / monkeypatched buffer implementations can
            # still observe them without requiring new constructor params.
            try:
                if self.buffer is not None:
                    if getattr(run_args, 'n_agents', None) is not None:
                        setattr(self.buffer, '_n_agents', int(getattr(run_args, 'n_agents')))
                    if getattr(run_args, 'obs_shape', None) is not None:
                        setattr(self.buffer, '_obs_dim', int(getattr(run_args, 'obs_shape')))
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            self.rolloutWorker = CommRolloutWorker(env=env, agents=self.agents, buffer=self.buffer, args=run_args)
        else:
            self.agents = Agents(run_args)
            self.buffer = ReplayBuffer(episode_capacity=run_args.buffer_size, seed=run_args.seed) \
                if getattr(run_args, "learn", True) else None
            try:
                if self.buffer is not None:
                    if getattr(run_args, 'n_agents', None) is not None:
                        setattr(self.buffer, '_n_agents', int(getattr(run_args, 'n_agents')))
                    if getattr(run_args, 'obs_shape', None) is not None:
                        setattr(self.buffer, '_obs_dim', int(getattr(run_args, 'obs_shape')))
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            self.rolloutWorker = RolloutWorker(env=env, agents=self.agents, buffer=self.buffer, args=run_args)

        # n_actions was computed centrally above; no further recomputation here.

        self.win_rates = []
        self.episode_rewards = []
        self.episode_durations = []
        self.wait_time_records = []

        # Use self.args (CLI-level settings) consistently for save path
        self.save_path = os.path.join(self.args.result_dir, self.args.alg, self.args.map)
        os.makedirs(self.save_path, exist_ok=True)
        # prefer history_dir supplied via run-local args; fall back to legacy path
        try:
            self.history_dir = str(getattr(self.run_args, 'history_dir', './my_data_and_graph/historydata'))
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            self.history_dir = './my_data_and_graph/historydata'
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
                        except Exception as e:
                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                            pass
                print("[Runner] History cleaned.")
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass

        # NOTE: removed pre-run creation/overwrite of scheduling_timeline.txt.
        # The timeline file is now written dynamically after episodes when
        # runtime gantt records exist. This prevents creating placeholder
        # timeline files before simulation data is produced.

        print(f"[Runner 8A.6.6] Initialized | alg={self.args.alg} | buffer={getattr(self.args,'buffer_size','-')} | batch={getattr(self.args,'batch_size','-')}")
        # Write initial job -> operations mapping for easy inspection
        # Optionally append initial job -> operations mapping for inspection.
        # This is opt-in only via run_args.log_initial_jobs (default False)
        try:
            if bool(getattr(self.run_args, 'log_initial_jobs', False)):
                init_path = os.path.join(self.history_dir, "initial_jobs.txt")
                # Append-only: do not overwrite or read this file. Treat it as a runtime log.
                try:
                    # write a short header with timestamp for traceability
                    import datetime
                    header = f"\n=== Initial Job -> Operation mapping (appended {datetime.datetime.utcnow().isoformat()}Z) ===\n"
                    with open(init_path, 'a') as hf:
                        hf.write(header)
                        hf.write("Format: JobID | OpIdx | OpType | Allowed_Machine_Indices | OpGroups | BaseDur\n\n")
                        for job in getattr(self.env, 'jobs', []):
                            hf.write(f"Job {int(job.id)}:\n")
                            for idx, op in enumerate(getattr(job, 'operations', [])):
                                try:
                                    # canonical formats:
                                    # legacy: (allowed_machine_indices, dur)
                                    # old: (op_type, allowed_machine_indices, base_dur)
                                    # new canonical: (op_type, allowed_machine_indices, per_machine_durations_dict)
                                    if isinstance(op, (list, tuple)) and len(op) == 2:
                                        allowed_machine_indices, dur = op
                                        op_type = 'legacy'
                                        hf.write(f"  Op {idx} | Type {op_type} | machines {allowed_machine_indices} | Dur {float(dur):.3f}\n")
                                    else:
                                        op_type = op[0]
                                        allowed_machine_indices = op[1]
                                        third = op[2]
                                        # if third is dict, print per-machine durations and both coarse/eligible operator info
                                        if isinstance(third, dict):
                                            per_machine = third
                                            groups_by_machine = []
                                            for m in allowed_machine_indices:
                                                try:
                                                    eligible = self.env.workcenters_meta.eligible_operator_groups_by_wc.get(int(m), [])
                                                except Exception:
                                                    eligible = []
                                                groups_by_machine.append({'machine': int(m), 'eligible_ops': eligible})
                                            hf.write(f"  Op {idx} | Type {op_type} | machines {allowed_machine_indices} | Groups {groups_by_machine} | base_per_machine_durations:\n")
                                            for m in allowed_machine_indices:
                                                try:
                                                    dur_m = float(per_machine.get(int(m), 0.0))
                                                except Exception:
                                                    dur_m = 0.0
                                                try:
                                                    eligible = self.env.workcenters_meta.eligible_operator_groups_by_wc.get(int(m), [])
                                                except Exception:
                                                    eligible = []
                                                hf.write(f"    M{m} -> dur={dur_m:.3f} | eligible_ops={eligible}\n")
                                        else:
                                            # legacy-ish third numeric
                                            base_dur = float(third)
                                            groups_info = []
                                            for m in allowed_machine_indices:
                                                try:
                                                    eligible = self.env.workcenters_meta.eligible_operator_groups_by_wc.get(int(m), [])
                                                except Exception:
                                                    eligible = []
                                                groups_info.append({'machine': int(m), 'eligible_ops': eligible})
                                            hf.write(f"  Op {idx} | Type {op_type} | machines {allowed_machine_indices} | Groups {groups_info} | base_dur {base_dur:.3f}\n")
                                except Exception:
                                    hf.write(f"  Op {idx} | malformed: {op}\n")
                            hf.write("\n")
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught while appending initial jobs mapping", exc_info=True)
                    print(f"[WARN] Could not append initial job mapping: {e}")
                # Append human-readable job list produced by env helper if available (append only)
                try:
                    if hasattr(self.env, 'print_jobs_human_readable') and callable(getattr(self.env, 'print_jobs_human_readable')):
                        try:
                            import io, sys as _sys
                            buf = io.StringIO()
                            old = _sys.stdout
                            _sys.stdout = buf
                            try:
                                self.env.print_jobs_human_readable()
                            finally:
                                _sys.stdout = old
                            with open(init_path, 'a') as hf:
                                hf.write('\nHuman-readable job list:\n')
                                hf.write(buf.getvalue())
                        except Exception:
                            # best-effort: attempt to append without capture
                            try:
                                with open(init_path, 'a') as hf:
                                    hf.write('\nHuman-readable job list (partial):\n')
                                    try:
                                        self.env.print_jobs_human_readable()
                                    except Exception:
                                        hf.write('  <could not capture human-readable output>\n')
                            except Exception:
                                logging.getLogger(__name__).exception("Exception caught while appending human-readable jobs", exc_info=True)
                                pass
                except Exception:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
            # Respect central gating: skip metric writes when history writes are disabled
            if not getattr(self, 'allow_history_writes', False):
                return
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
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                last_loss = ''
            try:
                tpath = os.path.join(self.history_dir, 'td_error.txt')
                if os.path.exists(tpath):
                    with open(tpath, 'r') as tf:
                        lines = [ln.strip() for ln in tf.readlines() if ln.strip()]
                        if lines:
                            last_td = lines[-1]
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                last_line = None

            if last_line != new_line:
                try:
                    with open(metrics_path, 'a') as mf:
                        if header_needed:
                            mf.write('episode,epoch,episode_reward,last_loss,last_td\n')
                        mf.write(new_line + '\n')
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                    pass
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                # Per-epoch Gantt PNG/CSV generation disabled.
                # We produce a single, authoritative gantt_last_evolution.png at
                # the per-evolution summary step to keep artifacts minimal and
                # readable. This avoids producing intermediate gantt_epoch*.png files.
                try:
                    # still keep the combined gantt in memory for downstream use
                    combined = list(all_gantt_data)
                    try:
                        if hasattr(self.env, 'gantt_records'):
                            combined.extend(list(self.env.gantt_records))
                    except Exception:
                        pass
                except Exception:
                    pass

            # collect per-epoch gantt records so we can compute per-evolution metrics
            episodes, avg_rewards = [], []
            epoch_gantt = []
            # Keep the last episode's gantt records (overwrite each episode)
            # so we can plot a clean Gantt for the most recent episode only.
            last_episode_gantt = []
            try:
                before_wait = float(getattr(self.env, 'total_wait_time', 0.0))
            except Exception:
                before_wait = 0.0
            try:
                # Compute finished job count on-demand (canonical)
                before_completed = len([j for j in getattr(self.env, 'jobs', []) if getattr(j, 'finished', False)])
            except Exception:
                before_completed = 0

            for _ in range(self.args.n_episodes):
                # Use the SimPy event-driven episode execution exclusively.
                episode, ep_r, win_tag, gantt_data = self._run_event_driven_episode(global_ep_idx)

                all_gantt_data.extend(gantt_data)
                # record gantt for this epoch specifically
                try:
                    epoch_gantt.extend(gantt_data)
                    # capture this episode's gantt separately; last assignment
                    # will therefore represent the latest episode in the epoch
                    try:
                        last_episode_gantt = list(gantt_data)
                    except Exception:
                        last_episode_gantt = []
                except Exception:
                    pass
                avg_rewards.append(ep_r)
                episodes.append(episode)
                global_ep_idx += 1

                # Record per-training-episode reward for visibility and post-run summaries.
                # Runner previously only appended rewards from evaluation runs; include
                # training episodes here as well so entrypoints (like main.py) can
                # print a concise per-episode summary after runner.run() completes.
                try:
                    self.episode_rewards.append(float(ep_r))
                except Exception:
                    # best-effort: ignore if unable to append (shouldn't happen)
                    pass

                # Append per-episode metrics via single writer method
                try:
                    self._append_learning_metrics(global_ep_idx-1, epoch, ep_r)
                except Exception as e:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                    # fallback: try env.total_wait_time / computed completed-count
                    try:
                        # Replace legacy counter usage with computed count
                        _completed_count = max(1, len([j for j in getattr(self.env, 'jobs', []) if getattr(j, 'finished', False)]))
                        self.wait_time_records.append({0: (self.env.total_wait_time / _completed_count)})
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        self.wait_time_records.append({0: ep_r / 20.0})

                # Optionally append a human-readable scheduling timeline for this episode
                try:
                    if getattr(self, 'allow_history_writes', False):
                            # Guard timeline generation so it only happens when runtime
                            # gantt records exist and the central IO gate permits writes.
                            try:
                                from utils.io_control import allow_history_writes as _allow_fn
                            except Exception:
                                def _allow_fn():
                                    return False

                            try:
                                gr_count = len(getattr(self.env, 'gantt_records', []) or [])
                            except Exception:
                                gr_count = 0

                            # Only attempt to generate/write the timeline if there are
                            # actual runtime records. Otherwise skip to avoid creating
                            # deterministic/placeholder timeline files at startup.
                            if _allow_fn() and gr_count > 0:
                                # Debug print confirming data presence before writing
                                try:
                                    print(f"[Runner] Writing Scheduling_Timeline.txt for episode {global_ep_idx-1}, gantt_records={gr_count}")
                                except Exception:
                                    pass

                                timeline = None
                                if gantt_utils is not None and hasattr(gantt_utils, 'generate_scheduling_timeline'):
                                    try:
                                        # The RolloutWorker writes the authoritative per-episode
                                        # block (after lifecycle END). Here we only request a
                                        # non-writing preview to avoid duplicate writes.
                                        timeline = gantt_utils.generate_scheduling_timeline(self.env, episode_id=global_ep_idx-1, episode_reward=ep_r, write_if_allowed=False, out_dir=self.history_dir)
                                    except Exception:
                                        timeline = None
                                if timeline is None:
                                    try:
                                        from utils.gantt import generate_scheduling_timeline
                                        # non-writing preview to avoid duplicate authoritative writes
                                        timeline = generate_scheduling_timeline(self.env, episode_id=global_ep_idx-1, episode_reward=ep_r, write_if_allowed=False, out_dir=self.history_dir)
                                    except Exception:
                                        timeline = None
                                if timeline:
                                    try:
                                        print(f"[Runner] Scheduling timeline appended for episode {global_ep_idx-1} to {self.history_dir}/scheduling_timeline.txt")
                                    except Exception:
                                        pass
                            else:
                                try:
                                    print(f"[Runner] Skipped timeline generation for episode {global_ep_idx-1} (no runtime data yet, gantt_records={gr_count}, allow_history_writes={_allow_fn()})")
                                except Exception:
                                    pass

                            # Runtime debug logs after each episode: report whether writes are allowed
                            # and how many gantt records the environment has collected.
                            try:
                                from utils.io_control import allow_history_writes as _allow_fn2
                                try:
                                    print(f"[Runner] Scheduling timeline updated for episode {global_ep_idx-1} (allow_history_writes={_allow_fn2()})")
                                except Exception:
                                    print(f"[Runner] Scheduling timeline updated for episode {global_ep_idx-1} (allow_history_writes=<error>)")
                            except Exception:
                                print(f"[Runner] Scheduling timeline updated for episode {global_ep_idx-1} (allow_history_writes=<import-error>)")
                            try:
                                gr_count = len(getattr(self.env, 'gantt_records', []) or [])
                            except Exception:
                                gr_count = 0
                            try:
                                print(f"[Runner] Gantt records collected: {gr_count}")
                            except Exception:
                                pass
                except Exception:
                    logging.getLogger(__name__).exception("Exception caught while appending scheduling_timeline", exc_info=True)
                    pass

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
                        mini_batch = self.buffer.sample(self.args.batch_size, n_actions=self.run_args.n_actions)
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
                                    # Only write snapshots if allowed
                                    if getattr(self, 'allow_history_writes', False):
                                        plot_gantt(snapshot_gantt, filename=os.path.join(self.history_dir, snap_name))
                                        if getattr(self.args, 'gantt_csv', False):
                                            csv_path = os.path.join(self.history_dir, f"gantt_snapshot_step{train_steps}.csv")
                                            # prefer centralized writer
                                            try:
                                                if gantt_utils is not None:
                                                    gantt_utils.write_scheduling_trace(csv_path, snapshot_gantt)
                                                else:
                                                    raise RuntimeError("gantt_utils unavailable")
                                            except Exception as e:
                                                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                                # fallback: legacy writer
                                                try:
                                                    with open(csv_path, 'w') as cf:
                                                        cf.write('start,end,op_idx,op_name,wc,job_id,operator_grp,arrival,duration\n')
                                                        for r in snapshot_gantt:
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
                                                                except Exception as e:
                                                                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                                                    op_name = str(op_idx)
                                                                vals = [start, end, op_idx, op_name, wc, job_id, op_grp, arrival, duration]
                                                                cf.write(','.join([str(x) for x in vals]) + '\n')
                                                            except Exception as e:
                                                                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                                                pass
                                                except Exception as e:
                                                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                                    pass
                                        # also write a human-readable summary for easier inspection
                                        def _unpack_rec(rec):
                                            # robustly unpack records of length 8, 6 or 5
                                            if not isinstance(rec, (list, tuple)):
                                                raise ValueError('invalid rec')
                                            if len(rec) >= 8:
                                                return rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7]
                                            if len(rec) == 6:
                                                return rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], None, None
                                            if len(rec) == 5:
                                                return rec[0], rec[1], rec[2], rec[3], rec[4], None, None, None
                                            raise ValueError('unsupported rec len')

                                        try:
                                            txt_path = os.path.join(self.history_dir, f"gantt_snapshot_step{train_steps}_readable.txt")
                                            with open(txt_path, 'w') as tf:
                                                # group by job_id
                                                jobs_map = {}
                                                for rec in snapshot_gantt:
                                                    try:
                                                        s, e, op_idx, wc, job_id, op_grp, arrival, duration = _unpack_rec(rec)
                                                        jobs_map.setdefault(int(job_id), []).append((s, e, int(op_idx), int(wc), op_grp, arrival, duration))
                                                    except Exception as e:
                                                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                                        continue
                                                for jid in sorted(jobs_map.keys()):
                                                    tf.write(f"Job {jid}:\n")
                                                    for (s, e, op_idx, wc, op_grp, arrival, duration) in sorted(jobs_map[jid], key=lambda x: x[0]):
                                                        dur_str = f"dur={duration:.3f}" if duration is not None else "dur=?"
                                                        tf.write(f"  Op {op_idx} @ WC{wc} (OpGrp {op_grp}) — start={s:.3f}, end={e:.3f} | arrival={arrival} {dur_str}\n")
                                                    tf.write('\n')
                                        except Exception as e:
                                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                                                except Exception as e:
                                                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                                                        except Exception as e:
                                                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                                            continue
                                                tf2.write('\n')
                                        except Exception as e:
                                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                                                # Compute average wait using on-demand completed count
                                                _completed_count = max(1, len([j for j in getattr(self.env, 'jobs', []) if getattr(j, 'finished', False)]))
                                                metrics['avg_wait'] = float(self.env.total_wait_time) / _completed_count
                                            except Exception as e:
                                                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                                metrics['avg_wait'] = None
                                            try:
                                                metrics['avg_reward'] = float(np.mean(self.episode_rewards)) if self.episode_rewards else None
                                            except Exception as e:
                                                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
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
                                            except Exception as e:
                                                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                                pass
                                            with open(metrics_path, 'w') as mf:
                                                json.dump(metrics, mf, indent=2)
                                        except Exception as e:
                                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                            pass
                                    print(f"[Runner] Gantt snapshot saved @ step {train_steps}")
                                except Exception as e:
                                    print(f"[WARN] Could not save gantt snapshot: {e}")
                        except Exception as e:
                            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                            pass

            # === KPI LOGGING (Step 8A.6.6) ===
            # KPI logging (opt-in)
            try:
                if getattr(self, 'allow_history_writes', False):
                    # compute avg_wait using computed completed count to avoid legacy counter
                    _completed_count = max(1, len([j for j in getattr(self.env, 'jobs', []) if getattr(j, 'finished', False)]))
                    avg_wait = self.env.total_wait_time / _completed_count
                    # Compute instantaneous machine utilization fraction (best-effort)
                    try:
                        mres = list(getattr(self.env, 'machine_resources', []) or [])
                        if mres:
                            occupied = 0
                            for r in mres:
                                try:
                                    users = getattr(r, 'users', None)
                                    if users is not None:
                                        if len(users) > 0:
                                            occupied += 1
                                        continue
                                    # fallback: treat a positive 'count' attribute as occupied
                                    if getattr(r, 'count', None) is not None:
                                        if int(getattr(r, 'count', 0)) > 0:
                                            occupied += 1
                                except Exception:
                                    continue
                            util_m = float(occupied) / max(1.0, float(len(mres)))
                        else:
                            util_m = 0.0
                    except Exception:
                        util_m = 0.0

                    # Compute instantaneous operator/utilization fraction (best-effort)
                    try:
                        og = list(getattr(self.env, 'operator_groups', []) or [])
                        if og:
                            occ_o = 0
                            for g in og:
                                try:
                                    users = getattr(g, 'users', None)
                                    if users is not None:
                                        if len(users) > 0:
                                            occ_o += 1
                                        continue
                                    if getattr(g, 'count', None) is not None:
                                        if int(getattr(g, 'count', 0)) > 0:
                                            occ_o += 1
                                except Exception:
                                    continue
                            util_o = float(occ_o) / max(1.0, float(len(og)))
                        else:
                            util_o = 0.0
                    except Exception:
                        util_o = 0.0
                    makespan = getattr(self.env, "t", getattr(self.env, "env", None) and getattr(self.env, "env").now or 0.0)
                    makespan = getattr(self.env, "t", makespan)
                    with open(os.path.join(self.history_dir, "kpi_log.txt"), "a") as f:
                        f.write(f"{epoch},{avg_wait:.4f},{util_m:.4f},{util_o:.4f},{makespan:.2f}\n")
            except Exception as e:
                print(f"[WARN] KPI logging failed: {e}")

            # === Per-evolution summary and plotting ===
            try:
                if getattr(self, 'allow_history_writes', False):
                    try:
                        # compute per-epoch deltas for wait/completed
                        after_wait = float(getattr(self.env, 'total_wait_time', 0.0))
                        # compute after_completed on-demand
                        after_completed = len([j for j in getattr(self.env, 'jobs', []) if getattr(j, 'finished', False)])
                        delta_wait = max(0.0, after_wait - float(before_wait))
                        delta_completed = max(0, after_completed - int(before_completed))
                    except Exception:
                        delta_wait = 0.0
                        delta_completed = 0

                    # infer machine/operator counts
                    try:
                        n_m = len(getattr(self.env, 'machine_resources', []) or [])
                        if n_m <= 0:
                            n_m = len(getattr(self.env.workcenters_meta, 'machine_list', []) or []) or int(getattr(self.env, 'num_wcs', 1))
                    except Exception:
                        n_m = int(getattr(self.env, 'num_wcs', 1) or 1)
                    try:
                        n_o = len(getattr(self.env, 'operator_groups', []) or [])
                        if n_o <= 0:
                            n_o = int(getattr(self.env, 'num_ops', 1) or 1)
                    except Exception:
                        n_o = int(getattr(self.env, 'num_ops', 1) or 1)

                    # Build a single-item env-like dict describing this epoch
                    # Build an epoch summary. Do NOT rely on legacy counters;
                    # instead include a serializable `jobs` list so downstream canonical
                    # metrics consumers can derive the completed count deterministically.
                    epoch_item = {
                        'gantt': list(epoch_gantt),
                        'total_wait_time': float(delta_wait),
                        # Provide a lightweight, serializable jobs list with minimal
                        # attributes needed by metrics (id, finished). Downstream
                        # collectors will compute completed counts from this list.
                        'jobs': [
                            {
                                'id': int(getattr(j, 'id', -1)),
                                'finished': bool(getattr(j, 'finished', False))
                            }
                            for j in getattr(self.env, 'jobs', [])
                        ],
                        'n_machines': int(n_m),
                        'n_ops': int(n_o),
                        # Ensure avg_epoch_reward is always present (fallback to 0.0)
                        'avg_epoch_reward': float(np.mean(avg_rewards)) if avg_rewards else 0.0,
                    }

                    try:
                        from my_data_and_graph.metrics import append_run_summary
                        from my_data_and_graph.plot_metrics import (
                            plot_utilization_and_makespan,
                            plot_per_machine_utilization,
                            plot_per_operator_utilization,
                            utilization_summary_grid,
                            check_timeline_consistency,
                        )

                        # Prefer asking the environment for the rich summary when available
                        util_item = None
                        try:
                            if hasattr(self.env, '_compute_utilization_summary') and callable(getattr(self.env, '_compute_utilization_summary')):
                                util_item = self.env._compute_utilization_summary()
                        except Exception:
                            util_item = None

                        # Fall back to the constructed epoch_item if env cannot produce summary
                        if util_item is None:
                            util_item = epoch_item

                        # Ensure avg_epoch_reward is present on the util summary and preserve any existing util fields
                        try:
                            avg_r = float(np.mean(avg_rewards)) if avg_rewards else 0.0
                        except Exception:
                            try:
                                avg_r = float(epoch_item.get('avg_epoch_reward', 0.0))
                            except Exception:
                                avg_r = 0.0
                        try:
                            # prefer not to overwrite existing avg values for machine/operator utils
                            if isinstance(util_item, dict):
                                util_item['avg_epoch_reward'] = float(avg_r)
                        except Exception:
                            pass

                        # Append the enriched util summary directly into run_summary.json
                        try:
                            append_run_summary(util_item, history_dir=self.history_dir)
                        except Exception:
                            # fallback: attempt to write via collect_evolution_summary if append fails
                            try:
                                from my_data_and_graph.metrics import collect_evolution_summary as _ces
                                _ces([util_item], epoch)
                            except Exception:
                                pass
                        # regenerate plots after appending
                        plot_utilization_and_makespan(history_dir=self.history_dir)
                        # also generate per-id bar charts for latest evolution
                        plot_per_machine_utilization(history_dir=self.history_dir)
                        plot_per_operator_utilization(history_dir=self.history_dir)
                        # combined grid (includes makespan & reward)
                        utilization_summary_grid(history_dir=self.history_dir, combine_plots=True)
                        # Generate a clean machine/job-level Gantt for the last
                        # episode of this evolution. This is a compact, readable
                        # job-focused Gantt that avoids accumulation across
                        # episodes.
                        try:
                            if getattr(self, 'allow_history_writes', False):
                                # Prefer the explicit last-episode gantt; if that's empty,
                                # fall back to any env-level gantt_records so we still
                                # produce a useful artifact when possible.
                                try:
                                    cand = list(last_episode_gantt) if last_episode_gantt else list(getattr(self.env, 'gantt_records', []) or [])
                                except Exception:
                                    cand = list(getattr(self.env, 'gantt_records', []) or [])
                                if cand:
                                    png_path = os.path.join(self.history_dir, 'gantt_last_evolution.png')
                                    try:
                                        saved = plot_gantt_last_evolution(cand, filename=png_path)
                                        if saved:
                                            print(f"[Runner] gantt_last_evolution written to {png_path}")
                                        else:
                                            logging.getLogger(__name__).info("gantt_last_evolution: plot routine ran but produced no output (skipped)")
                                    except Exception:
                                        logging.getLogger(__name__).exception("Could not write gantt_last_evolution", exc_info=True)
                                else:
                                    logging.getLogger(__name__).info("gantt_last_evolution: no gantt records available (last_episode and env.gantt_records empty)")
                        except Exception:
                            pass
                        # optional timeline consistency check (heuristic)
                        try:
                            check_timeline_consistency(history_dir=self.history_dir)
                        except Exception:
                            pass
                    except Exception:
                        # best-effort: don't break training if metrics plotting fails
                        logging.getLogger(__name__).exception("Exception while writing per-evolution summary/plots", exc_info=True)
            except Exception:
                logging.getLogger(__name__).exception("Exception in per-evolution summary block", exc_info=True)

        # === Save episode statistics ===
        try:
            if getattr(self, 'allow_history_writes', False):
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

        # === Write detailed scheduling trace and per-job timeline CSVs ===
        try:
            # combine collected gantt data and env records if any
            combined = list(all_gantt_data)
            try:
                if hasattr(self.env, 'gantt_records'):
                    combined.extend(list(self.env.gantt_records))
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            sched_path = os.path.join(self.history_dir, 'scheduling_trace.csv')
            jt_path = os.path.join(self.history_dir, 'job_timeline.csv')

            # delegate CSV formatting/writing to utils.gantt to centralize logic
            if getattr(self, 'allow_history_writes', False):
                if gantt_utils is not None:
                    try:
                        gantt_utils.write_scheduling_trace(sched_path, combined)
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        pass
                    try:
                        gantt_utils.write_job_timeline(jt_path, combined)
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        pass
                else:
                    # fallback: write a minimal scheduling_trace if utils unavailable
                    try:
                        with open(sched_path, 'w') as sf:
                            sf.write('start,end,op_idx,op_name,wc,job_id,operator_grp,arrival,duration\n')
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        pass

                print(f"[Runner] Scheduling trace and job timelines written to {self.history_dir}")
                # Optionally generate a plain-text scheduling timeline for easy human inspection.
                try:
                    if getattr(self, 'allow_history_writes', False):
                        try:
                            # Prefer the centralized gantt_utils wrapper if available
                            timeline = None
                            # Only generate the authoritative scheduling timeline if
                            # history writes are allowed AND the environment has
                            # collected runtime gantt records. This prevents creating
                            # a deterministic placeholder at startup.
                            try:
                                from utils.io_control import allow_history_writes as _allow_fn
                            except Exception:
                                def _allow_fn():
                                    return False

                            try:
                                gr_count = len(getattr(self.env, 'gantt_records', []) or [])
                            except Exception:
                                gr_count = 0

                            if _allow_fn() and gr_count > 0:
                                try:
                                    print(f"[Runner] Writing Scheduling_Timeline.txt (final) gantt_records={gr_count}")
                                except Exception:
                                    pass

                                if gantt_utils is not None and hasattr(gantt_utils, 'generate_scheduling_timeline'):
                                    try:
                                        # Do not perform an authoritative write here: the
                                        # per-episode generator (called by RolloutWorker)
                                        # already writes episode blocks. To avoid
                                        # duplicate episode blocks we only request a
                                        # non-writing preview from the final sweep.
                                        timeline = gantt_utils.generate_scheduling_timeline(self.env, write_if_allowed=False)
                                    except Exception:
                                        # fall through to direct import fallback
                                        timeline = None
                                if timeline is None:
                                    try:
                                        # direct import fallback — do not request an
                                        # authoritative write here to prevent duplicate
                                        # episode blocks (per-episode writes already
                                        # happen elsewhere).
                                        from utils.gantt import generate_scheduling_timeline
                                        timeline = generate_scheduling_timeline(self.env, write_if_allowed=False)
                                    except Exception:
                                        timeline = None
                                # If the generator returned text, log a short message indicating where it was written.
                                if timeline:
                                    try:
                                        tl_path = os.path.join(self.history_dir, 'scheduling_timeline.txt')
                                        print(f"[Runner] Scheduling timeline generated and written to {tl_path}")
                                    except Exception:
                                        print("[Runner] Scheduling timeline generated.")
                            else:
                                try:
                                    print(f"[Runner] Skipped final timeline generation (no runtime data yet, gantt_records={gr_count}, allow_history_writes={_allow_fn()})")
                                except Exception:
                                    pass
                        except Exception:
                            logging.getLogger(__name__).exception("Exception caught while generating scheduling_timeline", exc_info=True)
                except Exception:
                    # non-fatal: do not break the runner if timeline generation fails
                    pass
        except Exception as e:
            print("[WARN] Could not write scheduling/job timeline files:", e)

        # === Create run_summary.json with last-10 averages and utilization stats ===
        try:
            import json
            summary = {}

            # last 10 episode rewards
            try:
                rewards_path = os.path.join(self.history_dir, 'episode_rewards.txt')
                rewards = []
                if os.path.exists(rewards_path):
                    with open(rewards_path, 'r') as rf:
                        for ln in rf:
                            try:
                                parts = ln.strip().split(',')
                                if len(parts) >= 2:
                                    rewards.append(float(parts[1]))
                            except Exception as e:
                                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                continue
                # fallback to learning_metrics.csv if needed
                if not rewards:
                    lm_path = os.path.join(self.history_dir, 'learning_metrics.csv')
                    if os.path.exists(lm_path):
                        with open(lm_path, 'r') as lf:
                            lines = [l.strip() for l in lf.readlines() if l.strip()]
                            if len(lines) > 1:
                                for ln in lines[1:]:
                                    parts = ln.split(',')
                                    try:
                                        rewards.append(float(parts[2]))
                                    except Exception as e:
                                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                        continue
                summary['last_10_avg_reward'] = float(sum(rewards[-10:]) / max(1, len(rewards[-10:]))) if rewards else None
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                summary['last_10_avg_reward'] = None

            # average loss and td over last 10 entries
            try:
                def _read_last_floats(path, n=10):
                    vals = []
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            for ln in f:
                                try:
                                    v = float(ln.strip())
                                    vals.append(v)
                                except Exception as e:
                                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                    continue
                    return vals[-n:]

                loss_vals = _read_last_floats(os.path.join(self.history_dir, 'loss.txt'), 10)
                td_vals = _read_last_floats(os.path.join(self.history_dir, 'td_error.txt'), 10)
                summary['last_10_avg_loss'] = float(sum(loss_vals) / max(1, len(loss_vals))) if loss_vals else None
                summary['last_10_avg_td'] = float(sum(td_vals) / max(1, len(td_vals))) if td_vals else None
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                summary['last_10_avg_loss'] = None
                summary['last_10_avg_td'] = None

            # machine and operator utilization: read kpi_log.txt if present
            try:
                util_m_vals = []
                util_o_vals = []
                kpi_path = os.path.join(self.history_dir, 'kpi_log.txt')
                if os.path.exists(kpi_path):
                    with open(kpi_path, 'r') as kf:
                        for ln in kf:
                            parts = ln.strip().split(',')
                            if len(parts) >= 4:
                                try:
                                    util_m_vals.append(float(parts[2]))
                                    util_o_vals.append(float(parts[3]))
                                except Exception as e:
                                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                    continue
                # fallback to env util functions if kpi not available
                if not util_m_vals or not util_o_vals:
                    try:
                        # Best-effort fallback: compute instantaneous utilizations
                        try:
                            mres = list(getattr(self.env, 'machine_resources', []) or [])
                            if mres:
                                occupied = 0
                                for r in mres:
                                    try:
                                        users = getattr(r, 'users', None)
                                        if users is not None:
                                            if len(users) > 0:
                                                occupied += 1
                                            continue
                                        if getattr(r, 'count', None) is not None:
                                            if int(getattr(r, 'count', 0)) > 0:
                                                occupied += 1
                                    except Exception:
                                        continue
                                util_m = float(occupied) / max(1.0, float(len(mres)))
                            else:
                                util_m = 0.0
                        except Exception:
                            util_m = 0.0
                        try:
                            og = list(getattr(self.env, 'operator_groups', []) or [])
                            if og:
                                occ_o = 0
                                for g in og:
                                    try:
                                        users = getattr(g, 'users', None)
                                        if users is not None:
                                            if len(users) > 0:
                                                occ_o += 1
                                            continue
                                        if getattr(g, 'count', None) is not None:
                                            if int(getattr(g, 'count', 0)) > 0:
                                                occ_o += 1
                                    except Exception:
                                        continue
                                util_o = float(occ_o) / max(1.0, float(len(og)))
                            else:
                                util_o = 0.0
                        except Exception:
                            util_o = 0.0
                        util_m_vals.append(util_m); util_o_vals.append(util_o)
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        pass

                summary['avg_machine_utilization'] = float(sum(util_m_vals) / max(1, len(util_m_vals))) if util_m_vals else None
                summary['avg_operator_utilization'] = float(sum(util_o_vals) / max(1, len(util_o_vals))) if util_o_vals else None
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                summary['avg_machine_utilization'] = None
                summary['avg_operator_utilization'] = None

            # write summary (opt-in)
            try:
                if getattr(self, 'allow_history_writes', False):
                    # Merge top-level summary fields into existing run_summary.json if present
                    summary_path = os.path.join(self.history_dir, 'run_summary.json')
                    out_data = {}
                    try:
                        if os.path.exists(summary_path):
                            with open(summary_path, 'r', encoding='utf-8') as rf:
                                out_data = json.load(rf) or {}
                    except Exception:
                        out_data = {}

                    # Preserve existing evolutions list if present
                    evols = out_data.get('evolutions') if isinstance(out_data, dict) else None
                    # Update top-level fields with computed summary values (do not erase evolutions)
                    if not isinstance(out_data, dict):
                        out_data = {}
                    out_data['last_10_avg_reward'] = summary.get('last_10_avg_reward')
                    out_data['last_10_avg_loss'] = summary.get('last_10_avg_loss')
                    out_data['last_10_avg_td'] = summary.get('last_10_avg_td')
                    out_data['avg_machine_utilization'] = summary.get('avg_machine_utilization')
                    out_data['avg_operator_utilization'] = summary.get('avg_operator_utilization')

                    # restore evolutions if they existed previously
                    if evols is not None:
                        out_data['evolutions'] = evols

                    # If we have per-evolution summaries, mirror the most
                    # recent evolution's averaged metrics to the top-level
                    # fields so external consumers can read a concise
                    # summary without scanning evolutions.
                    try:
                        if isinstance(out_data, dict) and 'evolutions' in out_data and len(out_data.get('evolutions', [])) > 0:
                            last = out_data['evolutions'][-1]
                            out_data['avg_machine_utilization'] = last.get('avg_machine_utilization', 0)
                            out_data['avg_operator_utilization'] = last.get('avg_operator_utilization', 0)
                            out_data['avg_epoch_reward'] = last.get('avg_epoch_reward', 0)
                            out_data['avg_makespan'] = last.get('avg_makespan', last.get('average_makespan', 0))
                    except Exception:
                        # best-effort: do not fail the final write if mirroring fails
                        pass

                    with open(summary_path, 'w', encoding='utf-8') as sf:
                        json.dump(out_data, sf, indent=2)
                    print(f"[Runner] run_summary.json written to {summary_path}")
                    print("system fully functional")
            except Exception as e:
                print("[WARN] Could not write run_summary.json:", e)
        except Exception as e:
            print("[WARN] Exception while creating run_summary:", e)

        # === Post-training moving average plot (opt-in) ===
        try:
            if getattr(self, 'allow_history_writes', False):
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

        # === Save standalone PNGs for loss, td_error and episode rewards ===
        try:
            import csv
            import matplotlib.pyplot as _plt

            # Loss raw plot
            try:
                if not getattr(self, 'allow_history_writes', False):
                    raise RuntimeError('History writes disabled')
                loss_vals = np.loadtxt(os.path.join(self.history_dir, 'loss.txt'))
                _plt.figure()
                _plt.plot(loss_vals, label='Loss', color='tab:blue')
                _plt.xlabel('Train Step'); _plt.ylabel('Loss')
                _plt.title('Training Loss')
                _plt.legend(); _plt.tight_layout()
                _plt.savefig(os.path.join(self.history_dir, 'loss.png'), dpi=150)
                _plt.close()
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            # TD error raw plot
            try:
                if not getattr(self, 'allow_history_writes', False):
                    raise RuntimeError('History writes disabled')
                td_vals = np.loadtxt(os.path.join(self.history_dir, 'td_error.txt'))
                _plt.figure()
                _plt.plot(td_vals, label='TD Error', color='tab:orange')
                _plt.xlabel('Train Step'); _plt.ylabel('TD Error')
                _plt.title('TD Error')
                _plt.legend(); _plt.tight_layout()
                _plt.savefig(os.path.join(self.history_dir, 'td_error.png'), dpi=150)
                _plt.close()
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            # Episode reward plot from learning_metrics.csv
            try:
                lm_path = os.path.join(self.history_dir, 'learning_metrics.csv')
                episodes = []
                rewards = []
                if os.path.exists(lm_path):
                    with open(lm_path, 'r') as lf:
                        reader = csv.DictReader(lf)
                        for row in reader:
                            try:
                                episodes.append(int(row.get('episode', len(episodes))))
                                rewards.append(float(row.get('episode_reward', 0.0)))
                            except Exception as e:
                                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                                continue
                if rewards:
                    _plt.figure()
                    _plt.plot(rewards, label='Episode Reward', color='tab:green')
                    _plt.xlabel('Episode'); _plt.ylabel('Reward')
                    _plt.title('Episode Reward over Time')
                    _plt.legend(); _plt.tight_layout()
                    _plt.savefig(os.path.join(self.history_dir, 'episode_rewards.png'), dpi=150)
                    _plt.close()
            except Exception as e:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                pass

            print('[Runner] Loss/TD/Reward PNGs saved.')
        except Exception as e:
            print('[WARN] Could not create loss/reward PNGs:', e)

    def evaluate(self, all_gantt_data, global_ep_idx):
        win_number, episode_rewards, gantt_eval = 0, 0, []
        for _ in range(self.args.evaluate_epoch):
            # Use the SimPy event-driven execution exclusively.
            _, ep_reward, win_tag, gantt = self._run_event_driven_episode(global_ep_idx, evaluate=True)
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
    # Prefer per-machine 'avail_row' as the canonical mask. Operator-granular
    # masks (historically named 'avail_mask') are deprecated and removed
    # from runtime. When operator-granular actions are required, expand
    # the canonical per-machine row deterministically using
    # `np.repeat(avail_row, num_ops)`. (operator-level masks are deprecated since vX.Y)
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
                # Prefer canonical per-machine avail_row and expand deterministically
                # to per-(machine×operator) flattened space. If avail_row missing,
                # compute it deterministically via `build_machine_major_mask`.
                try:
                    ar = item.get('avail_row')
                    if ar is not None:
                        try:
                            ops = int(getattr(self.env, 'num_ops', 1))
                        except Exception:
                            ops = 1
                        r = np.asarray(ar, dtype=np.int32)
                        expanded = np.repeat(r.astype(np.int32), ops)
                        avail_batch.append(expanded.tolist())
                        continue
                except Exception:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                # Try to compute canonical per-machine row deterministically
                ar = None
                try:
                    if build_machine_major_mask is not None:
                        jid = item.get('job_id')
                        ar = build_machine_major_mask(self.env, jid)
                except Exception:
                    ar = None

                if ar is not None:
                    try:
                        ops = int(getattr(self.env, 'num_ops', 1))
                    except Exception:
                        ops = 1
                    r = np.asarray(ar, dtype=np.int32)
                    expanded = np.repeat(r.astype(np.int32), ops)
                    avail_batch.append(expanded.tolist())
                    continue

                # final deterministic permissive fallback: all ones
                try:
                    num_m = int(len(getattr(self.env.workcenters_meta, 'machine_list', []) or []))
                    ops = int(getattr(self.env, 'num_ops', 1))
                    avail_batch.append([1] * (max(1, num_m) * max(1, ops)))
                    continue
                except Exception:
                    avail_batch.append([1])
                    continue

            # Default behavior: provide per-machine availability (n_actions == num_wcs)
            # If a granular mask exists but user didn't request granular actions,
            # reduce it by OR-ing per-operator slots into a per-machine vector.
            # Prefer per-machine avail_row for agents; reduce operator-granular
            # masks only as a fallback for compatibility.
            ar = item.get('avail_row')
            if ar is not None:
                try:
                    per_machine = [1 if int(bool(x)) else 0 for x in list(ar)]
                    avail_batch.append(per_machine)
                    continue
                except Exception:
                    logging.getLogger(__name__).exception("Exception caught", exc_info=True)

            # Try to compute canonical per-machine row deterministically
            try:
                if build_machine_major_mask is not None:
                    jid = item.get('job_id')
                    row = build_machine_major_mask(self.env, jid)
                    if row:
                        per_machine = [1 if int(bool(x)) else 0 for x in list(row)]
                        avail_batch.append(per_machine)
                        continue
            except Exception:
                logging.getLogger(__name__).exception("Exception caught", exc_info=True)

            # final fallback: None (no mask info)
            avail_batch.append(None)

        # common agent APIs attempted (in order)
        try:
            if hasattr(self.agents, "select_actions"):
                return self.agents.select_actions(obs_batch, avail_batch, evaluate=evaluate)
            if hasattr(self.agents, "choose_actions"):
                return self.agents.choose_actions(obs_batch, avail_batch, evaluate=evaluate)
            if hasattr(self.agents, "act"):
                return self.agents.act(obs_batch, avail_batch, evaluate=evaluate)
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass

        # fallback to rolloutWorker if it provides a decision helper
        try:
            if hasattr(self.rolloutWorker, "decide_batch"):
                return self.rolloutWorker.decide_batch(batch, evaluate=evaluate)
        except Exception as e:
            logging.getLogger(__name__).exception("Exception caught", exc_info=True)
            pass

    # last resort: simple deterministic / random pick from allowed_machine_indices
        actions = []
        for item in batch:
            allowed = item.get("allowed_machine_indices", [])
            if not allowed:
                actions.append(None)
            else:
                # pick first available or random if evaluate==False
                if evaluate:
                    actions.append(int(allowed[0]))
                else:
                    try:
                        actions.append(int(self.rolloutWorker.rng.choice(allowed)))
                    except Exception as e:
                        logging.getLogger(__name__).exception("Exception caught", exc_info=True)
                        actions.append(int(np.random.choice(allowed)))
        return actions

    def _run_event_driven_episode(self, global_ep_idx, evaluate=False):
        # Delegate event-driven execution to the RolloutWorker exclusively.
        return self.rolloutWorker.run_event_driven_episode(global_ep_idx, evaluate=evaluate, runner_args=self.run_args)

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