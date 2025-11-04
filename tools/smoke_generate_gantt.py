import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from environment import MASAEnv

OUT_DIR = './my_data_and_graph/historydata'
os.makedirs(OUT_DIR, exist_ok=True)


def plot_gantt_local(for_gantt_data, filename="gantt.png"):
    if not for_gantt_data or not isinstance(for_gantt_data, (list, tuple)):
        return False
    if len(for_gantt_data) == 0:
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    job_ids = sorted(set([int(rec[4]) for rec in for_gantt_data if isinstance(rec, (list, tuple)) and len(rec) >= 5]))
    colors = list(mcolors.TABLEAU_COLORS.values())
    job_color_map = {jid: colors[i % len(colors)] for i, jid in enumerate(job_ids)}

    max_end = 0
    for rec in for_gantt_data:
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
            color = job_color_map.get(jid, 'gray')
            ax.barh(jid, float(end) - float(start), left=float(start), color=color, edgecolor='black')
            m_label = f"WC{workcenter}"
            op_grp_label = f"O{operator}" if operator is not None else "O?"
            label_text = f"{m_label} | {op_grp_label}"
            ax.text((float(start) + float(end)) / 2, jid, label_text, va='center', ha='center', fontsize=7, color='black')
            max_end = max(max_end, float(end))
        except Exception:
            continue

    ax.set_yticks(job_ids)
    ax.set_yticklabels([str(j) for j in job_ids])
    ax.set_xlabel('Simulation Time (SimPy clock)')
    ax.set_ylabel('JobAgent (ID)')
    ax.set_title('Smoke Gantt (job-color, WC + Operator labels)')
    ax.set_xlim(0, max_end + 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=job_color_map[j]) for j in job_ids]
    labels = [f"Job {j}" for j in job_ids]
    if handles:
        ax.legend(handles, labels, title='Job IDs', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return True


if __name__ == '__main__':
    env = MASAEnv(num_jobs=4, num_operators=2, episode_limit=40)
    # drive the event-driven decision loop: call wait_for_decisions and resume
    # choices deterministically (first allowed machine index) so job processes
    # actually proceed and produce gantt records.
    # reset environment (this starts job processes) then drive the event-driven decision loop
    try:
        env.reset()
    except Exception:
        try:
            env._generate_initial_jobs()
        except Exception:
            pass

    try:
        steps = 0
        while not getattr(env, 'done', False) and env.env.now < env.episode_limit and steps < 1000:
            batch, sim_t = env.wait_for_decisions()
            if not batch:
                break
            for item in batch:
                try:
                    allowed_inds = item.get('allowed_machine_indices') or []
                    if allowed_inds:
                        choice = int(allowed_inds[0])
                    else:
                        # fallback to first machine
                        choice = 0
                    item.get('resume')(choice)
                except Exception:
                    try:
                        item.get('resume')(0)
                    except Exception:
                        pass
            steps += 1
    except Exception:
        pass

    gantt = list(getattr(env, 'gantt_records', []))
    print('Collected gantt records:', len(gantt))
    for g in gantt[:10]:
        print(g)

    png_path = os.path.join(OUT_DIR, 'smoke_gantt.png')
    csv_path = os.path.join(OUT_DIR, 'smoke_gantt.csv')

    # save CSV using utils.gantt helper (consistent formatting)
    try:
        from utils.gantt import write_gantt_csv, format_gantt_records
        write_gantt_csv(csv_path, gantt)
        print('Gantt records sample:', format_gantt_records(gantt[:10]))
    except Exception:
        # fallback to legacy behavior if helpers unavailable
        with open(csv_path, 'w') as cf:
            cf.write('start,end,op_idx,wc,job_id,operator_grp,arrival,duration\n')
            for r in gantt:
                if not isinstance(r, (list, tuple)):
                    continue
                if len(r) >= 8:
                    vals = [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]]
                elif len(r) == 6:
                    vals = [r[0], r[1], r[2], r[3], r[4], r[5], '', '']
                elif len(r) == 5:
                    vals = [r[0], r[1], r[2], r[3], r[4], '', '', '']
                else:
                    continue
                cf.write(','.join([str(x) for x in vals]) + '\n')

    plot_gantt_local(gantt, filename=png_path)
    print('Saved:', png_path, csv_path)
