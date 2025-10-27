#!/usr/bin/env python3
"""
Simple watcher: monitor PID in run_long_clean.pid; when the process exits,
write a small summary file into my_data_and_graph/historydata/run_finished.txt
and append a marker to run_long_clean.log. This runs as a background process.
"""
import time
import os
import sys
import signal
import re
from datetime import datetime

PID_FILE = os.path.abspath('run_long_clean.pid')
LOG_FILE = os.path.abspath('run_long_clean.log')
OUT_DIR = os.path.abspath('my_data_and_graph/historydata')
OUT_FILE = os.path.join(OUT_DIR, 'run_finished.txt')


def read_pid():
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


def is_running(pid):
    if pid is None:
        return False
    return os.path.exists(f'/proc/{pid}')


def parse_final_summary(logpath, tail_lines=500):
    # read last tail_lines and extract last eval and last TRAIN summary
    from collections import deque
    last_eval=None
    last_train=None
    try:
        with open(logpath,'r') as f:
            tail=list(deque(f, tail_lines))
    except FileNotFoundError:
        return {'error':'log not found'}
    for line in reversed(tail):
        if last_train is None:
            m = re.search(r'TRAIN\] step=\s*(\d+),\s*loss=\s*([0-9.+\-eE,]+)', line)
            if m:
                try:
                    step=int(m.group(1))
                    loss=float(m.group(2).replace(',',''))
                    last_train=(step,loss)
                except:
                    pass
        if last_eval is None:
            m = re.search(r'\[Eval\] Epoch\s*(\d+)\s*\|\s*Reward=\s*([0-9.+\-eE,]+)', line)
            if m:
                try:
                    e=int(m.group(1))
                    r=float(m.group(2).replace(',',''))
                    last_eval=(e,r)
                except:
                    pass
        if last_train and last_eval:
            break
    return {'last_eval':last_eval, 'last_train':last_train}


def main():
    pid = read_pid()
    if pid is None:
        print('watcher: no PID file found at', PID_FILE)
        return 1
    print(f'watcher: monitoring pid {pid}')
    # Wait loop
    while True:
        if not is_running(pid):
            # process ended
            ts = datetime.utcnow().isoformat() + 'Z'
            summary = parse_final_summary(LOG_FILE)
            os.makedirs(OUT_DIR, exist_ok=True)
            try:
                with open(OUT_FILE, 'w') as f:
                    f.write(f'Run finished at {ts}\n')
                    f.write('Summary extracted from log:\n')
                    f.write(str(summary) + '\n')
            except Exception as e:
                print('watcher: failed to write out file', e)
            # append marker to log
            try:
                with open(LOG_FILE, 'a') as f:
                    f.write('\n=== RUN FINISHED at ' + ts + ' ===\n')
            except Exception:
                pass
            print('watcher: detected process end, wrote', OUT_FILE)
            return 0
        time.sleep(15)

if __name__ == '__main__':
    sys.exit(main())
