#!/usr/bin/env python3
"""
Episode 0 sonunda WIP analizi - neden 10+ operasyon bitmemiş?
"""

import re

# Episode 0'ı parse et
episode_0_lines = []
reading = False

with open('/tmp/experimental_scheduling_timeline.txt', 'r') as f:
    for line in f:
        if '=== EPISODE 0 ===' in line:
            reading = True
        elif '=== EPISODE 1 ===' in line:
            break
        elif reading:
            episode_0_lines.append(line)

# Operasyon tracking
ops_started = {}  # (Job, Op) -> start_time
ops_completed = set()  # (Job, Op)
jobs_completed = set()
last_time = 0

for line in episode_0_lines:
    # Time tracking
    time_match = re.search(r'\[t=([\d.]+)\]', line)
    if time_match:
        last_time = float(time_match.group(1))
    
    # Operation start
    if 'started on' in line:
        match = re.search(r'(Job_\d+)\.(Op\d+) started', line)
        if match:
            job = match.group(1)
            op = match.group(2)
            ops_started[(job, op)] = last_time
    
    # Job completion
    if 'completed all operations' in line:
        match = re.search(r'(Job_\d+) completed', line)
        if match:
            job = match.group(1)
            jobs_completed.add(job)
            # Tüm operasyonları tamamlanmış say
            for (j, o) in list(ops_started.keys()):
                if j == job:
                    ops_completed.add((j, o))

print(f"📊 EPISODE 0 ANALİZİ (t={last_time:.2f}'de bitti)")
print(f"{'='*70}")
print(f"Toplam başlatılan operasyon: {len(ops_started)}")
print(f"Tamamlanan operasyon: {len(ops_completed)}")
print(f"WIP operasyon (started but not completed): {len(ops_started) - len(ops_completed)}")
print(f"\nTamamlanan job: {len(jobs_completed)}")
print(f"{'='*70}")

# Hangi operasyonlar WIP'te?
wip_ops = []
for (job, op), start_time in ops_started.items():
    if (job, op) not in ops_completed:
        wip_ops.append((job, op, start_time))

print(f"\n🔧 WIP'TEKİ OPERASYONLAR (başlamış ama bitmemiş):")
wip_ops.sort(key=lambda x: x[2])  # Start time'a göre sırala

for job, op, start_time in wip_ops:
    running_time = last_time - start_time
    print(f"  {job}.{op} → başladı t={start_time:.2f}, çalışma süresi {running_time:.2f} (episode bittiğinde hala devam ediyordu)")

# Hangi joblar hiç tamamlanmadı?
started_jobs = set(job for job, _ in ops_started.keys())
incomplete_jobs = started_jobs - jobs_completed

print(f"\n📦 TAMAMLANMAMIŞ JOBLAR (en az 1 operasyona başlamış ama bitmemiş):")
for job in sorted(incomplete_jobs):
    job_ops_started = [op for (j, op) in ops_started.keys() if j == job]
    job_ops_completed = [op for (j, op) in ops_completed if j == job]
    print(f"  {job}: {len(job_ops_started)} op başladı, {len(job_ops_completed)} op bitti → {len(job_ops_started) - len(job_ops_completed)} WIP")
