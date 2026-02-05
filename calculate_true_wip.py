#!/usr/bin/env python3
"""
Timeline'dan gerçek WIP hesaplama:
- Hiç operasyona başlamamış işler → Queue (WIP değil)
- En az 1 operasyona başlamış ama bitirmemiş operasyonlar → WIP Operation
- En az 1 operasyona başlamış ama tamamlanmamış işler → WIP Job
"""

import re
import pandas as pd
from collections import defaultdict

timeline_file = '/tmp/experimental_scheduling_timeline.txt'

# Episode verilerini sakla
episode_data = []
current_episode = -1

# Her episode için tracking
jobs_arrived = set()  # Episode boyunca gelen tüm işler
jobs_started = set()  # En az 1 operasyona başlamış işler
jobs_completed = set()  # Tamamlanmış işler
job_total_ops = {}  # Job başına toplam operasyon sayısı (arrival'dan)
ops_started = defaultdict(set)  # Job başına başlatılan operasyonlar
ops_completed = defaultdict(set)  # Job başına tamamlanan operasyonlar
total_ops_arrived = 0  # Toplam gelen operasyon sayısı

def finalize_episode():
    """Episode sonunda WIP hesapla"""
    global current_episode, jobs_arrived, jobs_started, jobs_completed, ops_started, ops_completed, job_total_ops
    
    if current_episode < 0:
        return
    
    # Hiç operasyona başlamamış işler (queue'da bekleyenler)
    jobs_queued = jobs_arrived - jobs_started
    
    # WIP Jobs = En az 1 operasyona başlamış ama tamamlanmamış
    wip_jobs = jobs_started - jobs_completed
    
    # WIP Operations = Sadece WIP job'ların kalan operasyonları
    total_ops_for_wip_jobs = sum(job_total_ops.get(job, 0) for job in wip_jobs)
    ops_completed_for_wip_jobs = sum(len(ops_completed[job]) for job in wip_jobs)
    wip_operations = total_ops_for_wip_jobs - ops_completed_for_wip_jobs
    
    # Tüm operasyonlar (completed jobs dahil)
    total_ops_started = sum(len(ops) for ops in ops_started.values())
    total_ops_completed = sum(len(ops) for ops in ops_completed.values())
    
    episode_data.append({
        'episode': current_episode,
        'jobs_arrived': len(jobs_arrived),
        'jobs_queued': len(jobs_queued),  # Hiç işleme başlamamış
        'jobs_started': len(jobs_started),  # En az 1 operasyona başlamış
        'jobs_completed': len(jobs_completed),
        'wip_jobs': len(wip_jobs),  # Başlamış ama bitmemiş
        'ops_arrived': total_ops_arrived,  # Toplam gelen operasyon
        'ops_started': total_ops_started,
        'ops_completed': total_ops_completed,
        'wip_operations': wip_operations  # Sadece WIP job'ların kalan ops'ları
    })

def reset_episode():
    """Yeni episode için sıfırla"""
    global jobs_arrived, jobs_started, jobs_completed, ops_started, ops_completed, total_ops_arrived, job_total_ops
    jobs_arrived = set()
    jobs_started = set()
    jobs_completed = set()
    job_total_ops = {}
    ops_started = defaultdict(set)
    ops_completed = defaultdict(set)
    total_ops_arrived = 0

print("📖 Timeline dosyası parse ediliyor...")

with open(timeline_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        if line_num % 100000 == 0:
            print(f"  İşlenen satır: {line_num:,}")
        
        # Episode değişimi
        if '=== EPISODE' in line:
            match = re.search(r'=== EPISODE (\d+) ===', line)
            if match:
                new_episode = int(match.group(1))
                if current_episode >= 0:  # İlk episode değilse, öncekini kaydet
                    finalize_episode()
                current_episode = new_episode
                reset_episode()
                continue
        
        # Job arrival - sadece timeline'daki Job_X formatını say
        if 'New job' in line and 'arrived' in line and 'Job_' in line:
            # "New job Job_5 arrived with 3 ops"
            match = re.search(r'New job (Job_\d+) arrived with (\d+) ops', line)
            if match:
                job_id = match.group(1)
                num_ops = int(match.group(2))
                jobs_arrived.add(job_id)
                job_total_ops[job_id] = num_ops
                total_ops_arrived += num_ops
        
        # Operation start
        if 'started on' in line:
            # "Job_1.Op1 started on M3"
            match = re.search(r'(Job_\d+|[a-zA-Z_]*\d+)\.(Op\d+|[a-zA-Z]+\d+) started', line)
            if match:
                job_id = match.group(1)
                op_id = match.group(2)
                jobs_started.add(job_id)
                ops_started[job_id].add(op_id)        
        # Operation finish
        if 'finished' in line and '→' in line:
            # "Job_1.Op1 finished → next queued by O1"
            match = re.search(r'(Job_\d+)\.(Op\d+) finished', line)
            if match:
                job_id = match.group(1)
                op_id = match.group(2)
                ops_completed[job_id].add(op_id)        
        # Job completion
        if 'completed all operations' in line:
            # "Job_4 completed all operations"
            match = re.search(r'(Job_\d+|[a-zA-Z_]*\d+) completed all operations', line)
            if match:
                job_id = match.group(1)
                jobs_completed.add(job_id)


# Son episode'u kaydet
finalize_episode()

print(f"\n✅ Parse tamamlandı: {len(episode_data)} episode")

# DataFrame'e çevir
df = pd.DataFrame(episode_data)

# İstatistikler
print(f"\n📊 WIP İSTATİSTİKLERİ:")
print(f"  Ortalama jobs arrived/episode: {df['jobs_arrived'].mean():.1f}")
print(f"  Ortalama jobs queued (hiç işlenmemiş): {df['jobs_queued'].mean():.1f}")
print(f"  Ortalama jobs started: {df['jobs_started'].mean():.1f}")
print(f"  Ortalama jobs completed: {df['jobs_completed'].mean():.1f}")
print(f"  Ortalama WIP jobs: {df['wip_jobs'].mean():.1f}")
print(f"  Ortalama WIP operations: {df['wip_operations'].mean():.1f}")

# WIP oranları hesapla - hepsi "arrived"a göre (gelen işlere göre)
df['job_wip_ratio'] = (df['wip_jobs'] / df['jobs_arrived'] * 100).fillna(0)
df['job_completion_ratio'] = (df['jobs_completed'] / df['jobs_arrived'] * 100).fillna(0)
df['job_queued_ratio'] = (df['jobs_queued'] / df['jobs_arrived'] * 100).fillna(0)

# Operations için: gerçek ops_arrived kullan
df['op_wip_ratio'] = (df['wip_operations'] / df['ops_arrived'] * 100).fillna(0)
df['op_completion_ratio'] = (df['ops_completed'] / df['ops_arrived'] * 100).fillna(0)

print(f"\n📈 WIP ORANLARI:")
print(f"  Job WIP Ratio: {df['job_wip_ratio'].mean():.2f}%")
print(f"  Operation WIP Ratio: {df['op_wip_ratio'].mean():.2f}%")
print(f"  Job Queued Ratio (hiç işlenmemiş): {df['job_queued_ratio'].mean():.2f}%")

# Kaydet
output_file = '/tmp/true_wip_ratios.csv'
df.to_csv(output_file, index=False)
print(f"\n💾 Kaydedildi: {output_file}")

# İlk ve son 10 episode'u göster
print(f"\n📋 İLK 10 EPISODE:")
print(df.head(10)[['episode', 'jobs_arrived', 'jobs_queued', 'wip_jobs', 'wip_operations', 'job_wip_ratio', 'op_wip_ratio']].to_string(index=False))

print(f"\n📋 SON 10 EPISODE:")
print(df.tail(10)[['episode', 'jobs_arrived', 'jobs_queued', 'wip_jobs', 'wip_operations', 'job_wip_ratio', 'op_wip_ratio']].to_string(index=False))
