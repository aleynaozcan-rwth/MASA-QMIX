# Diagram Specifications for Thesis Presentation

## 🎯 İKİ DİAGRAM STRATEJİSİ

### Diagram 1: Standard MASA-QMIX
**Amaç:** Fixed agents, synchronous decisions, standard setup

### Diagram 2: Job-Centric MASA-QMIX  
**Amaç:** Dynamic agents, asynchronous/event-driven decisions, innovations vurgulu

---

## 📊 DIAGRAM 1: STANDARD MASA-QMIX

### Layout Önerisi:

```
┌─────────────────────────────────────────────────────────┐
│  Standard MASA-QMIX: Fixed Agents, Synchronous         │
└─────────────────────────────────────────────────────────┘

     Time Step t                    Time Step t+1
     
   ┌──────────┐                    ┌──────────┐
   │ Robot 1  │ ───o^1──→          │ Robot 1  │
   │ (Agent)  │          ↓          │          │
   └──────────┘          │          └──────────┘
                         │
   ┌──────────┐          │          ┌──────────┐
   │ Robot 2  │ ───o^2──→│          │ Robot 2  │
   │ (Agent)  │          ↓          │          │
   └──────────┘          │          └──────────┘
                    ┌────────┐
   ┌──────────┐     │  GRU   │      ┌──────────┐
   │ Robot 3  │ ───o^3─→Based ─→    │ Robot 3  │
   │ (Agent)  │     │Q-networks│     │          │
   └──────────┘     └────┬───┘      └──────────┘
                         │
                    Q₁, Q₂, Q₃
                         ↓
                  ┌─────────────┐
                  │   Mixing    │ ←─ Global State s_t
                  │   Network   │
                  └──────┬──────┘
                         ↓
                      Q_tot
                         ↓
                  Actions: u₁, u₂, u₃
                  (ALL AT ONCE - SYNCHRONOUS)
```

### Key Features to Show:
- ✅ **N = 3 (Fixed)** - Sayı değişmiyor
- ✅ **Synchronized** - Hepsi t anında karar veriyor
- ✅ **Simple actions** - Hareket komutları
- ✅ **Single box: "GRU-based Q-networks"** - Patronunuzun dediği gibi!

---

## 📊 DIAGRAM 2: JOB-CENTRIC MASA-QMIX (İMPROVED)

### Layout Önerisi - Asynchronous Decision Vurgulu:

```
┌──────────────────────────────────────────────────────────────┐
│  Job-Centric MASA-QMIX: Dynamic Agents, Event-Driven        │
└──────────────────────────────────────────────────────────────┘

TIMELINE (Simulation Time):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→
t=0    t=2.3    t=5.1    t=7.8    t=10.2  ...

  ↓      ↓        ↓        ↓         ↓
 Job1  Job2     Job1     Job3      Job2
arrive arrive  complete arrive   complete
         ↓        ↓        ↓         ↓
      DECISION DECISION DECISION DECISION
      (async!) (async!) (async!) (async!)


┌─────────────────────────────────────────────────────────┐
│  EVENT-TRIGGERED DECISION FLOW                          │
└─────────────────────────────────────────────────────────┘

When Job i needs decision (arrival OR operation complete):

   ┌─────────────────┐
   │ Job i (Agent)   │
   │ Status: ACTIVE  │
   └────────┬────────┘
            │
            │ o^i (observation)
            ↓
   ┌─────────────────┐
   │  GRU-based      │
   │  Q-networks     │ ←── Shared across all jobs
   │  (per-job h_i)  │
   └────────┬────────┘
            │
            │ Q-values for all (m,o) pairs
            ↓
   ┌─────────────────┐
   │  ACTION MASK    │ ←── ⚡ INNOVATION!
   │  Capability     │     C[operation, (m,o)]
   │  Matrix         │
   └────────┬────────┘
            │
            │ Q_masked (invalid → -1e9)
            ↓
   ┌─────────────────┐
   │  Select (m,o)   │
   │  argmax Q_mask  │
   └────────┬────────┘
            │
            │ u^i = (machine, operator)
            ↓
   ┌─────────────────┐
   │  Assign to      │
   │  Workcenter     │
   └─────────────────┘


CURRENT ACTIVE JOBS (N(t) varies!):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  t=5.1:  Job₁ ●  Job₂ ●  [N(t)=2]                      │
│           ↓                                             │
│         Decision                                        │
│           (Job₁ completes Op1)                          │
│                                                         │
│  t=7.8:  Job₁ ●  Job₂ ●  Job₃ ●  [N(t)=3] ⚡           │
│                           ↓                             │
│                         Decision                        │
│                         (Job₃ arrives)                  │
│                                                         │
│  t=10.2: Job₁ ●  Job₃ ●  [N(t)=2]                      │
│                    ↓                                    │
│                  Decision                               │
│                  (Job₂ completes)                       │
│                                                         │
└─────────────────────────────────────────────────────────┘

ALL Q-values → Mixing Network → Q_tot (for learning)
```

---

## 🎨 VİZUAL İYİLEŞTİRMELER

### 1. Color Coding (Önemli!)

**Standard MASA-QMIX:**
- 🔵 Mavi ton: Tüm elemanlar
- Sakin, klasik görünüm

**Job-Centric MASA-QMIX:**
- 🟢 Yeşil: Standard components (GRU-based Q-networks, Mixing Network)
- 🟠 Turuncu: **YENİ** innovations (Action Mask, Dynamic N(t))
- 🔴 Kırmızı: Event triggers (Job Arrival, Operation Complete)

### 2. Annotations (Oklar ve Notlar)

```
┌─────────────────────────────────────┐
│ ⚡ INNOVATION 1: Dynamic N(t)       │ ← TURUNCU KUTU
│ Agents come and go stochastically  │
└─────────────────────────────────────┘
          ↓ (kalın ok)
     Active Jobs


┌─────────────────────────────────────┐
│ ⚡ INNOVATION 2: Action Masking     │ ← TURUNCU KUTU
│ C[op, (m,o)] ∈ {0,1}               │
│ Machine capability + Operator skill │
└─────────────────────────────────────┘
          ↓ (kalın ok)
     Q_masked


┌─────────────────────────────────────┐
│ ⚡ INNOVATION 3: Event-Driven       │ ← KIRMIZI KUTU
│ Decision ONLY when:                 │
│  • Job arrives                      │
│  • Operation completes              │
└─────────────────────────────────────┘
          ↓ (kesik ok)
     Asynchronous Flow
```

### 3. Time Visualization (Kritik!)

**SİZİN ŞU ANKİ PROBLEM:** Job agents hepsi aynı anda karar veriyormuş gibi görünüyor

**ÇÖZÜM:** Timeline ile event-driven göstermek

```
Option A: Timeline üstte
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ Simulation Time
  ↓       ↓        ↓       ↓
Event1  Event2   Event3  Event4
(J1)    (J2)     (J1)    (J3)
  ↓       ↓        ↓       ↓
Decision Decision Decision Decision
```

```
Option B: Çerçeve içinde notlar
┌─────────────────────────────────────┐
│  ASYNCHRONOUS DECISIONS             │
│  ≠ All agents at same time          │
│  = Each job decides when needed     │
│                                     │
│  Job 1 at t=2.3  ← Event trigger   │
│  Job 2 at t=5.1  ← Event trigger   │
│  Job 3 at t=7.8  ← Event trigger   │
└─────────────────────────────────────┘
```

**SİZİN DEDİĞİNİZ GİBİ:**
- Bu kısmı bir çerçeveye alabilirsiniz
- Oklar ekleyerek event flow gösterebilirsiniz
- "⚠️ NOT SYNCHRONOUS!" diye vurgulayabilirsiniz

---

## 🔧 POWERPOINT/LATEX İMPLEMENTASYON

### PowerPoint için:

1. **Slide 1: Standard MASA-QMIX**
   - Layout: 3 robot boxes solda
   - Ortada: Tek kutu "GRU-based Q-networks"
   - Sağda: Mixing Network
   - Mavi renk tonu
   - Alt nota: "Fixed N=3, Synchronous, Time-stepped"

2. **Slide 2: Job-Centric MASA-QMIX**
   - Layout: Timeline üstte (stochastic arrivals)
   - Sol: Current active jobs (değişken sayı)
   - Orta: Decision flow with action masking
   - Sağ: Event triggers
   - Renkli: Yeşil (standard), Turuncu (innovations)
   - Alt nota: "Dynamic N(t), Asynchronous, Event-driven"

### LaTeX TikZ için:

```latex
% Standard MASA-QMIX
\begin{tikzpicture}[scale=0.8]
    % Fixed agents
    \node[agent, fill=blue!20] (a1) at (0,3) {Robot 1};
    \node[agent, fill=blue!20] (a2) at (0,1.5) {Robot 2};
    \node[agent, fill=blue!20] (a3) at (0,0) {Robot 3};
    
    % Q-network (single box!)
    \node[qnet, fill=green!20] (qnet) at (4,1.5) {GRU-based Q-networks};
    
    % Arrows
    \draw[->] (a1) -- (qnet);
    \draw[->] (a2) -- (qnet);
    \draw[->] (a3) -- (qnet);
    
    % Mixing
    \node[mix] (mix) at (8,1.5) {Mixing Network};
    \draw[->] (qnet) -- (mix);
    
    % Note
    \node[note, below] at (4,-1) {Synchronous: All at time t};
\end{tikzpicture}
```

```latex
% Job-Centric MASA-QMIX
\begin{tikzpicture}[scale=0.8]
    % Timeline
    \draw[thick, ->] (0,5) -- (10,5) node[right] {Sim Time};
    \node[event, fill=red!20] at (2,5) {↓ J1 arrive};
    \node[event, fill=red!20] at (5,5) {↓ J2 complete};
    \node[event, fill=red!20] at (8,5) {↓ J3 arrive};
    
    % Decision flow
    \node[innovate, fill=orange!20] at (5,3) {⚡ Event-Driven};
    \node[innovate, fill=orange!20] at (5,1) {⚡ Action Masking};
    
    % Note box
    \node[framebox, text width=8cm] at (5,-1) {
        ASYNCHRONOUS: Each job decides at its own event time,
        not synchronized with other jobs!
    };
\end{tikzpicture}
```

---

## 📝 DİAGRAM CAPTION'LARI

### Figure X: Standard MASA-QMIX
"Standard MASA-QMIX architecture with fixed agents (N=3). All agents observe the environment synchronously at time step t and make decisions simultaneously. Q-networks are GRU-based for handling partial observability."

### Figure Y: Job-Centric MASA-QMIX  
"Our job-centric adaptation showing event-driven decision making. Jobs (agents) arrive stochastically and make decisions asynchronously when triggered by events (arrival or operation completion). Action masking enforces manufacturing constraints, and N(t) varies dynamically throughout the episode."

---

## ✅ CHECKLIST - DİAGRAMDA OLMASI GEREKENLER

### Standard MASA-QMIX:
- [ ] Fixed N agents (3-4 tane yeterli)
- [ ] Synchronous timing vurgusu
- [ ] Tek kutu: "GRU-based Q-networks" (patronun dediği gibi!)
- [ ] Mixing network
- [ ] Mavi/nötr renkler
- [ ] Basit action space (movement)

### Job-Centric MASA-QMIX:
- [ ] **Timeline/event visualization** ← EN ÖNEMLİ!
- [ ] Variable N(t) - sayı değişiyor göster
- [ ] Asynchronous decision flow - HER JOB AYRI ZAMAN!
- [ ] Action masking box (turuncu)
- [ ] Event triggers (kırmızı/vurgulu)
- [ ] Capability matrix mention
- [ ] "⚡ INNOVATION" labels
- [ ] Çerçeve + oklar ile async vurgusu

---

## 🎯 SİZİN SORUNUNUZA ÖZEL ÇÖZÜM

**Problem:** "Job agentlarım hepsi aynı anda karar veriyormuş gibi görünüyor"

**Çözüm 1: Timeline + Event Markers**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ Simulation Time
  ↓         ↓          ↓         ↓
t=2.1     t=4.5      t=7.2     t=9.8
Job1      Job2       Job1      Job3
decides   decides    decides   decides
```

**Çözüm 2: Sequence Diagram Style**
```
Job1:  ────●──────────────●──────────
           decision       decision

Job2:  ──────────●──────────────────●
              decision            decision

Job3:  ────────────────────●─────────
                        decision

Time:  ━━━━━━━━━━━━━━━━━━━━━━━━━━━→
       (events at different times!)
```

**Çözüm 3: Çerçeve + Annotation (sizin dediğiniz gibi)**
```
┌─────────────────────────────────────┐
│  ⚠️ IMPORTANT: ASYNCHRONOUS!        │
│                                     │
│  Unlike standard QMIX where all     │
│  agents act at same time step,      │
│  here each job makes decision       │
│  ONLY when event occurs:            │
│    • Job arrives  ──→ t=2.1        │
│    • Op completes ──→ t=4.5        │
│                                     │
│  NO SYNCHRONIZED TIME STEPS!        │
└─────────────────────────────────────┘
```

İstediğiniz kombinasyonu kullanabilirsiniz!

---

## 🚀 NEXT STEPS

1. **Hangi tool kullanacaksınız?**
   - PowerPoint → Shapes + SmartArt
   - LaTeX → TikZ (daha profesyonel)
   - Draw.io → Hızlı prototyping
   - Python matplotlib → Programmatic

2. **Ben size şunu hazırlayayım mı?**
   - [ ] LaTeX TikZ kodu (kopyala-yapıştır)
   - [ ] Python script (matplotlib ile çiz)
   - [ ] Mermaid diagram (markdown)
   - [ ] Detaylı PowerPoint guide

3. **Priority:**
   - Önce ASYNCHRONOUS flow'u net göster
   - Sonra action masking ekle  
   - Son olarak renk/style polish

Hangi tool'u kullanacaksınız? Size o tool için tam implement edeyim! 🎨
