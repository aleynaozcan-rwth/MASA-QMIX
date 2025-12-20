# PowerPoint Diagram Updates
## Standard MASA-QMIX vs Job-Centric MASA-QMIX

Mevcut PowerPoint tasarımınızı iki versiyona dönüştürmek için yapılacak değişiklikler.

---

## 🎨 DIAGRAM 1: STANDARD MASA-QMIX

### Sol Panel - Agent Q-Network (Detaylı değil, basitleştirilmiş!)

```
┌─────────────────────────┐
│  Select action u^i      │  ← Yeşil oval (üstte)
│  Q_i(o^i, u^i)          │
└────────┬────────────────┘
         │
    ┌────▼────┐
    │ Action  │
    │  Mask   │  ← SİL! (Standard QMIX'te yok)
    │(invalid │
    │ → -1e9) │
    └────┬────┘
         │
    ┌────▼─────────────┐
    │  Q-values for    │  ← Turuncu kutu
    │  all actions     │
    │  Q₁, Q₂, ..., Qₙ │
    └────┬─────────────┘
         │
    ┌────▼──────────────┐
    │  GRU-based        │  ← TEK KUTU! (Patronun isteği)
    │  Q-networks       │     Koyu yeşil
    │                   │
    │  (maintains       │
    │  hidden states)   │
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │ Input:            │  ← Alttta
    │  Observation o_t^i│     ⚠️ DİKKAT: ZAMAN FARKI!
    │  Previous action  │
    │    u_{t-1}^i      │
    └───────────────────┘
         │
         ↓ Output: u_t^i
```

**DEĞİŞİKLİKLER:**
- ❌ "Action Mask" kutusunu SİL
- ❌ "Linear Layer", "RNN Layer (GRU)", "Linear Layer" üçlü yapıyı SİL
- ✅ TEK KUTU: "GRU-based Q-networks (maintains hidden states)"
- ✅ Altına ekle: "Observation o^i, Action u^i"

---

### Orta Panel - Agents

```
┌─────────────────────────────────────────────────┐
│          FIXED AGENTS (N = 3)                   │
└─────────────────────────────────────────────────┘

    ┌──────────────────┐
    │   Q₁(o¹, u¹)     │  ← Agent 1
    │      u¹          │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │   Q₂(o², u²)     │  ← Agent 2
    │      u²          │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │   Q₃(o³, u³)     │  ← Agent 3
    │      u³          │
    └────────┬─────────┘
             │
    
    ┌──────────────────┐
    │ Observation o¹   │  ← Altta, her agent için
    │ Action u¹        │
    └──────────────────┘
    
    ┌──────────────────┐
    │ Observation o²   │
    │ Action u²        │
    └──────────────────┘
    
    ┌──────────────────┐
    │ Observation o³   │
    │ Action u³        │
    └──────────────────┘
```

**DEĞİŞİKLİKLER:**
- ✅ "Job Agent 1, 2, ..., n" yerine "Agent 1, 2, 3"
- ✅ Üst nota ekle: "FIXED AGENTS (N = 3)"
- ✅ Robot/vehicle emojisi ekleyebilirsiniz 🤖
- ❌ "..." (dots) eklemeyebilirsiniz - 3 agent net

---

### Sağ Panel - Mixing Network

```
┌──────────────────────┐
│  Global State G      │  ← Üstte açık mavi
│                      │
│  (agent positions,   │
│   environment state) │
└──────────┬───────────┘
           │
   ┌───────▼────────────┐
   │  Mixing Network    │  ← Mavi kutu
   │                    │
   │  Q₁, Q₂, Q₃, G     │
   └───────┬────────────┘
           │
   ┌───────▼────────────┐
   │  Total Q-Value     │  ← Turuncu kutu (üst)
   │    Q_tot           │
   │ (for loss          │
   │  computation)      │
   └────────────────────┘
           │
   ┌───────▼────────────┐
   │  Joint Action u'   │  ← Turuncu kutu (alt)
   │                    │
   │ (for environment)  │
   └────────────────────┘
```

**DEĞİŞİKLİKLER:**
- ✅ Global State açıklamasını değiştir: "(agent positions, environment state)"
- ✅ Mixing Network input: "Q₁, Q₂, Q₃, G" (fixed 3 agent)

---

### Alt Nota (Önemli!)

```
┌─────────────────────────────────────────────────────────┐
│  STANDARD MASA-QMIX CHARACTERISTICS:                    │
│                                                         │
│  • Fixed N = 3 agents (robots/vehicles)                │
│  • Synchronous: All agents decide at time step t       │
│  • GRU-based Q-networks for partial observability      │
│  • Input: o_t (obs) + u_{t-1} (prev action)  ⚠️        │
│  • No action masking (all actions valid)               │
│  • Time-stepped episodes                               │
└─────────────────────────────────────────────────────────┘
```

---

## 🎨 DIAGRAM 2: JOB-CENTRIC MASA-QMIX (Your Adaptation)

### Sol Panel - Agent Q-Network (Action Masking ile!)

```
┌─────────────────────────┐
│  Select action u^j      │  ← Yeşil oval
│  Q_j(o^j, u^j)          │
└────────┬────────────────┘
         │
    ┌────▼────┐
    │ Action  │  ← ⚡ TEAL/Turkuaz (VUR GULU!)
    │  Mask   │     "INNOVATION 3"
    │(invalid │
    │ → -1e9) │
    │         │
    │ C[op,(m,o)] │  ← Capability matrix
    └────┬────┘
         │
    ┌────▼─────────────┐
    │  Q-values for    │  ← Turuncu kutu
    │  all actions     │
    │  Q₁, Q₂, ..., Qₙ │
    │                  │
    │ (machine-operator│  ← EKLE!
    │      pairs)      │
    └────┬─────────────┘
         │
    ┌────▼──────────────┐
    │  GRU-based        │  ← TEK KUTU (aynı!)
    │  Q-networks       │     Koyu yeşil
    │                   │
    │  (maintains       │
    │  per-job h_j)     │  ← "per-job" ekle!
    └────┬──────────────┘
         │
    ┌────▼───────────────────────┐
    │  Input:                    │  ← Alttta
    │  • Observation o_t^j       │  ⚠️ DİKKAT: ZAMAN FARKI!
    │    (current job state:     │
    │     operations, time, etc) │
    │  • Previous action         │
    │    u_{t-1}^j               │
    │    (prev. (m,o) pair)      │
    └────────────────────────────┘
         │
         ↓ Output: Q-values → select u_t^j
```

**DEĞİŞİKLİKLER:**
- ✅ Action Mask kutusunu EKLE (teal/turkuaz renk)
- ✅ Action Mask yanına: "⚡ INNOVATION 3"
- ✅ Q-values altına ekle: "(machine-operator pairs)"
- ✅ GRU-based altında: "(maintains per-job h_j)"
- ✅ Observation detaylandır: "(job state: operations, time, history)"

---

### Orta Panel - Dynamic Job Agents (Kritik!)

```
┌─────────────────────────────────────────────────┐
│     DYNAMIC JOB AGENTS (N(t) varies!) ⚡        │  ← TURUNCU BAŞLIK
│          INNOVATION 2                           │
└─────────────────────────────────────────────────┘

    ⏱️ EVENT-DRIVEN DECISIONS ⚡
    
    t=2.1 ──→  ┌──────────────────┐
               │ Q₁(o¹, u¹)  Job1 │  ← Pembe/açık somon
               │     u¹           │     
               └──────────────────┘
               ↑ Job 1 arrives
    
    t=4.5 ──→  ┌──────────────────┐
         ...   │ Q₂(o², u²)  Job2 │  ← Pembe/açık somon
               │     u²      ...  │
               └──────────────────┘
               ↑ Job 2 operation completes
    
    t=7.2 ──→  ┌──────────────────┐
               │ Qₙ(oⁿ, uⁿ) Jobn │  ← Pembe/açık somon
         ...   │     uⁿ      ...  │
               └──────────────────┘
               ↑ Job n arrives
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ Simulation Time

    ┌──────────────────────────────────┐
    │ INPUT (to Q-network):            │  ← Altta, soluk sarı
    │                                  │
    │ • Observation o_t^j              │  ⚠️ CURRENT TIME
    │   (current operation, remaining  │
    │    ops, time in system)          │
    │                                  │
    │ • Previous action u_{t-1}^j      │  ⚠️ PREVIOUS TIME
    │   (prev. machine-operator pair)  │
    │                                  │
    │ OUTPUT:                          │
    │ • Current action u_t^j           │
    │   (selected (m,o) pair)          │
    └──────────────────────────────────┘
```

**DEĞİŞİKLİKLER:**
- ✅ Başlık: "DYNAMIC JOB AGENTS (N(t) varies!) ⚡ INNOVATION 2" (turuncu)
- ✅ Timeline ekle: "━━━━━━━━━━━━━━━→ Simulation Time"
- ✅ Her agent yanına zaman damgası: "t=2.1", "t=4.5", "t=7.2"
- ✅ Her agent altına event açıklaması:
  - "↑ Job 1 arrives"
  - "↑ Job 2 operation completes"
  - "↑ Job n arrives"
- ✅ Observation ve Action detaylarını genişlet
- ✅ "..." işaretlerini ekle (variable N göstermek için)
- ⚡ **KESİK ÇIZGILER** kullan timeline'da asynchronous vurgulamak için

---

### Sağ Panel - Mixing Network

```
┌──────────────────────┐
│  Global State G      │  ← Üstte açık mavi
│                      │
│  (jobs, machines,    │  ← DEĞİŞTİR!
│   operators,         │
│   system metrics)    │
└──────────┬───────────┘
           │
   ┌───────▼────────────┐
   │  Mixing Network    │  ← Mavi kutu
   │                    │
   │  Q₁, Q₂, ... Qₙ, G │  ← "..." ekle!
   │                    │
   │  (variable N(t))   │  ← EKLE!
   └───────┬────────────┘
           │
   ┌───────▼────────────┐
   │  Total Q-Value     │  ← Turuncu kutu (üst)
   │    Q_tot           │
   │ (for loss          │
   │  computation)      │
   └────────────────────┘
           │
   ┌───────▼────────────┐
   │  Joint Action u'   │  ← Turuncu kutu (alt)
   │                    │
   │ (machine-operator  │  ← EKLE!
   │  assignments)      │
   └────────────────────┘
```

**DEĞİŞİKLİKLER:**
- ✅ Global State: "(jobs, machines, operators, system metrics)"
- ✅ Mixing Network input: "Q₁, Q₂, ... Qₙ, G" (dots için!)
- ✅ Altına ekle: "(variable N(t))"
- ✅ Joint Action altına: "(machine-operator assignments)"

---

### Alt Nota (Kritik!)

```
┌─────────────────────────────────────────────────────────┐
│  JOB-CENTRIC MASA-QMIX KEY INNOVATIONS:                 │
│                                                         │
│  ⚡ INNOVATION 1: Jobs as agents (job-centric)         │
│  ⚡ INNOVATION 2: Dynamic N(t) - stochastic arrivals   │
│  ⚡ INNOVATION 3: Action masking - capability matrix   │
│  ⚡ INNOVATION 4: Event-driven (NOT synchronous!)      │
│                                                         │
│  • Asynchronous: Each job decides at its event time    │
│  • Input: o_t^j (obs) + u_{t-1}^j (prev action) ⚠️     │
│  • Actions: Select machine-operator pair (m,o)         │
│  • Variable-length episodes (simulation-driven)        │
└─────────────────────────────────────────────────────────┘
```

---

## 🎨 RENK REHBERİ (PowerPoint için)

### Standard MASA-QMIX (Diagram 1):
- **Mavi tonları** (sakin, klasik)
  - Global State: Açık mavi (#B3D9FF)
  - Mixing Network: Orta mavi (#4A90E2)
  - Agent boxes: Açık mavi/gri (#E3F2FD)
- **Yeşil tonları**
  - GRU-based Q-networks: Koyu yeşil (#4CAF50)
  - Q-values: Açık yeşil/turuncu (#FFA726)
- **Temiz, minimal görünüm**

### Job-Centric MASA-QMIX (Diagram 2):
- **Turuncu vurgular** (innovations)
  - Başlıklar: Turuncu (#FF9800)
  - Innovation labels: Koyu turuncu (#F57C00)
- **Teal/Turkuaz** (action masking)
  - Action Mask box: Turkuaz (#00897B)
- **Pembe/Somon** (job agents)
  - Job boxes: Açık somon (#FFCDD2)
- **Kırmızı** (event triggers)
  - Timeline markers: Kırmızı (#E53935)
  - Event arrows: Kırmızı kesik çizgi
- **Yeşil** (standard components - aynı)
  - GRU-based Q-networks: Yeşil (#4CAF50)
  - Mixing Network: Mavi (aynı)

---

## 📐 POWERPOINT ADIM ADIM

### Diagram 1 (Standard) için:

1. **Sol panel - Q-network:**
   - Üç kutuyu SİL: "Linear", "RNN (GRU)", "Linear"
   - TEK kutu yap: "GRU-based Q-networks"
     - Shape: Rounded rectangle
     - Color: Yeşil (#4CAF50)
     - Text: "GRU-based Q-networks\n(maintains hidden states)"
   - Action Mask kutusunu SİL

2. **Orta panel - Agents:**
   - "Job Agent" yazılarını "Agent" yap
   - Üste text box ekle: "FIXED AGENTS (N = 3)"
   - Robot emoji ekle: 🤖 (opsiyonel)

3. **Sağ panel - Mixing:**
   - Global State text değiştir: "(agent positions, environment state)"

4. **Alt nota ekle:**
   - Text box (tüm genişlikte)
   - "STANDARD MASA-QMIX CHARACTERISTICS:..."

### Diagram 2 (Job-Centric) için:

1. **Sol panel - Q-network:**
   - Action Mask kutusunu EKLE (türkuaz renk)
   - Yanına "⚡ INNOVATION 3" yaz (turuncu)
   - GRU box text değiştir: "(maintains per-job h_j)"
   - Observation detaylandır

2. **Orta panel - CRITICAL!**
   - Başlığı değiştir: "DYNAMIC JOB AGENTS (N(t) varies!) ⚡" (turuncu)
   - Timeline çiz: Kalın ok "━━━━━━━→ Simulation Time"
   - Her job box yanına:
     - Text box: "t=2.1", "t=4.5", "t=7.2"
     - Arrow + text: "↑ Job arrives" / "↑ Op completes"
   - "..." ekle (variable N için)

3. **Sağ panel - Mixing:**
   - Global State değiştir: "(jobs, machines, operators,...)"
   - Mixing input: "Q₁, Q₂, ... Qₙ, G"
   - Altına: "(variable N(t))"

4. **Alt nota:**
   - 4 innovation liste (⚡ emojileriyle)
   - Turuncu highlight

---

## ⚡ EN ÖNEMLİ DEĞİŞİKLİKLER (ÖZET)

### Diagram 1 → 2 Dönüşümü:

| Element | Standard (1) | Job-Centric (2) |
|---------|-------------|-----------------|
| **Agents** | Fixed N=3 | Dynamic N(t), "..." ile göster |
| **Timing** | Synchronous | Timeline + event times (t=2.1, t=4.5...) |
| **Action Mask** | YOK | VAR (turkuaz kutu) ⚡ |
| **Q-network** | "maintains hidden states" | "maintains per-job h_j" |
| **Observations** | o_t + u_{t-1} (generic) | o_t^j + u_{t-1}^j (job-specific) |
| **Actions** | Generic movement | Specific: (machine, operator) pairs |
| **Global State** | Agent positions | Jobs, machines, operators, metrics |
| **Renk** | Mavi tonları | Turuncu/teal/kırmızı (innovations) |
| **Event flow** | YOK | Timeline + arrows + event labels |

---

## ✅ CHECKLIST

### Standard MASA-QMIX (Diagram 1):
- [ ] Üç ayrı layer yerine TEK "GRU-based Q-networks" kutusu
- [ ] Action Mask SİLİNDİ
- [ ] "Agent 1, 2, 3" (Job değil!)
- [ ] "FIXED N = 3" başlık
- [ ] Mavi renk tonu
- [ ] Alt nota: characteristics listesi

### Job-Centric MASA-QMIX (Diagram 2):
- [ ] Action Mask EKLENDI (turkuaz) ⚡
- [ ] Timeline + simulation time ok
- [ ] Event times: t=2.1, t=4.5, t=7.2
- [ ] Event açıklamaları: "↑ Job arrives", "↑ Op completes"
- [ ] "..." ile variable N göster
- [ ] "DYNAMIC N(t)" başlık (turuncu)
- [ ] 4 innovation listesi alt nota
- [ ] Renkli: turuncu (innovations), turkuaz (mask), kırmızı (events)
- [ ] GRU-based text: "per-job h_j"
- [ ] Observations detaylı
- [ ] Actions: "(m,o) pairs"

---

## 🚀 HIZLI BAŞLANGIÇ

Mevcut PowerPoint'inizde:

1. **İlk slide'ı kopyalayın** → "Standard MASA-QMIX" başlığı verin
2. **İkinci slide'ı kopyalayın** → "Job-Centric MASA-QMIX" başlığı verin
3. **Slide 1'de:**
   - Sol panel: 3 kutuyu 1 yapın
   - Action Mask silin
   - "Job" → "Agent" değiştirin
4. **Slide 2'de:**
   - Timeline ekleyin (üstte)
   - Event times ekleyin (t=2.1, ...)
   - Action Mask ekleyin (turkuaz)
   - Turuncu renk ekleyin (innovations)
   - "..." ekleyin (variable N)

Hazır! 🎉

---

**NOT:** Bu değişiklikler sadece görsel - teorik doğruluk için [thesis_presentation_comparison.md](thesis_presentation_comparison.md) dokümandaki açıklamaları kullanın!
