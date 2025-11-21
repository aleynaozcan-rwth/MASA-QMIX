# C8: Epsilon Decay Migration Plan - Step-Based → Episode-Based

**Hedef**: Epsilon decay'i decision-based (50x hızlı) yerine episode-based yapmak  
**Zorluk**: ⚡ Kolay (2 saat)  
**Dosya Sayısı**: 1 dosya, 2 değişiklik noktası  
**Yeni Parametre**: ❌ HAYIR - Mevcut `epsilon_anneal_steps` kullanılacak

---

## Mevcut Durum Analizi

### 1. Epsilon Decay Şu Anda Nerede Yapılıyor?

**Dosya**: `MARL/common/rollout.py`  
**Satır**: 944-950 (decision loop içinde)

```python
# --- Epsilon decay update (per decision step) ---
try:
    # Only decay during training (not evaluation)
    if not evaluate:
        # Linear annealing: epsilon = max(epsilon_end, epsilon - decay_rate)
        self.epsilon = max(float(self.epsilon_end), float(self.epsilon) - float(self._eps_decay))
except Exception:
    # Best-effort: don't break training if epsilon update fails
    pass
```

**Sorun**:
- Her `wait_for_decisions()` çağrısında decay oluyor
- 1 episode = ~200 decision → epsilon çok hızlı düşüyor
- `epsilon_anneal_steps=5000` → 25 episode'da 0'a iniyor (olması gereken: 5000 episode)

---

### 2. Epsilon Parametreleri Nereden Geliyor?

**Dosya**: `MARL/common/rollout.py`  
**Satırlar**: 58-81 (__init__ içinde)

```python
# Epsilon start
if hasattr(self.args, 'epsilon_start') and getattr(self.args, 'epsilon_start') is not None:
    self.epsilon_start = float(getattr(self.args, 'epsilon_start'))
elif epsilon_start is not None:
    self.epsilon_start = float(epsilon_start)
else:
    raise ValueError("epsilon_start must be provided via args or constructor")

# Epsilon end
if hasattr(self.args, 'epsilon_end') and getattr(self.args, 'epsilon_end') is not None:
    self.epsilon_end = float(getattr(self.args, 'epsilon_end'))
elif epsilon_end is not None:
    self.epsilon_end = float(epsilon_end)
else:
    raise ValueError("epsilon_end must be provided via args or constructor")

# Epsilon anneal steps
if hasattr(self.args, 'epsilon_anneal_steps') and getattr(self.args, 'epsilon_anneal_steps') is not None:
    self.epsilon_anneal_steps = int(getattr(self.args, 'epsilon_anneal_steps'))
elif epsilon_anneal_steps is not None:
    self.epsilon_anneal_steps = int(epsilon_anneal_steps)
else:
    raise ValueError("epsilon_anneal_steps must be provided via args or constructor")

self.epsilon = float(self.epsilon_start)
self._eps_decay = (self.epsilon_start - self.epsilon_end) / max(1, self.epsilon_anneal_steps)
```

**Mevcut args.epsilon_anneal_steps Semantiği**:
- Config'de: `epsilon_anneal_steps: 5000` (decision count olarak yorumlanıyor)
- Kod: `_eps_decay = (1.0 - 0.05) / 5000 = 0.00019` (per-decision decay rate)

---

### 3. Episode Counter Nereden Erişilir?

**Dosya**: `MARL/common/rollout.py`  
**Fonksiyon**: `run_event_driven_episode(self, global_ep_idx, evaluate=False, runner_args=None)`

```python
def run_event_driven_episode(self, global_ep_idx, evaluate=False, runner_args=None):
    """
    Run one episode using SimPy event-driven execution.
    
    Args:
        global_ep_idx: Episode index (0-based) across all training
        evaluate: If True, skip epsilon decay
        runner_args: Arguments from Runner
    """
```

**Episode counter**: `global_ep_idx` parametresi fonksiyona geliyor ✅

---

## Migration Stratejisi

### Yeni Semantik

**Config değişikliği**: ❌ HAYIR  
**Parametre ismi**: `epsilon_anneal_steps` → `epsilon_anneal_episodes` olarak yorumlanacak  
**Backwards compatibility**: ✅ Var (episode count olarak yorumlanacak)

**Yorum**:
```python
# Config'de:
epsilon_anneal_steps: 5000

# Eski yorum: "5000 decision sonra epsilon=0.05'e düş"
# Yeni yorum: "5000 episode sonra epsilon=0.05'e düş"
```

---

## Adım Adım Değişiklikler

### Değişiklik 1: Episode Counter Ekleme

**Dosya**: `MARL/common/rollout.py`  
**Konum**: `__init__` sonuna (line ~95)

**Eklenecek**:
```python
# Episode-based epsilon tracking
self.episode_count = 0  # Tracks episodes for epsilon decay
```

**Mantık**:
- `__init__`'te sıfırla
- Her episode sonunda artır

---

### Değişiklik 2: Decision Loop'tan Kaldırma

**Dosya**: `MARL/common/rollout.py`  
**Satır**: 944-950

**ÖNCE**:
```python
# --- Epsilon decay update (per decision step) ---
try:
    # Only decay during training (not evaluation)
    if not evaluate:
        # Linear annealing: epsilon = max(epsilon_end, epsilon - decay_rate)
        self.epsilon = max(float(self.epsilon_end), float(self.epsilon) - float(self._eps_decay))
except Exception:
    # Best-effort: don't break training if epsilon update fails
    pass
```

**SONRA** (KALDIRILIYOR):
```python
# C8 FIX: Epsilon decay moved to episode end (not decision boundary)
# Decay logic now in run_event_driven_episode after episode completes
```

---

### Değişiklik 3: Episode End'e Taşıma

**Dosya**: `MARL/common/rollout.py`  
**Fonksiyon**: `run_event_driven_episode`  
**Konum**: Return statement'tan hemen önce (line ~1130, gantt generation'dan sonra)

**Eklenecek**:
```python
# C8 FIX: Episode-based epsilon decay (moved from decision loop)
# Decay epsilon only after full episode completes (not per decision)
if not evaluate:
    try:
        # Increment episode counter
        self.episode_count += 1
        
        # Compute epsilon for next episode
        # Reinterpret epsilon_anneal_steps as epsilon_anneal_episodes
        if self.episode_count >= self.epsilon_anneal_steps:
            # Reached or exceeded annealing period → use epsilon_end
            self.epsilon = float(self.epsilon_end)
        else:
            # Linear decay: epsilon(e) = start - (start-end) * (e / total_episodes)
            decay_fraction = float(self.episode_count) / float(self.epsilon_anneal_steps)
            self.epsilon = self.epsilon_start - decay_fraction * (self.epsilon_start - self.epsilon_end)
            # Clamp to [epsilon_end, epsilon_start] for safety
            self.epsilon = max(float(self.epsilon_end), min(float(self.epsilon_start), self.epsilon))
        
        # Log epsilon change (periodic, non-fatal)
        try:
            if self.episode_count % 10 == 0:  # Log every 10 episodes
                msg = f"[EPSILON_DECAY] episode={self.episode_count}, epsilon={self.epsilon:.4f}"
                print(msg)
                try:
                    history_dir = getattr(self, 'history_dir', None) or getattr(self.env, 'history_dir', None) or './my_data_and_graph/historydata/'
                    os.makedirs(history_dir, exist_ok=True)
                    diag_path = os.path.join(history_dir, 'epsilon_decay_log.txt')
                    with open(diag_path, 'a', encoding='utf-8') as ef:
                        ef.write(f"{self.episode_count},{self.epsilon:.6f}\n")
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        # Best-effort: don't break training if epsilon update fails
        pass

return episode, ep_reward, bool(win_tag), gantt
```

**Yer**: `return episode, ep_reward, bool(win_tag), gantt` satırından HEMEN ÖNCE

---

## Detaylı Kod Lokasyonları

### __init__ Değişikliği

**Dosya**: `MARL/common/rollout.py`  
**Satır**: ~95 (sonunda `self.step_counter` tanımından sonra)

```python
# lightweight step counter for diagnostics (do not affect logic)
self.step_counter = int(getattr(self.args, 'start_step_counter', 0) or 0)
try:
    self.epsilon_log_every = int(getattr(self.args, 'epsilon_diagnostics_every', 50) or 50)
except Exception:
    self.epsilon_log_every = 50

# C8 FIX: Episode-based epsilon tracking
self.episode_count = 0  # Tracks completed episodes for epsilon decay

# log
print(f"[RolloutWorker] init | episode_limit={self.episode_limit} | device={self.device}")
```

---

### Decision Loop Değişikliği

**Dosya**: `MARL/common/rollout.py`  
**Satır**: 944-950

```python
actions = processed_actions

# C8 FIX: Epsilon decay moved to episode end (not decision boundary)
# Previously epsilon decayed here every decision (50x too fast)
# Now handled in run_event_driven_episode after episode completes

# --- Epsilon diagnostics (periodic, non-fatal) ---
try:
```

**Değişiklik**: 7 satırlık epsilon decay bloğu kaldırıldı, 3 satırlık yorum eklendi

---

### Episode End Değişikliği

**Dosya**: `MARL/common/rollout.py`  
**Satır**: ~1130 (return'den önce)

```python
        # Invoke the generator so TIMELINE and SUMMARY are written after
        # the lifecycle END footer. All I/O remains best-effort.
        if ended_ep is not None:
            # ... existing timeline generation code ...
            pass

    # C8 FIX: Episode-based epsilon decay (moved from decision loop)
    # Decay epsilon only after full episode completes (not per decision)
    if not evaluate:
        try:
            # Increment episode counter
            self.episode_count += 1
            
            # Compute epsilon for next episode
            # Reinterpret epsilon_anneal_steps as epsilon_anneal_episodes
            if self.episode_count >= self.epsilon_anneal_steps:
                # Reached or exceeded annealing period → use epsilon_end
                self.epsilon = float(self.epsilon_end)
            else:
                # Linear decay: epsilon(e) = start - (start-end) * (e / total_episodes)
                decay_fraction = float(self.episode_count) / float(self.epsilon_anneal_steps)
                self.epsilon = self.epsilon_start - decay_fraction * (self.epsilon_start - self.epsilon_end)
                # Clamp to [epsilon_end, epsilon_start] for safety
                self.epsilon = max(float(self.epsilon_end), min(float(self.epsilon_start), self.epsilon)))
            
            # Log epsilon change (periodic, non-fatal)
            try:
                if self.episode_count % 10 == 0:  # Log every 10 episodes
                    msg = f"[EPSILON_DECAY] episode={self.episode_count}, epsilon={self.epsilon:.4f}"
                    print(msg)
                    try:
                        history_dir = getattr(self, 'history_dir', None) or getattr(self.env, 'history_dir', None) or './my_data_and_graph/historydata/'
                        os.makedirs(history_dir, exist_ok=True)
                        diag_path = os.path.join(history_dir, 'epsilon_decay_log.txt')
                        with open(diag_path, 'a', encoding='utf-8') as ef:
                            ef.write(f"{self.episode_count},{self.epsilon:.6f}\n")
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            # Best-effort: don't break training if epsilon update fails
            pass

    return episode, ep_reward, bool(win_tag), gantt
```

---

## Parametreye Erişim Şekli

### args.epsilon_anneal_steps Erişimi

```python
# __init__'te zaten var:
self.epsilon_anneal_steps = int(getattr(self.args, 'epsilon_anneal_steps'))

# Episode decay'de kullanımı:
if self.episode_count >= self.epsilon_anneal_steps:
    # Episode count epsilon_anneal_steps'i geçti
    self.epsilon = float(self.epsilon_end)
else:
    # Hala decay aralığındayız
    decay_fraction = float(self.episode_count) / float(self.epsilon_anneal_steps)
    self.epsilon = self.epsilon_start - decay_fraction * (self.epsilon_start - self.epsilon_end)
```

**Yeni parametre gerekli mi?** ❌ HAYIR

**Neden?**
- `epsilon_anneal_steps` zaten mevcut
- Sadece yorumu değiştiriyoruz: "decision count" → "episode count"
- Backward compatible: Kullanıcı config'i değiştirmeden devam edebilir

---

## Config'de Değişiklik Gerekli Mi?

### Mevcut Config

**Dosya**: `configs/env_config.yaml`

```yaml
training_defaults:
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_anneal_steps: 5000  # ← Bu değer decision count olarak yorumlanıyordu
```

### Yeni Yorum (kod değişikliği YOK, sadece anlam değişti)

```yaml
training_defaults:
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_anneal_steps: 5000  # ← Artık episode count olarak yorumlanıyor
```

**Action Required**: ❌ HAYIR - Config değiştirmeye gerek yok  
**Neden**: Semantik değişiklik (decision→episode), parametre ismi aynı

---

## Davranış Değişikliği

### ÖNCE (Decision-Based Decay)

```
Episode 1: 200 decision → epsilon 200 kez düşer
Episode 2: 200 decision → epsilon 200 kez daha düşer
...
Episode 25: epsilon = 0.05'e ulaştı (5000 decision / 200 decision per episode = 25 episode)
```

**Sorun**: 25 episode'da decay tamamlandı (çok erken!)

---

### SONRA (Episode-Based Decay)

```
Episode 1: epsilon = 1.0 - (1.0 - 0.05) * (1/5000) = 0.9998
Episode 2: epsilon = 1.0 - (1.0 - 0.05) * (2/5000) = 0.9996
...
Episode 100: epsilon = 1.0 - (1.0 - 0.05) * (100/5000) = 0.981
Episode 500: epsilon = 1.0 - (1.0 - 0.05) * (500/5000) = 0.905
Episode 1000: epsilon = 1.0 - (1.0 - 0.05) * (1000/5000) = 0.81
...
Episode 5000: epsilon = 0.05 (decay tamamlandı)
```

**Düzeltme**: 5000 episode boyunca yavaşça decay oluyor ✅

---

## Testing & Verification

### 1. Epsilon Değerlerini Kontrol Etme

**Yeni log dosyası**: `my_data_and_graph/historydata/epsilon_decay_log.txt`

**Format**:
```
episode,epsilon
10,0.9981
20,0.9962
30,0.9943
...
5000,0.0500
```

**Beklenen değerler**:
```python
# Episode 10
epsilon = 1.0 - 0.95 * (10 / 5000) = 0.9981 ✓

# Episode 100
epsilon = 1.0 - 0.95 * (100 / 5000) = 0.981 ✓

# Episode 1000
epsilon = 1.0 - 0.95 * (1000 / 5000) = 0.81 ✓

# Episode 5000
epsilon = 0.05 ✓
```

---

### 2. Smoke Test

```bash
# 50 episode train et, epsilon'u gözle
python scripts/run_train_qmix.py --n_episodes=50 --n_epoch=1

# Beklenen çıktı:
[EPSILON_DECAY] episode=10, epsilon=0.9981
[EPSILON_DECAY] episode=20, epsilon=0.9962
[EPSILON_DECAY] episode=30, epsilon=0.9943
[EPSILON_DECAY] episode=40, epsilon=0.9924
[EPSILON_DECAY] episode=50, epsilon=0.9905
```

---

### 3. Verification Script

**Dosya**: `tests/test_epsilon_decay_episode_based.py`

```python
"""
Test C8 Fix: Epsilon decay is episode-based, not decision-based.
"""
import numpy as np

def test_epsilon_decay_formula():
    """Verify epsilon decay formula matches expected values."""
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_anneal_steps = 5000
    
    def compute_epsilon(episode_count):
        if episode_count >= epsilon_anneal_steps:
            return epsilon_end
        decay_fraction = float(episode_count) / float(epsilon_anneal_steps)
        return epsilon_start - decay_fraction * (epsilon_start - epsilon_end)
    
    # Test cases
    assert abs(compute_epsilon(0) - 1.0) < 1e-6, "Episode 0 should be 1.0"
    assert abs(compute_epsilon(10) - 0.9981) < 1e-3, "Episode 10 should be ~0.9981"
    assert abs(compute_epsilon(100) - 0.981) < 1e-3, "Episode 100 should be ~0.981"
    assert abs(compute_epsilon(1000) - 0.81) < 1e-3, "Episode 1000 should be ~0.81"
    assert abs(compute_epsilon(5000) - 0.05) < 1e-6, "Episode 5000 should be 0.05"
    assert abs(compute_epsilon(6000) - 0.05) < 1e-6, "Episode 6000+ should stay 0.05"
    
    print("✅ All epsilon decay formula tests passed")

def test_epsilon_log_file():
    """Verify epsilon_decay_log.txt contains expected entries."""
    import os
    
    log_path = "my_data_and_graph/historydata/epsilon_decay_log.txt"
    
    if not os.path.exists(log_path):
        print("⚠️  Log file not found (run training first)")
        return
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Check format: episode,epsilon
    for line in lines[:5]:  # Check first 5 lines
        parts = line.strip().split(',')
        assert len(parts) == 2, f"Invalid format: {line}"
        episode = int(parts[0])
        epsilon = float(parts[1])
        assert 0.05 <= epsilon <= 1.0, f"Epsilon out of range: {epsilon}"
    
    print(f"✅ Epsilon log file verified ({len(lines)} entries)")

if __name__ == "__main__":
    test_epsilon_decay_formula()
    test_epsilon_log_file()
```

**Çalıştırma**:
```bash
python tests/test_epsilon_decay_episode_based.py
```

---

## Özet: Hangi Satırları Değiştireceksin?

### Dosya: `MARL/common/rollout.py`

| Değişiklik | Satır | Ne Yapılacak |
|-----------|-------|--------------|
| 1. Episode counter ekle | ~95 (__init__ sonu) | `self.episode_count = 0` ekle |
| 2. Decision loop decay'i kaldır | 944-950 | 7 satır epsilon decay kodu → 3 satır yorum |
| 3. Episode end decay ekle | ~1130 (return öncesi) | 30 satırlık episode-based decay bloğu ekle |

**Toplam**:
- Eklenen satır: ~33
- Kaldırılan satır: ~7
- Net değişiklik: +26 satır

---

## Implementation Checklist

- [ ] 1. `__init__` içine `self.episode_count = 0` ekle
- [ ] 2. Decision loop'tan epsilon decay bloğunu kaldır (satır 944-950)
- [ ] 3. `run_event_driven_episode` return'ünden önce episode-based decay ekle
- [ ] 4. Smoke test: 50 episode train et, epsilon'u kontrol et
- [ ] 5. Verification script çalıştır: `test_epsilon_decay_episode_based.py`
- [ ] 6. Epsilon log dosyasını kontrol et: `epsilon_decay_log.txt`
- [ ] 7. Documentation güncelle: C8 fix tamamlandı olarak işaretle

---

## Beklenen Etki

### Training Quality
- ✅ Exploration 50x daha uzun sürecek
- ✅ Agent erken convergence yapmayacak
- ✅ Final policy daha iyi olacak

### Epsilon Curve
```
Before: Steep drop (0 after 25 episodes)
After:  Gradual linear decay (0.05 after 5000 episodes)
```

### Backwards Compatibility
- ✅ Config değişikliği gerekmiyor
- ✅ Parametre isimleri aynı
- ✅ Sadece semantik değişti (decision → episode)

---

## Sonraki Adım

Bu planı uygulamak için:

```bash
# Değişiklikleri uygula
# (yukarıdaki 3 değişiklik)

# Test et
python scripts/run_train_qmix.py --n_episodes=50 --n_epoch=1

# Verify
python tests/test_epsilon_decay_episode_based.py
```

**Hazır mısın?** Değişiklikleri uygulayayım mı?

---

*C8 Epsilon Decay Migration Plan*  
*Generated: November 20, 2025*
