# C1: Exception Swallowing - Global Kontrol Akışı - Migration Plan

**Hedef**: Catch-all exception handler'ları elimine etmek  
**Zorluk**: 🔨 Zor (8 saat)  
**Dosya Sayısı**: 4+ dosya, ~60 lokasyon  
**Öncelik**: Düşük (ama uzun vadede kritik)

---

## Sorun Derinlemesine Analizi

### Ne Oluyor?
```python
try:
    critical_operation()
except Exception:
    pass  # Sessizce yut
```

### Neden Tehlikeli?

1. **Crash'leri Gizliyor**
   - Gerçek bug'lar silent failure oluyor
   - Training divergence'ın kök sebebi bulunamıyor

2. **Debug İmkansız**
   - Hangi satırda hata olduğu belli değil
   - Stack trace kaybolmuş
   - Reproduction için ipucu yok

3. **State Corruption**
   - Exception sonrası state invalid ama kod devam ediyor
   - Downstream operations yanlış veriyle çalışıyor
   - Cascade failure

4. **Training Divergence**
   - Yanlış state → yanlış action → yanlış reward
   - Agent yanlış policy öğreniyor
   - Loss NaN oluyor ama sebep bilinmiyor

---

## Lokasyonlar ve Öncelik Analizi

### Kritik Lokasyonlar (Öncelik 1 - 3 saat)

#### 1. MARL/common/rollout.py (~15 örnek)

**Kritik Noktalar**:

```python
# Example 1: Epsilon Update (Line ~320)
try:
    self.epsilon = max(self.epsilon_end, self.epsilon - self._eps_decay)
except Exception:
    pass  # ← Epsilon stuck kalabilir!

# Example 2: Hidden State Reset (Line ~450)
try:
    self.hidden_states = torch.zeros(...)
except Exception:
    pass  # ← Hidden state corrupt kalabilir!

# Example 3: Batch Collection (Line ~680)
try:
    batch = self.env.get_obs_batch()
except Exception:
    batch = []  # ← Empty batch training'i bozar!

# Example 4: Action Processing (Line ~850)
try:
    actions = self.policy.select_actions(obs)
except Exception:
    actions = [0] * self.n_agents  # ← Always action 0!

# Example 5: Reward Computation (Line ~950)
try:
    reward = self.env.pop_decision_reward()
except Exception:
    reward = 0.0  # ← No learning signal!
```

**Neden Kritik?**
- Training loop içinde
- Her episode'da çağrılıyor
- Silent failure → divergence
- Debug çok zor

---

#### 2. environment.py (~20 örnek)

**Kritik Noktalar**:

```python
# Example 1: Duration Computation (Line ~850)
try:
    dur = float(decision_item['per_machine_durations'][chosen_idx])
except Exception:
    dur = 0.0  # ← Zero duration → simpy crash!

# Example 2: Job Completion (Line ~1200)
try:
    job.mark_completed(self.env.now)
except Exception:
    pass  # ← Job state inconsistent!

# Example 3: Gantt Record (Line ~1500)
try:
    self.gantt_records.append(record)
except Exception:
    pass  # ← Utilization yanlış hesaplanır!

# Example 4: Observation Build (Line ~600)
try:
    obs = self._build_agent_obs(job)
except Exception:
    obs = np.zeros(self.obs_shape)  # ← Fake observation!

# Example 5: Operator Selection (Line ~750)
try:
    operator = self.operators.find_free_operator(...)
except Exception:
    operator = None  # ← No operator → simpy stuck!
```

**Neden Kritik?**
- Core environment logic
- SimPy interaction
- State management
- Every decision boundary

---

### Orta Öncelik (Öncelik 2 - 3 saat)

#### 3. MARL/policy/*.py (~10 örnek)

```python
# epsilon_greedy.py
try:
    action = self.policy.select_action(obs)
except Exception:
    action = random.choice(available_actions)  # ← Random action!

# qmix.py
try:
    q_values = self.mixer(agent_qs, state)
except Exception:
    q_values = torch.zeros_like(agent_qs)  # ← Zero Q-values!

# actor_critic.py
try:
    value = self.critic(state)
except Exception:
    value = torch.tensor(0.0)  # ← No value estimation!
```

**Neden Önemli?**
- Policy learning
- Q-value estimation
- Action selection
- Her decision'da çağrılıyor

---

#### 4. utils/*.py (~15 örnek)

```python
# utils/env_obs.py
try:
    obs = build_agent_obs(env, job)
except Exception:
    obs = np.zeros(env.obs_shape)  # ← Fake observation!

# utils/gantt_utils.py
try:
    gantt_data = process_records(records)
except Exception:
    gantt_data = []  # ← Visualization fail (tolerable)

# utils/normalization.py
try:
    normalized = (value - mean) / std
except Exception:
    normalized = 0.0  # ← Wrong normalization!
```

---

### Düşük Öncelik (Öncelik 3 - 2 saat)

#### 5. Logging ve I/O (~20 örnek)

```python
# utils/metrics_writer.py
try:
    writer.write_metrics(episode_data)
except Exception:
    pass  # ← Metrics kaybı (acceptable)

# tools/save_checkpoint.py
try:
    torch.save(model.state_dict(), path)
except Exception:
    pass  # ← Checkpoint kaybı (warning yeterli)

# my_data_and_graph/*.py
try:
    plt.savefig(path)
except Exception:
    pass  # ← Plot kaybı (tolerable)
```

**Neden Düşük Öncelik?**
- Best-effort operations
- Training devam edebilir
- Data loss tolerable
- Warning log yeterli

---

## Çözüm Stratejileri (Tip Bazında)

### Tip 1: Tamamen Kaldır (En Kolay)
**Ne zaman?** Exception beklenmiyor, defensive programming

```python
# ÖNCE
try:
    self.epsilon = max(self.epsilon_end, self.epsilon - self._eps_decay)
except Exception:
    pass

# SONRA
# C1 FIX: Removed exception swallowing
# If epsilon update fails, training should crash explicitly
self.epsilon = max(self.epsilon_end, self.epsilon - self._eps_decay)
```

**Mantık**: Eğer burada exception oluyorsa zaten training crash etmeli.

---

### Tip 2: Spesifik Exception'a Dönüştür
**Ne zaman?** Belli bir hata türü bekleniyor

```python
# ÖNCE
try:
    value = config[key]
except Exception:
    value = default

# SONRA
# C1 FIX: Specific exception handling
try:
    value = config[key]
except KeyError:
    raise ValueError(
        f"Required config key '{key}' missing. "
        f"Available keys: {list(config.keys())}"
    ) from None
```

**Mantık**: Specific exception → açık error message → kolay debug.

---

### Tip 3: Validation'a Dönüştür
**Ne zaman?** Değer kontrolü gerekiyor

```python
# ÖNCE
try:
    result = risky_calculation()
except Exception:
    result = 0

# SONRA
# C1 FIX: Replaced exception handler with validation
result = risky_calculation()
if not np.isfinite(result):
    raise RuntimeError(
        f"Invalid calculation result: {result}. "
        f"Check input values and formula. "
        f"Inputs: {locals()}"
    )
```

**Mantık**: Validation explicit → fail-fast → clear error.

---

### Tip 4: Log and Re-raise (Best-Effort Operations)
**Ne zaman?** Non-critical operation (logging, I/O, visualization)

```python
# ÖNCE
try:
    save_checkpoint(model, path)
except Exception:
    pass

# SONRA
# C1 FIX: Log failure but allow training to continue
try:
    save_checkpoint(model, path)
except Exception as e:
    LOG.warning(
        "[C1] Checkpoint save failed (non-critical): %s. "
        "Training continues without checkpoint. "
        "Path: %s",
        e, path
    )
    # Don't re-raise - allow training to continue
```

**Mantık**: Log error, continue if non-critical.

---

### Tip 5: Wrap with Context
**Ne zaman?** Multiple operations, need cleanup

```python
# ÖNCE
try:
    resource = acquire()
    use(resource)
    release(resource)
except Exception:
    pass

# SONRA
# C1 FIX: Proper resource management with context
resource = None
try:
    resource = acquire()
    use(resource)
finally:
    if resource is not None:
        try:
            release(resource)
        except Exception as e:
            LOG.warning("Resource release failed: %s", e)
```

---

## Uygulama Planı: Aşamalı Yaklaşım

### Fase 1: Kritik Training Loop (3 saat)

**Step 1.1: rollout.py - Top 5 Kritik**

| Lokasyon | Tip | Çözüm | Satır |
|----------|-----|-------|-------|
| Epsilon update | 1 | Tamamen kaldır | ~320 |
| Hidden state reset | 1 | Tamamen kaldır | ~450 |
| Batch collection | 2 | ValueError raise | ~680 |
| Action processing | 2 | RuntimeError raise | ~850 |
| Reward computation | 2 | RuntimeError raise | ~950 |

**Örnek İmplementasyon**:
```python
# Line ~680: Batch Collection
def collect_batch(self):
    """Collect observation, state, and availability batches.
    
    C1 FIX: Removed exception swallowing. If batch collection fails,
    training should crash explicitly rather than silently using empty batches.
    
    Raises:
        RuntimeError: If any batch collection fails
    """
    try:
        obs_batch = self.env.get_obs_batch()
    except Exception as e:
        raise RuntimeError(
            f"[C1] Failed to collect observation batch at t={self.env.now:.2f}. "
            f"Check environment state and observation builder."
        ) from e
    
    try:
        state_batch = self.env.get_state_batch()
    except Exception as e:
        raise RuntimeError(
            f"[C1] Failed to collect state batch at t={self.env.now:.2f}. "
            f"Check environment state and state builder."
        ) from e
    
    try:
        avail_batch = self.env.get_avail_actions_batch()
    except Exception as e:
        raise RuntimeError(
            f"[C1] Failed to collect availability batch at t={self.env.now:.2f}. "
            f"Check mask generation logic."
        ) from e
    
    return obs_batch, state_batch, avail_batch
```

---

**Step 1.2: environment.py - Top 5 Kritik**

| Lokasyon | Tip | Çözüm | Satır |
|----------|-----|-------|-------|
| Duration computation | 3 | Validation ekle | ~850 |
| Job completion | 2 | RuntimeError raise | ~1200 |
| Gantt record | 2 | RuntimeError raise | ~1500 |
| Observation build | 2 | ValueError raise | ~600 |
| Operator selection | 2 | RuntimeError raise | ~750 |

**Örnek İmplementasyon**:
```python
# Line ~850: Duration Computation
# ÖNCE
try:
    dur = float(decision_item['per_machine_durations'][chosen_idx])
except Exception:
    dur = 0.0

# SONRA
# C1 FIX: Explicit validation instead of exception swallowing
if 'per_machine_durations' not in decision_item:
    raise ValueError(
        f"[C1] Missing 'per_machine_durations' in decision_item. "
        f"Job={job.id}, op_idx={job.current_op_idx}. "
        f"Available keys: {list(decision_item.keys())}"
    )

if chosen_idx not in decision_item['per_machine_durations']:
    raise ValueError(
        f"[C1] Machine index {chosen_idx} not in per_machine_durations. "
        f"Job={job.id}, op_idx={job.current_op_idx}. "
        f"Available machines: {list(decision_item['per_machine_durations'].keys())}"
    )

dur = float(decision_item['per_machine_durations'][chosen_idx])

# Additional validation (C17 synergy)
if dur <= 0:
    raise ValueError(
        f"[C1+C17] Invalid duration {dur} for machine {chosen_idx}. "
        f"Duration must be positive."
    )
```

---

### Fase 2: Policy ve Utils (3 saat)

**Step 2.1: MARL/policy/*.py (5 lokasyon)**

| Dosya | Lokasyon | Çözüm |
|-------|----------|-------|
| epsilon_greedy.py | Action selection | RuntimeError |
| qmix.py | Q-value mixing | RuntimeError |
| actor_critic.py | Value estimation | RuntimeError |
| ddpg.py | Policy forward | RuntimeError |
| td3.py | Target update | RuntimeError |

---

**Step 2.2: utils/*.py (5 lokasyon)**

| Dosya | Lokasyon | Çözüm |
|-------|----------|-------|
| env_obs.py | Observation build | ValueError |
| gantt_utils.py | Record processing | ValueError |
| normalization.py | Value normalization | RuntimeError |
| mask_utils.py | Mask generation | ValueError |
| config_loader.py | Config parsing | ValueError |

---

### Fase 3: Best-Effort Locations (2 saat)

**Step 3.1: Logging ve Metrics (10 lokasyon)**
- Tip 4 (Log and continue) kullan
- Training devam etsin
- Warning log yeterli

**Step 3.2: I/O ve Visualization (10 lokasyon)**
- Tip 4 (Log and continue) kullan
- Data loss tolerable
- Warning log yeterli

---

## Otomatik Detection Script

### Script: tools/find_exception_swallowing.py

```python
#!/usr/bin/env python3
"""
Detect catch-all exception handlers in codebase.

Usage:
    python tools/find_exception_swallowing.py
    python tools/find_exception_swallowing.py --verbose
    python tools/find_exception_swallowing.py --json output.json
"""
import re
import os
import json
import argparse
from pathlib import Path


def find_exception_swallowing(root_dir, verbose=False):
    """Find catch-all exception handlers.
    
    Patterns detected:
    1. except Exception: pass
    2. except Exception: return default
    3. except Exception: continue
    4. except: pass (even worse - catches everything including KeyboardInterrupt)
    """
    
    # Pattern 1: except Exception: pass
    pattern1 = re.compile(
        r'except\s+Exception\s*:\s*\n\s*pass\b',
        re.MULTILINE
    )
    
    # Pattern 2: except Exception: return/continue
    pattern2 = re.compile(
        r'except\s+Exception\s*:\s*\n\s*(return|continue)',
        re.MULTILINE
    )
    
    # Pattern 3: except: (no exception type)
    pattern3 = re.compile(
        r'except\s*:\s*\n\s*(pass|return|continue)',
        re.MULTILINE
    )
    
    patterns = [
        (pattern1, 'except Exception: pass'),
        (pattern2, 'except Exception: return/continue'),
        (pattern3, 'bare except'),
    ]
    
    results = []
    for root, dirs, files in os.walk(root_dir):
        # Skip non-code directories
        skip_dirs = ['archive', 'backup', '__pycache__', '.git', 'my_data_and_graph']
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if not file.endswith('.py'):
                continue
            
            path = os.path.join(root, file)
            rel_path = os.path.relpath(path, root_dir)
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, pattern_name in patterns:
                    matches = pattern.finditer(content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Extract context (3 lines before and after)
                        lines = content.split('\n')
                        start_line = max(0, line_num - 4)
                        end_line = min(len(lines), line_num + 3)
                        context = '\n'.join(lines[start_line:end_line])
                        
                        results.append({
                            'file': rel_path,
                            'line': line_num,
                            'pattern': pattern_name,
                            'snippet': match.group(0),
                            'context': context if verbose else None,
                        })
            except Exception as e:
                print(f"Warning: Failed to process {path}: {e}")
    
    return results


def categorize_results(results):
    """Categorize results by priority."""
    
    high_priority = ['rollout.py', 'environment.py', 'runner.py']
    medium_priority = ['policy', 'qmix', 'epsilon', 'actor_critic']
    
    categories = {
        'high': [],
        'medium': [],
        'low': [],
    }
    
    for r in results:
        file = r['file']
        
        if any(hp in file for hp in high_priority):
            categories['high'].append(r)
        elif any(mp in file for mp in medium_priority):
            categories['medium'].append(r)
        else:
            categories['low'].append(r)
    
    return categories


def print_summary(results, categories):
    """Print summary report."""
    
    print(f"\n{'='*70}")
    print(f"EXCEPTION SWALLOWING DETECTION REPORT")
    print(f"{'='*70}\n")
    
    print(f"Total locations found: {len(results)}\n")
    
    # By priority
    print("By Priority:")
    print(f"  🔴 HIGH:   {len(categories['high'])} locations")
    print(f"  🟡 MEDIUM: {len(categories['medium'])} locations")
    print(f"  🟢 LOW:    {len(categories['low'])} locations")
    print()
    
    # By file
    by_file = {}
    for r in results:
        file = r['file']
        if file not in by_file:
            by_file[file] = []
        by_file[file].append(r)
    
    print(f"By File ({len(by_file)} files):")
    for file, matches in sorted(by_file.items(), key=lambda x: -len(x[1])):
        priority = '🔴' if any(r in categories['high'] for r in matches) else \
                   '🟡' if any(r in categories['medium'] for r in matches) else '🟢'
        print(f"  {priority} {file}: {len(matches)} locations")
        for m in matches[:3]:  # Show first 3
            print(f"      Line {m['line']}: {m['pattern']}")
        if len(matches) > 3:
            print(f"      ... and {len(matches) - 3} more")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Find exception swallowing in code')
    parser.add_argument('--verbose', '-v', action='store_true', help='Include context')
    parser.add_argument('--json', '-j', type=str, help='Output JSON file')
    parser.add_argument('--root', '-r', type=str, default='.', help='Root directory')
    
    args = parser.parse_args()
    
    print(f"Scanning {args.root} for exception swallowing...")
    results = find_exception_swallowing(args.root, verbose=args.verbose)
    categories = categorize_results(results)
    
    print_summary(results, categories)
    
    # JSON output
    if args.json:
        output = {
            'total': len(results),
            'categories': {k: len(v) for k, v in categories.items()},
            'results': results,
        }
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Detailed results written to: {args.json}")


if __name__ == "__main__":
    main()
```

**Kullanım**:
```bash
# Basic scan
python tools/find_exception_swallowing.py

# Verbose (with context)
python tools/find_exception_swallowing.py --verbose

# JSON output
python tools/find_exception_swallowing.py --json exception_report.json
```

---

## Test Stratejisi

### Test 1: Detection Script Test
```bash
# Run detection
python tools/find_exception_swallowing.py --json before_fix.json

# Fix top 10
# ... apply fixes ...

# Run detection again
python tools/find_exception_swallowing.py --json after_fix.json

# Compare
python -c "
import json
before = json.load(open('before_fix.json'))
after = json.load(open('after_fix.json'))
print(f'Before: {before[\"total\"]} locations')
print(f'After:  {after[\"total\"]} locations')
print(f'Fixed:  {before[\"total\"] - after[\"total\"]} locations')
"
```

---

### Test 2: Integration Test (Smoke)
```python
# tests/test_c1_no_silent_failures.py
"""
Test C1 Fix: No silent failures in critical paths.

Strategy: Inject errors and verify they propagate (not swallowed).
"""

def test_c1_batch_collection_propagates_error(monkeypatch):
    """Test that batch collection errors propagate."""
    from MARL.common.rollout import RolloutWorker
    
    # Mock env.get_obs_batch to raise
    def mock_get_obs_batch():
        raise RuntimeError("Mock error")
    
    worker = create_test_worker()
    monkeypatch.setattr(worker.env, 'get_obs_batch', mock_get_obs_batch)
    
    # Should raise (not swallow)
    with pytest.raises(RuntimeError, match="Mock error|Failed to collect"):
        worker.collect_batch()
    
    print("✅ PASS: Batch collection error propagates")


def test_c1_duration_computation_validates(monkeypatch):
    """Test that duration computation validates inputs."""
    from environment import MASAEnv
    
    env = create_test_env()
    
    # Test missing duration key
    decision_item = {'job_id': 0}  # Missing 'per_machine_durations'
    
    with pytest.raises(ValueError, match="Missing 'per_machine_durations'"):
        # This should raise now (C1 fix)
        env._extract_duration(decision_item, chosen_idx=0)
    
    print("✅ PASS: Duration computation validates inputs")
```

---

### Test 3: Regression Test
```python
# tests/test_c1_training_stability.py
"""
Test that C1 fixes don't break training.

Run 10 episodes, verify:
- No crashes
- Rewards finite
- Episode completes
"""

def test_c1_training_runs_10_episodes():
    """Test training completes 10 episodes after C1 fixes."""
    from MARL.runner import Runner
    
    runner = create_test_runner()
    
    # Run 10 episodes
    for ep in range(10):
        episode_data = runner.run_episode(ep)
        
        # Verify episode completed
        assert episode_data is not None
        assert 'reward' in episode_data
        assert np.isfinite(episode_data['reward'])
    
    print("✅ PASS: Training stable after C1 fixes")
```

---

## Implementation Checklist

### Fase 1: Kritik Training Loop (3 saat)

**rollout.py (5 lokasyon)**:
- [ ] Line ~320: Epsilon update → Remove try/except
- [ ] Line ~450: Hidden state reset → Remove try/except
- [ ] Line ~680: Batch collection → Add explicit errors
- [ ] Line ~850: Action processing → Add explicit errors
- [ ] Line ~950: Reward computation → Add explicit errors

**environment.py (5 lokasyon)**:
- [ ] Line ~850: Duration computation → Add validation
- [ ] Line ~1200: Job completion → Add explicit errors
- [ ] Line ~1500: Gantt record → Add explicit errors
- [ ] Line ~600: Observation build → Add explicit errors
- [ ] Line ~750: Operator selection → Add explicit errors

**Test**:
- [ ] Run detection script (before)
- [ ] Apply fixes
- [ ] Run detection script (after)
- [ ] Verify count reduced by 10
- [ ] Run smoke test (10 episodes)
- [ ] Verify no regressions

---

### Fase 2: Policy ve Utils (3 saat)

**MARL/policy/*.py (5 lokasyon)**:
- [ ] epsilon_greedy.py: Action selection
- [ ] qmix.py: Q-value mixing
- [ ] actor_critic.py: Value estimation
- [ ] ddpg.py: Policy forward
- [ ] td3.py: Target update

**utils/*.py (5 lokasyon)**:
- [ ] env_obs.py: Observation build
- [ ] gantt_utils.py: Record processing
- [ ] normalization.py: Value normalization
- [ ] mask_utils.py: Mask generation
- [ ] config_loader.py: Config parsing

**Test**:
- [ ] Run detection script
- [ ] Verify count reduced by 10
- [ ] Run training test (50 episodes)

---

### Fase 3: Best-Effort (2 saat)

**Logging/Metrics (10 lokasyon)**:
- [ ] Add LOG.warning + continue
- [ ] Don't crash training

**I/O/Visualization (10 lokasyon)**:
- [ ] Add LOG.warning + continue
- [ ] Data loss tolerable

**Test**:
- [ ] Run detection script
- [ ] Verify count reduced by 20
- [ ] Total reduction: 40 locations

---

## Beklenen Etki

### Before (C1 Yok)
```
Episode 50: Training diverged
  - Loss: NaN
  - Reward: 0.0 (stuck)
  - Debug: "Hangi satırda hata var?"
  - Stack trace: Yok (exception swallowed)
```

### After (C1 Fix)
```
Episode 50: Training crashed
  - Error: "Failed to collect observation batch at t=125.5"
  - Cause: "Job 12 has invalid state (finished=True but active=True)"
  - Stack trace: Full stack available
  - Debug: Açık hata mesajı + context
  - Fix: 5 dakika (state management bug bulundu)
```

**Fayda**:
- ✅ Debug time: 2 saat → 5 dakika
- ✅ Training stability: Clear error messages
- ✅ Bug detection: Silent failure → explicit crash
- ✅ Maintenance: Kolay debugging

---

## Alternatif Yaklaşım: Gradual Migration

### Yaklaşım 1: Feature Flag
```python
# environment.py
STRICT_ERROR_HANDLING = os.getenv('STRICT_ERRORS', 'false').lower() == 'true'

if STRICT_ERROR_HANDLING:
    # C1 FIX: Explicit error
    raise RuntimeError(...)
else:
    # Legacy: Swallow
    pass
```

**Kullanım**:
```bash
# Test with strict errors
STRICT_ERRORS=true python scripts/run_train_qmix.py --n_episodes=10

# Revert if breaks
STRICT_ERRORS=false python scripts/run_train_qmix.py
```

---

### Yaklaşım 2: Logging-Only First
```python
# Phase 1: Log exception but don't crash
try:
    critical_operation()
except Exception as e:
    LOG.error("[C1] Exception swallowed (will become fatal): %s", e, exc_info=True)
    # Still swallow for now

# Phase 2 (later): Make fatal
try:
    critical_operation()
except Exception as e:
    raise RuntimeError(f"[C1] {e}") from e
```

---

## Özet

**Süre**: 8 saat (3 + 3 + 2)  
**Zorluk**: 🔨 Zor  
**Dosya**: 4+ dosya, 60 lokasyon  
**Öncelik**: Düşük (ama uzun vadede kritik)

**Strateji**:
1. **Fase 1**: Kritik training loop (3h, 10 fix)
2. **Fase 2**: Policy ve utils (3h, 10 fix)
3. **Fase 3**: Best-effort (2h, 20 fix)

**Alternatif**:
- Detection script ile başla (30 dk)
- Top 10 kritik fix (2 saat)
- Kalan lokasyonları later

**Tavsiye**: C1'i şimdilik **atla**, önce C7 (determinism) yap. C1 uzun vadeli cleanup işi, acil değil.

---

*C1 Exception Swallowing Migration Plan*  
*Generated: November 20, 2025*
