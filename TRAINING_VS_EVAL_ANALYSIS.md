# Training vs Evaluation Episodes Analizi

## Özet

Mevcut çıktılardan evaluation episode'ları başarıyla filtrelendi.

## Sonuçlar

### Orijinal Veri
- **Toplam episode sayısı:** 899
- **Episode aralığı:** 0-898

### Filtreleme Sonrası
- **Training episodes:** 554
- **Evaluation episodes (kaldırıldı):** 345
- **Training episode aralığı:** 0-553 (yeniden numaralandırıldı)

### Hesaplama
```
Training episodes: 554
Episodes per epoch: 4
Hesaplanan epoch sayısı: 554 ÷ 4 = 138.5 epoch
```

**Not:** Kod tam 200 epoch yerine ~138.5 epoch çalıştırılmış.

### İstatistikler (Training Only)
- **Ortalama reward:** 32.13 ± 3.47
- **Min reward:** ~23.73
- **Max reward:** ~40.39

## Filtreleme Mantığı

Kod şu düzeni kullanıyor:
- Her epoch'ta 4 training episode
- Her 2 epoch'ta bir evaluation (5 episode)

**Episode düzeni:**
```
Epoch 0: episodes 0-3 (training)
Epoch 1: episodes 4-7 (training)
Epoch 2: episodes 8-12 (8-12: evaluation, 5 episode)
         episodes 13-16 (training)
Epoch 3: episodes 17-20 (training)
Epoch 4: episodes 21-25 (21-25: evaluation, 5 episode)
         episodes 26-29 (training)
...
```

## Kaldırılan Evaluation Episodes

İlk 20 evaluation episode index'i:
```
[8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 34, 35, 36, 37, 38, 47, 48, 49, 50, 51]
```

Bu episode'lar artık `episode_rewards_training_only.txt` dosyasında yok.

## Çıktı Dosyaları

1. **episode_rewards_training_only.txt**
   - Sadece training episodes
   - 0'dan başlayarak yeniden numaralandırılmış
   - Format: `episode,reward`

2. **episode_rewards_training_original_idx.txt**
   - Sadece training episodes
   - Orijinal episode numaraları korunmuş
   - Format: `original_episode,reward`

3. **training_only_rewards.png**
   - Training episode rewards grafiği
   - 50-episode moving average ile

## Kod Düzeltmesi

`runner.py` dosyasında evaluation fonksiyonu güncellendi:
- Evaluation episodes artık `global_ep_idx`'i artırmıyor
- Ayrı bir `eval_ep_counter` kullanılıyor
- Bu düzeltme ile gelecekteki çalıştırmalarda:
  - 200 epoch × 4 episode = **800 training episode** (0-799)
  - Evaluation episodes ayrı sayılacak
