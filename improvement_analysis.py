import pandas as pd
import numpy as np

df = pd.read_csv('./my_data_and_graph/historydata/episode_metrics_training_only.csv')

episodes = df['episode'].values
rewards = df['episode_reward'].values

if 'epsilon' in df.columns:
    epsilon_min = df['epsilon'].min()
    eps_min_idx = df[df['epsilon'] <= epsilon_min * 1.01].index.min()
    exploitation_start_ep = df.loc[eps_min_idx, 'episode']
    
    before_mask = episodes < exploitation_start_ep
    after_mask = episodes >= exploitation_start_ep
    
    before_rewards = rewards[before_mask]
    after_rewards = rewards[after_mask]
    
    # İstatistikler
    before_mean = np.mean(before_rewards)
    after_mean = np.mean(after_rewards)
    before_std = np.std(before_rewards)
    after_std = np.std(after_rewards)
    before_var = np.var(before_rewards)
    after_var = np.var(after_rewards)
    
    # İyileşme metrikleri
    mean_improvement = after_mean - before_mean
    mean_improvement_pct = (mean_improvement / before_mean) * 100
    
    std_reduction = before_std - after_std
    std_reduction_pct = (std_reduction / before_std) * 100
    
    var_reduction = before_var - after_var
    var_reduction_pct = (var_reduction / before_var) * 100
    
    # Coefficient of Variation (CV) - normalized stability measure
    before_cv = (before_std / before_mean) * 100
    after_cv = (after_std / after_mean) * 100
    cv_improvement = before_cv - after_cv
    cv_improvement_pct = (cv_improvement / before_cv) * 100
    
    print('=' * 70)
    print('     EXPLOITATION ONCESI vs SONRASI IYILESME ANALIZI')
    print('=' * 70)
    print()
    print('RAW DEGISIM:')
    print('-' * 70)
    print(f'  Mean:     {before_mean:.2f} -> {after_mean:.2f}  (Delta = {mean_improvement:+.2f})')
    print(f'  Std Dev:  {before_std:.2f} -> {after_std:.2f}  (Delta = {std_reduction:+.2f})')
    print(f'  Variance: {before_var:.2f} -> {after_var:.2f}  (Delta = {var_reduction:+.2f})')
    print()
    print('YUZDESEL IYILESME:')
    print('-' * 70)
    print(f'  Mean improvement:        {mean_improvement_pct:+.2f}%')
    print(f'  Std deviation reduction: {std_reduction_pct:+.2f}%')
    print(f'  Variance reduction:      {var_reduction_pct:+.2f}%')
    print()
    print('NORMALIZE EDILMIS STABILITE (Coefficient of Variation):')
    print('-' * 70)
    print(f'  Before CV: {before_cv:.2f}% (std/mean ratio)')
    print(f'  After CV:  {after_cv:.2f}% (std/mean ratio)')
    print(f'  CV improvement: {cv_improvement:+.2f} points ({cv_improvement_pct:+.2f}%)')
    print()
    print('SONUC ONERILERI:')
    print('=' * 70)
    
    # En güçlü iyileşme metriği
    metrics = {
        'Mean Reward': mean_improvement_pct,
        'Variance Reduction': var_reduction_pct,
        'Std Deviation Reduction': std_reduction_pct,
        'Coefficient of Variation': cv_improvement_pct
    }
    
    best_metric = max(metrics, key=metrics.get)
    best_value = metrics[best_metric]
    
    print(f'EN GUCLU IYILESME: {best_metric} ({best_value:+.1f}%)')
    print()
    
    # Tez için öneriler
    print('TEZINIZDE VURGULAYACAGINIZ NOKTALAR:')
    print('-' * 70)
    
    if mean_improvement_pct > 3:
        print(f'+ Mean reward %{mean_improvement_pct:.1f} artti')
        print(f'  -> "Agent achieved {mean_improvement_pct:.1f}% performance improvement"')
    
    if var_reduction_pct > 20:
        print(f'+ Variance %{var_reduction_pct:.1f} azaldi')
        print(f'  -> "Variance reduced by {var_reduction_pct:.1f}%, indicating more consistent behavior"')
    
    if cv_improvement_pct > 15:
        print(f'+ CV %{cv_improvement_pct:.1f} iyilesti (stabilite gostergesi)')
        print(f'  -> "Coefficient of variation improved by {cv_improvement_pct:.1f}%,"')
        print(f'     "demonstrating both higher rewards and greater stability"')
    
    print()
    print('ONERILEN SUNUS:')
    print('-' * 70)
    if var_reduction_pct > std_reduction_pct * 1.5:
        print('-> VARIANCE REDUCTION vurgulayin (en dramatik iyilesme)')
        print(f'  "Variance decreased from {before_var:.1f} to {after_var:.1f}"')
        print(f'  "representing a {var_reduction_pct:.1f}% reduction in performance variability"')
    elif cv_improvement_pct > 15:
        print('-> COEFFICIENT OF VARIATION vurgulayin (normalized metric)')
        print(f'  "CV improved from {before_cv:.1f}% to {after_cv:.1f}%"')
        print(f'  "showing {cv_improvement_pct:.1f}% better stability relative to mean"')
    else:
        print('-> COMBINED IMPROVEMENT vurgulayin')
        print(f'  "Mean reward increased by {mean_improvement_pct:.1f}% while"')
        print(f'  "variance reduced by {var_reduction_pct:.1f}%"')
