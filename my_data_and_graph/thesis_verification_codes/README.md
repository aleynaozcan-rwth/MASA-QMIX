# Thesis Verification Codes

This directory contains visualization scripts created for thesis verification and analysis.

## Files Overview

### Stochasticity Analysis
1. **analyze_job_arrival_stochasticity.py**
   - Basic stochasticity analysis with CDF comparison
   - Compares empirical job arrival inter-arrival times against theoretical exponential distribution
   - Outputs: CDF comparison plot, histogram with exponential overlay
   - Key metrics: KS statistic, CV (Coefficient of Variation)

2. **analyze_job_arrival_advanced_plots.py**
   - Advanced probability visualizations (KDE, Q-Q plot, combined panel)
   - Manual KDE implementation with Silverman's bandwidth
   - Outputs: KDE plot, Q-Q plot, 2x2 combined visualization
   - Features: Near Perfect Stochasticity annotation, CV explanations

### Job and Operation Level Analysis
3. **plot_arrivals_departures.py**
   - Job-level arrival and departure analysis
   - Three visualizations: dual-line with moving averages, completion ratio, combined view
   - Filters: 10 training failure episodes removed (0 completions)
   - Features: Moving average (window=20), color coordination (purple/orange)

4. **plot_operation_arrivals_completions.py**
   - Operation-level granular analysis
   - Parallel structure to job-level analysis
   - Different color scheme (purple/yellow/turquoise zones)
   - Features: Moving average, thin raw data visualization

### Decision Point Analysis
5. **plot_cumulative_dp_staircase.py**
   - Cumulative decision points staircase comparison (original data)
   - Compares event-driven vs fixed interval approaches
   - Episode 10: 37 decision points vs 25 standard DPs
   - Features: Timestamp annotations, vertical lines showing DP clustering

6. **plot_cumulative_dp_staircase_POISSON.py**
   - Cumulative decision points with TRUE POISSON arrivals
   - Same structure as non-POISSON version
   - Balanced Poisson generation (λ=0.602, seed=100, CV=0.91)
   - Separates job arrivals (blue) and operation completions (green)
   - Features: Color-coded annotations, legend with arrival rate information

## Data Sources
- Timeline file: `../historydata/scheduling_timeline.txt`
- Output directory: `../historydata/plots/`

## Key Parameters
- **Balanced Poisson**: λ=0.602, seed=100, CV=0.91, max_gap=4.44
- **Stochasticity Analysis**: λ=0.125 (mean interval=8.0)
- **Moving Average Window**: 20 episodes
- **Training Failures Filtered**: 10 episodes (0.77%)
- **Valid Episodes**: 1285 (down from 1295)

## Usage
Each script can be run independently:
```bash
python3 analyze_job_arrival_stochasticity.py
python3 analyze_job_arrival_advanced_plots.py
python3 plot_arrivals_departures.py
python3 plot_operation_arrivals_completions.py
python3 plot_cumulative_dp_staircase.py
python3 plot_cumulative_dp_staircase_POISSON.py
```

## Dependencies
- matplotlib
- numpy
- pathlib (standard library)
- re (standard library)

All scripts use Agg backend for non-interactive plotting.
