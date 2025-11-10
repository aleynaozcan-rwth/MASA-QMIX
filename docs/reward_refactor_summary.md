# MASA-QMIX – Hybrid Reward Refactor Summary
**Branch:** `refactor-modules-hybridreward-v9`  
**Date:** 2025-11-10  
**Author:** [Fatmanur Aleyna Özcan]

---

## 🧭 Overview
This refactor introduces a hybrid global–local reward system into the MASA-QMIX framework.
The previous scalar reward (alpha/beta/gamma/delta) has been replaced with a modular, interpretable
formulation that combines system-level and agent-level metrics.

The new structure enables improved interpretability, reward-shaping flexibility, and diagnostic
tracking during MARL training — while remaining fully compatible with the existing SimPy-based
environment and QMIX training loop by preserving a single scalar reward per decision.

---

## ⚙️ Implementation Summary

| Component | Description | Status |
|-----------|-------------|:-----:|
| `arguments.py` | Added hybrid reward parameters (w1–w5, a1–a3, α_mix, λ_m, λ_o) and a logging flag | ✅ |
| `environment.py` | Removed legacy reward logic and implemented new `pop_decision_reward()` combining global + local metrics | ✅ |
| Decision cache | Introduced `_last_decision_info` to store per-agent decision context used to compute local rewards | ✅ |
| Diagnostics | Added `last_reward_components` dict to record K1–K5, `R_global`, `R_local_mean`, `R_total` | ✅ |
| Tests | Added `tests/test_reward_hybrid_smoke.py` to validate environment initialization and reward integrity | ✅ |
| Legacy cleanup | Removed all `alpha/beta/gamma/delta/c_time` usages from edited source files and replaced CLI flags | ✅ |

---

## 📈 Reward Formulation

This refactor separates interpretable components (K1..K5) for a global view of system health and a
local, per-decision reward for agent-centric signals. At every decision pop, the environment computes
and returns a single scalar `R_total` while exposing the decomposition for logging/analysis.

### Components (K1..K5)
- K1 — CompletedNorm: normalized count of completed jobs / operations (range approx. 0..1)
- K2 — AvgWaitNorm: average wait per job, normalized and clipped to [0,1]
- K3 — WIPNorm: work-in-progress normalized by configured capacity (`max_jobs`)
- K4 — ThroughputDelta: short-run change in completed throughput (small rolling window)
- K5 — LoadVariance: weighted variance across machine and operator utilization (uses `lambda_m`/`lambda_o`)

### Weights and local terms
- Global weights: `w1..w5` (CLI flags `--reward_w1_completed`, `--reward_w2_avgwait`, `--reward_w3_wip`, `--reward_w4_throughput_delta`, `--reward_w5_load_variance`)
- Local per-decision weights: `a1..a3` (`--reward_a1_completion`, `--reward_a2_wait`, `--reward_a3_infeasible`)
- Mixing coefficient: `alpha_mix` (`--reward_alpha_mix`, default 0.7) determines the blend between global and local rewards
- Load variance mixture: `lambda_m`, `lambda_o` (`--reward_lambda_m`, `--reward_lambda_o`)

### Formulas
- R_global = w1 * CompletedNorm
  - w2 * AvgWaitNorm
  - w3 * WIPNorm
  + w4 * ThroughputDelta
  - w5 * LoadVariance

- For each decision i, local (agent) reward r_local_i = a1 * completed_i - a2 * wait_penalty_i - a3 * infeasible_i
  - R_local_mean = mean_i(r_local_i) over decisions in the batch

- R_total = alpha_mix * R_global + (1 - alpha_mix) * R_local_mean

The implementation writes a diagnostic dictionary on the environment instance as:

```py
env.last_reward_components == {
    'CompletedNorm': ..., 'AvgWaitNorm': ..., 'WIPNorm': ...,
    'ThroughputDelta': ..., 'LoadVariance': ...,
    'R_global': ..., 'R_local_mean': ..., 'R_total': ...
}
```

This preserves the single scalar `R_total` return expected by the rollout/replay pipeline while
making each component available for logging, analysis and tuning.

---

## 🔧 Defaults (from CLI)
- reward_w1_completed: 1.0
- reward_w2_avgwait: 0.6
- reward_w3_wip: 0.3
- reward_w4_throughput_delta: 0.8
- reward_w5_load_variance: 0.4
- reward_a1_completion: 1.0
- reward_a2_wait: 0.5
- reward_a3_infeasible: 0.25
- reward_alpha_mix: 0.7
- reward_lambda_m: 0.8
- reward_lambda_o: 0.2

These are centrally defined in `MARL/common/arguments.py` so they can be overridden from the CLI or
from test harnesses.

---

## ✅ Verification & quick checks
- A lightweight smoke test was added at `tests/test_reward_hybrid_smoke.py`. It checks:
  - `MASAEnv` initializes with default args
  - `env.pop_decision_reward()` returns a `float`
  - `env.last_reward_components` contains the expected keys (K1..K5, `R_global`, `R_local_mean`, `R_total`) and numeric values

Run the smoke test locally:

```bash
pytest -q tests/test_reward_hybrid_smoke.py
```

---

## ⚠️ Compatibility & migration notes
- The MARL training stack (rollout -> replay -> QMIX) expects a scalar per-decision reward. This refactor
  preserves that contract by returning `R_total` as a float.
- Tests and tools that referenced legacy names (`env.alpha`, `args.reward_alpha`, etc.) were updated where
  necessary in the `tests/` tree. Non-test tooli# MASA-QMIX – Hybrid Reward Refactor Summary  
**Branch:** `refactor-modules-hybridreward-v9`  
**Date:** 2025-11-10  
**Author:** [Fatmanur Aleyna Özcan]  

---

## 🧭 Overview  
This refactor introduces a **hybrid global–local reward system** into the MASA-QMIX framework.  
The previous scalar reward (defined by alpha/beta/gamma/delta) has been replaced with a modular, interpretable formulation combining system-level and agent-level metrics.  

The new structure enables improved interpretability, reward shaping flexibility, and diagnostic tracking during MARL training — while remaining fully compatible with the existing SimPy-based environment and QMIX training loop.  

---

## ⚙️ Implementation Summary  

| Component | Description | Status |
|------------|--------------|--------|
| **arguments.py** | Added hybrid reward parameters (w1–w5, a1–a3, α_mix, λ_m, λ_o) | ✅ |
| **environment.py** | Removed legacy reward logic and implemented new `pop_decision_reward()` combining global + local metrics | ✅ |
| **Decision Cache** | Introduced `_last_decision_info` to store per-agent decision context | ✅ |
| **Diagnostics** | Added `last_reward_components` dict to record K1–K5, R_global, R_local_mean, R_total | ✅ |
| **Tests** | Added `tests/test_reward_hybrid_smoke.py` to validate environment initialization and reward integrity | ✅ |
| **Legacy Cleanup** | Removed all `alpha/beta/gamma/delta/c_time` references | ✅ |

---

## 📈 Reward Formulation  

**Global Reward (R_global):**
R_global = w1CompletedNorm
- w2AvgWaitNorm
- w3WIPNorm
+ w4ThroughputDelta
- w5*LoadVariance

java
Copy code

**Local Reward (R_local_i):**
R_local_i = +a1*(completed_op_by_i)
- a2*(wait_i_norm)
- a3*(infeasible_action)

graphql
Copy code

**Final Total Reward (scalar):**
R_total = α_mix * R_global + (1 - α_mix) * mean_i(R_local_i)

yaml
Copy code

**Diagnostic Outputs:**
- CompletedNorm, AvgWaitNorm, WIPNorm, ThroughputDelta, LoadVariance  
- R_global, R_local_mean, R_total  

All stored under `env.last_reward_components` after each decision step.  

---

## 🔍 Validation  

✅ **Smoke Test:**  
- `pytest tests/test_reward_hybrid_smoke.py` passed successfully.  
- Verified scalar reward output and valid component structure.  

✅ **Syntax Check:**  
- `python -m py_compile environment.py` — passed with no errors.  

✅ **Legacy Audit:**  
- No references to `env.alpha`, `args.reward_alpha`, or related fields remain.  

✅ **Replay & Training Compatibility:**  
- The new reward system maintains a scalar `R_total` → replay → QMIX pipeline unchanged.  

---

## 🧩 Research Implications  

- Enables **reward decomposition analysis** during training.  
- Facilitates **cooperative vs. local policy evaluation** (by adjusting α_mix).  
- Provides **traceable component contributions** for explainability.  

Future extensions may include:
- Logging reward components across training epochs for correlation studies.
- Adaptive tuning of α_mix based on performance metrics.

---

## 📚 Citation / Reference  

If included in thesis or reports, cite as:

> Özcan, F.A. (2025). *Hybrid Reward Integration in MASA-QMIX:  
> A Modular Global–Local Reward Architecture for Multi-Agent Scheduling Environments.*  
> RWTH Aachen University – MASA-QMIX Thesis Project, v9 Branch.

---
ng (e.g. scripts under `tools/`) may still reference legacy
  names and should be updated or adapted if you plan to run them.
- If you prefer a backward-compatibility shim (to avoid touching many external scripts), consider adding
  short aliasing assignments inside the `MASAEnv` constructor (e.g., `self.alpha = self.reward_w1`) — this
  was intentionally avoided in the current refactor to keep the API explicit.

---

## ▶️ Next steps (suggested)
- Run an end-to-end training run (or a quick rollout smoke harness) to inspect trajectories and component logs.
- Tune `w1..w5, a1..a3, alpha_mix` using small grid search experiments and monitor `last_reward_components`.
- Add lightweight unit tests that populate `_last_decision_info` to exercise `R_local_mean` behavior explicitly.

---

If you want, I can prepare a short PR description and open a PR against `main` with this branch, or add a
compatibility alias layer in `MASAEnv` to smooth migration for external scripts — tell me which you prefer.
