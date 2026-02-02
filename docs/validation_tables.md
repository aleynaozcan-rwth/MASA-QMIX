# Validation Chapter Tables - Environmental Setup

## Table 1: Operation-Machine Compatibility and Processing Times

This table shows which operations can be performed on which machines and their respective mean processing times (in time units).

| Operation | M0 (WC1) | M1 (WC1) | M2 (WC2) | M3 (WC2) | M4 (WC3) |
|-----------|----------|----------|----------|----------|----------|
| **Op1**   | 1.225    | —        | 1.575    | 1.400    | 1.050    |
| **Op2**   | 1.050    | —        | 1.750    | —        | —        |
| **Op3**   | 1.575    | —        | —        | 1.750    | 1.925    |
| **Op4**   | —        | 1.575    | 1.680    | —        | 2.100    |
| **Op5**   | 2.275    | 1.750    | —        | —        | —        |
| **Op6**   | —        | —        | 2.100    | —        | 2.800    |
| **Op7**   | —        | —        | —        | 1.820    | —        |
| **Op8**   | —        | 2.975    | —        | 2.625    | 1.575    |
| **Op9**   | 2.975    | —        | 3.150    | —        | 1.925    |

**Note:** "—" indicates that the operation cannot be performed on that machine.

**Summary:**
- **Total Machines:** 5 (M0, M1, M2, M3, M4)
- **Total Operations:** 9 (Op1 through Op9)
- **Work Centers:** 3 (WC1: M0-M1, WC2: M2-M3, WC3: M4)

---

## Table 2: Operator-Machine Compatibility Matrix

This table shows which operators are qualified to work on which machines.

| Operator | M0 (WC1) | M1 (WC1) | M2 (WC2) | M3 (WC2) | M4 (WC3) |
|----------|----------|----------|----------|----------|----------|
| **O1**   | ✓        | ✗        | ✗        | ✓        | ✓        |
| **O2**   | ✗        | ✓        | ✓        | ✗        | ✓        |
| **O3**   | ✓        | ✗        | ✗        | ✓        | ✓        |
| **O4**   | ✗        | ✓        | ✓        | ✗        | ✓        |

**Legend:**
- ✓ = Operator is qualified to work on this machine
- ✗ = Operator is not qualified to work on this machine

**Summary:**
- **Total Operators:** 4 (O1, O2, O3, O4)
- **Operator Qualifications:**
  - O1 & O3: Qualified for M0, M3, M4
  - O2 & O4: Qualified for M1, M2, M4
- **Note:** All operators can work on machine M4, creating flexibility in scheduling

---

## LaTeX Version (for direct thesis inclusion)

### Table 1: Operation-Machine Compatibility (LaTeX)

```latex
\begin{table}[H]
\centering
\caption{Operation-Machine Compatibility and Processing Times}
\label{tab:op-machine-compatibility}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Operation} & \textbf{M0 (WC1)} & \textbf{M1 (WC1)} & \textbf{M2 (WC2)} & \textbf{M3 (WC2)} & \textbf{M4 (WC3)} \\ \hline
\textbf{Op1} & 1.225 & — & 1.575 & 1.400 & 1.050 \\ \hline
\textbf{Op2} & 1.050 & — & 1.750 & — & — \\ \hline
\textbf{Op3} & 1.575 & — & — & 1.750 & 1.925 \\ \hline
\textbf{Op4} & — & 1.575 & 1.680 & — & 2.100 \\ \hline
\textbf{Op5} & 2.275 & 1.750 & — & — & — \\ \hline
\textbf{Op6} & — & — & 2.100 & — & 2.800 \\ \hline
\textbf{Op7} & — & — & — & 1.820 & — \\ \hline
\textbf{Op8} & — & 2.975 & — & 2.625 & 1.575 \\ \hline
\textbf{Op9} & 2.975 & — & 3.150 & — & 1.925 \\ \hline
\hline
\multicolumn{6}{|l|}{\small Note: "—" indicates that the operation cannot be performed on that machine.} \\ 
\multicolumn{6}{|l|}{\small Processing times are in time units.} \\
\end{tabular}
\end{table}
```

### Table 2: Operator-Machine Compatibility (LaTeX)

```latex
\begin{table}[H]
\centering
\caption{Operator-Machine Compatibility Matrix}
\label{tab:operator-machine-compatibility}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Operator} & \textbf{M0 (WC1)} & \textbf{M1 (WC1)} & \textbf{M2 (WC2)} & \textbf{M3 (WC2)} & \textbf{M4 (WC3)} \\ \hline
\textbf{O1} & \checkmark & $\times$ & $\times$ & \checkmark & \checkmark \\ \hline
\textbf{O2} & $\times$ & \checkmark & \checkmark & $\times$ & \checkmark \\ \hline
\textbf{O3} & \checkmark & $\times$ & $\times$ & \checkmark & \checkmark \\ \hline
\textbf{O4} & $\times$ & \checkmark & \checkmark & $\times$ & \checkmark \\ \hline
\hline
\multicolumn{6}{|l|}{\small Note: \checkmark~indicates operator is qualified, $\times$ indicates not qualified.} \\
\end{tabular}
\end{table}
```

**LaTeX Package Requirements:**
```latex
\usepackage{amssymb}  % for \checkmark
% Optional: \usepackage{float} for [H] placement (if using [H] instead of [!htbp])
```

---

## Additional Environmental Configuration Summary

For your validation chapter, you may also want to include:

### Table 3: Environment Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Number of Machines | 5 | M0, M1, M2, M3, M4 |
| Number of Work Centers | 3 | WC1 (M0-M1), WC2 (M2-M3), WC3 (M4) |
| Number of Operators | 4 | O1, O2, O3, O4 |
| Number of Operations | 9 | Op1 through Op9 |
| Operations per Job (range) | 1-9 | Uniformly distributed |
| Initial Jobs | 4 | Heterogeneous jobs arrive at episode start |
| Job Arrival Rate (λ) | 0.4 | Mean arrival rate (jobs/time unit) |
| Episode Limit | 50 | Continuous simulation time units |

### Table 4: Training and Learning Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Training Loop** | | |
| Number of Epochs | 200 | Total training epochs |
| Episodes per Epoch | 4 | Episodes in each epoch |
| Evaluation Frequency | Every 2 epochs | Model evaluation interval |
| **Learning Algorithm** | | |
| Algorithm | QMIX | Multi-agent Q-learning |
| Discount Factor (γ) | 0.99 | Future reward discount |
| Learning Rate | 5×10⁻⁴ | Optimizer step size |
| Optimizer | RMSprop | Gradient descent optimizer |
| **Experience Replay** | | |
| Replay Buffer Size | 5000 | Experience storage capacity |
| Batch Size | 32 | Training batch size |
| Min Warmup Size | 800 | Min samples before training |
| Training Steps per Update | 30 | Gradient updates per cycle |
| Target Network Update | Every 50 steps | Target network sync frequency |
| **Exploration** | | |
| ε-greedy (start) | 1.0 | Initial exploration rate |
| ε-greedy (end) | 0.1 | Final exploration rate |
| ε Anneal Fraction | 0.6 | Fraction of time for decay |
| **Neural Network** | | |
| RNN Hidden Dimension | 64 | Recurrent layer size |
| RNN Number of Layers | 1 | Number of recurrent layers |
| Mixer Embedding Dimension | 32 | QMIX mixing network size |
| Gradient Clipping | 10.0 | Max gradient norm |
| **Normalization** | | |
| Reward Normalization | True | Welford's algorithm, clipped to [-10, +10] |
| Observation Normalization | True | Parametric min-max scaling to [0, 1] |
| State Normalization | True | Parametric min-max scaling to [0, 1] |

### Table 3: Environment Configuration Parameters (LaTeX)

```latex
\begin{table}[H]
\centering
\caption{Environment Configuration Parameters}
\label{tab:environment-config}
\begin{tabular}{|l|c|p{6.5cm}|}
\hline
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\ \hline
Number of Machines & 5 & M0, M1, M2, M3, M4 \\ \hline
Number of Work Centers & 3 & WC1 (M0-M1), WC2 (M2-M3), WC3 (M4) \\ \hline
Number of Operators & 4 & O1, O2, O3, O4 \\ \hline
Number of Operations & 9 & Op1 through Op9 \\ \hline
Operations per Job (range) & 1-9 & Uniformly distributed \\ \hline
Initial Jobs & 4 & Heterogeneous jobs arrive at episode start \\ \hline
Job Arrival Rate ($\lambda$) & 0.4 & Mean arrival rate (jobs/time unit) \\ \hline
Episode Limit & 50 & Continuous simulation time units \\ \hline
\end{tabular}
\end{table}
```

### Table 4: Training and Learning Configuration (LaTeX)

```latex
\begin{table}[H]
\centering
\caption{Training and Learning Configuration}
\label{tab:training-config}
\begin{tabular}{|l|c|p{5.5cm}|}
\hline
\textbf{Parameter} & \textbf{Value} & \textbf{Description} \\ \hline
\multicolumn{3}{|c|}{\textbf{Training Loop}} \\ \hline
Number of Epochs & 200 & Total training epochs \\ \hline
Episodes per Epoch & 4 & Episodes in each epoch \\ \hline
Evaluation Frequency & Every 2 epochs & Model evaluation interval \\ \hline
\multicolumn{3}{|c|}{\textbf{Learning Algorithm}} \\ \hline
Algorithm & QMIX & Multi-agent Q-learning \\ \hline
Discount Factor ($\gamma$) & 0.99 & Future reward discount \\ \hline
Learning Rate & 5$\times$10$^{-4}$ & Optimizer step size \\ \hline
Optimizer & RMSprop & Gradient descent optimizer \\ \hline
\multicolumn{3}{|c|}{\textbf{Experience Replay}} \\ \hline
Replay Buffer Size & 5000 & Experience storage capacity \\ \hline
Batch Size & 32 & Training batch size \\ \hline
Min Warmup Size & 800 & Min samples before training \\ \hline
Training Steps per Update & 30 & Gradient updates per cycle \\ \hline
Target Network Update & Every 50 steps & Target network sync frequency \\ \hline
\multicolumn{3}{|c|}{\textbf{Exploration}} \\ \hline
$\epsilon$-greedy (start) & 1.0 & Initial exploration rate \\ \hline
$\epsilon$-greedy (end) & 0.1 & Final exploration rate \\ \hline
$\epsilon$ Anneal Fraction & 0.6 & Fraction of time for decay \\ \hline
\multicolumn{3}{|c|}{\textbf{Neural Network Architecture}} \\ \hline
RNN Hidden Dimension & 64 & Recurrent layer size \\ \hline
RNN Number of Layers & 1 & Number of recurrent layers \\ \hline
Mixer Embedding Dimension & 32 & QMIX mixing network size \\ \hline
Gradient Clipping & 10.0 & Maximum gradient norm \\ \hline
\multicolumn{3}{|c|}{\textbf{Normalization}} \\ \hline
Reward Normalization & Enabled & Welford's algorithm, clipped to [-10, +10] \\ \hline
Observation Normalization & Enabled & Parametric  scaling to [0, 1] \\ \hline
State Normalization & Enabled & Parametric  scaling to [0, 1] \\ \hline
\end{tabular}
\end{table}
```

### Table 5: Computing Infrastructure

| Component | Specification | Description |
|-----------|---------------|-------------|
| **Cluster** | | |
| Institution | RWTH Aachen HPC | High-Performance Computing Cluster |
| CPU Partition | c23ms | Multi-core CPU nodes |
| GPU Partition | c23g | GPU-enabled nodes |
| **Hardware Resources** | | |
| CPU Cores | 8 | Per task allocation |
| Memory (RAM) | 32-64 GB | 32GB (CPU), 64GB (GPU jobs) |
| GPU | 1× NVIDIA GPU | CUDA-enabled (GPU jobs only) |
| **Job Scheduler** | | |
| System | SLURM | Workload manager |
| Max Wall Time | 8-12 hours | 8h (GPU), 12h (CPU) |

### Table 6: Software Environment

| Component | Version | Purpose |
|-----------|---------|---------|
| **Core Environment** | | |
| Operating System | Linux | HPC cluster environment |
| Python | 3.10.8 | Main programming language |
| GCC Core | 12.2.0 | C/C++ compiler toolchain |
| **Simulation Framework** | | |
| SimPy | 4.1.1 | Discrete-event simulation |
| **Deep Learning** | | |
| PyTorch | 2.8.0+cu128 | Neural network framework |
| CUDA | 12.8 | GPU acceleration (when enabled) |
| **Scientific Computing** | | |
| NumPy | 2.3.3 | Numerical computations |
| Pandas | 2.3.3 | Data analysis and manipulation |
| PyYAML | ≥6.0 | Configuration management |
| **Visualization** | | |
| Matplotlib | 3.10.7 | Plotting and visualization |
| Seaborn | 0.13.2 | Statistical data visualization |
| **Reinforcement Learning** | | |
| Gym (OpenAI) | 0.26.2 | RL environment interface |
| **Development Tools** | | |
| pytest | ≥7.0 | Unit testing framework |
| Git | 2.47.3 | Version control |

### Table 5: Computing Infrastructure (LaTeX)

```latex
\begin{table}[H]
\centering
\caption{Computing Infrastructure}
\label{tab:computing-infrastructure}
\begin{tabular}{|l|l|p{5cm}|}
\hline
\textbf{Component} & \textbf{Specification} & \textbf{Description} \\ \hline
\multicolumn{3}{|c|}{\textbf{Cluster}} \\ \hline
Institution & RWTH Aachen HPC & High-Performance Computing Cluster \\ \hline
CPU Partition & c23ms & Multi-core CPU nodes \\ \hline
GPU Partition & c23g & GPU-enabled nodes \\ \hline
\multicolumn{3}{|c|}{\textbf{Hardware Resources}} \\ \hline
CPU Cores & 8 & Per task allocation \\ \hline
Memory (RAM) & 32-64 GB & 32GB (CPU), 64GB (GPU jobs) \\ \hline
GPU & 1$\times$ NVIDIA GPU & CUDA-enabled (GPU jobs only) \\ \hline
\multicolumn{3}{|c|}{\textbf{Job Scheduler}} \\ \hline
System & SLURM & Workload manager \\ \hline
Max Wall Time & 8-12 hours & 8h (GPU), 12h (CPU) \\ \hline
\end{tabular}
\end{table}
```

### Table 6: Software Environment (LaTeX)

```latex
\begin{table}[H]
\centering
\caption{Software Environment}
\label{tab:software-environment}
\begin{tabular}{|l|c|p{5.5cm}|}
\hline
\textbf{Component} & \textbf{Version} & \textbf{Purpose} \\ \hline
\multicolumn{3}{|c|}{\textbf{Core Environment}} \\ \hline
Operating System & Linux & HPC cluster environment \\ \hline
Python & 3.10.8 & Main programming language \\ \hline
GCC Core & 12.2.0 & C/C++ compiler toolchain \\ \hline
\multicolumn{3}{|c|}{\textbf{Simulation Framework}} \\ \hline
SimPy & 4.1.1 & Discrete-event simulation \\ \hline
\multicolumn{3}{|c|}{\textbf{Deep Learning}} \\ \hline
PyTorch & 2.8.0+cu128 & Neural network framework \\ \hline
CUDA & 12.8 & GPU acceleration (when enabled) \\ \hline
\multicolumn{3}{|c|}{\textbf{Scientific Computing}} \\ \hline
NumPy & 2.3.3 & Numerical computations \\ \hline
Pandas & 2.3.3 & Data analysis and manipulation \\ \hline
PyYAML & $\geq$6.0 & Configuration management \\ \hline
\multicolumn{3}{|c|}{\textbf{Visualization}} \\ \hline
Matplotlib & 3.10.7 & Plotting and visualization \\ \hline
Seaborn & 0.13.2 & Statistical data visualization \\ \hline
\multicolumn{3}{|c|}{\textbf{Reinforcement Learning}} \\ \hline
Gym (OpenAI) & 0.26.2 & RL environment interface \\ \hline
\multicolumn{3}{|c|}{\textbf{Development Tools}} \\ \hline
pytest & $\geq$7.0 & Unit testing framework \\ \hline
Git & 2.47.3 & Version control \\ \hline
\end{tabular}
\end{table}
```

---

### Machine Capability Summary

- **M0:** 5 operations (Op1, Op2, Op3, Op5, Op9)
- **M1:** 3 operations (Op4, Op5, Op8)
- **M2:** 5 operations (Op1, Op2, Op4, Op6, Op9)
- **M3:** 4 operations (Op1, Op3, Op7, Op8)
- **M4:** 6 operations (Op1, Op3, Op4, Op6, Op8, Op9) - Most versatile machine

### Bottleneck Analysis

- **Critical Operations:** Op7 (only M3), Op2 (only M0, M2)
- **Most Flexible Operation:** Op1 (can be performed on all machines except M1)
- **Most Flexible Machine:** M4 (capable of 6/9 operations)
- **Operator Distribution:** Balanced with redundancy (2 operators per machine group)
