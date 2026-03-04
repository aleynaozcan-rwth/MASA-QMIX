# MASA-QMIX

**Job-centric Multi-Agent Reinforcement Learning Framework for Dynamic Flexible Job Shop Scheduling (FJSP)**

This repository implements a QMIX-based job-centric multi-agent reinforcement learning system for dynamic flexible job shop scheduling problems with stochastic job arrivals, operator qualifications, and event-driven decision-making.

---


## 🎯 Overview

MASA-QMIX solves flexible job scheduling problem where:
- **Jobs arrive dynamically** following Poisson processes with exponential inter-arrival times
- **Machines require qualified operators** to process operations (dual-resource constraint)
- **Decisions are event-driven**, triggered by job arrivals and operation completions
- **Multiple agents coordinate** to minimize wait times and maximize throughput
- **Parallel decision making and conflicts between agents enabled** to reinforce experience gaining and don´t lose learning oppurtunities in a stochastic environment



The system uses QMIX value decomposition to train job agents that learn coordinated scheduling policies through centralized training and decentralized execution.

---

## ✨ Key Features

- **Event-Driven Simulation**: SimPy-based discrete-event simulation (not fixed timesteps)
- **Dual-Resource Constraints**: Machine availability + operator qualifications
- **Stochastic Environment**: Random job arrivals (λ=0.4), heterogeneous job structures (2-4 operations/job)
- **Hybrid Reward Function**: Balances completed jobs, wait times, throughput, and load balancing
- **Action Masking**: Dynamic feasibility constraints prevent invalid machine-operator assignments
- **Reproducibility**: Full seeding support for NumPy, PyTorch, and simulation RNGs
- **Real-time Visualization**: Gantt charts, training curves, KPI breakdowns

---

## 📁 Project Structure

### **Root Directory**
```
MASA-QMIX/
├── main.py                          # Entry point: training/evaluation orchestration
├── environment.py                   # MASAEnv: SimPy-based MARL environment 
├── requirements.txt                 # Python dependencies
├── run_masa_qmix.sh                # SLURM batch script for HPC clusters
├── run_cpu.sh                       # Local CPU training script
└── README.md                        # This file
```

#### **`main.py`**
- **Purpose**: Main entry point for training and evaluation
- **Key Functions**:
  - `marl_agent_wrapper()`: Standard MARL training loop with reproducibility controls
  - `random_agent_baseline()`: Random policy baseline for comparison
- **Usage**: `python main.py --alg qmix --seed 123 --n_epoch 200`

#### **`environment.py`** 
- **Purpose**: Core simulation environment implementing Gym-like interface
- **Key Components**:
  - `MASAEnv` class: Main environment with `reset()` and `step()` methods
  - `RunningMeanStd`: Welford's algorithm for reward normalization
  - Event-driven scheduling logic (job arrivals + operation completions)
  - Observation/state construction (7D per-agent obs, 10D global state)
  - Hybrid reward calculation (5 weighted components)
- **Decision Points**: ~39,000 decision points over 800 episodes (variable per episode: 30-70)

---

### **`utils/` - Simulation Components**

All domain-specific logic for job shop scheduling simulation.

#### **`utils/jobagent.py`** (286 lines)
- **Purpose**: Job-level agents that execute operations on machines
- **Key Classes**:
  - `JobAgent`: Represents a single job with operation sequence
    - Tracks operation progress, wait times, start/completion times
    - Records machine/operator assignments
  - `JobAgents`: Container managing all JobAgent instances
- **Key Methods**:
  - `execute_task(machine_id, workcenter_id)`: Processes operation at machine-level
  - `record_action()`: Logs decision traces for Gantt visualization

#### **`utils/workcenter.py`** (348 lines)
- **Purpose**: Machine and work center registry (topology definition)
- **Default Topology**:
  - 3 Work Centers (WC1, WC2, WC3)
  - 5 Machines (M0-M4) distributed across work centers
  - Machine capabilities: Each machine can process 3-6 operation types (Op1-Op9)
- **Key Classes**:
  - `WorkCenter`: Single work center with SimPy resources for machines
  - `WorkCenters`: Global registry maintaining all machines and their capabilities
- **Data**: `DEFAULT_WORKCENTERS` dict defines machine→operations mapping

#### **`utils/operator.py`** (507 lines)
- **Purpose**: Operator resource management with qualification constraints
- **Key Classes**:
  - `Operator`: Single operator with qualified machine list
    - Owns SimPy Resource for concurrency control
    - Tracks assignment history and availability
  - `Operators`: Container managing all operators (default: 4 operators)
- **Qualification Matrix**:
  - O1, O3: Qualified for M0, M3, M4
  - O2, O4: Qualified for M1, M2, M4
  - M4 is bottleneck (all operators qualified)
- **Key Methods**:
  - `allocate()`: Assigns operator to job (SimPy request)
  - `release()`: Frees operator after operation completion

#### **`utils/job.py`** (97 lines)
- **Purpose**: Job metadata registry
- **Key Classes**:
  - `Job`: Single job with ID, name, and code
  - `Jobs`: Registry of job definitions
- **Note**: Actual job *instances* are generated by `TaskGenerator` at runtime, not from static registry

#### **`utils/task_generator.py`** (382 lines)
- **Purpose**: Dynamic job generation with stochastic arrivals
- **Key Features**:
  - Poisson arrival process (λ=0.4, configurable)
  - Random operation sequences (2-4 operations per job)
  - Processing time sampling from exponential distributions (machine-specific means)
  - Valid sequence validation (ensures all operations can be processed)
- **Key Methods**:
  - `generate_random_job_sequence()`: Creates valid operation sequence
  - Injects jobs into SimPy environment using `env.process()`

#### **`utils/gantt.py`**
- **Purpose**: Gantt chart visualization utilities
- **Generates**: Timeline plots showing job execution, wait times, machine utilization

#### **`utils/io_control.py`**
- **Purpose**: File I/O utilities for saving/loading training data

---

### **`MARL/` - Reinforcement Learning Framework**

#### **`MARL/runner.py`** (3102 lines)
- **Purpose**: Training orchestration and experiment management
- **Key Classes**:
  - `Runner`: Main training loop coordinator
    - Episode rollout collection
    - Replay buffer management
    - Agent training steps
    - Metrics logging and visualization
- **Key Methods**:
  - `run()`: Main training loop (epochs → episodes → steps)
  - `evaluate()`: Evaluation without exploration
  - `generate_episode()`: Single episode rollout with environment interaction
  - `visualize_training_progress()`: Matplotlib-based training curves
- **Visualization**: Generates TD error, loss, reward, Q-value plots

#### **`MARL/agent/agent.py`** (186 lines)
- **Purpose**: Agent wrapper managing policy selection
- **Key Classes**:
  - `Agents`: Main agent container
    - Supports 8 algorithms: VDN, QMIX, COMA, QTRAN, MAVEN, Central-V, REINFORCE
    - Routes learning to selected policy
- **Key Methods**:
  - `learn_from_replay()`: Batch learning from replay buffer
  - `choose_action()`: Action selection (epsilon-greedy with masking)

---

### **`MARL/policy/` - MARL Algorithms**

Each file implements a specific multi-agent RL algorithm.

#### **`MARL/policy/qmix.py`**
- **Purpose**: QMIX value decomposition (default algorithm)
- **Key Features**:
  - Hypernetwork-based mixing network
  - Monotonicity constraint (ensures IGM: Individual-Global-Max)
  - Target networks for stable learning
- **Loss**: TD(0) loss with gradient clipping (max_norm=10.0)

#### **`MARL/policy/vdn.py`**
- **Purpose**: Value Decomposition Networks (simpler baseline)
- **Mixing**: Linear sum of agent Q-values

#### **`MARL/policy/coma.py`**
- **Purpose**: Counterfactual Multi-Agent policy gradients
- **Architecture**: Actor-critic with centralized critic

#### Other Policies
- `qtran_alt.py`, `qtran_base.py`: QTRAN variants
- `maven.py`: MAVEN (exploration via latent space)
- `central_v.py`, `reinforce.py`: Baseline algorithms

---

### **`MARL/network/` - Neural Network Architectures**

#### **`MARL/network/base_net.py`**
- **Purpose**: Base RNN architecture for agents
- **Architecture**: GRU-based recurrent network (64 hidden units, 1 layer)
- **Input**: 7D observation → Embedding → GRU → Q-values (n_actions)

#### **`MARL/network/qmix_net.py`**
- **Purpose**: QMIX mixer network
- **Architecture**: Hypernetwork generates weights for mixing network
- **Input**: Global state (10D) → Hypernetworks → Mixer weights
- **Output**: Global Q_tot from individual Q-values

#### Other Networks
- `vdn_net.py`: Simple summation mixer
- `coma_critic.py`: COMA centralized critic
- `commnet.py`, `g2anet.py`: Communication networks
- `qtran_net.py`, `maven_net.py`: Specialized architectures

---

### **`MARL/common/` - Shared Utilities**

#### **`MARL/common/arguments.py`** (368 lines)
- **Purpose**: Central configuration management (single source of truth)
- **Key Functions**:
  - `get_mutable_args()`: Returns argparse namespace with all hyperparameters
  - `get_common_args()`, `get_mixer_args()`: Category-specific argument groups
- **Key Parameters**:
  ```python
  # Training
  n_epoch = 200                    # Training epochs
  n_episodes = 4                   # Episodes per epoch
  episode_limit = 50.0             # Simulation time limit
  
  # Environment
  arrival_lambda = 0.4             # Job arrival rate
  num_operators = 4                # Operator count
  n_agents = 5                     # Machine agents
  n_actions = 18                   # Action space (5 machines × 4 operators - duplicates)
  
  # Network
  rnn_hidden_dim = 64              # GRU hidden size
  qmix_hidden_dim = 32             # Mixer embedding dimension
  
  # Learning
  lr = 5e-4                        # Learning rate
  gamma = 0.99                     # Discount factor
  epsilon_anneal_scale = 0.6       # Exploration decay (60% of training)
  grad_norm_clip = 10.0            # Gradient clipping
  
  # Reward weights
  reward_w1_completed = 3.0        # Completed jobs weight
  reward_w2_avgwait = 2.0          # Average wait time penalty
  reward_w4_throughput_delta = 4.0 # Throughput improvement bonus
  reward_w5_load_variance = 0.3    # Load balancing regularization
  reward_scale = 2.0               # Global scaling factor
  ```

#### **`MARL/common/mask_utils.py`**
- **Purpose**: Action masking for invalid machine-operator pairs
- **Key Function**: `build_machine_major_mask(env, job_id)`
  - Returns binary mask: 1 = valid action, 0 = invalid
  - Machine available ⟺ machine free AND ≥1 qualified operator free
- **Critical**: Ensures reproducible, deterministic rollouts

#### **`MARL/common/replay_buffer.py`**
- **Purpose**: Experience replay storage
- **Capacity**: 5,000 transitions (configurable via `buffer_size`)
- **Sampling**: Random minibatch sampling (batch_size=32)

#### **`MARL/common/rollout.py`**
- **Purpose**: Episode rollout worker for data collection
- **Key Class**: `RolloutWorker`
  - Interacts with environment
  - Constructs observations/states
  - Applies action masks
  - Stores transitions in buffer

#### **`MARL/common/utils.py`**
- **Purpose**: Miscellaneous utilities (tensor conversions, logging helpers)

#### **`MARL/common/terms.py`**
- **Purpose**: Unified terminology helper (maps legacy variable names)

#### **`MARL/common/analyse.py`**
- **Purpose**: Post-training analysis utilities

---

### **Visualization & Analysis Scripts**

#### **`my_data_and_graph/`**
- `metrics.py`: Metrics computation 
- `plot_metrics.py`: Plotting utilities
- `historydata/`: Saved training metrics (png, CSV, txt, pickle)

---

## 🚀 Installation & Usage

### **Prerequisites**
```bash
Python 3.11+
CUDA 11.8+ (optional, for GPU training)
```

### **Installation**
```bash
# Clone repository
cd MASA-QMIX

# Install dependencies
pip install -r requirements.txt
```

### **Training**
```bash
# Local CPU training (default)
python main.py --alg qmix --seed 123

# GPU training
python main.py --alg qmix --seed 123 --cuda

# Custom configuration
python main.py \
  --alg qmix \
  --n_epoch 200 \
  --n_episodes 4 \
  --lr 5e-4 \
  --arrival_lambda 0.4 \
  --seed 123
```

### **HPC Cluster (SLURM)**
```bash
sbatch run_masa_qmix.sh
```

---



## 🔬 System Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│              (Training orchestration)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    MARL/runner.py                            │
│         (Episode rollout, buffer, training loop)             │
└──────┬──────────────────────────────────────────────────┬───┘
       │                                                   │
       ▼                                                   ▼
┌──────────────┐                                  ┌──────────────┐
│ environment  │                                  │ MARL/agent/  │
│   .py        │◄────observe/state────────────────┤  agent.py    │
│ (SimPy Env)  │                                  │ (QMIX Policy)│
└──────┬───────┘────────reward──────────────────►└──────┬───────┘
       │                                                 │
       │ ┌─────────────────────────────────────────┐   │
       │ │        utils/ (Simulation)               │   │
       │ │  ┌──────────────┐  ┌─────────────┐     │   │
       └►│  │ jobagent.py  │  │ operator.py │     │   │
         │  │ (Jobs)       │  │ (Operators) │     │   │
         │  └──────────────┘  └─────────────┘     │   │
         │  ┌──────────────┐  ┌─────────────┐     │   │
         │  │workcenter.py │  │task_gen.py  │     │   │
         │  │ (Machines)   │  │ (Arrivals)  │     │   │
         │  └──────────────┘  └─────────────┘     │   │
         └─────────────────────────────────────────┘   │
                                                        │
                      ┌─────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │     MARL/policy/qmix.py        │
         │  (Value decomposition learning)│
         └────────────────┬───────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │   MARL/network/qmix_net.py     │
         │ (RNN agents + Hypernetwork     │
         │  mixer)                        │
         └────────────────────────────────┘
```

---

## 📖 Citation

this work extends standart static MASA-QMIX algorithm introduced in this paper:

```bibtex
@article{wang2022solving,
  title={Solving job scheduling problems in a resource preemption environment with multi-agent reinforcement learning},
  author={Wang, Xiaohan and Zhang, Lin and Lin, Tingyu and Zhao, Chun and Wang, Kunyu and Chen, Zhen},
  journal={Robotics and Computer-Integrated Manufacturing},
  volume={77},
  pages={102324},
  year={2022},
  publisher={Elsevier}
}
```

---
