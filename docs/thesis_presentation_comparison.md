# MASA-QMIX Adaptation for Dynamic FJSP
## Thesis & Presentation Content

---

## 📊 PRESENTATION SLIDE STRUCTURE

### Slide 1: Standard MASA-QMIX Architecture

**Title:** "MASA-QMIX: Foundational Multi-Agent RL Framework"

**Visual Elements:**
- Diagram showing 3-4 fixed agents (robots/vehicles)
- Q-network architecture (GRU + MLP)
- Mixing network with hypernetworks
- All agents acting synchronously

**Key Points:**
- Fixed number of agents (N = constant)
- Synchronous decision-making (all agents act together)
- Time-stepped episodes (T = fixed)
- Monotonicity constraint: ∂Q_tot/∂Q_i ≥ 0

**Speaker Notes:**
"MASA-QMIX is a value-based multi-agent reinforcement learning algorithm that learns decentralized policies while training in a centralized manner. Each agent has its own Q-network that processes local observations, and a mixing network combines these individual Q-values into a total Q-value while maintaining the monotonicity constraint."

---

### Slide 2: Job-Centric MASA-QMIX for Dynamic FJSP

**Title:** "Our Adaptation: Job-Centric MASA-QMIX"

**Visual Elements:**
- Jobs as agents (arriving stochastically)
- Asynchronous decision flow
- Action masking visualization
- Variable-length episode timeline

**Key Innovations Highlighted:**
1. **Paradigm Shift**: Jobs as agents ⚡
2. **Dynamic Agent Set**: N(t) varies with arrivals ⚡
3. **Action Masking**: Capability constraints ⚡
4. **Event-Driven**: Asynchronous decisions ⚡

**Speaker Notes:**
"To address dynamic FJSP with stochastic arrivals, we adapted MASA-QMIX with four key innovations: First, we redefine agents as jobs rather than physical entities. Second, our architecture handles dynamic agent sets where jobs arrive stochastically. Third, we implement action masking to enforce manufacturing constraints. Finally, decisions are event-driven rather than synchronized time steps."

---

## 📝 THESIS THEORETICAL BACKGROUND SECTION

### Section: "Multi-Agent Reinforcement Learning: MASA-QMIX"

#### Subsection 1: QMIX Fundamentals

QMIX (Rashid et al., 2018) is a value-based multi-agent reinforcement learning algorithm designed for cooperative tasks under partial observability. The algorithm addresses the challenge of learning in decentralized execution while training with centralized information.

**Core Principle:**

QMIX represents the joint action-value function Q_tot as a monotonic combination of individual agent Q-values:

```
Q_tot(τ, u) = f_mix(Q_1(τ^1, u^1), ..., Q_n(τ^n, u^n), s)
```

where:
- τ^i represents the action-observation history for agent i
- u^i is the action selected by agent i
- s is the global state
- f_mix is the mixing network

**Monotonicity Constraint:**

The key innovation is maintaining monotonicity:

```
∂Q_tot/∂Q_i ≥ 0, ∀i
```

This ensures that individual greedy action selection (argmax_u^i Q_i) is consistent with team coordination, enabling decentralized execution.

**Architecture Components:**

1. **Agent Networks**: Each agent i has a GRU-based Q-network that processes its local observation history τ^i and outputs Q-values for all actions.

2. **Mixing Network**: Combines individual Q-values using hypernetworks that generate mixing weights and biases from the global state, ensuring non-negative weights.

3. **Training**: Uses centralized training with decentralized execution (CTDE) paradigm, where the mixing network has access to global state during training but individual agents use only local observations during execution.

---

#### Subsection 2: MASA Extension

MASA (Multi-Agent Soft Actor-Critic) extends QMIX with:
- Improved exploration through entropy regularization
- Soft Q-value updates for more stable learning
- Enhanced credit assignment mechanisms

The fundamental architecture remains similar to QMIX, with modifications in the training procedure for better sample efficiency.

---

## 🔧 THESIS METHODOLOGY SECTION

### Section: "Adapting MASA-QMIX for Dynamic FJSP"

The dynamic FJSP with stochastic job arrivals presents unique challenges that require significant adaptations to the standard MASA-QMIX framework. This section details our job-centric reformulation and the technical modifications required to handle the problem's distinctive characteristics.

---

#### 3.1 Job-Centric Multi-Agent Formulation

**Paradigm Shift:**

Unlike traditional MARL applications where agents are physical entities (robots, vehicles), we redefine the agent concept to represent jobs (tasks to be scheduled). This reformulation naturally maps FJSP to a multi-agent problem:

- **Agent Definition**: Each job j ∈ J is an independent agent
- **Agent Action**: Select a machine-operator pair (m, o) for the current operation
- **Agent Goal**: Minimize its contribution to system makespan while ensuring valid routing

**Justification:**

This job-centric view offers several advantages:
1. Natural representation of decentralized decision-making in manufacturing
2. Scalability to varying problem sizes without architectural changes
3. Direct correspondence between agent policies and scheduling decisions
4. Intuitive credit assignment (each job's impact on system performance)

---

#### 3.2 Dynamic Agent Set Management

**Challenge:**

Standard MASA-QMIX assumes a fixed number of agents throughout an episode. In dynamic FJSP, jobs arrive stochastically during the scheduling horizon, creating a variable agent set N(t) that changes over time.

**Our Adaptation:**

We developed a flexible architecture that handles variable agent sets without requiring fixed-size inputs:

**Observation Processing:**
```
o^j_t = [operation_features^j, time_features^j, previous_action^j]
```

Each job's observation is independently processed by the Q-network (shared parameters across all jobs). The architecture uses:
- **RNN hidden states**: Maintained per job, tracking decision history
- **Variable-length concatenation**: Global state aggregates all active jobs
- **Dynamic indexing**: Jobs referenced by active indices, not fixed positions

**Technical Implementation:**

Instead of padding to a maximum size (which would introduce artifacts), we:
1. Process each job's observation independently through shared Q-network
2. Aggregate individual Q-values into variable-length tensors
3. Pass variable-size global state to mixing network
4. Use PyTorch's dynamic computation graphs for flexibility

**Mathematical Formulation:**

At time t, with N(t) active jobs:
```
Q_tot,t = f_mix([Q_1,t, ..., Q_N(t),t], s_t)
```

where N(t) varies across decision points, and the mixing network processes variable-length inputs through adaptive pooling operations.

---

#### 3.3 Action Masking for Manufacturing Constraints

**Motivation:**

Not all machine-operator pairs can process every operation. Manufacturing capabilities and skill requirements create hard constraints that must be enforced during action selection.

**Capability Matrix:**

We define a capability matrix C ∈ {0,1}^(O×P) where P is the set of machine-operator pairs:
- C[op, (m,o)] = 1 if machine-operator pair (m,o) can process operation type op
- C[op, (m,o)] = 0 otherwise

This captures both machine capabilities and operator skills simultaneously.

**Action Masking Procedure:**

For job j requiring operation op_j:

1. **Mask Generation**:
   ```
   mask^j = [C[op_j, (m_1,o_1)], ..., C[op_j, (m_k,o_k)]]
   ```
   where k is the number of available machine-operator pairs

2. **Q-Value Masking**:
   ```
   Q^j_masked[(m,o)] = Q^j[(m,o)]  if mask^j[(m,o)] = 1
                       -1e9         otherwise
   ```

3. **Action Selection**:
   ```
   u^j = argmax_(m,o) Q^j_masked[(m,o)]
   ```

**Integration with Exploration:**

During ε-greedy exploration, random action sampling respects the mask:
```
u^j = random choice from {(m,o) | mask^j[(m,o)] = 1}  with probability ε
      argmax_(m,o) Q^j_masked[(m,o)]                   with probability 1-ε
```

**Key Advantage:**

Action masking eliminates the need for penalty-based constraint handling, ensuring constraint satisfaction while maintaining natural Q-value distributions. This is crucial for FJSP where both machine capabilities and operator skills must be simultaneously considered.

**Future Extension:**

The framework can be extended to include distance constraints between workcenters (spatial layout considerations) by incorporating workcenter proximity information into the capability matrix or state representation.

---

#### 3.4 Event-Driven Decision Making

**Departure from Time-Stepped MARL:**

Standard MASA-QMIX operates in synchronized time steps where all agents make decisions simultaneously. Dynamic FJSP requires asynchronous, event-triggered decisions.

**Our Approach:**

We integrate MASA-QMIX with SimPy discrete-event simulation:

**Decision Triggers:**
1. **Job Arrival**: New job enters the system
   - Trigger: Lottery-based stochastic process
   - Decision: Select workcenter for first operation

2. **Operation Completion**: Job finishes processing at a workcenter
   - Trigger: Processing time expires
   - Decision: Select workcenter for next operation (or job completion)

**Episode Flow:**

```
Initialize SimPy environment
while active_jobs or pending_arrivals:
    event = env.next_event()
    
    if event == JobArrival:
        job = create_new_job()
        observation = get_job_observation(job)
        action = select_action(observation, mask)
        assign_to_workcenter(job, action)
        
    elif event == OperationComplete:
        job = event.job
        if job.has_remaining_operations():
            observation = get_job_observation(job)
            action = select_action(observation, mask)
            assign_to_workcenter(job, action)
        else:
            mark_job_complete(job)
```

**Advantages:**
- Natural integration with manufacturing simulation
- No artificial time discretization
- Decisions only when needed (computational efficiency)
- Realistic representation of manufacturing dynamics

---

#### 3.5 Variable Episode Length and Exploration

**Challenge:**

Episodes have variable length due to stochastic arrivals and varying processing times. Standard ε-decay based on episode count or time steps is inappropriate.

**Our Solution:**

We adapt ε-greedy exploration to use cumulative simulation time:

```
ε(T_sim) = ε_start - (ε_start - ε_end) × min(1, T_sim / T_anneal)
```

where:
- T_sim: Cumulative SimPy simulation time across all episodes
- T_anneal: Total simulation time for ε decay (e.g., 60% of training)

**Rationale:**

Using simulation time rather than episode count provides:
- Consistent exploration regardless of episode length variability
- Natural adaptation to problem scale (longer episodes = more exploration time)
- Smooth decay independent of stochastic arrival patterns

---

#### 3.6 State Representation Design

**Global State Composition:**

Our global state captures both job-level and system-level information:

```
s_t = [job_states_t, machine_states_t, operator_states_t, system_metrics_t]
```

**Job States** (for all N(t) active jobs):
- Operation sequence progress: [ops_done / total_ops]
- Current operation features: [processing_time, operation_type]
- Time in system: [arrival_time, current_time - arrival_time]
- Previous machine-operator pair assignment

**Machine and Operator States**:

*Machine States* (for all M machines):
- Queue length: [|queue_m|]
- Utilization: [busy_time_m / total_time]
- Current job being processed (if any)
- Expected completion time

*Operator States* (for all O operators):
- Current assignment (which machine, if any)
- Utilization: [busy_time_o / total_time]
- Skill level indicators

**System Metrics:**
- Current makespan: max(completion_times)
- Load balance score: std(workcenter_utilizations)
- Number of completed jobs
- Simulation time

**Local Observation** (for job j):
```
o^j = [
    remaining_operations_j / total_operations_j,
    current_operation_features_j,
    previous_machine_j (one-hot),
    previous_operator_j (one-hot),
    time_in_system_j (normalized),
    operation_sequence_j (embedded)
]
```

**Design Rationale:**

- **Job observations**: Enable individual routing decisions
- **Global state**: Facilitates coordination through mixing network
- **Normalization**: All features scaled to [0, 1] for stable learning
- **Temporal information**: Captures urgency and progress

---

#### 3.7 Reward Function Design

**Multi-Objective Formulation:**

```
r_t = α × r_makespan + β × r_loadbalance + γ × r_penalty
```

**Component 1: Makespan Minimization**
```
r_makespan = -(current_makespan - previous_makespan)
```
Dense reward signal penalizing increase in system completion time.

**Component 2: Load Balancing**
```
r_loadbalance = -(std(workcenter_utilization))
```
Encourages balanced resource utilization to avoid bottlenecks.

**Component 3: Constraint Penalties**
```
r_penalty = -1e9  if invalid action (should never occur with masking)
           -idle_time_penalty if workcenter idle while jobs waiting
```

**Hyperparameters:**
- α = 1.0 (primary objective weight)
- β = 0.1 (secondary objective weight)
- γ = 1.0 (hard constraint weight)

**Justification:**

Dense reward structure provides:
- Continuous learning signal (every decision impacts reward)
- Multi-objective optimization (makespan + load balance)
- Clear credit assignment (each action's immediate impact)

---

#### 3.8 Network Architecture Adaptations

**Q-Network Structure:**

Each agent uses a GRU-based Q-network that:
- Processes job observations o^j
- Maintains per-job hidden states h^j for temporal learning
- Outputs Q-values for each machine-operator pair |P|

**Key Feature:** Shared parameters across all jobs, with per-job hidden states

**Mixing Network Structure:**

```
Individual Q-values: [Q_1, ..., Q_N(t)] (variable N)
Global state: s_t (variable dimension)
  ↓
Hypernetwork_weights(s_t) → w_1, w_2 (non-negative via abs())
Hypernetwork_bias(s_t) → b_1, b_2
  ↓
Layer 1: f_1 = ELU(w_1 · [Q_1, ..., Q_N(t)] + b_1)
  ↓
Layer 2: Q_tot = w_2 · f_1 + b_2
```

**Monotonicity Enforcement:**
- Weights w_1, w_2 constrained to be non-negative (absolute value)
- Ensures ∂Q_tot/∂Q_i ≥ 0

**Variable Input Handling:**
- Adaptive pooling for variable-length Q-value lists
- Global state processed through fully connected layers
- No fixed-size padding required

---

## 🎤 PRESENTATION TALKING POINTS

### For Standard MASA-QMIX Slide:

"Before discussing our adaptation, let me briefly introduce MASA-QMIX. It's a value-based multi-agent RL algorithm where each agent learns its own Q-network for local decision-making. The key innovation is the mixing network, which combines individual Q-values into a total Q-value while maintaining a monotonicity constraint. This constraint ensures that greedy action selection by individual agents leads to coordinated team behavior."

"In typical MASA-QMIX applications, you have a fixed number of agents—say, robots or vehicles—that all act synchronously at each time step. The episode has a fixed length, and all actions are generally valid."

### For Job-Centric MASA-QMIX Slide:

"Now, let's see how we adapted this framework for dynamic flexible job shop scheduling."

"**First innovation**: We reformulated the problem by treating jobs as agents. Each job independently decides which workcenter to visit for its operations. This job-centric view naturally maps scheduling to multi-agent RL."

"**Second innovation**: Unlike standard QMIX with fixed agents, we handle dynamic agent sets. Jobs arrive stochastically during the episode, so the number of active agents N(t) changes over time. Our architecture processes variable-length inputs dynamically without requiring fixed-size representations."

"**Third innovation**: We implement action masking based on a capability matrix. Not all machine-operator pairs can process all operations—we must consider both machine capabilities and operator skills simultaneously. Action masking enforces these manufacturing constraints by masking invalid pairs before action selection, ensuring only feasible assignments."

"**Fourth innovation**: Decisions are event-driven rather than synchronized. When a job arrives or completes an operation, we trigger a decision. This integrates naturally with discrete-event simulation and avoids artificial time discretization."

### For Methodology Defense:

**Question: "Why not just pad to maximum agents?"**

**Answer:** "While padding is a common approach, it introduces several issues. Padding creates artificial 'dummy' agents that the network must learn to ignore, adding computational overhead and potentially confusing the learning process. Instead, our flexible architecture processes variable-length inputs directly through dynamic computation graphs. The mixing network uses adaptive operations that naturally handle varying numbers of agents, leading to cleaner learning signals and better computational efficiency."

**Question: "How does action masking affect learning?"**

**Answer:** "Action masking is crucial for two reasons. First, it enforces hard manufacturing constraints—specifically, the joint constraint that a machine-operator pair must have both the machine capability AND the operator skill to process an operation. This eliminates the need for penalty-based approaches that can destabilize learning. Second, by masking invalid pairs before Q-value comparison, we maintain natural Q-value distributions without extreme negative values. This leads to faster, more stable learning. The approach is particularly powerful because it handles the combinatorial complexity of pairing machines with operators efficiently."

**Question: "Why treat jobs as agents instead of machines?"**

**Answer:** "The job-centric formulation offers several advantages. First, it naturally represents the scheduling decision: each job needs to decide where to go next. Second, it handles dynamic arrivals elegantly—new jobs simply become new agents. Third, it scales automatically with problem size. A machine-centric view would struggle with variable numbers of jobs and would make the action space (which job to process) dynamically sized, which is more complex to handle."

---

## 📊 COMPARISON TABLE FOR THESIS

### Table 3.1: Comparison of Standard MASA-QMIX and Job-Centric Adaptation

| Aspect | Standard MASA-QMIX | Job-Centric MASA-QMIX |
|--------|-------------------|----------------------|
| **Agent Definition** | Physical entity (robot, vehicle) | Job (task to be scheduled) |
| **Agent Set** | Fixed N agents throughout episode | Dynamic N(t) agents (stochastic arrivals) |
| **Decision Timing** | Synchronous (all agents per time step) | Asynchronous (event-triggered) |
| **Episode Length** | Fixed time steps (e.g., T=100) | Variable (simulation-driven) |
| **Action Space** | Movement/control commands | Workcenter selection |
| **Action Validity** | All actions typically valid | Capability-constrained (masked) |
| **Observation** | Local sensor data | Job state (operations, time, history) |
| **Global State** | Agent positions and states | Jobs + workcenters + system metrics |
| **State Dimension** | Fixed | Variable (adapts to N(t)) |
| **Reward** | Sparse team reward | Dense multi-objective reward |
| **Exploration** | ε-decay by episode/step | ε-decay by cumulative simulation time |
| **Integration** | Standalone RL environment | Discrete-event simulation (SimPy) |

---

## 🎯 KEY MESSAGES TO EMPHASIZE

### For Thesis Committee:

1. **Theoretical Contribution**: "We extend MASA-QMIX from fixed, synchronized agent systems to dynamic, event-driven multi-agent problems, maintaining the theoretical guarantees while handling variable agent sets."

2. **Technical Innovation**: "Our architecture processes variable-length inputs through flexible network design and adaptive operations, eliminating the need for padding-based workarounds."

3. **Domain Integration**: "We bridge multi-agent RL with discrete-event simulation, enabling realistic modeling of manufacturing systems while leveraging modern deep RL techniques."

4. **Practical Impact**: "The job-centric formulation scales naturally to different problem sizes and handles the inherent stochasticity of real-world manufacturing environments."

### For Industrial Audience:

1. "Jobs make their own routing decisions based on system state"
2. "Handles unpredictable job arrivals automatically"
3. "Enforces manufacturing constraints (which machines can do which operations)"
4. "Optimizes both completion time and resource utilization"
5. "Learns from experience, improving over time"

---

## 📚 RELATED WORK POSITIONING

**Compared to Traditional FJSP:**
- "Unlike optimization-based approaches that require complete problem knowledge upfront, our framework handles stochastic arrivals online"
- "Learns adaptive policies that generalize across different arrival patterns"

**Compared to Other MARL for Scheduling:**
- "Most MARL scheduling works assume fixed agent sets; we explicitly handle dynamic arrivals"
- "Action masking enforces constraints without penalty-based learning"
- "Event-driven decisions avoid artificial time discretization"

**Compared to Standard QMIX:**
- "We extend QMIX's centralized training, decentralized execution paradigm to variable agent sets"
- "Maintain monotonicity guarantees while processing variable-length inputs"

---

## ✅ FINAL CHECKLIST

### Thesis Writing:
- [ ] Explain standard MASA-QMIX in Theoretical Background
- [ ] Justify job-centric paradigm shift
- [ ] Detail each adaptation with technical specifications
- [ ] Provide mathematical formulations
- [ ] Compare with related work
- [ ] Avoid mentioning padding (focus on "flexible architecture")

### Presentation:
- [ ] One slide for standard MASA-QMIX (with diagram)
- [ ] One slide for your adaptation (with key innovations highlighted)
- [ ] Clear visual distinction between the two
- [ ] Prepare talking points for each innovation
- [ ] Practice defense questions

### Key Phrases to Use:
- ✅ "Flexible architecture supporting variable agent sets"
- ✅ "Dynamic input processing through adaptive operations"
- ✅ "Event-driven decision making integrated with discrete-event simulation"
- ✅ "Capability-based action masking for constraint enforcement"
- ✅ "Job-centric multi-agent formulation"

### Phrases to Avoid:
- ❌ "We pad observations to maximum size"
- ❌ "Dummy agents for padding"
- ❌ "Fixed-size input with masking"

---

## 📐 SUGGESTED VISUAL IMPROVEMENTS FOR YOUR DIAGRAM

### Current Diagram Issues:
- Not enough visual distinction from standard QMIX
- Missing: stochastic arrival visualization
- Missing: action masking flow
- Missing: event-driven decision triggers

### Suggested Additions:

**For Job-Centric Diagram:**

1. **Top Section**: Show job arrival process
   ```
   t=0: Job1 arrives → Select WC
   t=5: Job2 arrives → Select WC
   t=8: Job1 completes Op1 → Select WC for Op2
   ...
   ```

2. **Middle Section**: Your current architecture with additions:
   - Add "Action Mask" box between Q-network output and action selection
   - Show "Capability Matrix" feeding into mask generation
   - Highlight "Variable N" with visual indicator (e.g., dashed boxes for potential future agents)

3. **Bottom Section**: Show episode dynamics
   ```
   Episode timeline: [----Job1----][--Job2--][---Job3---]
                         ↑            ↑          ↑
                      Decision    Decision   Decision
   ```

4. **Color Coding**:
   - Standard components (from QMIX): Blue
   - Your innovations: Orange/Red
   - Constraints/masking: Green

5. **Annotations**:
   - "⚡ NEW" labels on innovations
   - "Variable N(t)" instead of just "N agents"
   - "Event-triggered" near decision points
   - "Masked Q-values" near action selection

---

This comprehensive guide should help you clearly communicate your contributions without getting into implementation details like padding!
