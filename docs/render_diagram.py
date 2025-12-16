#!/usr/bin/env python3
"""
Render Episode Scheduling Diagram
Usage: python render_diagram.py
Output: masa_qmix_episode.png
"""

import subprocess
import os

EPISODE_SCHEDULING = """
graph LR
    subgraph ARR["📦 Job Arrivals"]
        direction TB
        J1["J1: OpA→OpB→OpC"]
        J2["J2: OpD→OpA→OpE→OpB"]
        J3["J3: OpC→OpB"]
        JDOTS["⋮"]
        Jn["Jn: OpF→OpA→OpE"]
        J1 --> J2 --> J3 -.-> JDOTS -.-> Jn
    end
    
    subgraph SIMPY["⚙️ SimPy Environment - Discrete Event Simulation"]
        direction TB
        
        subgraph FLOOR["🏭 Job Shop Floor State"]
            direction LR
            MACHINES["<b>Machines</b><br/>M1:BUSY J2-OpB+O1<br/>M2:BUSY J5-OpA+O3<br/>M3:IDLE | M4:BUSY J1-OpC+O2 | M5:IDLE"]
            OPERATORS["<b>Operators</b><br/>O1:BUSY@M1 | O2:BUSY@M4<br/>O3:BUSY@M2 | O4:IDLE"]
            QUEUE["<b>Queue</b><br/>J3:5s→OpC<br/>J4:2s→OpA<br/>J7:8s→OpE"]
            AGENTS["<b>Active Agents</b><br/>J1→A1 | J2→A2 | J3→A3<br/>J4→A4 | J5→A5 | J7→A7"]
        end
        
        OBS_BUILD["<b>📊 Obs Builder</b><br/>Per-agent: obs_i dim n_obs<br/>Global: state dim n_state | Norm [0,1]"]
        REWARD_CALC["<b>💰 Reward - Every Step</b><br/>R_global = w1×Completed - w2×Wait + w4×Throughput∆ + w5×LoadBalance<br/>R_total = R_global / reward_scale"]
        EXECUTOR["<b>▶️ Executor</b><br/>Execute joint actions in parallel<br/>Assign to mc+op | Advance time | Update floor"]
        COMPLETED["<b>✅ Completion</b><br/>When done: Remove agent | Update metrics"]
        
        FLOOR ~~~ OBS_BUILD ~~~ REWARD_CALC ~~~ COMPLETED ~~~ EXECUTOR
        FLOOR --> OBS_BUILD --> REWARD_CALC --> COMPLETED
        EXECUTOR --> FLOOR
    end
    
    EXIT["🎉 Completed Jobs<br/>.<br/>J0, J8, J12, ...<br/>Exit system"]
    
    subgraph AGENT["🤖 Multi-Agent Network - Parallel Decision Making"]
        direction TB
        
        RNN1["<b>Agent-1 RNN</b><br/>obs_1 → GRU rnn_hidden<br/>→ Q_1 o,a"]
        RNNDOTS["⋮<br/>more agents"]
        RNNn["<b>Agent-n RNN</b><br/>obs_n → GRU rnn_hidden<br/>→ Q_n o,a"]
        
        RNN1 ~~~ RNNDOTS ~~~ RNNn
        
        POLICY["<b>ε-greedy Policy</b><br/>For EACH agent i:<br/>.<br/>if rand < ε: random valid action<br/>else: argmax_a Q_i o,a<br/>.<br/>ε decays from epsilon_start to epsilon_finish<br/>.<br/>Output: Joint actions [a_1, a_2, ..., a_n]"]
    end
    
    subgraph BUFFER["💾 Experience Storage"]
        REPLAY["<b>Replay Buffer</b><br/>Store transition tuple:<br/>.<br/>state: s_t dimension n_state<br/>obs: [obs_1, ..., obs_n] dimension n_agents × n_obs<br/>actions: [a_1, ..., a_n]<br/>reward: r_t global scalar<br/>next_state: s_t+1<br/>next_obs: [obs_1', ..., obs_n']<br/>done: episode_end flag<br/>.<br/>Size: buffer_size episodes<br/>Currently: N episodes stored"]
        
        TRAIN["<b>Training Module</b><br/>if learn=True:<br/>.<br/>Sample: batch_size episodes<br/>→ QMIX Mixer<br/>→ Compute TD loss<br/>→ Backprop gradient<br/>.<br/>Every train_steps steps"]
    end
    
    %% Main Flow
    ARR -->|"Arrive"| FLOOR
    OBS_BUILD -->|"obs batch"| RNN1
    OBS_BUILD -->|"obs batch"| RNNn
    RNN1 --> POLICY
    RNNn --> POLICY
    POLICY -->|"joint actions"| EXECUTOR
    REWARD_CALC -->|"store tuple"| REPLAY
    COMPLETED -->|"exit"| EXIT
    REPLAY -.->|"if learn=True"| TRAIN
    TRAIN -.->|"∇θ"| RNN1
    TRAIN -.->|"∇θ"| RNNn
    
    %% Loop
    FLOOR -.->|"next decision"| OBS_BUILD
    
    %% Styling
    classDef arr fill:#E0F2FE,stroke:#0284C7,stroke-width:3px
    classDef simpy fill:#FEF9C3,stroke:#CA8A04,stroke-width:4px
    classDef floor fill:#FFF4E6,stroke:#F59E0B,stroke-width:2px
    classDef agent fill:#FEF3C7,stroke:#F59E0B,stroke-width:3px
    classDef buf fill:#E0E7FF,stroke:#6366F1,stroke-width:2px
    classDef exit fill:#D1FAE5,stroke:#059669,stroke-width:2px
    
    class J1,J2,J3,JDOTS,Jn arr
    class MACHINES,OPERATORS,QUEUE,AGENTS,OBS_BUILD,REWARD_CALC,EXECUTOR,COMPLETED floor
    class RNN1,RNNDOTS,RNNn,POLICY agent
    class REPLAY,TRAIN buf
    class EXIT exit
"""

MERMAID_CODE = EPISODE_SCHEDULING

def render_diagram():
    """Render Episode Scheduling diagram to PNG"""
    
    # Write mermaid code to file
    mmd_file = "masa_qmix_episode.mmd"
    with open(mmd_file, "w") as f:
        f.write(MERMAID_CODE)
    
    print(f"✓ Episode scheduling diagram code written to {mmd_file}")
    
    # Check if mermaid-cli is installed
    try:
        subprocess.run(["mmdc", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n❌ mermaid-cli not installed!")
        print("\nInstall with:")
        print("  npm install -g @mermaid-js/mermaid-cli")
        print("\nOr use online: https://mermaid.live")
        return
    
    # Render to PNG (PowerPoint optimized: compact layout)
    output_file = "masa_qmix_episode.png"
    cmd = [
        "mmdc",
        "-i", mmd_file,
        "-o", output_file,
        "-c", "docs/mermaid_config.json",  # Config for compact spacing
        "-w", "1920",  # Width (standard PowerPoint width)
        "-H", "1080",  # Height (16:9 ratio for slides)
        "-b", "white"   # Background
    ]
    
    print(f"\n⏳ Rendering episode diagram...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Episode diagram saved to: {output_file}")
        print(f"   Size: {os.path.getsize(output_file)} bytes")
        
        # Also render SVG
        svg_file = "masa_qmix_episode.svg"
        cmd[4] = svg_file
        subprocess.run(cmd, capture_output=True)
        print(f"✅ SVG version: {svg_file}")
    else:
        print(f"❌ Error rendering diagram:")
        print(result.stderr)

if __name__ == "__main__":
    render_diagram()
