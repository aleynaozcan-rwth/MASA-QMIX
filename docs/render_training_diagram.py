#!/usr/bin/env python3
"""
Render QMIX Training Diagram
Usage: python render_training_diagram.py
Output: masa_qmix_training.png
"""

import subprocess
import os

TRAINING_DIAGRAM = """
graph TB
    subgraph EPISODE["📝 Episode Experience"]
        TRANS["<b>Transitions</b><br/>Each step stores:<br/>(s_t, obs_t, a_t, r_t, s_t+1, obs_t+1, done)<br/>━━━━━━━<br/>s: global state (10D)<br/>obs: local observations (8D per agent)<br/>a: joint actions<br/>r: global reward"]
    end
    
    subgraph BUFFER["💾 Replay Buffer"]
        STORE["<b>Experience Storage</b><br/>Capacity: 5000 episodes<br/>Current size: N episodes<br/>━━━━━━━<br/>Sample batch_size=32 episodes<br/>for training"]
    end
    
    subgraph SAMPLE["🎲 Batch Sampling"]
        BATCH["<b>Random Sample</b><br/>Select 32 episodes<br/>━━━━━━━<br/>Each episode contains:<br/>T timesteps of transitions<br/>Full trajectory data"]
    end
    
    subgraph QNET["🤖 Q-Networks"]
        CURRENT["<b>Current Network</b><br/>θ (trainable)<br/>━━━━━━━<br/>For each agent i:<br/>Q_i(obs_i, a_i | θ)"]
        TARGET["<b>Target Network</b><br/>θ' (frozen)<br/>━━━━━━━<br/>For each agent i:<br/>Q_i'(obs_i', a_i' | θ')<br/>Updated every 200 steps"]
    end
    
    subgraph MIXER["🎯 QMIX Mixer"]
        HYPER["<b>Hypernetwork</b><br/>Input: global state s<br/>Output: mixing weights w_i(s)<br/>━━━━━━━<br/>Ensures monotonicity:<br/>∂Q_tot/∂Q_i ≥ 0"]
        QTOT["<b>Total Q-Value</b><br/>Q_tot(s,a) = f([Q_1,...,Q_n], s)<br/>━━━━━━━<br/>Q_tot = Σ w_i(s) · Q_i<br/>Non-linear mixing"]
    end
    
    subgraph LOSS["📉 Loss Computation"]
        TD["<b>TD Error</b><br/>y = r + γ · max_a' Q_tot'(s', a')<br/>δ = y - Q_tot(s, a)<br/>━━━━━━━<br/>Loss = δ²<br/>Mean over batch"]
    end
    
    subgraph OPTIM["⚙️ Optimization"]
        GRAD["<b>Backpropagation</b><br/>Compute ∇_θ Loss<br/>Gradient clipping: ||∇|| ≤ 10<br/>━━━━━━━<br/>Adam optimizer<br/>lr = 1e-4"]
        UPDATE["<b>Parameter Update</b><br/>θ ← θ - α·∇_θ Loss<br/>━━━━━━━<br/>Every training_steps=15 steps"]
    end
    
    subgraph TARGET_UPDATE["🔄 Target Update"]
        COPY["<b>Hard Update</b><br/>θ' ← θ<br/>━━━━━━━<br/>Every target_update_cycle=200 steps<br/>Stabilizes learning"]
    end
    
    %% Flow
    TRANS -->|"Store"| STORE
    STORE -->|"When buffer > batch_size"| BATCH
    BATCH -->|"Sampled episodes"| CURRENT
    BATCH -->|"Next states"| TARGET
    CURRENT -->|"Q_i(s,a)"| HYPER
    TARGET -->|"Q_i'(s',a')"| TD
    HYPER -->|"w_i(s)"| QTOT
    CURRENT -->|"Individual Q_i"| QTOT
    QTOT -->|"Q_tot(s,a)"| TD
    TD -->|"Loss = δ²"| GRAD
    GRAD -->|"∇_θ Loss"| UPDATE
    UPDATE -->|"Updated θ"| CURRENT
    UPDATE -.->|"Every 200 steps"| COPY
    COPY -.->|"θ' ← θ"| TARGET
    
    %% Styling
    classDef episode fill:#E0F2FE,stroke:#0284C7,stroke-width:3px
    classDef buffer fill:#E0E7FF,stroke:#6366F1,stroke-width:3px
    classDef sample fill:#DBEAFE,stroke:#3B82F6,stroke-width:2px
    classDef qnet fill:#FEF3C7,stroke:#F59E0B,stroke-width:3px
    classDef mixer fill:#FFE4E6,stroke:#F43F5E,stroke-width:3px
    classDef loss fill:#FED7AA,stroke:#EA580C,stroke-width:3px
    classDef optim fill:#BBF7D0,stroke:#16A34A,stroke-width:3px
    classDef target fill:#E9D5FF,stroke:#A855F7,stroke-width:2px
    
    class TRANS episode
    class STORE buffer
    class BATCH sample
    class CURRENT,TARGET qnet
    class HYPER,QTOT mixer
    class TD loss
    class GRAD,UPDATE optim
    class COPY target
"""

def render_diagram():
    """Render Training diagram to PNG"""
    
    # Write mermaid code to file
    mmd_file = "masa_qmix_training.mmd"
    with open(mmd_file, "w") as f:
        f.write(TRAINING_DIAGRAM)
    
    print(f"✓ Training diagram code written to {mmd_file}")
    
    # Check if mermaid-cli is installed
    try:
        subprocess.run(["mmdc", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n❌ mermaid-cli not installed!")
        print("\nInstall with:")
        print("  npm install -g @mermaid-js/mermaid-cli")
        print("\nOr use online: https://mermaid.live")
        return
    
    # Render to PNG (vertical layout for training flow)
    output_file = "masa_qmix_training.png"
    cmd = [
        "mmdc",
        "-i", mmd_file,
        "-o", output_file,
        "-w", "2400",
        "-H", "2800",
        "-b", "white"
    ]
    
    print(f"\n⏳ Rendering training diagram...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Training diagram saved to: {output_file}")
        print(f"   Size: {os.path.getsize(output_file)} bytes")
        
        # Also render SVG
        svg_file = "masa_qmix_training.svg"
        cmd[4] = svg_file
        subprocess.run(cmd, capture_output=True)
        print(f"✅ SVG version: {svg_file}")
    else:
        print(f"❌ Error rendering diagram:")
        print(result.stderr)

if __name__ == "__main__":
    render_diagram()
