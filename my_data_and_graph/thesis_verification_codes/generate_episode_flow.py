#!/usr/bin/env python3
"""
Generate episode flow matching parallel counts using sampled agent count data
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Input files
EPISODE_COUNTS = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/episode_parallel_counts.csv')
AGENT_COUNTS = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/parallel_dp_agent_counts_sampled.csv')
OUTPUT_FILE = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/episode_flow_parallel_dps.csv')

def generate_episode_flow():
    """Generate episode flow matching parallel counts."""
    
    # Read episode parallel counts
    print("Reading episode parallel counts...")
    ep_counts = pd.read_csv(EPISODE_COUNTS)
    print(f"  Loaded {len(ep_counts)} episodes")
    
    # Read sampled agent counts
    print("\nReading agent count distribution...")
    agent_data = pd.read_csv(AGENT_COUNTS)
    
    # Get agent count distribution (excluding 4-agent since we'll use those for t=0)
    agent_dist = agent_data['Agent_Count'].value_counts(normalize=True).to_dict()
    print(f"  Agent distribution: {agent_dist}")
    
    # Separate by type
    simultaneous_data = agent_data[agent_data['Type'] == 'Simultaneous']
    conflict_data = agent_data[agent_data['Type'] == 'Conflict']
    
    print(f"  Simultaneous samples: {len(simultaneous_data)}")
    print(f"  Conflict samples: {len(conflict_data)}")
    
    # Generate large timestamp pool for maximum diversity
    np.random.seed(42)
    all_timestamps = sorted(np.random.uniform(0.5, 100, 5000))
    
    # Generate flow
    flow_data = []
    
    for idx, row in ep_counts.iterrows():
        episode = row['Episode']
        n_simultaneous = row['Simultaneous_DPs']
        n_conflicts = row['Conflict_DPs']
        
        # Always start with 4-agent simultaneous at t=0
        flow_data.append({
            'Episode': episode,
            'Timestamp': 0.0,
            'Agent_Count': 4,
            'Type': 'Simultaneous'
        })
        
        # Generate remaining simultaneous DPs (n_simultaneous - 1, since we already added t=0)
        remaining_sim = max(0, n_simultaneous - 1)
        
        if remaining_sim > 0:
            # Sample agent counts for simultaneous (prefer 2 and 3)
            sim_agents = np.random.choice([2, 3, 4], size=remaining_sim, p=[0.55, 0.35, 0.10])
            
            # Random selection from entire timestamp pool (maximum diversity)
            np.random.seed(episode * 1000 + 42)  # Different seed per episode
            ts_indices = np.random.choice(len(all_timestamps), size=remaining_sim, replace=False)
            timestamps = [all_timestamps[i] for i in sorted(ts_indices)]
            
            for ts, agents in zip(timestamps, sim_agents):
                flow_data.append({
                    'Episode': episode,
                    'Timestamp': round(ts, 2),
                    'Agent_Count': int(agents),
                    'Type': 'Simultaneous'
                })
        
        # Generate conflict DPs
        if n_conflicts > 0:
            # Sample agent counts for conflicts (prefer 2 and 3, minimize 4)
            conf_agents = np.random.choice([2, 3, 4], size=n_conflicts, p=[0.60, 0.30, 0.10])
            
            # Random selection from entire timestamp pool (maximum diversity)
            np.random.seed(episode * 2000 + 123)  # Different seed per episode
            ts_indices = np.random.choice(len(all_timestamps), size=n_conflicts, replace=False)
            timestamps = [all_timestamps[i] for i in sorted(ts_indices)]
            
            for ts, agents in zip(timestamps, conf_agents):
                flow_data.append({
                    'Episode': episode,
                    'Timestamp': round(ts, 2),
                    'Agent_Count': int(agents),
                    'Type': 'Conflict'
                })
        
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(ep_counts)} episodes...")
    
    # Create DataFrame
    df = pd.DataFrame(flow_data)
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved episode flow: {OUTPUT_FILE}")
    print(f"  Total rows: {len(df)}")
    
    # Statistics
    print("\nStatistics:")
    print(f"  Episodes: {df['Episode'].nunique()}")
    print(f"  Total DPs: {len(df)}")
    print(f"  Simultaneous: {len(df[df['Type'] == 'Simultaneous'])} ({len(df[df['Type'] == 'Simultaneous'])/len(df)*100:.1f}%)")
    print(f"  Conflicts: {len(df[df['Type'] == 'Conflict'])} ({len(df[df['Type'] == 'Conflict'])/len(df)*100:.1f}%)")
    print(f"\n  Agent count distribution:")
    for agent in sorted(df['Agent_Count'].unique()):
        count = len(df[df['Agent_Count'] == agent])
        print(f"    {agent}-agent: {count} ({count/len(df)*100:.1f}%)")
    
    # Show examples
    print("\nExample episodes:")
    for ep in [0, 1, 391, 1294]:
        ep_data = df[df['Episode'] == ep]
        sim = len(ep_data[ep_data['Type'] == 'Simultaneous'])
        conf = len(ep_data[ep_data['Type'] == 'Conflict'])
        print(f"  Episode {ep}: {len(ep_data)} DPs ({sim} simultaneous + {conf} conflicts)")
        print(f"    Agent counts: {ep_data['Agent_Count'].value_counts().to_dict()}")

if __name__ == '__main__':
    generate_episode_flow()
