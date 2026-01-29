#!/usr/bin/env python3
"""
Sample 4072 random DPs from the CSV with specific distribution:
- Minimize 4-agent DPs
- Mostly 2-agent DPs
- Some 3-agent DPs
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility but with high randomness
np.random.seed(42)

INPUT_CSV = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/parallel_dp_agent_counts.csv')
OUTPUT_CSV = Path('/home/cc253232/MASA-QMIX/my_data_and_graph/historydata/plots/parallel_dp_agent_counts_sampled.csv')

print("Reading CSV...")
df = pd.read_csv(INPUT_CSV)

print(f"\nOriginal data: {len(df)} rows")
print(f"Distribution:")
for agent_count in sorted(df['Agent_Count'].unique()):
    count = len(df[df['Agent_Count'] == agent_count])
    print(f"  {agent_count} agents: {count} rows ({100*count/len(df):.1f}%)")

# Separate by agent count
df_2 = df[df['Agent_Count'] == 2].copy()
df_3 = df[df['Agent_Count'] == 3].copy()
df_4 = df[df['Agent_Count'] == 4].copy()

# Target: 4072 rows
# Distribution: Mostly 2-agent, some 3-agent, minimize 4-agent
target_total = 4072

# Check available counts
avail_2 = len(df_2)
avail_3 = len(df_3)
avail_4 = len(df_4)

print(f"\nAvailable counts:")
print(f"  2 agents: {avail_2}")
print(f"  3 agents: {avail_3}")
print(f"  4 agents: {avail_4}")

# Target distribution (adjusted to available data)
# Use ALL available 2-agent to maximize sample size
target_3 = avail_3   # Use all 3-agent (1339)
target_2 = avail_2   # Use all 2-agent (2153)
target_4 = target_total - target_3 - target_2  # Fill remaining with 4-agent

# If we need more than available 4-agent, adjust
if target_4 > avail_4:
    target_4 = avail_4
    print(f"\n⚠ Warning: Cannot reach {target_total} rows with available data")
    print(f"Maximum possible: {target_2 + target_3 + target_4} rows")
elif target_4 < 0:
    # We have too much data, reduce 2-agent
    target_4 = 200  # Minimize 4-agent
    target_2 = target_total - target_3 - target_4

print(f"\nTarget sample: {target_total} rows")
print(f"  2 agents: {target_2} rows")
print(f"  3 agents: {target_3} rows")
print(f"  4 agents: {target_4} rows")

# Random sampling with high shuffle
sampled_2 = df_2.sample(n=target_2, random_state=np.random.randint(0, 100000))
sampled_3 = df_3.sample(n=target_3, random_state=np.random.randint(0, 100000))
sampled_4 = df_4.sample(n=target_4, random_state=np.random.randint(0, 100000))

# Combine and shuffle heavily
sampled_df = pd.concat([sampled_2, sampled_3, sampled_4], ignore_index=True)

# Shuffle multiple times for high randomness
for _ in range(5):
    sampled_df = sampled_df.sample(frac=1, random_state=np.random.randint(0, 100000)).reset_index(drop=True)

# Sort by DP_Index to maintain chronological order
sampled_df = sampled_df.sort_values('DP_Index').reset_index(drop=True)

# Save
sampled_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✓ Saved sampled CSV: {OUTPUT_CSV}")
print(f"  Total rows: {len(sampled_df)}")
print(f"\nFinal distribution:")
for agent_count in sorted(sampled_df['Agent_Count'].unique()):
    count = len(sampled_df[sampled_df['Agent_Count'] == agent_count])
    print(f"  {agent_count} agents: {count} rows ({100*count/len(sampled_df):.1f}%)")

# Show some statistics
print(f"\nDP Index range: {sampled_df['DP_Index'].min()} to {sampled_df['DP_Index'].max()}")
print(f"Episode range: {sampled_df['Episode'].min()} to {sampled_df['Episode'].max()}")
print(f"Type distribution:")
print(f"  Simultaneous: {len(sampled_df[sampled_df['Type'] == 'simultaneous'])} ({100*len(sampled_df[sampled_df['Type'] == 'simultaneous'])/len(sampled_df):.1f}%)")
print(f"  Conflict: {len(sampled_df[sampled_df['Type'] == 'conflict'])} ({100*len(sampled_df[sampled_df['Type'] == 'conflict'])/len(sampled_df):.1f}%)")
