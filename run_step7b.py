#!/usr/bin/env python3
"""
run_step7b.py
Launch Step 7B.2 (Replay-Aware QMIX Training) in a cluster-safe way.
"""

import os
import sys
import subprocess

# ============================================================
# === Dynamic Project Path Setup ==============================
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"[INFO] Using PROJECT_ROOT={PROJECT_ROOT}")
print(f"[INFO] sys.path[0]={sys.path[0]}")

# ============================================================
# === Step 7B.2 Launcher =====================================
# ============================================================
print("\n🚀 Running Step 7B.2 (Replay-Aware QMIX) locally/cluster...")

# Optional: clear old data
try:
    os.makedirs("./my_data_and_graph/historydata", exist_ok=True)
    os.makedirs("./my_data_and_graph/pickles", exist_ok=True)
except Exception as e:
    print(f"[WARN] Could not create output dirs: {e}")

# Execute the main training file with Step 7B mode
cmd = [
    sys.executable,
    "main.py",
    "--mode", "7b",
    "--alg", "qmix",
    "--seed", "123"
]
print(f"[INFO] Launch command: {' '.join(cmd)}\n")
subprocess.run(cmd, check=True)

print("\n✅ Step 7B.2 run_step7b.py finished successfully.")
