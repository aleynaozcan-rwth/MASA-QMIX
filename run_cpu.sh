#!/bin/bash
#SBATCH --job-name=masa_cpu
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=12:00:00
#SBATCH --partition=c23m
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

# === MODULES ===
module purge
module load GCCcore/12.2.0
module load Python/3.10.8

# === ACTIVATE VENV ===
source ~/masa-qmix-env-gpu/bin/activate

# === MOVE TO PROJECT ROOT ===
cd ~/MASA-QMIX

# === PYTHONPATH SETUP ===
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# === JOB METADATA ===
echo "=========================================================="
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "START TIME: $(date)"
echo "PYTHON: $(which python)"
echo "GIT BRANCH: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a')"
echo "=========================================================="

# === CLEANUP OLD RUNS ===
echo "[CLEANUP] Removing old logs, checkpoints, and outputs..."
rm -f ./my_data_and_graph/historydata/*.txt 2>/dev/null || true
rm -f ./my_data_and_graph/historydata/*.csv 2>/dev/null || true
rm -f ./my_data_and_graph/historydata/*.png 2>/dev/null || true
rm -f ./my_data_and_graph/pickles/*.pk 2>/dev/null || true
rm -rf ./result/qmix/masa_schedule/*/ 2>/dev/null || true
echo "[CLEANUP] Checkpoint directory cleared (fresh training)"

# === ADAPTIVE FIXES TRAINING ===
echo "[RUN] Launching main.py with adaptive fixes..."
echo "[INFO] Expected improvements:"
echo "  - Epsilon: stable decay to 0.05 by episode 240"
echo "  - Loss: < 10^6 within 50 episodes"
echo "  - LoadBalanceScore: active (non-zero values)"
echo "------------------------------------------------------------"
python main.py | tee training_${SLURM_JOB_ID}.txt
echo "[RUN] Training finished at $(date)"
echo "------------------------------------------------------------"

echo "=========================================================="
echo "JOB FINISHED at $(date)"
echo "=========================================================="
