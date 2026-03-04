#!/bin/bash
#SBATCH --job-name=masa_qmix_7b2
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=08:00:00
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
echo "[CLEANUP] Removing old logs and outputs..."
rm -f ./my_data_and_graph/historydata/*.{txt,csv,png} 2>/dev/null || true
rm -f ./my_data_and_graph/pickles/*.pk 2>/dev/null || true


# === LAUNCH TRAINING ===
echo "[RUN] Launching main.py with GPU/CUDA support..."
echo "[INFO] CUDA enabled: true (if available)"
python main.py --cuda | tee training_${SLURM_JOB_ID}.txt
echo "[RUN] Training finished at $(date)"

echo "=========================================================="
echo "JOB FINISHED at $(date)"
echo "=========================================================="
#--------------------------------------------------------------