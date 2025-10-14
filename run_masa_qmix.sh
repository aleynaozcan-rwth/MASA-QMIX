#!/bin/bash
#SBATCH --job-name=masa_qmix_7b2
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=03:00:00
#SBATCH --partition=c23g
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

# === STEP 7B.2 EXECUTION ===
echo "[RUN] Launching run_step7b.py ..."
python run_step7b.py | tee sanity_${SLURM_JOB_ID}.txt

# === POST-TRAINING ANALYSIS ===
echo "[ANALYZE] Running analyse_rewards.py..."
python analyse_rewards.py | tee analyse_log_${SLURM_JOB_ID}.txt
echo "[ANALYZE] Finished at $(date)"
echo "------------------------------------------------------------"

# === AUTO LEARNING SUMMARY ===
echo "[SUMMARY] Running automatic learning analysis..."
python analyze_learning_progress.py | tee summary_${SLURM_JOB_ID}.txt
echo "[SUMMARY] Learning analysis complete."
echo "------------------------------------------------------------"

echo "=========================================================="
echo "JOB FINISHED at $(date)"
echo "=========================================================="
