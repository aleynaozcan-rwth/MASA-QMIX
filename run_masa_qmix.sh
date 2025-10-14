#!/bin/bash
#SBATCH --job-name=masa_qmix_7b2
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=02:00:00
#SBATCH --partition=c23g
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ============================================================
# === Environment Setup =======================================
# ============================================================
module purge
module load GCCcore/12.2.0
module load Python/3.10.8

# Activate GPU Python environment
source ~/masa-qmix-env-gpu/bin/activate

# Move to project directory
cd ~/MASA-QMIX

# ============================================================
# === Cleanup old outputs =====================================
# ============================================================
echo "[CLEANUP] Removing old training outputs..."
rm -f ./my_data_and_graph/historydata/*.txt
rm -f ./my_data_and_graph/historydata/*.csv
rm -f ./my_data_and_graph/historydata/*.png
rm -f ./my_data_and_graph/pickles/*.pk

# ============================================================
# === Job Metadata ============================================
# ============================================================
echo "=== JOB $SLURM_JOB_ID STARTED at $(date) ==="
echo "Python: $(which python)"
echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git log -1 --pretty=format:'%h - %s (%ci)')"
echo "------------------------------------------------------------"

# ============================================================
# === Step 7B.2 — Replay-Aware QMIX Training ==================
# ============================================================
echo "[TRAIN] Starting Step 7B.2 replay-aware QMIX training..."
python main.py --mode 7b --alg qmix --seed 123 | tee training_log_${SLURM_JOB_ID}.txt
echo "[TRAIN] Finished at $(date)"
echo "------------------------------------------------------------"

# ============================================================
# === Post-Training Analysis ==================================
# ============================================================
echo "[ANALYZE] Running analyse_rewards.py..."
python analyse_rewards.py | tee analyse_log_${SLURM_JOB_ID}.txt
echo "[ANALYZE] Finished at $(date)"
echo "------------------------------------------------------------"

# ============================================================
# === Job Finished ============================================
# ============================================================
echo "=== JOB $SLURM_JOB_ID FINISHED at $(date) ==="
echo "Check generated plots and training data in:"
echo "→ ./my_data_and_graph/historydata/"
