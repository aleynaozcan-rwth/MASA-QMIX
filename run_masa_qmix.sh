#!/bin/bash
#SBATCH --job-name=masa_qmix
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=c23g
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# --- Environment setup ---
module purge
module load GCCcore/12.2.0
module load Python/3.10.8

# Activate GPU Python environment
source ~/masa-qmix-env-gpu/bin/activate

# Move to project directory
cd ~/MASA-QMIX

# --- Cleanup old outputs (start fresh for each run) ---
rm -f ./my_data_and_graph/historydata/*.txt
rm -f ./my_data_and_graph/historydata/*.csv
rm -f ./my_data_and_graph/historydata/*.png
rm -f ./my_data_and_graph/pickles/*.pk

# --- Job started ---
echo "=== JOB $SLURM_JOB_ID STARTED at $(date) ==="

# --- Run training ---
echo "=== Training started at $(date) ==="
python main.py | tee training_log_${SLURM_JOB_ID}.txt
echo "=== Training finished at $(date) ==="

# --- Run analysis after training ---
echo "=== Running analyse_rewards.py ==="
python analyse_rewards.py | tee analyse_log_${SLURM_JOB_ID}.txt
echo "=== Analyse finished at $(date) ==="

# --- Job finished ---
echo "=== JOB $SLURM_JOB_ID FINISHED at $(date) ==="
echo "Check generated plots in: ./my_data_and_graph/historydata/"