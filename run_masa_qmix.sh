#!/bin/bash
#SBATCH --job-name=masa_qmix         # Job name
#SBATCH --output=output_%j.txt       # Standard output (%j will be replaced with job ID)
#SBATCH --error=error_%j.txt         # Error output
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=04:00:00              # Maximum walltime (4 hours)
#SBATCH --partition=c23g             # GPU partition/queue
#SBATCH --mem=32G                    # Memory allocation (32 GB)
#SBATCH --cpus-per-task=8            # Number of CPU cores

# --- Environment setup ---
module purge
module load GCCcore/12.2.0
module load Python/3.10.8

# Activate GPU Python environment
source ~/masa-qmix-env-gpu/bin/activate

# Move to project directory
cd ~/MASA-QMIX

# --- Run training ---
echo "=== Training started at $(date) ==="
python main.py | tee training_log_${SLURM_JOB_ID}.txt

# --- Run analysis after training ---
echo "=== Running analyse_rewards.py ==="
python analyse_rewards.py | tee analyse_log_${SLURM_JOB_ID}.txt

echo "=== Analyse finished at $(date) ==="
echo "Check generated plots in: ./my_data_and_graph/historydata/"
