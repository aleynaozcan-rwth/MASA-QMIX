#!/bin/bash
#SBATCH --job-name=masa_qmix
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=02:00:00
#SBATCH --partition=c23g_low
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# --- Environment setup ---
module purge
module load GCCcore/12.2.0
module load Python/3.10.8

# Activate GPU Python environment
source ~/masa-qmix-env-gpu/bin/activate

# Move to project directory
cd ~/MASA-QMIX

# --- Ensure folders exist & cleanup old outputs ---
mkdir -p ./my_data_and_graph/historydata ./my_data_and_graph/pickles
rm -f ./my_data_and_graph/historydata/*.txt
rm -f ./my_data_and_graph/historydata/*.csv
rm -f ./my_data_and_graph/historydata/*.png
rm -f ./my_data_and_graph/pickles/*.pk

# --- Runtime env (threads & CUDA allocator) ---
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export PYTHONUNBUFFERED=1

# --- Job started ---
echo "=== JOB $SLURM_JOB_ID STARTED at $(date) ==="
which python
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); 
print('device_count=', torch.cuda.device_count()); 
print('device_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
nvidia-smi || true

# --- Run training (GPU ON) ---
echo "=== Training started at $(date) ==="
python -u main.py --cuda True --learn True | tee training_log_${SLURM_JOB_ID}.txt
echo "=== Training finished at $(date) ==="

# --- Run analysis after training ---
echo "=== Running analyse_rewards.py ==="
python -u analyse_rewards.py | tee analyse_log_${SLURM_JOB_ID}.txt
echo "=== Analyse finished at $(date) ==="

# --- Job finished ---
python - <<'EOF'
import torch, gc
gc.collect()
try:
    torch.cuda.empty_cache()
except Exception:
    pass
EOF

echo "=== JOB $SLURM_JOB_ID FINISHED at $(date) ==="
echo "Check generated plots in: ./my_data_and_graph/historydata/"
