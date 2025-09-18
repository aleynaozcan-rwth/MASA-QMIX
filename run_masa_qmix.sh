#!/bin/bash
#SBATCH --job-name=masa_qmix         # İşin ismi
#SBATCH --output=output_%j.txt       # Normal çıktılar (%j job ID ile değişir)
#SBATCH --error=error_%j.txt         # Hata çıktıları
#SBATCH --gres=gpu:1                 # 1 GPU talep et
#SBATCH --time=04:00:00              # Maksimum süre (4 saat)
#SBATCH --partition=c23g             # GPU partition
#SBATCH --mem=32G                    # RAM miktarı
#SBATCH --cpus-per-task=8            # CPU çekirdeği

# --- Ortam Ayarları ---
module purge
module load GCCcore/12.2.0
module load Python/3.10.8

# GPU ortamını aktive et
source ~/masa-qmix-env-gpu/bin/activate

# Proje klasörüne gir
cd ~/MASA-QMIX

# --- Kodunu Çalıştır ---
echo "=== Training started at $(date) ==="
python main.py | tee training_log_%j.txt

# --- Training sonrası analiz ---
echo "=== Running analyse_rewards.py ==="
python analyse_rewards.py | tee analyse_log_%j.txt

echo "=== Analyse finished at $(date) ==="
echo "Check generated plots in: ./my_data_and_graph/historydata/"

