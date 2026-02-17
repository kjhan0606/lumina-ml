#!/bin/bash
#SBATCH -J oide_a40
#SBATCH -p a40
#SBATCH -N 1
#SBATCH --gres=gpu:7
#SBATCH -c 64
#SBATCH --mem=128G
#SBATCH -t 30-00:00:00
#SBATCH -o logs/oide_a40_%j.out
#SBATCH -e logs/oide_a40_%j.err

export PYTHONUNBUFFERED=1

# Disable OMP threading: avoid barrier/futex synchronization overhead
# GPU handles the heavy computation; CPU single-threaded is cleaner
export OMP_NUM_THREADS=1
unset OMP_PLACES

echo "=== Node: $(hostname), $(date) ==="
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
nvidia-smi -L

# Clean rebuild for A40 (sm_86)
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
make clean
make cuda GPU_ARCH=sm_86 2>&1
echo ""

# Run OIDE pipeline
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-ML
python3 -u scripts/00_oide_pipeline.py \
    --n-gpus 7 \
    --n-packets 200000 \
    --n-iters 10 \
    --timeout 1800 \
    --oide-dir data/oide \
    --base-values data/oide_base_values.json
