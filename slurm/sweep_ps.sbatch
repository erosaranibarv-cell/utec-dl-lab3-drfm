#!/bin/bash
# Diagnostico/ablacion de L_ps: el DRFM completo (lambda=1.5) se desplomo en la epoca 26. Se prueba
#   a) CRFF sin L_ps      b) CRFF con lambda = 0.3
#SBATCH -J lab3-sweep-ps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME="$HOME/lab3/cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$HOME/lab3/venv/bin/python"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "################ a) CRFF sin L_ps";        "$PY" -m src.train.train_drfm --tag drfm_sin_ps --sin-ps --resume auto
echo "################ b) CRFF, lambda_ps = 0.3"; "$PY" -m src.train.train_drfm --tag drfm_lps03 --lambda-ps 0.3 --resume auto
