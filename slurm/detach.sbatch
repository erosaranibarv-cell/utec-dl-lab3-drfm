#!/bin/bash
# Diagnostico: CRFF sin gradiente hacia los encoders (y sin L_ps, que fue lo que desestabilizo el entrenamiento).
#SBATCH -J lab3-detach
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:15:00
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
"$PY" -m src.train.train_drfm --tag drfm_detach --sin-ps --crff-detach --resume auto
