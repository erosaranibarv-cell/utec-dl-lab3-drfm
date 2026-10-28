#!/bin/bash
# Tres entrenamientos de desarrollo (eligen configuracion con VALIDACION), uno tras otro en una sola GPU:
#   1) DRFM completo  2) DRFM sin CRFF (ablacion)  3) modelo base (clustering de TODAS las imagenes: el mas pesado)
#SBATCH -J lab3-dev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME="$HOME/lab3/cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # menos fragmentacion si la GPU es chica
PY="$HOME/lab3/venv/bin/python"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

# si un entrenamiento falla, seguimos con el siguiente (cada uno es independiente)
echo "################ 1/3 DRFM completo";        "$PY" -m src.train.train_drfm --tag drfm --resume auto
echo "################ 2/3 DRFM sin CRFF";        "$PY" -m src.train.train_drfm --tag drfm_sin_crff --sin-crff --resume auto
echo "################ 3/3 modelo base";          "$PY" -m src.train.train_baseline --tag base
