#!/bin/bash
# Referencia ResNet-50 sin entrenar en CPU (validacion: solo ~2 mil imagenes).
#SBATCH -J lab3-zeroshot-cpu
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME="$HOME/lab3/cache/torch"
export HF_HOME="$HOME/lab3/cache/hf"
PY="$HOME/lab3/venv/bin/python"

"$PY" -m scripts.zero_shot --particion validation
