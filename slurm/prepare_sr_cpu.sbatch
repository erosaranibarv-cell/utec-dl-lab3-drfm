#!/bin/bash
# Preproceso de super-resolucion en CPU: son ~16 mil imagenes de 32x16, no hace falta GPU.
#SBATCH -J lab3-sr-cpu
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME="$HOME/lab3/cache/torch"
export HF_HOME="$HOME/lab3/cache/hf"
export HF_HUB_OFFLINE=1
PY="$HOME/lab3/venv/bin/python"

"$PY" -m scripts.prepare_sr
