#!/bin/bash
#SBATCH -J lab3-eps-cpu
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME="$HOME/lab3/cache/torch"
PY="$HOME/lab3/venv/bin/python"

"$PY" -m scripts.calibrar_eps
