#!/bin/bash
#SBATCH -J lab3-zeroshot
#SBATCH --partition=gpu
#SBATCH --gres=shard:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export TORCH_HOME="$HOME/lab3/cache/torch"   # pesos ya descargados en el login
export HF_HOME="$HOME/lab3/cache/hf"
PY="$HOME/lab3/venv/bin/python"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

"$PY" -m scripts.zero_shot --particion validation
