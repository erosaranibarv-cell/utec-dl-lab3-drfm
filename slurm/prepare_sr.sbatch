#!/bin/bash
#SBATCH -J lab3-prepare-sr
#SBATCH --partition=gpu
#SBATCH --gres=shard:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export TORCH_HOME="$HOME/lab3/cache/torch"
export HF_HOME="$HOME/lab3/cache/hf"
export HF_HUB_OFFLINE=1          # pesos ya estan en cache; no depender de la red
PY="$HOME/lab3/venv/bin/python"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

"$PY" -m scripts.prepare_sr
