#!/bin/bash
# Ejecuta el cuaderno final (lee results/ y data/) y guarda Lab3_final_ejecutado.ipynb
#SBATCH -J lab3-notebook
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME="$HOME/lab3/cache/torch"
export HF_HOME="$HOME/lab3/cache/hf"

venv/bin/python -m ipykernel install --user --name lab3venv --display-name lab3venv >/dev/null
venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=lab3venv Lab3_final.ipynb --output Lab3_final_ejecutado.ipynb
