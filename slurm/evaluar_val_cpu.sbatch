#!/bin/bash
# Control: evaluar_final sobre VALIDACION en CPU (debe reproducir los numeros de los logs de entrenamiento).
#SBATCH -J lab3-evalval-cpu
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME="$HOME/lab3/cache/torch"
PY="$HOME/lab3/venv/bin/python"

"$PY" -m scripts.evaluar_final --particion validation --drfm drfm_lps03_best \
      --extra-drfm drfm_v2_best drfm_sin_ps_best drfm_detach_best
