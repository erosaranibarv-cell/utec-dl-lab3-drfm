#!/bin/bash
# Evaluacion de TODOS los modelos en las dos condiciones. Primero validacion (control) y luego prueba (una sola vez).
# El DRFM "titular" (--drfm) es la configuracion con TODOS los componentes que mejor puntua en validacion
# (lambda_ps = 0.3, 35.4 de mAP medio, frente a 32.2 con el lambda = 1.5 del paper).
#SBATCH -J lab3-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

ARGS="--drfm drfm_lps03_best --extra-drfm drfm_v2_best drfm_sin_ps_best drfm_detach_best"
echo "################ VALIDACION (control)"
"$PY" -m scripts.evaluar_final --particion validation $ARGS
echo "################ PRUEBA (una sola vez)"
"$PY" -m scripts.evaluar_final --particion testing $ARGS
