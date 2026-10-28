#!/bin/bash
# Evaluacion en la particion de PRUEBA en CPU (32 nucleos): alternativa a la GPU cuando la cola esta llena.
# Mismo script y mismos checkpoints que evaluar_final.sbatch; el resultado es identico.
#SBATCH -J lab3-evaltest-cpu
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME="$HOME/lab3/cache/torch"
PY="$HOME/lab3/venv/bin/python"

"$PY" -m scripts.evaluar_final --particion testing --drfm drfm_lps03_best \
      --extra-drfm drfm_v2_best drfm_sin_ps_best drfm_detach_best
