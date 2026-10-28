#!/bin/bash
# Prueba de humo en CPU: recorre TODO el codigo de entrenamiento con un subconjunto diminuto,
# para encontrar errores antes de gastar un turno de GPU.
#SBATCH -J lab3-smoke
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$HOME/lab3"
module load python3/3.11.11
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME="$HOME/lab3/cache/torch"
PY="$HOME/lab3/venv/bin/python"

echo "################ DRFM completo (humo)"
"$PY" -m src.train.train_drfm --tag humo_drfm --modo smoke --epochs 2 --iters 3 --eps 0.55 --inicio-ps 1 --resume no
echo "################ DRFM sin CRFF (humo)"
"$PY" -m src.train.train_drfm --tag humo_nocrff --sin-crff --modo smoke --epochs 2 --iters 3 --eps 0.55 --resume no
echo "################ modelo base (humo)"
"$PY" -m src.train.train_baseline --tag humo_base --modo smoke --epochs 2 --iters 3 --eps 0.55
