#!/bin/bash
# Ejecutar UNA vez en el nodo login de Khipu.
#   bash slurm/env_setup.sh
#
# Usa el modulo python3/3.11.11 (consistente entre login y nodos de computo).
# El python3 del sistema NO sirve: en los nodos de computo es 3.6.
set -e

cd "$HOME/lab3"
module load python3/3.11.11
echo "python de referencia: $(which python3)  ($(python3 --version 2>&1))"

rm -rf venv
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip wheel

# torch + torchvision desde PyPI (el wheel de PyPI ya trae CUDA).
venv/bin/pip install torch torchvision
venv/bin/pip install -r requirements.txt

venv/bin/python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| CUDA build:', torch.version.cuda)"

# Pesos preentrenados: se descargan AQUI (login tiene internet) a una cache local
# que los jobs leen luego sin necesitar red.
export TORCH_HOME="$HOME/lab3/cache/torch"
export HF_HOME="$HOME/lab3/cache/hf"
mkdir -p "$TORCH_HOME" "$HF_HOME"
venv/bin/python - <<'PY'
from torchvision.models import resnet50, ResNet50_Weights
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
print("resnet50 IMAGENET1K_V1: ok")
from transformers import Swin2SRForImageSuperResolution
m = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x4-64")
print("swin2SR classical x4: ok,", round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "M parametros")
PY

echo
echo "Entorno listo. Verifica en un nodo GPU:"
echo "  srun -p gpu --gres=gpu:1 -t 00:10:00 --pty bash -c \\"
echo "    'module load python3/3.11.11; ~/lab3/venv/bin/python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"'"
