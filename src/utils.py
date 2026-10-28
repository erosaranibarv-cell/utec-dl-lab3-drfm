"""Utilidades comunes: configuracion, semilla y dispositivo."""
import random
from pathlib import Path

import numpy as np
import torch
import yaml

RAIZ = Path(__file__).resolve().parent.parent  # carpeta del proyecto (Lab3 o ~/lab3)


def cargar_config(ruta="configs/base.yaml"):
    """Lee el yaml de hiperparametros y rutas."""
    with open(RAIZ / ruta) as f:
        return yaml.safe_load(f)


def fijar_semilla(seed):
    """Misma semilla en python, numpy y torch para que las corridas sean repetibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
