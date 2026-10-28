"""Extraccion de vectores de caracteristicas para evaluar."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def extraer(fn, imgs_uint8, device, bs=256):
    """Pasa las imagenes (N,3,H,W uint8) por `fn` en lotes y devuelve vectores normalizados L2.

    `fn` recibe un lote de floats en [0,1] y devuelve (B, D). Normalizar L2 hace que
    el producto punto entre dos vectores sea justo su similitud coseno.
    """
    salidas = []
    for i in range(0, len(imgs_uint8), bs):
        x = torch.as_tensor(imgs_uint8[i:i + bs]).to(device).float() / 255.0   # numpy o tensor
        salidas.append(F.normalize(fn(x), dim=1).cpu())
    return torch.cat(salidas).numpy().astype(np.float32)
