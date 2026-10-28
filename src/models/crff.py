"""CRFF: Cross-Resolution Feature Fusion (Sec. 3.3 del paper, Ec. 10-13).

Junta dos caracteristicas de la MISMA imagen vistas desde espacios distintos:
  * `f_enc_h`: la que saca el encoder HR (ej. de la imagen super-resuelta SHR, o de la HR)
  * `f_enc_l`: la que saca el encoder LR (ej. de la imagen LR, o de la SLR)
y devuelve una sola caracteristica fusionada (la "invariante a la resolucion").
"""
import torch
import torch.nn as nn


class CRFF(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.F1 = nn.Linear(dim, dim)       # capa FC que mira la caracteristica del encoder HR
        self.F2 = nn.Linear(dim, dim)       # capa FC que mira la caracteristica del encoder LR
        self.F3 = nn.Linear(2 * dim, dim)   # capa FC final de la fusion (4096 -> 2048)

    def forward(self, f_enc_h, f_enc_l):
        # CRFF se calcula SIEMPRE en fp32, aunque el resto vaya en media precision: el producto
        # F1(f)*F2(f) puede superar 65504 (maximo de fp16) -> inf -> softmax da NaN.
        with torch.autocast(device_type=f_enc_h.device.type, enabled=False):
            f_enc_h, f_enc_l = f_enc_h.float(), f_enc_l.float()
            # Ec. 10: "que tanto se parecen" -> producto elemento a elemento y softmax sobre los canales
            sim = torch.softmax(self.F1(f_enc_h) * self.F2(f_enc_l), dim=1)
            # Ec. 11-12: cada caracteristica se refuerza con esa similitud (conexion residual)
            f_h_mejorada = f_enc_h + f_enc_h * sim
            f_l_mejorada = f_enc_l + f_enc_l * sim
            # Ec. 13: se concatenan y una capa FC las mezcla en un solo vector
            return self.F3(torch.cat([f_h_mejorada, f_l_mejorada], dim=1))
