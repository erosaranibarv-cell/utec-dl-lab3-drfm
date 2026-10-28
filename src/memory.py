"""Memoria de caracteristicas y centroides de resolucion (Ec. 1-3 y 14-15 del paper).

Guarda UNA caracteristica por cada imagen con pseudo-etiqueta (la ultima vista, suavizada).
Cada caracteristica tiene un `tipo` (0 o 1) que dice de que resolucion viene:
  espacio HR:     tipo 0 = f_h  (imagen HR)         tipo 1 = f_sh  (imagen super-resuelta)
  espacio LR:     tipo 0 = f_sl (imagen reducida)   tipo 1 = f_l   (imagen LR)
  espacio fusion: tipo 0 = f_hsl                    tipo 1 = f_lsh
Un "centroide de resolucion" es el promedio de las caracteristicas de un cluster y un tipo.
Por eso un cluster puede tener dos centroides (uno por tipo) y se usan ambos como positivos.
"""
import torch
import torch.nn.functional as F


class Memoria:
    def __init__(self, feats, labels, tipos, n_clusters, alpha):
        feats = feats.detach().clone()
        self.n_no_finitas = int((~torch.isfinite(feats).all(dim=1)).sum())   # para el diagnostico
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)       # un NaN aqui contaminaria TODOS los centroides
        self.feats = F.normalize(feats, dim=1)                                 # (M, D)
        self.labels = labels                                       # (M,) cluster de cada entrada
        self.tipos = tipos                                         # (M,) 0 o 1
        self.K = n_clusters
        self.alpha = alpha

    @torch.no_grad()
    def actualizar(self, idx, nuevas):
        """Ec. 1-3:  v <- alpha * v + (1 - alpha) * f   (promedio movil), y se renormaliza."""
        nuevas = nuevas.detach()
        ok = torch.isfinite(nuevas).all(dim=1)          # solo se actualizan las filas con valores finitos
        if not bool(ok.all()):
            self.n_no_finitas += int((~ok).sum())
            idx, nuevas = idx[ok], nuevas[ok]
        nuevas = F.normalize(nuevas, dim=1)
        mezcla = self.alpha * self.feats[idx] + (1 - self.alpha) * nuevas
        self.feats[idx] = F.normalize(mezcla, dim=1)

    @torch.no_grad()
    def centroides(self):
        """Ec. 14-15: promedio de la memoria por (cluster, tipo).

        Devuelve C de forma (K, 2, D) con centroides normalizados y `presente` (K, 2) que indica
        si ese cluster tiene al menos una caracteristica de ese tipo.
        """
        D = self.feats.shape[1]
        clave = self.labels * 2 + self.tipos
        suma = torch.zeros(self.K * 2, D, device=self.feats.device)
        suma.index_add_(0, clave, self.feats)
        cuenta = torch.bincount(clave, minlength=self.K * 2)
        presente = (cuenta > 0).view(self.K, 2)
        prom = suma / cuenta.clamp(min=1).unsqueeze(1)
        return F.normalize(prom, dim=1).view(self.K, 2, D), presente
