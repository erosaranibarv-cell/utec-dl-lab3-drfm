"""Contenedor de DRFM: los cuatro modulos entrenables de la Figura 2 del paper.

  E_h   encoder de alta resolucion  (procesa imagenes HR y SHR)
  E_l   encoder de baja resolucion  (procesa imagenes LR y SLR)
  Pred  predictor (para la perdida de consistencia de identidad)
  CRFF  modulo de fusion
Hay dos "ramas" segun el tipo de imagen, y cada una produce 3 vectores:
  rama HR: imagen HR + su version reducida SLR  -> f_h,  f_sl, f_hsl (fusionada)
  rama LR: imagen LR + su version super-resuelta -> f_l,  f_sh, f_lsh (fusionada)
"""
import torch.nn as nn

from src.models.crff import CRFF
from src.models.encoder import EncoderReID


class DRFM(nn.Module):
    def __init__(self, preentrenado=True, usa_crff=True, detach_crff=False):
        super().__init__()
        self.E_h = EncoderReID(preentrenado)
        self.E_l = EncoderReID(preentrenado)
        self.pred = nn.Linear(2048, 2048)
        self.crff = CRFF(2048) if usa_crff else None   # None = ablacion (sin fusion)
        # diagnostico: si es True, CRFF recibe las caracteristicas SIN gradiente hacia los encoders
        # (los encoders se entrenan solo con las perdidas de los espacios HR y LR)
        self.detach_crff = detach_crff

    def rama_hr(self, x_h, x_sl):
        """x_h: imagen original; x_sl: la misma imagen reducida x4 y vuelta a ampliar."""
        f_h = self.E_h(x_h)
        f_sl = self.E_l(x_sl)
        d = (lambda t: t.detach()) if self.detach_crff else (lambda t: t)
        return f_h, f_sl, (self.crff(d(f_h), d(f_sl)) if self.crff is not None else None)

    def rama_lr(self, x_l, x_sh):
        """x_l: imagen LR (ampliada borrosa); x_sh: esa misma imagen super-resuelta."""
        f_l = self.E_l(x_l)
        f_sh = self.E_h(x_sh)
        d = (lambda t: t.detach()) if self.detach_crff else (lambda t: t)
        fusion = self.crff(d(f_sh), d(f_l)) if self.crff is not None else None   # (la del encoder HR va primero)
        return f_l, f_sh, fusion
