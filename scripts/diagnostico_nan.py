"""Diagnostico del NaN de la corrida drfm (Lcc=nan desde la epoca 3).

Hipotesis: en fp16 el producto F1(f)*F2(f) de CRFF se desborda (max fp16 = 65504) -> inf -> softmax da NaN
-> el NaN entra a la memoria de centroides y contamina todas las perdidas.
Aqui, con el checkpoint de la ultima epoca, medimos las magnitudes reales en fp32 y contamos cuantos
elementos se desbordarian en fp16.   Uso: python -m scripts.diagnostico_nan --ckpt drfm_last
"""
import argparse

import numpy as np
import torch

from src.data.h5data import ampliar_x4_bicubica, cargar_particion, reducir_x4
from src.eval.features import extraer
from src.models.drfm import DRFM
from src.utils import RAIZ, cargar_config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="drfm_last")
    args = ap.parse_args()
    cfg = cargar_config()
    ck = torch.load(RAIZ / cfg["ckpt_dir"] / f"{args.ckpt}.pt", map_location="cpu", weights_only=False)
    modelo = DRFM(preentrenado=False, usa_crff=True).eval()
    modelo.load_state_dict(ck["modelo"])
    print(f"checkpoint de la epoca {ck['epoca']}")

    # 1) pesos finitos?
    malos = [n for n, p in modelo.named_parameters() if not torch.isfinite(p).all()]
    print("parametros con NaN/inf:", malos if malos else "ninguno")
    for n in ("crff.F1.weight", "crff.F2.weight", "crff.F3.weight"):
        w = dict(modelo.named_parameters())[n]
        print(f"  |{n}|: max={w.abs().max():.3f} media={w.abs().mean():.4f}")

    # 2) magnitudes en la rama HR sobre las imagenes de validacion (fp32)
    val = cargar_particion(RAIZ / cfg["data_dir"], "validation")
    imgs = np.concatenate([val["q_images"], val["g_images"]])[:600]
    deg = ampliar_x4_bicubica(reducir_x4(imgs))
    f_h = torch.from_numpy(extraer(lambda x: modelo.E_h(x), imgs, torch.device("cpu"))).float()   # normalizadas
    # las caracteristicas CRUDAS (sin normalizar) son las que entran a CRFF: se recalculan
    fh_raw, fsl_raw = [], []
    for i in range(0, len(imgs), 100):
        a = torch.from_numpy(imgs[i:i + 100]).float() / 255
        b = torch.from_numpy(deg[i:i + 100]).float() / 255
        fh_raw.append(modelo.E_h(a)); fsl_raw.append(modelo.E_l(b))
    fh_raw, fsl_raw = torch.cat(fh_raw), torch.cat(fsl_raw)
    print(f"\ncaracteristicas crudas: max E_h={fh_raw.max():.1f}  max E_l={fsl_raw.max():.1f}  "
          f"norma media={fh_raw.norm(dim=1).mean():.1f}")
    A, B = modelo.crff.F1(fh_raw), modelo.crff.F2(fsl_raw)
    P = A * B
    print(f"|F1(f)| max={A.abs().max():.1f} | |F2(f)| max={B.abs().max():.1f} | |F1*F2| max={P.abs().max():.1f}")
    desb = int((P.abs() > 65504).sum())
    print(f"elementos del producto que se desbordan en fp16 (>65504): {desb} de {P.numel()}")
    print(f"  inf tras convertir a fp16: {int(torch.isinf(P.half()).sum())}")
    s = torch.softmax(P.half().float(), dim=1)
    print(f"  NaN en softmax tras emular el desborde fp16: {int(torch.isnan(s).sum())}")
    s32 = torch.softmax(P, dim=1)
    print(f"  NaN en softmax en fp32: {int(torch.isnan(s32).sum())}")
    fus = modelo.crff(fh_raw, fsl_raw)
    print(f"salida de CRFF en fp32: finita={bool(torch.isfinite(fus).all())}, |max|={fus.abs().max():.1f}")


if __name__ == "__main__":
    main()
