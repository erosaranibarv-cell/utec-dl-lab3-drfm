"""Evaluacion de DRFM en las dos condiciones del laboratorio, con el MISMO modelo entrenado.

Como se arma el vector de cada imagen (la parte que el paper no detalla; lo inferimos de su Sec. 3.3):
  * galeria (siempre original, alta resolucion):  f_hsl = CRFF( E_h(imagen), E_l(su version reducida) )
  * consulta original:                            f_hsl  (igual que la galeria)
  * consulta de baja resolucion (32x16):          f_lsh = CRFF( E_h(SR(consulta)), E_l(consulta ampliada) )
Se comparan con similitud coseno entre vectores normalizados L2.
Tambien se calcula la variante "sin fusion" (solo el encoder HR) para medir el aporte de CRFF:
  original: E_h(consulta) vs E_h(galeria)      baja res.: E_h(SR(consulta)) vs E_h(galeria)
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.data.h5data import ampliar_x4_bicubica, reducir_x4
from src.eval.reid_metrics import evaluar_reid, similitud_coseno


@torch.no_grad()
def extraer_ramas(modelo, imgs_a, imgs_b, rama, dev, bs=256):
    """Pasa pares de imagenes uint8 por una rama de DRFM. Devuelve 3 matrices normalizadas (numpy):
       rama 'hr': a = original, b = su SLR  ->  (f_h, f_sl, f_hsl)
       rama 'lr': a = LR ampliada, b = SHR   ->  (f_l, f_sh, f_lsh)"""
    modelo.eval()
    salidas = ([], [], [])
    for i in range(0, len(imgs_a), bs):
        a = torch.as_tensor(imgs_a[i:i + bs]).to(dev).float() / 255.0   # acepta numpy o tensor
        b = torch.as_tensor(imgs_b[i:i + bs]).to(dev).float() / 255.0
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            fs = modelo.rama_hr(a, b) if rama == "hr" else modelo.rama_lr(a, b)
        for lista, f in zip(salidas, fs):
            if f is not None:                       # sin CRFF no hay vector fusionado
                lista.append(F.normalize(f.float(), dim=1).cpu())
    return tuple(torch.cat(l).numpy() if l else None for l in salidas)


def caracteristicas_drfm(modelo, datos, shr_consultas, dev, bs=256):
    """Vectores normalizados de consultas y galeria para las dos variantes (con y sin fusion).
    datos: dict de cargar_particion; shr_consultas: SR de las consultas LR (N,3,128,64 uint8)."""
    g_deg = ampliar_x4_bicubica(reducir_x4(datos["g_images"]))
    q_deg = ampliar_x4_bicubica(reducir_x4(datos["q_images"]))
    q_lr_up = ampliar_x4_bicubica(datos["q_images_lr"])

    g_h, _, g_hsl = extraer_ramas(modelo, datos["g_images"], g_deg, "hr", dev, bs)
    q_h, _, q_hsl = extraer_ramas(modelo, datos["q_images"], q_deg, "hr", dev, bs)
    _, q_sh, q_lsh = extraer_ramas(modelo, q_lr_up, shr_consultas, "lr", dev, bs)
    # variante -> (consulta original, consulta de baja resolucion, galeria)
    variantes = {"sin_fusion": (q_h, q_sh, g_h)}
    if g_hsl is not None:
        variantes = {"fusion": (q_hsl, q_lsh, g_hsl), **variantes}
    return variantes


def evaluar_modelo(modelo, datos, shr_consultas, dev, bs=256):
    """Devuelve metricas de ambas condiciones, con y sin fusion."""
    variantes = caracteristicas_drfm(modelo, datos, shr_consultas, dev, bs)
    m = (datos["q_pids"], datos["g_pids"], datos["q_cams"], datos["g_cams"])
    res = {}
    for nombre, (q_orig, q_lr, g) in variantes.items():
        r_o = evaluar_reid(similitud_coseno(q_orig, g), *m)
        r_l = evaluar_reid(similitud_coseno(q_lr, g), *m)
        res[nombre] = {"original": r_o, "baja_resolucion": r_l, "caida_mAP": r_o["mAP"] - r_l["mAP"]}
    return res


def resumen(res_variante):
    """Texto de una linea con las metricas principales de una variante."""
    o, l = res_variante["original"], res_variante["baja_resolucion"]
    return (f"orig mAP {o['mAP']:.1f} R1 {o['Rank-1']:.1f} R5 {o['Rank-5']:.1f} R10 {o['Rank-10']:.1f} | "
            f"LR mAP {l['mAP']:.1f} R1 {l['Rank-1']:.1f} R5 {l['Rank-5']:.1f} R10 {l['Rank-10']:.1f} | "
            f"caida {res_variante['caida_mAP']:.1f}")
