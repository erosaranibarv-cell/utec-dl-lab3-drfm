"""Evaluacion final de TODOS los modelos en las dos condiciones del laboratorio:
  * imagenes originales (consulta y galeria en resolucion original)
  * consultas de baja resolucion (32x16, galeria original)
con el MISMO modelo entrenado, vectores normalizados L2 y similitud coseno.
Reporta mAP, Rank-1/5/10 y la caida de mAP. Guarda tambien el detalle por consulta (para el
analisis de aciertos y fallos).

Uso:  python -m scripts.evaluar_final --particion validation
      python -m scripts.evaluar_final --particion testing        # SOLO al final, una vez
"""
import argparse
import json

import numpy as np
import torch

from src.data.h5data import ampliar_x4_bicubica, cargar_particion
from src.eval.evaluar_drfm import caracteristicas_drfm
from src.eval.features import extraer
from src.eval.reid_metrics import agregar, evaluar_reid_detallado, similitud_coseno
from src.models.drfm import DRFM
from src.models.encoder import EncoderReID
from src.utils import RAIZ, cargar_config, dispositivo, fijar_semilla


def medir(nombre, q_orig, q_lr, g, datos, detalle):
    """Metricas de las dos condiciones a partir de los vectores; guarda el detalle por consulta."""
    m = (datos["q_pids"], datos["g_pids"], datos["q_cams"], datos["g_cams"])
    res = {}
    for cond, q in (("original", q_orig), ("baja_resolucion", q_lr)):
        d = evaluar_reid_detallado(similitud_coseno(q, g), *m, topk=10)
        res[cond] = agregar(d)
        for k in ("ap", "primer_acierto", "top"):
            detalle[f"{nombre}__{cond}__{k}"] = d[k]
    res["caida_mAP"] = res["original"]["mAP"] - res["baja_resolucion"]["mAP"]
    return res


def cargar_ckpt(ruta, dev):
    return torch.load(ruta, map_location=dev, weights_only=False)["modelo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", default="validation", choices=["validation", "testing"])
    ap.add_argument("--drfm", default="drfm_best")
    ap.add_argument("--base", default="base_best")
    ap.add_argument("--nocrff", default="drfm_sin_crff_best")
    ap.add_argument("--extra-drfm", nargs="*", default=[], help="otros checkpoints DRFM con CRFF (ej. drfm_sin_ps_best)")
    args = ap.parse_args()

    cfg = cargar_config()
    fijar_semilla(cfg["seed"])
    dev = dispositivo()
    datos = cargar_particion(RAIZ / cfg["data_dir"], args.particion)
    corto = "val" if args.particion == "validation" else "test"
    shr_q = np.load(RAIZ / cfg["cache_dir"] / f"shr_{corto}_q.npy")
    ckpts = RAIZ / cfg["ckpt_dir"]
    resultados, detalle = {}, {}
    q_lr_up = ampliar_x4_bicubica(datos["q_images_lr"])

    # --- 1. ResNet-50 de ImageNet sin entrenar (piso) y 2. modelo base: un solo encoder ---
    for nombre, ruta in (("zero_shot", None), ("base", ckpts / f"{args.base}.pt")):
        if ruta is not None and not ruta.exists():
            print(f"[aviso] no existe {ruta}; se omite {nombre}")
            continue
        enc = EncoderReID(preentrenado=ruta is None).to(dev).eval()
        if ruta is not None:
            enc.load_state_dict(cargar_ckpt(ruta, dev))
        fn = lambda x: enc(x)
        g, q_o, q_l = (extraer(fn, a, dev) for a in (datos["g_images"], datos["q_images"], q_lr_up))
        resultados[nombre] = medir(nombre, q_o, q_l, g, datos, detalle)

    # --- 3. DRFM completo y 4. DRFM sin CRFF ---
    principal = args.drfm.replace("_best", "")            # el DRFM "titular": se elige con validacion
    lista_drfm = [(principal, args.drfm, ("fusion", "sin_fusion"))]
    lista_drfm += [(c.replace("_best", ""), c, ("fusion", "sin_fusion")) for c in args.extra_drfm
                   if c.replace("_best", "") != principal]
    lista_drfm += [("drfm_sin_crff", args.nocrff, ("sin_fusion",))]
    for etiqueta, ckpt, variantes in lista_drfm:
        ruta = ckpts / f"{ckpt}.pt"
        if not ruta.exists():
            print(f"[aviso] no existe {ruta}; se omite {etiqueta}")
            continue
        modelo = DRFM(preentrenado=False, usa_crff="fusion" in variantes).to(dev).eval()
        modelo.load_state_dict(cargar_ckpt(ruta, dev))
        feats = caracteristicas_drfm(modelo, datos, shr_q, dev)
        for v in variantes:
            resultados[f"{etiqueta}__{v}"] = medir(f"{etiqueta}__{v}", *feats[v], datos, detalle)

    # --- tabla ---
    print(f"\nRESULTADOS [{args.particion}] — {resultados[next(iter(resultados))]['original']['n_consultas']} consultas")
    print(f"{'modelo':<26} | {'ORIGINAL':^33} | {'BAJA RESOLUCION':^33} | caida")
    print(f"{'':<26} | {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}   | {'mAP':>6} {'R1':>6} {'R5':>6} {'R10':>6}   | mAP")
    for nombre, r in resultados.items():
        o, l = r["original"], r["baja_resolucion"]
        print(f"{nombre:<26} | {o['mAP']:6.1f} {o['Rank-1']:6.1f} {o['Rank-5']:6.1f} {o['Rank-10']:6.1f}   "
              f"| {l['mAP']:6.1f} {l['Rank-1']:6.1f} {l['Rank-5']:6.1f} {l['Rank-10']:6.1f}   | {r['caida_mAP']:5.1f}")

    salida = RAIZ / cfg["results_dir"]
    salida.mkdir(exist_ok=True)
    resultados["_meta"] = {"principal": principal, "base": args.base, "nocrff": args.nocrff, "extra": args.extra_drfm}
    with open(salida / f"resultados_{args.particion}.json", "w") as f:
        json.dump(resultados, f, indent=2)
    np.savez_compressed(salida / f"detalle_{args.particion}.npz", **detalle)
    print("\nguardado en", salida)


if __name__ == "__main__":
    main()
