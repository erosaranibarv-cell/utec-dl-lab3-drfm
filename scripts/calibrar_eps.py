"""Calibra como agrupar con DBSCAN (el paper no da eps ni la distancia): agrupa las imagenes de
VALIDACION con las caracteristicas del ResNet-50 sin entrenar y compara contra las identidades reales.

Las identidades reales se usan SOLO aqui, para elegir hiperparametros con validacion
(permitido por el enunciado). El entrenamiento nunca las ve.
Uso:  python -m scripts.calibrar_eps
"""
import numpy as np
import torch

from src.cluster.cpg import agrupar_dbscan, calidad_clusters
from src.data.h5data import cargar_particion
from src.eval.features import extraer
from src.models.encoder import EncoderReID
from src.utils import RAIZ, cargar_config, dispositivo, fijar_semilla


def fila(nombre, etq, pids):
    if etq.max() < 0:
        print(f"{nombre:<26} (ningun cluster)")
        return
    c = calidad_clusters(etq, pids)
    print(f"{nombre:<26} {c['n_clusters']:9d} {c['frac_agrupadas']:10.2f} {c['precision']:10.2f} "
          f"{c['recall']:8.2f} {c['F']:6.2f}")


def main():
    cfg = cargar_config()
    fijar_semilla(cfg["seed"])
    dev = dispositivo()
    ms = cfg["drfm"]["dbscan_min_samples"]
    val = cargar_particion(RAIZ / cfg["data_dir"], "validation")
    imgs = np.concatenate([val["q_images"], val["g_images"]])
    pids = np.concatenate([val["q_pids"], val["g_pids"]])
    print(f"{len(imgs)} imagenes de validacion, {len(np.unique(pids))} personas reales")

    enc = EncoderReID(preentrenado=True).to(dev).eval()
    f = torch.from_numpy(extraer(lambda x: enc(x), imgs, dev)).to(dev)
    print(f"cos. medio entre imagenes distintas (sin centrar): {float((f @ f.T).mean()):.3f}")

    cab = f"{'configuracion':<26} {'clusters':>9} {'agrupadas':>10} {'precision':>10} {'recall':>8} {'F':>6}"
    print("\n(a) coseno sin centrar, eps fijo\n" + cab)
    for eps in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]:
        fila(f"eps={eps}", agrupar_dbscan(f, eps, ms), pids)

    print("\n(b) coseno CENTRADO, eps fijo\n" + cab)
    for eps in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        fila(f"centrado eps={eps}", agrupar_dbscan(f, eps, ms, centrar=True), pids)

    print("\n(c) centrado + eps ADAPTATIVO (percentil de la distancia al 4o vecino)\n" + cab)
    for q in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        fila(f"centrado percentil={q}", agrupar_dbscan(f, 0, ms, centrar=True, percentil=q), pids)

    d = cfg["drfm"]
    print(f"\n(d) distancia JACCARD de k-reciprocos (k1={d['jaccard_k1']}, k2={d['jaccard_k2']})\n" + cab)
    for eps in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]:
        fila(f"jaccard eps={eps}",
             agrupar_dbscan(f, eps, ms, distancia="jaccard", k1=d["jaccard_k1"], k2=d["jaccard_k2"]), pids)


if __name__ == "__main__":
    main()
