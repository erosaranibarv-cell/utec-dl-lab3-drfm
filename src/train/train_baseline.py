"""Modelo base (paso A1 de la Tabla 3 del paper): SIN ninguna de las ideas de DRFM.

  * un solo encoder ResNet-50 para todas las imagenes (HR y LR mezcladas; las LR entran
    reducidas x4 y vueltas a ampliar, sin super-resolucion);
  * DBSCAN directo sobre TODAS las imagenes para generar pseudo-etiquetas (sin CPG);
  * perdida de clasificacion softmax sobre esas pseudo-etiquetas (como BUC), sin fusion ni L_cc.
Mismo presupuesto que DRFM (epocas, iteraciones, batch, lr) para que la comparacion sea justa.

Uso:  python -m src.train.train_baseline --tag base --modo dev
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.cluster.cpg import agrupar_dbscan, centroides_por_cluster
from src.data.dataset_drfm import DatosDRFM, MuestreadorClusters, aumentar_par
from src.data.h5data import ampliar_x4_bicubica, cargar_particion
from src.eval.features import extraer
from src.eval.reid_metrics import evaluar_reid, similitud_coseno
from src.models.encoder import EncoderReID
from src.utils import RAIZ, cargar_config, dispositivo, fijar_semilla

TEMPERATURA = 0.05   # temperatura del softmax coseno del clasificador


def evaluar_base(enc, datos, dev):
    """Misma evaluacion que DRFM pero con un solo encoder: la consulta LR entra ampliada (borrosa)."""
    fn = lambda x: enc(x)
    gf = extraer(fn, datos["g_images"], dev)
    qo = extraer(fn, datos["q_images"], dev)
    ql = extraer(fn, ampliar_x4_bicubica(datos["q_images_lr"]), dev)
    m = (datos["q_pids"], datos["g_pids"], datos["q_cams"], datos["g_cams"])
    r_o = evaluar_reid(similitud_coseno(qo, gf), *m)
    r_l = evaluar_reid(similitud_coseno(ql, gf), *m)
    return {"original": r_o, "baja_resolucion": r_l, "caida_mAP": r_o["mAP"] - r_l["mAP"]}


def entrenar(args):
    cfg = cargar_config()
    d = cfg["drfm"]
    for k, nombre in (("eps", "dbscan_eps"), ("epochs", "epochs"), ("iters", "iters_por_epoca")):
        if getattr(args, k) is not None:
            d[nombre] = getattr(args, k)
    fijar_semilla(cfg["seed"])
    dev = dispositivo()
    print(f"dispositivo: {dev} | modo={args.modo} | tag={args.tag}")

    datos = DatosDRFM(cfg, args.modo, dev, max_imgs=args.smoke_imgs)
    # entrada del modelo base: HR tal cual; LR (segun camara) reducida x4 y re-ampliada
    X_in = datos.X.clone()
    idx_l = torch.from_numpy(datos.idx_l)
    X_in[idx_l] = datos.X_deg[idx_l]

    enc = EncoderReID(preentrenado=True).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=d["lr"], weight_decay=d["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=d["lr_decay_epoca"], gamma=d["lr_decay_gamma"])
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    ckpt_dir, res_dir = RAIZ / cfg["ckpt_dir"], RAIZ / cfg["results_dir"]
    ckpt_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)
    val = cargar_particion(RAIZ / cfg["data_dir"], "validation") if args.modo in ("dev", "smoke") else None

    historial, mejor = [], -1.0
    for epoca in range(d["epochs"]):
        t0 = time.time()
        # 1-2. caracteristicas y DBSCAN sobre TODAS las imagenes juntas (esto es lo que DRFM evita)
        f = torch.from_numpy(extraer(lambda x: enc(x), X_in, dev)).to(dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        etq = agrupar_dbscan(f, d["dbscan_eps"], d["dbscan_min_samples"],
                             distancia=d["dbscan_distancia"], k1=d["jaccard_k1"], k2=d["jaccard_k2"])
        K = int(etq.max()) + 1
        if K < 2:
            print(f"[ep {epoca + 1}] DBSCAN no encontro clusters (eps={d['dbscan_eps']}); se salta")
            continue
        ok = etq >= 0
        filas_img = np.flatnonzero(ok)
        clf = torch.nn.Parameter(centroides_por_cluster(f, torch.from_numpy(etq).to(dev), K).clone())
        opt_clf = torch.optim.Adam([clf], lr=d["lr"])

        # 3. iteraciones con perdida softmax
        if dev.type == "cuda":
            torch.cuda.empty_cache()      # libera lo que uso el clustering antes de entrenar
        enc.train()
        muest = MuestreadorClusters(etq[ok], d["p_clusters"], d["k_instancias"], seed=cfg["seed"] + epoca)
        perdida_acum = 0.0
        for it in range(d["iters_por_epoca"]):
            filas = muest.lote()
            img = filas_img[filas]
            a = X_in[torch.as_tensor(img)].to(dev)
            x, _ = aumentar_par(a, a)
            y = torch.from_numpy(etq[img]).to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                feats = enc(x)
            logits = F.normalize(feats.float(), dim=1) @ F.normalize(clf, dim=1).T / TEMPERATURA
            perdida = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            opt_clf.zero_grad(set_to_none=True)
            scaler.scale(perdida).backward()
            scaler.step(opt)
            scaler.step(opt_clf)
            scaler.update()
            perdida_acum += float(perdida.detach())
        sched.step()

        reg = {"epoca": epoca + 1, "n_clusters": K, "frac_agrupadas": float(ok.mean()),
               "L_softmax": perdida_acum / d["iters_por_epoca"], "seg": time.time() - t0,
               "mem_pico_GB": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0}
        linea = f"[ep {epoca + 1:02d}/{d['epochs']}] K={K} agrup={ok.mean():.2f} Lsoftmax={reg['L_softmax']:.3f}"
        if val is not None:
            res = evaluar_base(enc.eval(), val, dev)
            reg["val"] = res
            puntaje = 0.5 * (res["original"]["mAP"] + res["baja_resolucion"]["mAP"])
            o, l = res["original"], res["baja_resolucion"]
            linea += (f"\n      val: orig mAP {o['mAP']:.1f} R1 {o['Rank-1']:.1f} | LR mAP {l['mAP']:.1f} "
                      f"R1 {l['Rank-1']:.1f} | caida {res['caida_mAP']:.1f}")
            if puntaje > mejor:
                mejor = puntaje
                torch.save({"modelo": enc.state_dict(), "epoca": epoca + 1}, ckpt_dir / f"{args.tag}_best.pt")
        print(linea + f" | {reg['seg']:.0f}s", flush=True)
        historial.append(reg)
        with open(res_dir / f"{args.tag}_historial.json", "w") as fh:
            json.dump(historial, fh, indent=1, default=float)

    if args.modo == "final":
        torch.save({"modelo": enc.state_dict(), "epoca": d["epochs"]}, ckpt_dir / f"{args.tag}_best.pt")
    print("entrenamiento terminado. mejor puntaje de validacion:", mejor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--modo", default="dev", choices=["dev", "final", "smoke"])
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--smoke-imgs", type=int, default=None)
    entrenar(ap.parse_args())


if __name__ == "__main__":
    main()
