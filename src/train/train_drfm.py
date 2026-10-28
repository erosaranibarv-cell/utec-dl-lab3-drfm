"""Entrenamiento de DRFM (Sec. 3 y 4.2 del paper) adaptado a Market-1501.

Cada epoca hace, en orden:
  1. extraer las caracteristicas de TODAS las imagenes (sin aumentacion);
  2. DBSCAN sobre las imagenes HR  -> clusters (pseudo-identidades);
  3. CPG: pasar esos clusters a las imagenes LR mirando LR y SHR a la vez;
  4. llenar las memorias con las caracteristicas de las imagenes etiquetadas;
  5. iterar: lotes de 16 clusters x 4 imagenes, perdida L_cc + L_ic (+ lambda * L_ps despues de la epoca 10).

Uso:
  python -m src.train.train_drfm --tag drfm --modo dev           # elegir configuracion con validacion
  python -m src.train.train_drfm --tag drfm_sin_crff --sin-crff  # ablacion
  python -m src.train.train_drfm --tag drfm_final --modo final   # entrenamiento final (train + val sin etiquetas)
"""
import argparse
import json
import time

import numpy as np
import torch

from src.cluster.cpg import agrupar_dbscan, calidad_clusters, generar_pseudoetiquetas
from src.data.dataset_drfm import DatosDRFM, MuestreadorClusters, aumentar_par
from src.data.h5data import ampliar_x4_bicubica, cargar_particion, reducir_x4
from src.eval.evaluar_drfm import evaluar_modelo, extraer_ramas, resumen
from src.losses.drfm_losses import afilado_probabilidad, consistencia_identidad, contraste_entre_resoluciones
from src.memory import Memoria
from src.models.drfm import DRFM
from src.utils import RAIZ, cargar_config, dispositivo, fijar_semilla


def a_dev(arrays, dev):
    return [torch.from_numpy(a).to(dev) if a is not None else None for a in arrays]


@torch.no_grad()
def diagnostico_clusters(modelo, val, d, dev):
    """Calidad del DBSCAN sobre las imagenes de validacion (aqui SI conocemos las identidades).
    Sirve para ajustar eps; nunca se usa para entrenar."""
    imgs = np.concatenate([val["q_images"], val["g_images"]])
    pids = np.concatenate([val["q_pids"], val["g_pids"]])
    f_h, _, _ = extraer_ramas(modelo, imgs, ampliar_x4_bicubica(reducir_x4(imgs)), "hr", dev)
    etq = agrupar_dbscan(torch.from_numpy(f_h).to(dev), d["dbscan_eps"], d["dbscan_min_samples"],
                         distancia=d["dbscan_distancia"], k1=d["jaccard_k1"], k2=d["jaccard_k2"])
    return calidad_clusters(etq, pids)


def entrenar(args):
    cfg = cargar_config()
    d = cfg["drfm"]
    nombres = {"eps": "dbscan_eps", "t": "t", "epochs": "epochs", "iters": "iters_por_epoca",
               "inicio_ps": "epoca_inicio_ps", "lambda_ps": "lambda_ps"}
    for k, nombre in nombres.items():                   # sobreescrituras por linea de comandos
        if getattr(args, k) is not None:
            d[nombre] = getattr(args, k)
    usa_crff, usa_ps = not args.sin_crff, not (args.sin_crff or args.sin_ps)   # L_ps necesita la fusion

    fijar_semilla(cfg["seed"])
    dev = dispositivo()
    print(f"dispositivo: {dev} | modo={args.modo} | CRFF={usa_crff} | L_ps={usa_ps} | tag={args.tag}")

    datos = DatosDRFM(cfg, args.modo, dev, max_imgs=args.smoke_imgs)
    modelo = DRFM(preentrenado=True, usa_crff=usa_crff, detach_crff=args.crff_detach).to(dev)
    opt = torch.optim.Adam(modelo.parameters(), lr=d["lr"], weight_decay=d["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=d["lr_decay_epoca"], gamma=d["lr_decay_gamma"])
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    ckpt_dir = RAIZ / cfg["ckpt_dir"]
    res_dir = RAIZ / cfg["results_dir"]
    ckpt_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)
    ultimo = ckpt_dir / f"{args.tag}_last.pt"

    val = cargar_particion(RAIZ / cfg["data_dir"], "validation") if args.modo in ("dev", "smoke") else None
    shr_val_q = np.load(RAIZ / cfg["cache_dir"] / "shr_val_q.npy") if val is not None else None

    historial, mejor, epoca0 = [], -1.0, 0
    if args.resume == "auto" and ultimo.exists():
        ck = torch.load(ultimo, map_location=dev, weights_only=False)   # es nuestro propio archivo
        modelo.load_state_dict(ck["modelo"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        historial, mejor, epoca0 = ck["historial"], ck["mejor"], ck["epoca"]
        print(f"reanudando desde la epoca {epoca0}")

    rng = np.random.default_rng(cfg["seed"])
    for epoca in range(epoca0, d["epochs"]):
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ---- 1. caracteristicas de todo el conjunto (modo eval, sin aumentacion) ----
        f_h, f_sl, f_hsl = a_dev(extraer_ramas(modelo, datos.X[datos.idx_h], datos.X_deg[datos.idx_h], "hr", dev), dev)
        f_l, f_sh, f_lsh = a_dev(extraer_ramas(modelo, datos.X_deg[datos.idx_l], datos.SHR[datos.idx_l], "lr", dev), dev)

        # ---- 2. DBSCAN sobre las HR ----
        etq_h = agrupar_dbscan(f_h, d["dbscan_eps"], d["dbscan_min_samples"],
                               distancia=d["dbscan_distancia"], k1=d["jaccard_k1"], k2=d["jaccard_k2"])
        K = int(etq_h.max()) + 1
        if K < 2:
            print(f"[ep {epoca + 1}] DBSCAN no encontro clusters con eps={d['dbscan_eps']}; se salta la epoca")
            historial.append({"epoca": epoca + 1, "n_clusters": K})
            continue

        # ---- 3. CPG: pseudo-etiquetas para las LR ----
        etq_l, conf, _ = generar_pseudoetiquetas(f_h, f_sl, torch.from_numpy(etq_h).to(dev), f_l, f_sh,
                                                 d["tau_l"], d["tau_h"], d["t"])
        etq_l = etq_l.cpu().numpy()
        ok_h, ok_l = etq_h >= 0, etq_l >= 0

        # ---- 4. lista de imagenes etiquetadas (filas) y memorias ----
        fila_img = np.concatenate([datos.idx_h[ok_h], datos.idx_l[ok_l]])           # indice global de imagen
        fila_grupo = np.concatenate([np.zeros(ok_h.sum(), int), np.ones(ok_l.sum(), int)])  # 0=HR 1=LR
        fila_cluster = np.concatenate([etq_h[ok_h], etq_l[ok_l]])
        ambiguas = datos.idx_l[~ok_l]
        grupo_t, cluster_t = torch.from_numpy(fila_grupo).to(dev), torch.from_numpy(fila_cluster).to(dev)
        oh, ol = torch.from_numpy(ok_h).to(dev), torch.from_numpy(ok_l).to(dev)

        def mem(f_hr, f_lr):
            return Memoria(torch.cat([f_hr[oh], f_lr[ol]]), cluster_t, grupo_t, K, d["alpha_memoria"])
        M_hh, M_ll = mem(f_h, f_sh), mem(f_sl, f_l)             # espacio HR (f_h, f_sh) y LR (f_sl, f_l)
        M_fu = mem(f_hsl, f_lsh) if usa_crff else None          # espacio de fusion (f_hsl, f_lsh)

        # ---- 5. iteraciones ----
        if dev.type == "cuda":
            torch.cuda.empty_cache()      # libera lo que uso el clustering antes de entrenar
        modelo.train()
        muest = MuestreadorClusters(fila_cluster, d["p_clusters"], d["k_instancias"], seed=cfg["seed"] + epoca)
        acum = {"cc": 0.0, "ic": 0.0, "ps": 0.0}
        saltadas = 0                                   # iteraciones con perdida no finita (se descartan)
        nf_inicio = int(M_hh.n_no_finitas + M_ll.n_no_finitas + (M_fu.n_no_finitas if usa_crff else 0))
        for it in range(d["iters_por_epoca"]):
            filas = muest.lote()
            filas = np.concatenate([filas[fila_grupo[filas] == 0], filas[fila_grupo[filas] == 1]])  # HR primero
            n_h = int((fila_grupo[filas] == 0).sum())
            img = fila_img[filas]
            cl = cluster_t[torch.from_numpy(filas).to(dev)]
            filas_t = torch.from_numpy(filas).to(dev)

            C_hh, p_hh = M_hh.centroides()
            C_ll, p_ll = M_ll.centroides()
            if usa_crff:
                C_fu, p_fu = M_fu.centroides()

            # forward en media precision (rapido); las PERDIDAS se calculan despues en fp32
            # porque con temperatura 0.04 los productos punto necesitan precision.
            hay_ps = usa_ps and epoca >= d["epoca_inicio_ps"] and len(ambiguas) > 0
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                partes = {"hh": [], "ll": [], "fu": []}
                f_l_raw = f_sh_raw = pred_sh = None
                if n_h > 0:
                    xh, xsl = aumentar_par(*datos.par_hr(img[:n_h]))
                    a, b, c = modelo.rama_hr(xh, xsl)                       # a=f_h  b=f_sl  c=f_hsl
                    partes["hh"].append(a); partes["ll"].append(b)
                    if usa_crff:
                        partes["fu"].append(c)
                if n_h < len(img):
                    xl, xsh = aumentar_par(*datos.par_lr(img[n_h:]))
                    a, b, c = modelo.rama_lr(xl, xsh)                       # a=f_l  b=f_sh  c=f_lsh
                    partes["hh"].append(b); partes["ll"].append(a)
                    if usa_crff:
                        partes["fu"].append(c)
                    f_l_raw, pred_sh = a, modelo.pred(b)
                f_lsh2 = None
                if hay_ps:
                    amb = rng.choice(ambiguas, d["batch_ps"], replace=len(ambiguas) < d["batch_ps"])
                    xl2, xsh2 = aumentar_par(*datos.par_lr(amb))
                    f_lsh2 = modelo.rama_lr(xl2, xsh2)[2]

            f_hh, f_ll = (torch.cat(partes[k]).float() for k in ("hh", "ll"))
            L_cc = (contraste_entre_resoluciones(f_hh, cl, C_hh, p_hh, d["n_negativos"], d["tau_c"])
                    + contraste_entre_resoluciones(f_ll, cl, C_ll, p_ll, d["n_negativos"], d["tau_c"]))
            if usa_crff:
                f_fu = torch.cat(partes["fu"]).float()
                L_cc = L_cc + contraste_entre_resoluciones(f_fu, cl, C_fu, p_fu, d["n_negativos"], d["tau_c"])
            L_ic = (consistencia_identidad(pred_sh.float(), f_l_raw.float())
                    if f_l_raw is not None else torch.zeros((), device=dev))
            L_ps = (afilado_probabilidad(f_lsh2.float(), C_fu[:, 0], d["tau_s"], d["tau_c"])
                    if f_lsh2 is not None else torch.zeros((), device=dev))
            total = L_cc + L_ic + d["lambda_ps"] * L_ps

            if not torch.isfinite(total):               # nunca entrenar ni actualizar memoria con NaN/inf
                saltadas += 1
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), d["grad_clip"])   # red de seguridad contra explosiones
            scaler.step(opt)
            scaler.update()

            # memorias: v <- alpha*v + (1-alpha)*f   (Ec. 1-3)
            M_hh.actualizar(filas_t, f_hh)
            M_ll.actualizar(filas_t, f_ll)
            if usa_crff:
                M_fu.actualizar(filas_t, f_fu)
            for k, v in (("cc", L_cc), ("ic", L_ic), ("ps", L_ps)):
                acum[k] += float(v.detach())
        sched.step()
        n_it = max(d["iters_por_epoca"] - saltadas, 1)
        nf_total = int(M_hh.n_no_finitas + M_ll.n_no_finitas + (M_fu.n_no_finitas if usa_crff else 0))

        # ---- validacion (solo modo dev) ----
        reg = {"epoca": epoca + 1, "n_clusters": K, "frac_hr_agrupadas": float(ok_h.mean()),
               "iters_saltadas": saltadas, "feats_no_finitas": nf_total,
               "lr_etiquetadas": int(ok_l.sum()), "lr_ambiguas": int((~ok_l).sum()),
               **{f"L_{k}": v / n_it for k, v in acum.items()}, "seg": time.time() - t0,
               "mem_pico_GB": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0}
        linea = (f"[ep {epoca + 1:02d}/{d['epochs']}] K={K} agrup={ok_h.mean():.2f} LRetq={ok_l.sum()} "
                 f"Lcc={reg['L_cc']:.3f} Lic={reg['L_ic']:.3f} Lps={reg['L_ps']:.3f} "
                 f"saltadas={saltadas} nofinitas={nf_total} mem={reg['mem_pico_GB']:.1f}GB")
        if val is not None:
            res = evaluar_modelo(modelo, val, shr_val_q, dev)
            clave = "fusion" if usa_crff else "sin_fusion"
            reg["val"] = res
            reg["val_clusters"] = diagnostico_clusters(modelo, val, d, dev)
            puntaje = 0.5 * (res[clave]["original"]["mAP"] + res[clave]["baja_resolucion"]["mAP"])
            reg["val_puntaje"] = puntaje
            linea += f"\n      val[{clave}]: {resumen(res[clave])} | F-clusters(val)={reg['val_clusters']['F']:.2f}"
            if puntaje > mejor:
                mejor = puntaje
                torch.save({"modelo": modelo.state_dict(), "epoca": epoca + 1, "cfg": cfg}, ckpt_dir / f"{args.tag}_best.pt")
        print(linea + f" | {reg['seg']:.0f}s", flush=True)
        historial.append(reg)

        torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoca": epoca + 1, "historial": historial, "mejor": mejor}, ultimo)
        with open(res_dir / f"{args.tag}_historial.json", "w") as f:
            json.dump(historial, f, indent=1, default=float)

    if args.modo == "final":
        torch.save({"modelo": modelo.state_dict(), "epoca": d["epochs"], "cfg": cfg}, ckpt_dir / f"{args.tag}_best.pt")
    print("entrenamiento terminado. mejor puntaje de validacion:", mejor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--modo", default="dev", choices=["dev", "final", "smoke"])
    ap.add_argument("--sin-crff", action="store_true", help="ablacion: sin fusion (tampoco L_ps, que la necesita)")
    ap.add_argument("--sin-ps", action="store_true", help="ablacion: sin L_ps")
    ap.add_argument("--crff-detach", action="store_true",
                    help="diagnostico: CRFF no manda gradiente a los encoders")
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--inicio-ps", type=int, default=None, help="epoca (0-based) desde la que entra L_ps")
    ap.add_argument("--lambda-ps", type=float, default=None, help="peso de L_ps (el paper usa 1.5)")
    ap.add_argument("--smoke-imgs", type=int, default=None, help="prueba rapida con un subconjunto")
    entrenar(ap.parse_args())


if __name__ == "__main__":
    main()
