"""CPG: Cross-resolution Pseudo-label Generation (Sec. 3.2 del paper, Ec. 4-9).

Idea: agrupar solo las imagenes HR (tienen mas detalle) y de ahi "pasar" las etiquetas a las
imagenes LR mirandolas en DOS espacios a la vez (LR y su version super-resuelta SHR).
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

from src.cluster.jaccard import distancia_jaccard


@torch.no_grad()
def agrupar_dbscan(f, eps, min_samples=4, centrar=False, percentil=None, distancia="coseno", k1=30, k2=6):
    """Agrupa con DBSCAN. f: (N, D) normalizado L2.
    Devuelve etiquetas (N,) numpy: 0..K-1 para cada cluster y -1 para los "ruido" (outliers).

    distancia="jaccard": distancia Jaccard de k-reciprocos (la que usan SpCL y Cluster Contrast;
    ver jaccard.py). Con "coseno" las caracteristicas de ImageNet encadenan todo en un cluster.

    Otros dos ajustes con distancia coseno (el paper solo dice "DBSCAN, min_samples=4"):
      * centrar: se resta el promedio de las caracteristicas antes de normalizar. Las salidas de
        ResNet son positivas (vienen de una ReLU), asi que TODOS los cosenos son altos y sin
        centrar un eps razonable junta todo en un solo cluster.
      * percentil: en vez de un eps fijo, eps = ese percentil de la distancia de cada imagen a su
        `min_samples`-esimo vecino. Se adapta solo cuando la escala de las distancias cambia
        durante el entrenamiento (un eps fijo no).
    """
    if distancia == "jaccard":
        dist = distancia_jaccard(f, k1, k2)
        return DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1).fit_predict(
            dist.cpu().numpy().astype(np.float64))
    if centrar:
        f = F.normalize(f - f.mean(dim=0, keepdim=True), dim=1)
    dist = (1.0 - f @ f.T).clamp(min=0)
    dist.fill_diagonal_(0)
    if percentil is not None:
        vecino_k = dist.topk(min_samples + 1, dim=1, largest=False).values[:, -1]   # (incluye a si misma en 0)
        eps = float(torch.quantile(vecino_k, percentil))
    dist = dist.cpu().numpy().astype(np.float64)
    return DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1).fit_predict(dist)


def calidad_clusters(etiquetas, pids_reales):
    """Mide que tan buenos son los clusters comparando contra identidades REALES (solo para
    diagnostico en validacion; nunca se usa para entrenar). Devuelve precision, recall y F-score
    sobre PARES de imagenes (como el F-score de la Fig. 6 del paper).
      precision = de los pares que el cluster junta, cuantos son la misma persona
      recall    = de los pares de la misma persona, cuantos quedaron juntos (los outliers cuentan en contra)
    """
    pares = lambda n: n * (n - 1) / 2.0
    ok = etiquetas >= 0
    cont = {}
    for e, p in zip(etiquetas[ok], pids_reales[ok]):
        cont[(e, p)] = cont.get((e, p), 0) + 1
    tp = sum(pares(n) for n in cont.values())
    _, tam_cluster = np.unique(etiquetas[ok], return_counts=True)
    _, tam_real = np.unique(pids_reales, return_counts=True)
    prec = tp / max(sum(pares(n) for n in tam_cluster), 1)
    rec = tp / max(sum(pares(n) for n in tam_real), 1)
    f = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"precision": prec, "recall": rec, "F": f,
            "n_clusters": int(len(tam_cluster)), "frac_agrupadas": float(ok.mean())}


@torch.no_grad()
def centroides_por_cluster(f, etiquetas, K):
    """Promedio de las caracteristicas de cada cluster (solo muestras con etiqueta >= 0), normalizado."""
    ok = etiquetas >= 0
    suma = torch.zeros(K, f.shape[1], device=f.device)
    suma.index_add_(0, etiquetas[ok], f[ok])
    return F.normalize(suma, dim=1)


@torch.no_grad()
def generar_pseudoetiquetas(f_h, f_sl, etiquetas_h, f_l, f_sh, tau_l, tau_h, t):
    """Pseudo-etiquetas para las imagenes LR a partir de los clusters de las HR.

    f_h, f_sl : caracteristicas (normalizadas) de las HR y de sus SLR   (N1, D)
    etiquetas_h: clusters de las HR por DBSCAN (-1 = sin cluster)       (N1,) tensor
    f_l, f_sh : caracteristicas (normalizadas) de las LR y de sus SHR   (N2, D)
    Devuelve etiquetas para las N2 imagenes LR (-1 = ambigua) y la confianza de cada una.
    """
    K = int(etiquetas_h.max().item()) + 1
    C_h = centroides_por_cluster(f_h, etiquetas_h, K)       # Ec. 7: centroides HR (espacio HR)
    C_sl = centroides_por_cluster(f_sl, etiquetas_h, K)     # Ec. 5: centroides SLR (espacio LR); la SLR hereda el cluster de su HR

    # Ec. 4 y 6: "vector de clase" = a que cluster se parece cada imagen (en logaritmo, por estabilidad)
    log_s_l = F.log_softmax(f_l @ C_sl.T / tau_l, dim=1)     # la LR, vista en el espacio LR
    log_s_sh = F.log_softmax(f_sh @ C_h.T / tau_h, dim=1)    # su SHR, vista en el espacio HR

    # Ec. 8: se multiplican (Hadamard) y se renormaliza a suma 1 => softmax de la suma de logs
    fusion = F.softmax(log_s_l + log_s_sh, dim=1)
    conf, clase = fusion.max(dim=1)       # si ambos "votan" la misma clase, conf es alta

    # Ec. 9: se queda solo la proporcion T de las mas confiables; el resto queda ambiguo
    T = (etiquetas_h >= 0).float().mean().item() * t
    n_keep = int(round(T * len(conf)))
    etiquetas_l = torch.full_like(clase, -1)
    mejores = conf.argsort(descending=True)[:n_keep]
    etiquetas_l[mejores] = clase[mejores]
    return etiquetas_l, conf, K
