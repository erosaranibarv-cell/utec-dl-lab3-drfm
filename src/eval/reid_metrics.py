"""Metricas de re-identificacion (protocolo estandar de Market-1501): mAP y CMC (Rank-k)."""
import numpy as np


def similitud_coseno(qf, gf):
    """Con vectores ya normalizados L2, el coseno es solo un producto punto."""
    return qf @ gf.T


def evaluar_reid_detallado(sim, q_pids, g_pids, q_cams, g_cams, topk=10):
    """Evalua cada consulta por separado. Para cada una:
      1. ordena la galeria de mas a menos parecida;
      2. descarta las imagenes "junk" (g_pid == -1) y las de la MISMA persona en la
         MISMA camara (esas no cuentan ni como acierto ni como error);
      3. los distractores (g_pid == 0) se quedan: son negativos que ensucian el ranking;
      4. un acierto es misma persona y OTRA camara.
    Devuelve, por consulta: ap, primer_acierto (posicion 0-based en el ranking filtrado, -1 si no
    hay), top (indices de galeria de las `topk` mejores ya filtradas) y valida.
    AP = promedio de la precision en cada acierto."""
    n_q = sim.shape[0]
    ap = np.zeros(n_q)
    primer = np.full(n_q, -1)
    valida = np.zeros(n_q, dtype=bool)
    top = np.zeros((n_q, topk), dtype=np.int64)
    for i in range(n_q):
        orden = np.argsort(-sim[i])  # de mayor a menor similitud
        gp, gc = g_pids[orden], g_cams[orden]
        junk = (gp == -1) | ((gp == q_pids[i]) & (gc == q_cams[i]))
        orden, gp, gc = orden[~junk], gp[~junk], gc[~junk]
        top[i] = orden[:topk]
        acierto = (gp == q_pids[i]) & (gc != q_cams[i])
        if acierto.sum() == 0:  # consulta sin posible acierto: no se evalua
            continue
        valida[i] = True
        pos = np.flatnonzero(acierto)  # posiciones (0-based) de los aciertos en el ranking
        ap[i] = np.mean(np.arange(1, len(pos) + 1) / (pos + 1))   # precision en el k-esimo acierto = k/(pos+1)
        primer[i] = pos[0]
    return {"ap": ap, "primer_acierto": primer, "top": top, "valida": valida}


def agregar(d, ranks=(1, 5, 10)):
    """De los resultados por consulta a las metricas globales.
    mAP = promedio de AP. Rank-k = % de consultas con al menos un acierto entre las k primeras (CMC)."""
    v = d["valida"]
    resultado = {"mAP": float(d["ap"][v].mean()) * 100, "n_consultas": int(v.sum())}
    for k in ranks:
        resultado[f"Rank-{k}"] = float(np.mean(d["primer_acierto"][v] < k)) * 100
    return resultado


def evaluar_reid(sim, q_pids, g_pids, q_cams, g_cams, ranks=(1, 5, 10)):
    """Atajo: metricas globales directamente desde la matriz de similitud."""
    return agregar(evaluar_reid_detallado(sim, q_pids, g_pids, q_cams, g_cams, topk=1), ranks)
