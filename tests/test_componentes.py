"""Pruebas de los componentes de DRFM con datos sinteticos (CPU, segundos).
Ejecutar desde la raiz del proyecto:   python -m tests.test_componentes
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.cluster.cpg import agrupar_dbscan, calidad_clusters, generar_pseudoetiquetas
from src.losses.drfm_losses import afilado_probabilidad, consistencia_identidad, contraste_entre_resoluciones
from src.memory import Memoria
from src.models.crff import CRFF

torch.manual_seed(0)
np.random.seed(0)
ok = lambda nombre: print(f"  [OK] {nombre}")


def test_crff():
    m = CRFF(dim=64)
    a, b = torch.randn(5, 64, requires_grad=True), torch.randn(5, 64, requires_grad=True)
    out = m(a, b)
    assert out.shape == (5, 64)
    out.sum().backward()
    assert a.grad is not None and b.grad is not None and torch.isfinite(a.grad).all()
    ok("CRFF: forma (B,dim) y gradiente llega a las dos entradas")


def test_memoria():
    D, K = 8, 3
    feats = F.normalize(torch.randn(10, D), dim=1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    tipos = torch.tensor([0, 0, 1, 0, 1, 1, 1, 0, 0, 0])      # el cluster 2 NO tiene tipo 1
    mem = Memoria(feats, labels, tipos, K, alpha=0.2)
    C, presente = mem.centroides()
    assert C.shape == (K, 2, D) and presente.tolist() == [[True, True], [True, True], [True, False]]
    esperado = F.normalize(feats[[0, 1]].mean(0), dim=0)       # cluster 0, tipo 0 = filas 0 y 1
    assert torch.allclose(C[0, 0], esperado, atol=1e-6)
    nueva = F.normalize(torch.randn(1, D), dim=1)
    antes = mem.feats[4].clone()
    mem.actualizar(torch.tensor([4]), nueva)
    esperado = F.normalize(0.2 * antes + 0.8 * nueva[0], dim=0)
    assert torch.allclose(mem.feats[4], esperado, atol=1e-6)
    ok("Memoria: centroides por (cluster,tipo), tipo ausente marcado, actualizacion con momentum (alpha=0.2)")


def test_lcc_contra_version_ingenua():
    D, K, B, Q, tau = 16, 6, 7, 4, 0.1
    C = F.normalize(torch.randn(K, 2, D), dim=2)
    presente = torch.ones(K, 2, dtype=torch.bool)
    presente[2, 1] = False
    presente[4, 0] = False
    f = torch.randn(B, D)
    clusters = torch.tensor([0, 1, 2, 3, 4, 5, 2])
    obtenido = contraste_entre_resoluciones(f, clusters, C, presente, Q, tau)

    # version ingenua con bucles, directa de la Ec. 16
    fn = F.normalize(f, dim=1)
    total = 0.0
    for i in range(B):
        k = int(clusters[i])
        pos = [C[k, t] for t in range(2) if presente[k, t]]
        otros = [C[j, t] for j in range(K) if j != k for t in range(2) if presente[j, t]]
        s_otros = torch.stack([fn[i] @ c / tau for c in otros]).sort(descending=True).values[:Q]
        s_pos = torch.stack([fn[i] @ c / tau for c in pos])
        denom = torch.logsumexp(torch.cat([s_pos, s_otros]), 0)
        total += -(s_pos - denom).mean()
    assert torch.allclose(obtenido, total / B, atol=1e-5), (obtenido.item(), (total / B).item())
    ok(f"L_cc coincide con la implementacion ingenua de la Ec. 16 ({obtenido.item():.4f})")

    # propiedades: baja si f se parece a su centroide; el gradiente es finito
    cerca = C[clusters, 0] + 0.01 * torch.randn(B, D)
    assert contraste_entre_resoluciones(cerca, clusters, C, presente, Q, tau) < obtenido
    fg = f.clone().requires_grad_()
    contraste_entre_resoluciones(fg, clusters, C, presente, Q, tau).backward()
    assert torch.isfinite(fg.grad).all()
    ok("L_cc: menor cuando f esta cerca de su centroide y con gradiente finito (aun con tipos ausentes)")


def test_otras_perdidas():
    x = torch.randn(4, 16)
    assert abs(consistencia_identidad(x, x).item() + 1.0) < 1e-5
    ok("L_ic = -1 cuando el predictor ya coincide con f_l")
    C = F.normalize(torch.randn(6, 16), dim=1)
    f = torch.randn(5, 16, requires_grad=True)
    perd = afilado_probabilidad(f, C, tau_s=0.02, tau_c=0.04)
    perd.backward()
    assert torch.isfinite(perd) and torch.isfinite(f.grad).all()
    ok("L_ps finita y con gradiente")


def test_cpg():
    P, D, n_hr, n_lr = 12, 64, 8, 6                 # 12 identidades; 8 imagenes HR y 6 LR cada una
    ids_vec = F.normalize(torch.randn(P, D), dim=1)
    ruido = lambda n, s: s * torch.randn(n, D)
    id_hr = torch.arange(P).repeat_interleave(n_hr)
    id_lr = torch.arange(P).repeat_interleave(n_lr)
    f_h = F.normalize(ids_vec[id_hr] + ruido(len(id_hr), 0.06), dim=1)
    f_sl = F.normalize(ids_vec[id_hr] + ruido(len(id_hr), 0.10), dim=1)
    f_l = F.normalize(ids_vec[id_lr] + ruido(len(id_lr), 0.12), dim=1)
    f_sh = F.normalize(ids_vec[id_lr] + ruido(len(id_lr), 0.10), dim=1)

    etq_h = agrupar_dbscan(f_h, eps=0.5, min_samples=4)
    cal = calidad_clusters(etq_h, id_hr.numpy())
    assert cal["F"] > 0.95 and cal["n_clusters"] == P, cal
    ok(f"DBSCAN recupera los {P} clusters (F={cal['F']:.3f})")

    perfecta = calidad_clusters(id_hr.numpy(), id_hr.numpy())
    azar = calidad_clusters(np.random.randint(0, P, len(id_hr)), id_hr.numpy())
    assert abs(perfecta["F"] - 1) < 1e-9 and azar["F"] < 0.3
    ok(f"calidad_clusters: perfecto=1.0, al azar={azar['F']:.2f}")

    etq_l, conf, K = generar_pseudoetiquetas(f_h, f_sl, torch.from_numpy(etq_h), f_l, f_sh, 0.04, 0.05, t=0.8)
    asignadas = etq_l >= 0
    esperado = int(round((etq_h >= 0).mean() * 0.8 * len(conf)))
    assert int(asignadas.sum()) == esperado
    # cada cluster <-> identidad mayoritaria, para medir si la etiqueta de las LR es correcta
    mapa = {k: np.bincount(id_hr.numpy()[etq_h == k]).argmax() for k in range(K)}
    aciertos = np.mean([mapa[int(e)] == int(i) for e, i in zip(etq_l[asignadas], id_lr[asignadas])])
    assert aciertos > 0.95, aciertos
    ok(f"CPG: etiqueta {int(asignadas.sum())}/{len(conf)} LR (T=0.8*frac) con {aciertos * 100:.0f}% de acierto")
    # las no asignadas deben ser de menor confianza que las asignadas
    assert conf[asignadas].min() >= conf[~asignadas].max() - 1e-9
    ok("CPG: las ambiguas son siempre las de menor confianza")


if __name__ == "__main__":
    print("Pruebas de componentes:")
    test_crff()
    test_memoria()
    test_lcc_contra_version_ingenua()
    test_otras_perdidas()
    test_cpg()
    print("Todo OK")
