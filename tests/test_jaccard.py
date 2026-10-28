"""Compara nuestra distancia Jaccard vectorizada contra una version ingenua con bucles que sigue,
paso a paso, el algoritmo original de re-ranking (Zhong et al. 2017 / codigo de SpCL).
Ejecutar:  python -m tests.test_jaccard
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.cluster.jaccard import distancia_jaccard


def jaccard_ingenuo(f, k1, k2):
    N = len(f)
    d = 2 - 2 * f @ f.T
    rank = np.argsort(d, axis=1, kind="stable")               # rank[i,0] = i (la mas cercana es ella misma)

    def recip(i, k):
        fw = rank[i, :k + 1]
        bw = rank[fw, :k + 1]
        return fw[np.where(bw == i)[0]]

    nn1 = [recip(i, k1) for i in range(N)]
    nnh = [recip(i, int(np.around(k1 / 2))) for i in range(N)]

    V = np.zeros((N, N))
    for i in range(N):
        exp = nn1[i]
        for c in nn1[i]:
            if len(np.intersect1d(nnh[c], nn1[i])) > 2 / 3 * len(nnh[c]):
                exp = np.append(exp, nnh[c])
        exp = np.unique(exp)
        w = np.exp(-d[i, exp])
        V[i, exp] = w / w.sum()
    if k2 != 1:
        V = np.stack([V[rank[i, :k2]].mean(0) for i in range(N)])

    J = np.zeros((N, N))
    for i in range(N):                                         # 1 - sum(min) / sum(max)
        for j in range(N):
            J[i, j] = 1 - np.minimum(V[i], V[j]).sum() / np.maximum(V[i], V[j]).sum()
    return J


def test_contra_version_ingenua():
    rng = np.random.default_rng(0)
    centros = rng.normal(size=(8, 24))
    f = np.concatenate([c + 0.25 * rng.normal(size=(12, 24)) for c in centros])
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    k1, k2 = 10, 3
    ref = jaccard_ingenuo(f, k1, k2)
    ft = torch.from_numpy(f).float()
    for bloque in (1024, 32, 7):      # un bloque, varios, y uno que no divide a N: el resultado no debe cambiar
        obt = distancia_jaccard(ft, k1, k2, bloque=bloque).numpy()
        err = np.abs(ref - obt).max()
        assert err < 1e-4, (bloque, err)
        print(f"  [OK] Jaccard por bloques (bloque={bloque:4d}) = version ingenua (error maximo {err:.2e}, N={len(f)})")
    # propiedades: simetrica, diagonal 0, rango [0,1]
    assert np.allclose(obt, obt.T, atol=1e-5) and np.allclose(np.diag(obt), 0, atol=1e-5)
    assert obt.min() >= -1e-6 and obt.max() <= 1 + 1e-6
    # misma identidad -> mas cerca que identidades distintas
    ids = np.repeat(np.arange(8), 12)
    mismo = obt[ids[:, None] == ids[None, :]].mean()
    distinto = obt[ids[:, None] != ids[None, :]].mean()
    assert mismo < distinto
    print(f"  [OK] simetrica, diagonal 0, en [0,1]; misma persona {mismo:.2f} < distinta {distinto:.2f}")


if __name__ == "__main__":
    print("Prueba de la distancia Jaccard:")
    test_contra_version_ingenua()
    print("Todo OK")
