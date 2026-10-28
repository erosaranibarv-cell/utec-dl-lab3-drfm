"""Distancia Jaccard de k-reciprocos (Zhong et al., 2017, "Re-ranking person re-identification").

Es la distancia que usan los metodos de re-ID no supervisado (SpCL, Cluster Contrast...) antes de
DBSCAN. Con el coseno puro, las caracteristicas de ImageNet "encadenan" a todos en un solo cluster;
exigir que dos imagenes sean vecinas MUTUAS corta ese encadenamiento.

Pasos (para cada imagen i):
  1. vecinos reciprocos: j esta entre los k1 mas cercanos de i  Y  i esta entre los k1 de j;
  2. expansion: se suman los reciprocos (con k1/2) de los reciprocos de i que se solapan >2/3;
  3. peso de cada vecino = softmax(-distancia) -> un vector disperso V_i que suma 1;
  4. "query expansion": V_i se promedia con el de sus k2 vecinos mas cercanos;
  5. distancia = 1 - interseccion/union de los vectores V_i y V_j (Jaccard generalizado).
Los pasos 1-4 se escriben con multiplicaciones de matrices (rapido en GPU). El paso 5 se
simplifica: como cada V_i suma 1,  sum(min(Vi,Vj)) = 1 - L1(Vi,Vj)/2,
asi que  Jaccard = 2*L1 / (2 + L1)  con L1 la distancia de Manhattan entre filas de V.

Todo se hace POR BLOQUES de filas para no materializar muchas matrices NxN a la vez: asi cabe en
una GPU chica (un trozo de 5 GB de la A100 compartida). El resultado se devuelve en la CPU.
"""
import torch


@torch.no_grad()
def distancia_jaccard(f, k1=30, k2=6, bloque=1024):
    """f: (N, D) normalizado L2 (en cualquier dispositivo). Devuelve (N, N) float32 en CPU, en [0, 1]."""
    N, dev = f.shape[0], f.device
    tipo = torch.float16 if dev.type == "cuda" else torch.float32   # 0/1 y conteos pequenos: exactos en fp16
    k_max = max(k1 + 1, k2)

    def dist_bloque(filas):                                       # distancia euclidiana^2 de unas filas a todas
        d = (2 - 2 * f[filas] @ f.T).clamp(min=0)
        d[torch.arange(len(filas), device=dev), filas] = 0         # a si misma: exactamente 0
        return d

    # 1. vecinos mas cercanos de cada imagen (ella misma primero), por bloques
    idx = torch.empty(N, k_max, dtype=torch.long, device=dev)
    for i in range(0, N, bloque):
        filas = torch.arange(i, min(i + bloque, N), device=dev)
        idx[filas] = dist_bloque(filas).topk(k_max, dim=1, largest=False).indices

    def reciprocos(k):
        A = torch.zeros(N, N, dtype=torch.bool, device=dev)
        A.scatter_(1, idx[:, :k], True)                           # j entre los k mas cercanos de i
        R = A & A.T                                               # ...y viceversa (reciprocidad)
        del A
        return R

    R1 = reciprocos(k1 + 1)
    Rh = reciprocos(int(round(k1 / 2)) + 1)
    Rhf = Rh.to(tipo)
    tam_h = Rh.sum(1).float()                                     # |Rh[c]|
    del Rh

    # 2-3. expansion y pesos, por bloques de filas -> V (N, N), cada fila suma 1
    V = torch.empty(N, N, dtype=torch.float32, device=dev)
    for i in range(0, N, bloque):
        filas = torch.arange(i, min(i + bloque, N), device=dev)
        R1b = R1[filas]
        solape = R1b.to(tipo) @ Rhf.T                             # |R1[i] interseccion Rh[c]|
        cond = R1b & (solape.float() > (2.0 / 3.0) * tam_h.unsqueeze(0))   # candidatos que cumplen la regla 2/3
        E = R1b | ((cond.to(tipo) @ Rhf) > 0)                     # expansion: R1[i] + los Rh[c] aceptados
        V[filas] = torch.softmax((-dist_bloque(filas)).masked_fill(~E, float("-inf")), dim=1)
        del solape, cond, E, R1b
    del R1, Rhf

    # 4. query expansion: cada V_i se promedia con el de sus k2 vecinos mas cercanos
    if k2 > 1:
        Vqe = torch.empty_like(V)
        for i in range(0, N, bloque):
            Vqe[i:i + bloque] = V[idx[i:i + bloque, :k2]].mean(dim=1)
        V = Vqe
        del Vqe

    # 5. Jaccard = 2*L1/(2+L1), por bloques, acumulando en la CPU
    salida = torch.empty(N, N, dtype=torch.float32)
    for i in range(0, N, bloque):
        L1 = torch.cdist(V[i:i + bloque], V, p=1)
        salida[i:i + bloque] = (2 * L1 / (2 + L1)).cpu()
    return salida
