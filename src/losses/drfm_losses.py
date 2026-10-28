"""Las tres perdidas de DRFM (Sec. 3.4 del paper): L_cc, L_ic y L_ps."""
import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def contraste_entre_resoluciones(f, clusters, C, presente, n_neg, tau):
    """L_cc (Ec. 16): una especie de InfoNCE contra CENTROIDES en lugar de contra muestras.

    Para cada caracteristica `f`:
      * positivos: los centroides de resolucion de SU cluster (1 o 2, uno por tipo);
      * negativos: los `n_neg` centroides mas parecidos de OTROS clusters;
    y se maximiza la probabilidad de los positivos frente al conjunto positivos + negativos.
    f: (B, D); clusters: (B,) cluster de cada muestra; C: (K, 2, D); presente: (K, 2)
    """
    f = F.normalize(f, dim=1)
    K = C.shape[0]
    valido = presente.reshape(-1)                              # (2K,) centroides que existen
    sim = (f @ C.reshape(K * 2, -1).T) / tau                   # (B, 2K) similitud / temperatura
    sim = sim.masked_fill(~valido, NEG_INF)

    pos = (torch.arange(K * 2, device=f.device).unsqueeze(0) // 2) == clusters.unsqueeze(1)
    pos = pos & valido                                         # (B, 2K): centroides del cluster propio
    sim_pos = sim.masked_fill(~pos, NEG_INF)
    sim_neg = sim.masked_fill(pos, NEG_INF)
    neg_top = sim_neg.topk(min(n_neg, K * 2), dim=1).values    # los |Q| negativos mas dificiles

    denom = torch.logsumexp(torch.cat([sim_pos, neg_top], dim=1), dim=1)       # (B,)
    log_prob = (sim_pos - denom.unsqueeze(1)).masked_fill(~pos, 0.0)
    return (-(log_prob.sum(1) / pos.sum(1).clamp(min=1))).mean()


def consistencia_identidad(pred_de_f_sh, f_l):
    """L_ic (Ec. 18): estilo SimSiam. El predictor transforma f_sh y se acerca (coseno) a f_l.
    f_l va con stop-gradient: el encoder LR no recibe gradiente de esta perdida (evita colapso)."""
    return -F.cosine_similarity(pred_de_f_sh, f_l.detach(), dim=1).mean()


def afilado_probabilidad(f_lsh, C_hsl, tau_s, tau_c):
    """L_ps (Ec. 19): para muestras AMBIGUAS (sin pseudo-etiqueta).
    No sabemos su clase correcta, pero si cuales clases NO son. Se toma como objetivo la
    distribucion "afilada" (temperatura baja tau_s, sin gradiente) y se obliga a la distribucion
    normal (tau_c) a parecerse: eso apaga las probabilidades de las clases claramente negativas.
    f_lsh: (B, D); C_hsl: (K, D) un centroide fusionado por cluster."""
    logits = F.normalize(f_lsh, dim=1) @ C_hsl.T
    objetivo = F.softmax(logits / tau_s, dim=1).detach()
    return -(objetivo * F.log_softmax(logits / tau_c, dim=1)).sum(1).mean()
