"""Genera Lab3_final.ipynb (sin ejecutar). Luego se ejecuta en Khipu con nbconvert.
Uso (desde la raiz del proyecto):  python notebooks/construir_notebook.py
"""
import pathlib

import nbformat as nbf

RAIZ = pathlib.Path(__file__).resolve().parent.parent
EXPLICACIONES = (RAIZ / "notebooks" / "explicaciones_casos.md")
texto_casos = EXPLICACIONES.read_text(encoding="utf8") if EXPLICACIONES.exists() else \
    "_(las explicaciones de los casos se agregan despues de ver las figuras)_"

C = []   # lista de (tipo, contenido)
md = lambda s: C.append(("md", s.strip("\n")))
code = lambda s: C.append(("code", s.strip("\n")))

# ------------------------------------------------------------------------------------------------
md('''
# Laboratorio 3 — Re-identificación de personas no supervisada con baja resolución (DRFM)

**Paper:** Pang, Zhao y Wang (2024). *Dual-Resolution Fusion Modeling for Unsupervised Cross-Resolution Person
Re-Identification*. ACM Multimedia '24.  **Dataset:** Market-1501 (6 cámaras).

**Problema.** Dada una imagen de consulta, ordenar una galería por similitud para encontrar a la misma persona en otra
cámara, **sin usar etiquetas de identidad al entrenar**, y evaluar el mismo modelo con consultas originales y con
consultas de baja resolución (32×16).

**Método (DRFM).** Dos encoders ResNet-50 (uno para alta y otro para baja resolución) y tres ideas:
1. **CPG** (generación de pseudo-etiquetas entre resoluciones): se agrupan solo las imágenes HR con DBSCAN y las
   etiquetas se pasan a las imágenes LR multiplicando sus "vectores de clase" en el espacio LR y en el de su versión
   super-resuelta (SHR); solo se conservan las más confiables.
2. **CRFF** (fusión de características): une las características de la misma imagen vistas por los dos encoders.
3. **Pérdidas:** `L_cc` (contraste contra centroides de resolución), `L_ic` (consistencia de identidad, tipo SimSiam)
   y `L_ps` (afilado de probabilidad para las muestras ambiguas).

## Cómo ejecutar
```bash
bash slurm/env_setup.sh                      # entorno + pesos (ResNet-50 ImageNet, Swin2SR) en Khipu
sbatch slurm/prepare_sr_cpu.sbatch           # super-resolución de las imágenes (cache/shr_*.npy)
python -m scripts.calibrar_eps               # calibra DBSCAN con validación
python -m src.train.train_drfm --tag drfm_v2                  # DRFM completo
python -m src.train.train_drfm --tag drfm_sin_crff --sin-crff # ablación
python -m src.train.train_baseline --tag base                 # modelo base
python -m scripts.evaluar_final --particion testing           # métricas en test (una sola vez, al final)
```
Los resultados de este cuaderno se leen de `results/` (generados por esos scripts) y las imágenes de `data/`.
''')

md('''
## 0. Diferencias importantes respecto a la propuesta original

| Aspecto | Paper | Esta implementación | Motivo |
|---|---|---|---|
| Datos LR de entrenamiento | MLR-Market-1501: las imágenes de una cámara se reducen ×2, ×3 o ×4 al azar | las cámaras **1, 2 y 5** actúan como "LR" (reducción ×4 bicúbica); **3, 4 y 6** como "HR" | el enunciado pide generar las LR desde las imágenes de entrenamiento; ×4 es exactamente la degradación de las consultas LR del dataset |
| Protocolo de evaluación | consulta LR contra una galería con 1 imagen HR por persona; solo Rank-k | protocolo estándar de Market-1501 (3 368 consultas, 19 732 de galería, *junk* y distractores), mAP y Rank-k, en **dos condiciones** | el enunciado |
| Super-resolución | Swin2SR (checkpoint no indicado) | `caidas/swin2SR-classical-sr-x4-64` | entrenado con degradación bicúbica, igual que el dataset |
| Distancia para DBSCAN | no se indica (solo `min_samples = 4`) | **Jaccard de k-recíprocos** (k1 = 30, k2 = 6), `eps = 0.5` | con distancia coseno DBSCAN junta todo en un cluster con características de ImageNet (F ≤ 0.06, medido en la sección 4) |
| Tamaño de entrada | no se indica | 128×64 (nativo); las LR se reescalan con bicúbica a 128×64; `layer4` con stride 1 | tamaño nativo del dataset |
| Iteraciones por época, schedule de lr | no se indican | 200 iteraciones/época; lr ×0.1 en la época 20 | decisión nuestra (convención de Cluster Contrast) |
| Estabilidad numérica | no se menciona | CRFF en fp32, recorte de gradiente (10), descarte de iteraciones no finitas | la primera corrida divergió a NaN (ver sección 4) |
| Características en test | no se detallan | galería y consultas originales: `f_hsl`; consultas LR: `f_lsh` | inferido de la Sec. 3.3 del paper |
| Entrenamiento final | — | los modelos se entrenan solo con `training.h5`; la configuración y la mejor época se eligen con validación | simplificación; el archivo permitiría reentrenar con las imágenes de validación sin etiquetas |
| Operaciones no implementadas | — | no se usan GNN/AM/CNN del paper original de comparación | no son parte de DRFM |
''')

code('''
import inspect, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path.cwd()                      # el cuaderno se ejecuta desde la raiz del proyecto
sys.path.insert(0, str(RAIZ))
from src.utils import cargar_config
from src.data.h5data import ampliar_x4_bicubica, cargar_entrenamiento, cargar_particion, reducir_x4

cfg = cargar_config()
RES, DATOS, CACHE = RAIZ / cfg["results_dir"], RAIZ / cfg["data_dir"], RAIZ / cfg["cache_dir"]
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 200)
''')

# ------------------------------------------------------------------------------------------------
md('''
## 1. Carga de datos y particiones

`training.h5` solo trae imágenes y cámaras (sin identidades). `validation.h5` y `testing.h5` traen consultas
(originales y de baja resolución) y galería con identidades y cámaras, que se usan **solo para calcular métricas**.
''')
code('''
tr = cargar_entrenamiento(DATOS)
val = cargar_particion(DATOS, "validation")
test = cargar_particion(DATOS, "testing")

resumen = pd.DataFrame({
    "imágenes de entrenamiento": [len(tr["images"]), "-", "-"],
    "consultas": ["-", len(val["q_pids"]), len(test["q_pids"])],
    "galería": ["-", len(val["g_pids"]), len(test["g_pids"])],
    "personas (consultas)": ["-", len(np.unique(val["q_pids"])), len(np.unique(test["q_pids"]))],
    "galería: junk (id -1)": ["-", int((val["g_pids"] == -1).sum()), int((test["g_pids"] == -1).sum())],
    "galería: distractores (id 0)": ["-", int((val["g_pids"] == 0).sum()), int((test["g_pids"] == 0).sum())],
}, index=["training.h5", "validation.h5", "testing.h5"])
display(resumen)
print("forma de las imágenes:", tr["images"].shape[1:], tr["images"].dtype, "| consultas LR:", test["q_images_lr"].shape[1:])

fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
cams, cuenta = np.unique(tr["camera_ids"], return_counts=True)
roles = np.isin(cams, cfg["lr_cameras"])
ax[0].bar(cams[roles], cuenta[roles], color="tab:orange", label="rol LR (se reducen x4)")
ax[0].bar(cams[~roles], cuenta[~roles], color="tab:blue", label="rol HR")
ax[0].set_title("Imágenes de entrenamiento por cámara"); ax[0].set_xlabel("cámara"); ax[0].legend()
cams_t, cuenta_t = np.unique(test["q_cams"], return_counts=True)
ax[1].bar(cams_t, cuenta_t, color="tab:green"); ax[1].set_title("Consultas de prueba por cámara"); ax[1].set_xlabel("cámara")
plt.tight_layout(); plt.show()
''')

# ------------------------------------------------------------------------------------------------
md('''
## 2. Preprocesamiento y transformaciones

Para cada imagen de entrenamiento se construyen tres versiones (todas de 128×64):

* **original** `X`;
* **SLR / LR "borrosa"** `X_deg`: reducida ×4 con bicúbica y vuelta a ampliar (lo que ve el encoder LR);
* **SHR**: la imagen reducida ×4 pasada por **Swin2SR** (lo que ve el encoder HR cuando la imagen es LR).

Cada imagen juega **un solo rol según su cámara** (el paper ata la resolución a la cámara): las de cámaras `lr_cameras`
solo se usan como par (LR, SHR) — su original nunca entra al modelo — y las demás como par (HR, SLR).

Aumentación: volteo horizontal y recorte aleatorio, aplicados **igual a las dos imágenes de cada par** (CRFF fusiona dos
vistas de la misma imagen y deben estar alineadas).
''')
code('''
# 1) La version LR que trae el dataset es EXACTAMENTE la bicubica x4 de PIL -> podemos generar las LR de entrenamiento igual
n = 200
dif = np.abs(reducir_x4(val["q_images"][:n]).astype(int) - val["q_images_lr"][:n].astype(int)).max()
print(f"diferencia máxima entre 'bicúbica x4 de PIL' y la LR del dataset (consultas de validación): {dif}")

# 2) Cuatro vistas de las mismas imágenes
shr_val_q = np.load(CACHE / "shr_val_q.npy")
idx = [0, 7, 21, 40, 63]
filas = {
    "original (128x64)": val["q_images"][idx],
    "LR del dataset (32x16)": val["q_images_lr"][idx],
    "bicúbica x4 (LR borrosa)": ampliar_x4_bicubica(val["q_images_lr"][idx]),
    "Swin2SR x4 (SHR)": shr_val_q[idx],
}
fig, ax = plt.subplots(len(filas), len(idx), figsize=(1.6 * len(idx), 3.1 * len(filas) / 2))
for r, (nombre, ims) in enumerate(filas.items()):
    for c in range(len(idx)):
        im = ims[c].transpose(1, 2, 0)
        if im.shape[0] == 32:                         # la LR de 32x16 se muestra con tamano real aumentado sin suavizar
            im = np.kron(im, np.ones((4, 4, 1), dtype=np.uint8))
        ax[r, c].imshow(im); ax[r, c].set_xticks([]); ax[r, c].set_yticks([]); ax[r, c].grid(False)
    ax[r, 0].set_ylabel(nombre, fontsize=7)
plt.tight_layout(); plt.show()
print(open(RES / "prepare_sr_log.txt", encoding="utf8").read())
''')

# ------------------------------------------------------------------------------------------------
md('''
## 3. Implementación del método del paper

El código está en `src/`. Estas son las piezas centrales, con su correspondencia con el paper:

| Pieza | Archivo | Ecuaciones del paper |
|---|---|---|
| CPG (pseudo-etiquetas entre resoluciones) | `src/cluster/cpg.py` | 4 – 9 |
| CRFF (fusión de características) | `src/models/crff.py` | 10 – 13 |
| Memoria y centroides de resolución | `src/memory.py` | 1 – 3, 14 – 15 |
| `L_cc`, `L_ic`, `L_ps` | `src/losses/drfm_losses.py` | 16, 18, 19 |
| Distancia Jaccard de k-recíprocos (nuestra) | `src/cluster/jaccard.py` | — (usada por DBSCAN) |
| Bucle de entrenamiento | `src/train/train_drfm.py` | 20 |

Pruebas con datos sintéticos (`tests/`): `L_cc` coincide con una implementación ingenua con bucles de la Ec. 16; CPG recupera
las identidades; el Jaccard por bloques coincide con el algoritmo original de re-ranking (error 1e-7).
''')
code('''
from src.cluster.cpg import generar_pseudoetiquetas
from src.models.crff import CRFF
print(inspect.getsource(generar_pseudoetiquetas))
print(inspect.getsource(CRFF))
''')
code('''
from src.losses.drfm_losses import afilado_probabilidad, consistencia_identidad, contraste_entre_resoluciones
for f in (contraste_entre_resoluciones, consistencia_identidad, afilado_probabilidad):
    print(inspect.getsource(f))
''')
code('''
import torch
from src.models.drfm import DRFM
m = DRFM(preentrenado=False)
cuenta = lambda mod: sum(p.numel() for p in mod.parameters()) / 1e6
print(f"E_h: {cuenta(m.E_h):.1f} M | E_l: {cuenta(m.E_l):.1f} M | CRFF: {cuenta(m.crff):.1f} M | Pred: {cuenta(m.pred):.1f} M | total: {cuenta(m):.1f} M parámetros")
del m
''')

# ------------------------------------------------------------------------------------------------
md('''
## 4. Entrenamiento y validación

**Protocolo.** 40 épocas de 200 iteraciones; lotes de 16 clusters × 4 imágenes = 64; Adam (lr 3.5·10⁻⁴, decaimiento 5·10⁻⁴);
`L_ps` entra después de la época 10 con λ = 1.5; temperaturas y demás hiperparámetros del paper (ver `configs/base.yaml`).
En cada época: (1) se extraen las características de todas las imágenes, (2) DBSCAN sobre las HR, (3) CPG pasa las
pseudo-etiquetas a las LR, (4) se llenan las memorias y (5) se entrena. **La configuración y la mejor época se eligen con
la partición de validación** (100 consultas, 1 793 de galería).

### 4.1 Calibración de DBSCAN con validación
El paper solo indica `min_samples = 4`. Con las imágenes de validación (donde sí conocemos las identidades) medimos la calidad de
los clusters que produce el ResNet-50 sin entrenar, por pares de imágenes (precisión = pares juntados que son la misma persona,
recall = pares de la misma persona que quedaron juntos, F = su media armónica):
''')
code('''
print(open(RES / "calibrar_eps.txt", encoding="utf8").read())
''')
md('''
Con **distancia coseno** (con o sin centrar) todas las configuraciones dan F ≤ 0.06: DBSCAN "encadena" y agrupa casi todo en un
solo cluster, porque las características de ResNet salen de una ReLU y todos los cosenos son altos (media 0.72). La
**distancia Jaccard de k-recíprocos** exige que dos imágenes sean vecinas mutuas, lo que corta el encadenamiento (F = 0.14 con
`eps = 0.55`, precisión 0.53 con `eps = 0.4`). Usamos `eps = 0.5`, un punto intermedio.

### 4.2 Primera corrida de DRFM: divergencia a NaN
La primera corrida de DRFM completo dio `Lcc = NaN` desde la época 3 y quedó estancada (mAP ≈ 10 durante 38 épocas), mientras que la
ablación sin CRFF y el modelo base entrenaban bien. La diferencia entre ambos es solo el camino de fusión. Nuestra explicación más probable: el producto
`F1(f)·F2(f)` de CRFF llegaba a ≈ 38 000 sobre imágenes de validación (el máximo de fp16 es 65 504, de modo que en entrenamiento, con
aumentación, puede desbordarse), y un NaN escrito en la memoria de centroides contamina todas las pérdidas siguientes. **No logramos
reproducir el desborde exacto** con el checkpoint final, así que esto es una hipótesis compatible con las medidas, no una demostración;
lo que sí está comprobado es que las correcciones eliminaron el NaN. Correcciones: CRFF en fp32, la memoria rechaza valores no finitos,
se descartan las iteraciones con pérdida no finita (y se cuentan), y se recorta la norma del gradiente. Las corridas corregidas registran
`saltadas` y `nofinitas` en cada época. La corrida fallida se conserva como evidencia (`drfm_historial.json`).
''')
code('''
def cargar_hist(nombre):
    return json.load(open(RES / f"{nombre}_historial.json"))

CANDIDATOS = {"base": ("Modelo base (A1)", None),
              "drfm_sin_crff": ("DRFM sin CRFF ni L_ps", "sin_fusion"),
              "drfm_v2": ("DRFM: CRFF + L_ps (λ=1.5, paper)", "fusion"),
              "drfm_sin_ps": ("DRFM: CRFF, sin L_ps", "fusion"),
              "drfm_lps03": ("DRFM: CRFF + L_ps (λ=0.3)", "fusion"),
              "drfm_detach": ("DRFM: CRFF sin gradiente a los encoders", "fusion")}
H = {nombre: (arch, clave) for arch, (nombre, clave) in CANDIDATOS.items() if (RES / f"{arch}_historial.json").exists()}
hist = {n: cargar_hist(f) for n, (f, _) in H.items()}

def serie(nombre, cond):
    f, clave = H[nombre]
    h = [e for e in hist[nombre] if "val" in e]
    v = [(e["val"][clave] if clave else e["val"])[cond]["mAP"] for e in h]
    return [e["epoca"] for e in h], v

COLOR = {n: f"C{i}" for i, n in enumerate(H)}          # un color fijo por modelo en los tres graficos
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for n in H:
    ax[0].plot(*serie(n, "original"), label=n, color=COLOR[n]); ax[1].plot(*serie(n, "baja_resolucion"), label=n, color=COLOR[n])
ax[0].set_title("mAP en validación — consultas originales"); ax[1].set_title("mAP en validación — consultas de baja resolución")
for a in ax[:2]: a.set_xlabel("época"); a.set_ylabel("mAP (%)"); a.legend(fontsize=8)
for n in [n for n in H if n.startswith("DRFM")]:
    h = [e for e in hist[n] if "val_clusters" in e]
    ax[2].plot([e["epoca"] for e in h], [e["val_clusters"]["F"] for e in h], label=n, color=COLOR[n])
ax[2].set_title("Calidad de los clusters (F por pares, validación)"); ax[2].set_xlabel("época"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()

# primera corrida fallida (NaN) para comparar
h1 = [e for e in cargar_hist("drfm") if "val" in e]
print("Primera corrida de DRFM (NaN): épocas con Lcc = NaN:", sum(1 for e in cargar_hist("drfm") if e.get("L_cc") != e.get("L_cc")), "de 40;",
      "mAP original máximo en validación:", round(max(e["val"]["fusion"]["original"]["mAP"] for e in h1), 1))
''')
code('''
filas = []
for n, (f, clave) in H.items():
    h = [e for e in hist[n] if "val" in e]
    mejor = max(h, key=lambda e: 0.5 * ((e["val"][clave] if clave else e["val"])["original"]["mAP"] + (e["val"][clave] if clave else e["val"])["baja_resolucion"]["mAP"]))
    v = mejor["val"][clave] if clave else mejor["val"]
    filas.append({"modelo": n, "mejor época": mejor["epoca"],
                  "mAP orig": v["original"]["mAP"], "R1 orig": v["original"]["Rank-1"],
                  "mAP LR": v["baja_resolucion"]["mAP"], "R1 LR": v["baja_resolucion"]["Rank-1"], "caída mAP": v["caida_mAP"]})
zs = json.load(open(RES / "zero_shot_validation.json"))
filas.insert(0, {"modelo": "ResNet-50 sin entrenar", "mejor época": "-", "mAP orig": zs["original"]["mAP"], "R1 orig": zs["original"]["Rank-1"],
                 "mAP LR": zs["baja_resolucion"]["mAP"], "R1 LR": zs["baja_resolucion"]["Rank-1"], "caída mAP": zs["caida_mAP"]})
print("Validación (100 consultas, 1 793 de galería) — se elige la época con mejor mAP medio de las dos condiciones")
display(pd.DataFrame(filas).round(1).set_index("modelo"))
''')

# ------------------------------------------------------------------------------------------------
md('''
## 5. Evaluación en la partición de prueba (ambas condiciones)

Mismo modelo entrenado para las dos condiciones, vectores normalizados L2 y similitud coseno. Condición **original**:
consultas y galería en resolución original. Condición **baja resolución**: consultas de 32×16 (con la galería original).
La partición de prueba se evaluó **una sola vez**, después de fijar la configuración y la mejor época con validación.
Protocolo estándar de Market-1501: se descartan de la galería las imágenes *junk* y las de la misma persona y misma cámara que la consulta.
''')
code('''
R = json.load(open(RES / "resultados_testing.json"))
META = R.pop("_meta")
PRINCIPAL = META["principal"]                       # el DRFM elegido con validacion
TAGS = {"drfm_v2": "CRFF + L_ps (λ=1.5, paper)", "drfm_sin_ps": "CRFF, sin L_ps", "drfm_lps03": "CRFF + L_ps (λ=0.3)",
        "drfm_detach": "CRFF sin gradiente a los encoders", "drfm_sin_crff": "sin CRFF ni L_ps"}
NOMBRES = {"zero_shot": "ResNet-50 sin entrenar", "base": "Modelo base (A1)"}
for k in R:
    if k.endswith("__fusion"):
        t = k[:-8]; NOMBRES[k] = f"DRFM {TAGS.get(t, t)}" + (" ★ elegido" if t == PRINCIPAL else "") + " — con fusión"
    elif k.endswith("__sin_fusion"):
        t = k[:-12]; NOMBRES[k] = f"DRFM {TAGS.get(t, t)}" + (" ★ elegido" if t == PRINCIPAL else "") + (" — sin fusión (solo encoder HR)" if t != "drfm_sin_crff" else "")
filas = []
for k, nombre in NOMBRES.items():
    if k not in R: continue
    o, l = R[k]["original"], R[k]["baja_resolucion"]
    filas.append({"modelo": nombre, "mAP": o["mAP"], "Rank-1": o["Rank-1"], "Rank-5": o["Rank-5"], "Rank-10": o["Rank-10"],
                  "mAP (LR)": l["mAP"], "Rank-1 (LR)": l["Rank-1"], "Rank-5 (LR)": l["Rank-5"], "Rank-10 (LR)": l["Rank-10"],
                  "caída de mAP": R[k]["caida_mAP"]})
tabla = pd.DataFrame(filas).set_index("modelo")
print(f"PRUEBA: {R['zero_shot']['original']['n_consultas']} consultas | columnas sin sufijo = imágenes originales; (LR) = consultas de baja resolución")
display(tabla.round(1))

def corto(k):
    """Etiqueta corta para los graficos."""
    if k == "zero_shot": return "Sin entrenar (ImageNet)"
    if k == "base": return "Modelo base (A1)"
    t, v = k.rsplit("__", 1)
    et = {"drfm_v2": "λ=1.5 (paper)", "drfm_sin_ps": "CRFF sin L_ps", "drfm_lps03": "λ=0.3 ★",
          "drfm_detach": "CRFF sin grad. a enc.", "drfm_sin_crff": "sin CRFF ni L_ps"}.get(t, t)
    return f"DRFM {et}" + (" · fusión" if v == "fusion" else (" · enc. HR" if t != "drfm_sin_crff" else ""))
CLAVES = [k for k in NOMBRES if k in R]
fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(CLAVES))[::-1]; h = 0.38
ax.barh(y + h / 2, [R[k]["original"]["mAP"] for k in CLAVES], h, label="consultas originales")
ax.barh(y - h / 2, [R[k]["baja_resolucion"]["mAP"] for k in CLAVES], h, label="consultas de baja resolución")
ax.set_yticks(y); ax.set_yticklabels([corto(k) for k in CLAVES], fontsize=8); ax.set_xlabel("mAP (%)")
ax.set_title("mAP en la partición de prueba"); ax.legend(); plt.tight_layout(); plt.show()
''')
code('''
det = np.load(RES / "detalle_testing.npz")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
for cond, a in zip(("original", "baja_resolucion"), ax):
    for i, k in enumerate(CLAVES):
        if f"{k}__{cond}__primer_acierto" not in det: continue
        p = det[f"{k}__{cond}__primer_acierto"]
        cmc = [100 * np.mean(p < r) for r in range(1, 21)]
        a.plot(range(1, 21), cmc, marker="o", ms=3, color=plt.cm.tab20(i), label=corto(k))
    a.set_title("Curva CMC — " + ("consultas originales" if cond == "original" else "consultas de baja resolución"))
    a.set_xlabel("rango k"); a.set_ylabel("% de consultas con acierto en el top-k"); a.set_xticks([1, 5, 10, 15, 20])
ax[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5)); plt.tight_layout(); plt.show()
''')

# ------------------------------------------------------------------------------------------------
md('''
## 6. Comparación con el modelo base y estudio de ablación

**Modelo base** = el paso A1 de la Tabla 3 del paper: un solo encoder, DBSCAN directo sobre *todas* las imágenes (HR y LR
mezcladas, sin CPG) y pérdida softmax sobre las pseudo-etiquetas; mismo presupuesto de entrenamiento que DRFM.

**Ablación.** Componente estudiado: **CRFF** (la fusión). Se entrena DRFM *sin* CRFF: se mantienen CPG, `L_cc` (en los espacios HR y LR) y `L_ic`;
se quitan el módulo de fusión y, con él, `L_ps` (que opera sobre la característica fusionada). Además, evaluando el DRFM completo
usando solo el encoder HR (sin fusión) se separa el efecto de la fusión **en la inferencia** del efecto de haber **entrenado** con ella.
''')
code('''
t = tabla
base = t.loc["Modelo base (A1)"]
mejora = (t[["mAP", "Rank-1", "mAP (LR)", "Rank-1 (LR)"]] - base[["mAP", "Rank-1", "mAP (LR)", "Rank-1 (LR)"]]).round(1)
mejora.columns = ["Δ mAP", "Δ Rank-1", "Δ mAP (LR)", "Δ Rank-1 (LR)"]
print("Diferencia frente al modelo base (puntos porcentuales):")
display(mejora)
''')

# ------------------------------------------------------------------------------------------------
md('''
## 7. Análisis de errores

Se analiza el **DRFM elegido con validación** (con fusión) en la condición de **baja resolución** (la más difícil).
''')
code('''
M = f"{PRINCIPAL}__fusion"
ap_o, ap_l = det[f"{M}__original__ap"], det[f"{M}__baja_resolucion__ap"]
pr_o, pr_l = det[f"{M}__original__primer_acierto"], det[f"{M}__baja_resolucion__primer_acierto"]
top_l = det[f"{M}__baja_resolucion__top"]
q_pids, q_cams, g_pids, g_cams = test["q_pids"], test["q_cams"], test["g_pids"], test["g_cams"]

ok_o, ok_l = pr_o == 0, pr_l == 0
print(f"Rank-1 original: {100 * ok_o.mean():.1f}% | Rank-1 baja resolución: {100 * ok_l.mean():.1f}%")
print(pd.DataFrame({"consultas": [int((ok_o & ok_l).sum()), int((ok_o & ~ok_l).sum()), int((~ok_o & ok_l).sum()), int((~ok_o & ~ok_l).sum())]},
                   index=["aciertan en ambas", "aciertan solo con original (la baja resolución las rompe)", "aciertan solo con baja resolución", "fallan en ambas"]))

# Rank-1 por camara de la consulta
por_cam = pd.DataFrame({"consultas": [int((q_cams == c).sum()) for c in range(1, 7)],
                        "Rank-1 original": [100 * ok_o[q_cams == c].mean() for c in range(1, 7)],
                        "Rank-1 LR": [100 * ok_l[q_cams == c].mean() for c in range(1, 7)],
                        "mAP LR": [100 * ap_l[q_cams == c].mean() for c in range(1, 7)]}, index=range(1, 7)).round(1)
por_cam.index.name = "cámara de la consulta"
display(por_cam)

# ¿Qué es lo que se recupera en primer lugar cuando se falla? (condición baja resolución)
falla = ~ok_l
t1 = top_l[falla, 0]
tipo = np.where(g_pids[t1] == 0, "distractor (persona fuera de las 750)", "otra persona")
misma_cam = g_cams[t1] == q_cams[falla]
print("\\nCuando el top-1 es incorrecto (baja resolución):")
print(pd.Series(tipo).value_counts().to_string())
print(f"  top-1 incorrecto tomado de la MISMA cámara que la consulta: {100 * misma_cam.mean():.1f}%")
print(f"  consultas sin ningún acierto en el top-10: {100 * (pr_l >= 10).mean():.1f}% ({int((pr_l >= 10).sum())})")
''')
code('''
rng = np.random.default_rng(7)
def elegir(mask, n=5):
    """Elige n consultas de `mask` con camaras distintas."""
    idx, usadas = [], set()
    for i in rng.permutation(np.flatnonzero(mask)):
        if q_cams[i] not in usadas:
            idx.append(int(i)); usadas.add(q_cams[i])
        if len(idx) == n: break
    return idx

aciertos = elegir(ok_l & (ap_l >= 0.9))
fallos = elegir(pr_l >= 10)
print("aciertos (consulta, cámara):", [(i, int(q_cams[i])) for i in aciertos])
print("fallos   (consulta, cámara):", [(i, int(q_cams[i])) for i in fallos])

def mosaico(casos, titulo):
    fig, ax = plt.subplots(len(casos), 7, figsize=(8.6, 2.3 * len(casos)))
    for r, i in enumerate(casos):
        q_lr = np.kron(test["q_images_lr"][i].transpose(1, 2, 0), np.ones((4, 4, 1), dtype=np.uint8))
        celdas = [(q_lr, "consulta LR", "tab:blue"), (test["q_images"][i].transpose(1, 2, 0), f"original (id {q_pids[i]})", "tab:blue")]
        for j in top_l[i, :5]:
            if g_pids[j] == q_pids[i] and g_cams[j] != q_cams[i]: col, txt = "tab:green", f"id {g_pids[j]} ✓"
            elif g_pids[j] == 0: col, txt = "tab:orange", "distractor"
            else: col, txt = "tab:red", f"id {g_pids[j]} ✗"
            celdas.append((test["g_images"][j].transpose(1, 2, 0), txt, col))
        for c, (im, txt, col) in enumerate(celdas):
            ax[r, c].imshow(im); ax[r, c].grid(False); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for s in ax[r, c].spines.values(): s.set_edgecolor(col); s.set_linewidth(3)
            ax[r, c].set_title(txt, fontsize=7)
        ax[r, 0].set_ylabel(f"#{i} cam {q_cams[i]}\\nAP={ap_l[i]:.2f}", fontsize=8)
    fig.suptitle(titulo, fontsize=10); plt.tight_layout(); plt.show()

mosaico(aciertos, "5 ACIERTOS (consulta de baja resolución, DRFM): verde = misma persona en otra cámara")
mosaico(fallos, "5 FALLOS (ninguna coincidencia correcta en el top-10): rojo = otra persona, naranja = distractor")
''')
md(texto_casos)

# ------------------------------------------------------------------------------------------------
md('''
## 8. Limitaciones y conclusiones
''')
md((RAIZ / "notebooks" / "conclusiones.md").read_text(encoding="utf8") if (RAIZ / "notebooks" / "conclusiones.md").exists()
   else "_(se completa con los resultados finales)_")

# ------------------------------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "lab3venv", "display_name": "lab3venv", "language": "python"}
nb.cells = [nbf.v4.new_markdown_cell(c) if t == "md" else nbf.v4.new_code_cell(c) for t, c in C]
destino = RAIZ / "Lab3_final.ipynb"
nbf.write(nb, destino)
print("notebook escrito en", destino, f"({len(nb.cells)} celdas)")
