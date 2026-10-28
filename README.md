# Laboratorio 3 — Re-identificación de personas no supervisada con baja resolución (DRFM)

Implementación propia (PyTorch) del método **DRFM** — *Dual-Resolution Fusion Modeling for Unsupervised Cross-Resolution
Person Re-Identification* (Pang, Zhao, Wang; ACM MM 2024) — adaptado a Market-1501, con modelo base, estudio de ablación y
evaluación en dos condiciones (consultas originales y consultas de baja resolución 32×16).

El resultado completo (tablas, curvas, análisis de errores con 5 aciertos y 5 fallos) está en **`Lab3_final.ipynb`**.

## Estructura

```
configs/base.yaml          hiperparámetros del paper y decisiones propias (comentadas)
src/data/h5data.py         lectura de los .h5 y degradación bicúbica x4 (idéntica a la del dataset)
src/data/dataset_drfm.py   las 3 versiones de cada imagen, roles por cámara, aumentación en pares, muestreo 16x4
src/cluster/jaccard.py     distancia Jaccard de k-recíprocos (propia, por bloques)
src/cluster/cpg.py         DBSCAN + CPG (pseudo-etiquetas entre resoluciones, Ec. 4-9)
src/models/{encoder,crff,drfm}.py   ResNet-50, CRFF (Ec. 10-13), contenedor DRFM
src/memory.py              memoria y centroides de resolución (Ec. 1-3, 14-15)
src/losses/drfm_losses.py  L_cc, L_ic, L_ps (Ec. 16, 18, 19)
src/train/train_drfm.py    entrenamiento de DRFM (y de la ablación con --sin-crff)
src/train/train_baseline.py  modelo base (paso A1 del paper)
src/eval/                  métricas (mAP, CMC; protocolo estándar de Market-1501) y evaluación en las dos condiciones
scripts/                   preproceso SR, calibración de DBSCAN, referencia sin entrenar, evaluación final
slurm/                     scripts de Khipu (SLURM)
tests/                     pruebas con datos sintéticos (python -m tests.test_componentes / tests.test_jaccard)
```

## Cómo ejecutar (Khipu, desde `~/lab3`)

```bash
bash slurm/env_setup.sh                     # venv con módulo python3/3.11.11 + descarga de pesos a cache/
sbatch slurm/prepare_sr_cpu.sbatch          # super-resolución Swin2SR x4 (cache/shr_*.npy), ~17 min en CPU
python -m scripts.calibrar_eps              # calidad de DBSCAN con validación (identidades solo para elegir eps)
python -m src.train.train_drfm --tag drfm_v2                    # DRFM completo, ~35 min en A100
python -m src.train.train_drfm --tag drfm_sin_crff --sin-crff   # ablación (sin CRFF ni L_ps)
python -m src.train.train_baseline --tag base                   # modelo base
sbatch slurm/evaluar_final.sbatch           # métricas en validación y prueba (una sola vez)
sbatch slurm/notebook.sbatch                # ejecuta Lab3_final.ipynb
```
Pruebas rápidas del código (CPU, segundos): `python -m tests.test_componentes && python -m tests.test_jaccard`.

## Decisiones y diferencias respecto al paper

Ver la tabla de la sección 0 del cuaderno. Las más importantes: roles LR/HR por cámara (cámaras 1, 2, 5 = LR);
distancia **Jaccard de k-recíprocos** para DBSCAN (con distancia coseno el clustering falla: F ≤ 0.06); CRFF y la
memoria protegidas contra desbordes numéricos tras una primera corrida que divergió a NaN; los modelos se entrenan solo
con `training.h5` y la mejor época se elige con validación.

## Resultados (partición de prueba: 3 368 consultas, 19 732 de galería)

| Modelo | mAP orig. | R1 orig. | mAP LR | R1 LR | caída de mAP |
|---|---|---|---|---|---|
| ResNet-50 ImageNet sin entrenar | 2.5 | 7.8 | 1.4 | 4.2 | 1.1 |
| Modelo base (paso A1 del paper) | 13.6 | 30.0 | 8.3 | 14.3 | 5.4 |
| **DRFM completo (λ = 0.3, elegido con validación)** | 13.2 | 28.1 | 12.7 | 25.8 | **0.5** |
| DRFM completo con λ = 1.5 (valor del paper) | 12.3 | 27.4 | 11.5 | 24.2 | 0.9 |
| DRFM sin CRFF ni L_ps | **19.0** | **38.2** | **18.6** | **36.8** | 0.4 |

DRFM elimina casi por completo la caída con consultas de baja resolución. En nuestra adaptación, CRFF y L_ps no mejoraron
el resultado (ver ablación y diagnóstico en el cuaderno, secciones 4 y 6). Resultados completos: `results/resultados_testing.json`;
puntos de control: `khipu:~/lab3/ckpts/` (`drfm_lps03_best.pt`, `drfm_sin_crff_best.pt`, `base_best.pt`, ...).

Otros archivos: `guia_estudio.md` (mapa paper → código y preguntas probables), `guion_video.md`, `results/logs/` (registros de entrenamiento, incluida la corrida inicial que divergió a NaN).

## Datos (no incluidos en este repositorio)

Market-1501 se distribuye solo para investigación y **no se redistribuye aquí**. Para reproducir, colocar `training.h5`,
`validation.h5` y `testing.h5` (los entregados por el curso) directamente en `data/`, y generar la super-resolución con
`slurm/prepare_sr_cpu.sbatch` (crea `cache/shr_*.npy`). El paper (PDF) y el enunciado tampoco se incluyen por derechos de autor.
Los puntos de control entrenados están en Khipu (`~/lab3/ckpts/`) y no se suben por su tamaño.
