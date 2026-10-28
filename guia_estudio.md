# Guía de estudio — Lab 3 (DRFM)

## 1. El método en 10 líneas
1. Queremos re-identificar personas **sin etiquetas de identidad**, y que funcione aunque la consulta sea de **baja resolución**.
2. Los métodos no supervisados fabrican etiquetas falsas ("pseudo-etiquetas") **agrupando** imágenes con DBSCAN. Si mezclas fotos nítidas y
   borrosas en el mismo clustering, los grupos salen ruidosos. El paper evita eso:
3. **Agrupa solo las nítidas (HR)** y de ahí "pasa" las etiquetas a las borrosas (LR) → **CPG**.
4. Para pasar la etiqueta mira la imagen LR **y** su versión super-resuelta (SHR) y exige que **las dos coincidan** en el mismo cluster
   (multiplica sus probabilidades). Solo se queda con las más seguras (fracción *T*).
5. Hay **dos encoders** ResNet-50: uno para imágenes nítidas (HR y SHR) y otro para borrosas (LR y la versión reducida SLR).
6. **CRFF** fusiona las dos características de la misma imagen (la del encoder HR y la del LR) en un solo vector: el vector "invariante a la resolución".
7. **`L_cc`**: acerca cada vector al **centroide** de su cluster y lo aleja de los 20 centroides de otros clusters más parecidos.
8. **`L_ic`**: el encoder HR, mirando la imagen super-resuelta, debe parecerse (coseno) a lo que ve el encoder LR en la imagen original (con *stop-gradient*).
9. **`L_ps`**: para las imágenes ambiguas (sin pseudo-etiqueta) no sabemos su clase, pero sí cuáles clases **no** son; se "afila" su distribución.
10. En cada época se repite: extraer características → clustering → CPG → memorias → entrenar.

## 2. Mapa ecuación → código

| Ecuación del paper | Qué hace | Dónde está |
|---|---|---|
| 1–3 | memoria con promedio móvil `v ← αv + (1−α)f` | `src/memory.py::Memoria.actualizar` |
| 4–7 | vectores de clase y centroides | `src/cluster/cpg.py::generar_pseudoetiquetas`, `centroides_por_cluster` |
| 8 | producto de los dos vectores de clase (Hadamard) y renormalización | `generar_pseudoetiquetas` (suma de log-softmax y `softmax`) |
| 9 | `T = N_c / N_1 · t`: fracción que se conserva | `generar_pseudoetiquetas` |
| 10–13 | CRFF | `src/models/crff.py` |
| 14–15 | centroides de resolución (por cluster y por tipo) | `Memoria.centroides` |
| 16 | `L_cc` | `src/losses/drfm_losses.py::contraste_entre_resoluciones` |
| 18 | `L_ic` | `consistencia_identidad` |
| 19 | `L_ps` | `afilado_probabilidad` |
| 20 | pérdida total `L_cc + L_ic + λ L_ps` | `src/train/train_drfm.py` (bucle de iteraciones) |

## 3. Una época, paso a paso (`src/train/train_drfm.py`)
1. **Extraer** características de todo el conjunto con los modelos en modo `eval` y sin aumentación: `f_h`, `f_sl` (grupo HR) y `f_l`, `f_sh` (grupo LR), y las fusionadas.
2. **DBSCAN** sobre `f_h` con distancia Jaccard → `etq_h` (−1 = ruido).
3. **CPG** → `etq_l` para las LR (−1 = ambigua).
4. Con las imágenes etiquetadas se arman **tres memorias**: espacio HR (`f_h`, `f_sh`), espacio LR (`f_sl`, `f_l`) y espacio de fusión (`f_hsl`, `f_lsh`). Cada entrada tiene un *tipo* (0 = viene de una imagen HR, 1 = de una LR).
5. **200 iteraciones**: se toman 16 clusters × 4 imágenes; se calculan las características con aumentación; `L_cc` en los 3 espacios + `L_ic` (+ `L_ps` con un lote aparte de 16 ambiguas, desde la época 10); backprop; se actualizan las memorias.
6. Validación (mAP/Rank-k en las 2 condiciones) y guardado del mejor checkpoint.

## 4. Decisiones nuestras y por qué
* **Roles por cámara** (cámaras 1, 2, 5 = LR): el paper ata la resolución a la cámara; así una misma persona aparece en HR y LR. La imagen original de las del grupo LR nunca entra al modelo.
* **LR = bicúbica ×4 de PIL**: comprobamos que reproduce *exactamente* (diferencia 0) las consultas LR del dataset.
* **Swin2SR clásico ×4**: entrenado con degradación bicúbica; sube el PSNR 2.5 dB frente a ampliar con bicúbica.
* **Distancia Jaccard para DBSCAN**: con coseno, DBSCAN junta todo en un cluster (F ≤ 0.06). Jaccard exige vecindad **mutua**. La implementamos por bloques (cabe en GPU chica) y la verificamos contra el algoritmo original con bucles (error 1e-7).
* **CRFF en fp32**, memoria que rechaza no-finitos, recorte de gradiente: la primera corrida divergió a NaN.
* **Mejor época por validación** (media del mAP de las dos condiciones). El test se evalúa una sola vez.

## 5. Preguntas probables (y respuesta corta)

**¿Qué hace DBSCAN y qué son `eps` y `min_samples`?** Agrupa puntos densos: un punto es "núcleo" si tiene al menos `min_samples` (4) vecinos a distancia ≤ `eps`; los clusters crecen conectando núcleos; lo que no cae en ninguno es ruido (etiqueta −1). No hay que decirle cuántos clusters.

**¿Por qué no coseno?** Medimos la calidad de los clusters en validación (donde conocemos las personas): con coseno F ≤ 0.06, con Jaccard 0.14 en el mejor `eps`. Las características de ResNet son positivas (vienen de ReLU), todos los cosenos son altos (media 0.72) y DBSCAN encadena.

**¿Qué es la distancia Jaccard de k-recíprocos?** Dos imágenes son vecinas si cada una está entre los *k1* más cercanos de la otra; cada imagen se describe por el conjunto ponderado de sus vecinos y la distancia es 1 − intersección/unión de esos conjuntos. Es más robusta que la distancia directa.

**¿Cómo se pasan las etiquetas a las LR (CPG)?** Se calcula a qué cluster se parece la LR (con el encoder LR contra centroides de las SLR) y a cuál su SHR (con el encoder HR contra centroides de las HR). Se multiplican: si ambas apuntan al mismo cluster el producto es grande y la muestra es confiable. Se conserva la fracción `T` más confiable; el resto queda ambigua.

**¿Qué es un centroide de resolución?** El promedio de las características de un cluster **separado por tipo** (HR o LR). Un cluster puede tener hasta 2; ambos son positivos de `L_cc`.

**¿Qué hace `L_cc`?** Es un InfoNCE: maximiza la probabilidad de los centroides positivos del propio cluster frente a ellos más los 20 negativos más cercanos de otros clusters (temperatura 0.04). Usa centroides (de la memoria) en vez de muestras.

**¿Para qué el *stop-gradient* en `L_ic`?** Evita el colapso (todas las salidas iguales): como en SimSiam, un lado de la comparación no recibe gradiente.

**¿Por qué `L_ps` tiene dos temperaturas?** `τ_s < τ_c`: con `τ_s` la distribución objetivo es más "afilada" (sin gradiente) y se obliga a la normal a parecerse; así bajan las probabilidades de las clases claramente incorrectas.

**¿Cómo se evalúa?** Vectores normalizados L2 y coseno; para cada consulta se ordena la galería, se descartan *junk* (id −1) y las imágenes de la misma persona **en la misma cámara**; **mAP** = promedio de la precisión media; **Rank-k** = % de consultas con un acierto entre las k primeras.

**¿Qué vector se usa en test?** Galería y consultas originales: `f_hsl` = CRFF(E_h(imagen), E_l(su reducida)). Consultas LR: `f_lsh` = CRFF(E_h(SR(consulta)), E_l(consulta)). (El paper no lo detalla: lo inferimos de la Sec. 3.3.)

**¿Qué pasó con el NaN?** La corrida inicial dio `Lcc = NaN` desde la época 3. Nuestra explicación más probable: el producto `F1(f)·F2(f)` llegaba a ~38 000 en validación (límite fp16 = 65 504), puede desbordarse en entrenamiento, y un NaN escrito en la memoria contamina todas las pérdidas. **No pudimos reproducir el desborde exacto** (es una hipótesis); lo comprobado es que el NaN desapareció al aplicar CRFF en fp32, memoria que rechaza no-finitos, descarte de iteraciones no finitas y recorte de gradiente.

**¿Qué es AMP / GradScaler?** Entrenamiento en media precisión (más rápido, menos memoria); el `GradScaler` multiplica la pérdida para evitar que los gradientes pequeños se redondeen a 0 y omite el paso si aparecen inf/NaN.

## 6. Preguntas sobre NUESTROS hallazgos (validación; 100 consultas → ruido de ±2–3 puntos)

Resumen de lo medido (mAP en validación, originales / baja resolución, en la mejor época):
modelo base **36.4 / 28.1** · DRFM sin CRFF ni `L_ps` **42.4 / 43.0** · DRFM completo λ=1.5 (paper) 32.8 / 31.6 ·
DRFM completo λ=0.3 35.6 / 35.2 · CRFF sin `L_ps` 33.9 / 31.3 · CRFF sin gradiente a los encoders 42.3 / 41.1 (con fusión) y 45.0 / 41.9 (solo encoder).

**¿Por qué el DRFM completo no supera a su ablación sin CRFF?** Tres hechos medidos: (1) con CRFF y `L_ps` λ=1.5 el entrenamiento **se desploma
en la época 26** (la calidad de los clusters cae de F=0.40 a ≈0.26 y las pseudo-etiquetas se degradan en cadena); (2) sin `L_ps` no hay colapso,
pero el modelo igual queda peor que sin CRFF; (3) si cortamos el gradiente que va de CRFF a los encoders (diagnóstico), los encoders recuperan su
calidad (45.0 / 41.9) → **el gradiente de la pérdida del espacio de fusión interfiere con los encoders**. Aun así la característica fusionada no
supera al encoder solo (42.3 / 41.1 frente a 45.0 / 41.9).

**¿Por qué colapsó `L_ps`?** Es una hipótesis, no algo demostrado: `L_ps` es auto-entrenamiento (la distribución "afilada" es el objetivo de sí misma) y con
λ=1.5 refuerza también las equivocaciones sobre las muestras ambiguas, lo que desplaza los centroides y arrastra al clustering. Con λ=0.3 no colapsa
y rinde mejor (35.6 / 35.2 frente a 32.8 / 31.6), por eso el DRFM "titular" usa λ=0.3, elegido con validación.

**¿Cómo saben que el gradiente de la fusión es el problema?** Por el experimento `--crff-detach`: mismo modelo, mismo entrenamiento, pero CRFF recibe las
características sin gradiente hacia los encoders. Es una evidencia fuerte pero de una sola semilla y 100 consultas de validación.

**¿Por qué el modelo base cae tanto con baja resolución (−8 de mAP) y DRFM casi nada?** El modelo base mezcla fotos nítidas y borrosas en un mismo clustering y
un mismo encoder; la consulta borrosa queda en otra zona del espacio. DRFM entrena la rama LR con imágenes super-resueltas y exige coincidencia entre ambas
vistas, y al evaluar pasa la consulta por la super-resolución.

**¿Qué limitaciones tienen los resultados?** Una sola semilla; validación con solo 100 consultas (diferencias de 2–3 puntos no son concluyentes); el paper no da
`eps`, distancia ni iteraciones por época, así que son decisiones nuestras; en test se evalúa una sola vez; el modelo se entrenó solo con `training.h5`.
