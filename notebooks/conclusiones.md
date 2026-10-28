### Resultados principales (partición de prueba: 3 368 consultas, 19 732 de galería)

| | mAP orig. | Rank-1 orig. | mAP LR | Rank-1 LR | caída de mAP |
|---|---|---|---|---|---|
| ResNet-50 de ImageNet sin entrenar | 2.5 | 7.8 | 1.4 | 4.2 | 1.1 |
| Modelo base (A1) | 13.6 | 30.0 | 8.3 | 14.3 | **5.4** |
| **DRFM completo (λ = 0.3, elegido con validación)** | 13.2 | 28.1 | **12.7** | **25.8** | **0.5** |
| DRFM completo con el λ = 1.5 del paper | 12.3 | 27.4 | 11.5 | 24.2 | 0.9 |
| DRFM sin CRFF ni `L_ps` (mejor configuración) | **19.0** | **38.2** | **18.6** | **36.8** | 0.4 |

* **Efecto principal reproducido:** el modelo base sufre mucho con la baja resolución (cae de 13.6 a 8.3 de mAP, −39 % relativo, y de 30.0 a 14.3 de Rank-1) y **todas las variantes de DRFM casi no la sufren** (caída ≤ 1.8 puntos). En consultas de baja resolución, el DRFM completo mejora al modelo base en +4.4 de mAP y +11.5 de Rank-1; la mejor configuración, en +10.3 y +22.4.
* **Con consultas originales** el DRFM completo *empata* con el modelo base (−0.4 de mAP); la mejora grande solo aparece en la configuración sin CRFF ni `L_ps` (+5.3 de mAP).
* **No reproducimos que CRFF y `L_ps` mejoren el resultado.** Los experimentos de ablación (validación y prueba coinciden en el orden) muestran: (1) `L_ps` con el λ = 1.5 del paper desestabiliza el entrenamiento (el mAP de validación se desploma en la época 26; con λ = 0.3 o sin `L_ps` no ocurre); (2) añadir CRFF empeora los encoders, porque el gradiente de la pérdida del espacio de fusión interfiere con ellos (si se corta ese gradiente, los encoders recuperan su nivel: 20.7 / 18.9 de mAP); (3) aun así, el vector fusionado **no supera** al del encoder solo (17.8 / 17.1 frente a 20.7 / 18.9). Lo que sí aporta es la combinación **CPG + `L_cc` + `L_ic`** (la configuración sin CRFF ni `L_ps`); no aislamos el efecto de cada uno de esos tres por separado.
* **Los valores absolutos** de mAP son bajos porque el protocolo es exigente (750 identidades, galería de 19 732 con distractores y *junk*) y porque el entrenamiento es no supervisado; no son comparables con los de la Tabla 2 del paper, que usa otro protocolo (una sola imagen HR por persona en la galería) y solo reporta Rank-k.

### Limitaciones

* **Una sola semilla** por configuración y una **validación de solo 100 consultas**: diferencias de 2–3 puntos en validación no son concluyentes. Las diferencias grandes (el colapso de `L_ps`, el efecto del modelo base, la ventaja de la ablación) son consistentes entre validación y prueba.
* El paper **no especifica** varios detalles (distancia y `eps` de DBSCAN, iteraciones por época, schedule, qué vector se usa en test); son decisiones nuestras. Solo se exploró λ ∈ {1.5, 0.3} y se calibró `eps` con características sin entrenar; no hubo búsqueda sistemática de hiperparámetros.
* La relación **CRFF + gradiente** se estudió con un único diagnóstico; no se probaron otras alternativas (por ejemplo normalizar las entradas de CRFF o reducir el peso de la pérdida de fusión), que podrían dar a la fusión un papel positivo.
* Los modelos se entrenaron solo con `training.h5`; el protocolo permitiría reentrenar con las imágenes de validación sin etiquetas.
* **Errores típicos del método** (sección 7): ropa de colores neutros o con detalles finos que la reducción ×4 destruye, personas vestidas casi igual, la cámara 6 (baja resolución nativa), imágenes muy oscuras y confusión con personas de la misma cámara (43 % de los primeros resultados incorrectos).

### Conclusión
La idea central del paper —agrupar solo lo nítido y pasar las etiquetas a lo borroso mirando también su versión super-resuelta— **funciona**: elimina casi por completo la caída de rendimiento con consultas de baja resolución y supera al modelo base en esa condición (de +4 a +10 puntos de mAP según la variante). En nuestra adaptación, sin embargo, los módulos de fusión (CRFF) y de afilado (`L_ps`) no aportaron mejora medible y, tal como los implementamos, perjudicaron el entrenamiento.
