# Guion del video — Lab 3 (máx. 18 min: 6 + 6 + 6)

Tono: conversacional, frases cortas. Entre corchetes, qué proyectar.

---

# METODOLOGÍA (6 min)

**[0:00 — el problema]**
Nuestro laboratorio es de **re-identificación de personas**. Te doy la foto de alguien captada por una cámara, y tengo que encontrar a esa misma persona en fotos de otras cámaras. La gracia es doble: **no podemos usar etiquetas** de quién es quién para entrenar, y además la foto de consulta puede venir en **muy baja resolución**.
**[PROYECTAR: montaje con original / LR 32×16 / SHR]**

**[0:50 — la idea del paper]**
El paper que nos tocó se llama **DRFM**. Los métodos sin etiquetas funcionan así: agrupan las fotos parecidas y tratan cada grupo como si fuera una persona, o sea, fabrican **etiquetas falsas**. Pero hay un problema: si mezclas fotos nítidas con fotos borrosas, los grupos salen mal. DRFM resuelve eso con tres ideas.
**[PROYECTAR: Figura 2 del paper]**

**[1:40 — preparar las imágenes]**
Primero, de cada foto nítida hacemos una versión **reducida**. Y de cada foto borrosa hacemos una versión **super-resuelta** con una red de super-resolución. Así tenemos cuatro tipos de imagen y **dos redes**: una para lo nítido y otra para lo borroso.

**[2:30 — idea 1: CPG, las etiquetas falsas]**
**[PROYECTAR: src/cluster/cpg.py, generar_pseudoetiquetas]**
Primero agrupamos **solo las fotos nítidas**, que son las más confiables. Para ponerle etiqueta a una foto borrosa la miramos de dos maneras: tal cual, y ya super-resuelta. Cada una nos dice "me parezco a este grupo". **Multiplicamos las dos respuestas**: si coinciden, la etiqueta es de fiar. Solo nos quedamos con las más seguras; las dudosas quedan como "ambiguas".

**[3:40 — idea 2: CRFF, fusionar]**
**[PROYECTAR: src/models/crff.py]**
Cada foto produce dos descripciones, una por cada red. El módulo **CRFF** las junta en un solo vector. La idea es que ese vector ya no dependa tanto de la resolución.

**[4:20 — idea 3: las tres pérdidas]**
**[PROYECTAR: src/losses/drfm_losses.py]**
Una: **acercar cada foto al centro de su grupo** y alejarla de los grupos vecinos. Dos: que lo que ve la red nítida en la foto super-resuelta se parezca a lo que ve la red borrosa en la foto original. Tres: para las fotos ambiguas, aunque no sepamos quién son, **sí sabemos quién no son**, y usamos eso.

**[5:20 — cómo se usa al final]**
Se repite en cada época: sacar descripciones, agrupar, poner etiquetas falsas, entrenar. Al final, para buscar, usamos el vector fusionado y comparamos con **similitud coseno**.

---

# IMPLEMENTACIÓN (6 min)

**[0:00 — los datos]**
**[PROYECTAR: sección 1 del notebook]**
11 mil fotos de entrenamiento **sin identidades**, solo con la cámara. El dataset ya trae las consultas de baja resolución: comprobamos que son **exactamente** una reducción bicúbica por cuatro, así que generamos las nuestras igual. Para fingir que hay fotos nítidas y borrosas, usamos **la cámara**: las cámaras 1, 2 y 5 hacen de "baja resolución" y las otras de "alta". Como en el paper, la resolución depende de la cámara.

**[1:20 — super-resolución]**
Usamos **Swin2SR**, una red ya entrenada. Mejora la calidad unos **2.5 dB** respecto a solo agrandar la imagen: no es solo suavizarla, recupera detalle.

**[2:00 — el problema del clustering]**
**[PROYECTAR: results/calibrar_eps.txt]**
Aquí encontramos un problema real. El paper dice "usamos DBSCAN", pero no dice con qué distancia. Probamos la distancia coseno y **todo caía en un solo grupo**: medimos la calidad con las personas reales de validación y salió casi cero. La razón: las descripciones de la red son todas positivas y todas se parecen. La solución, que usan los códigos estándar, es la **distancia Jaccard de vecinos recíprocos**: dos fotos son vecinas solo si **cada una** está entre las más cercanas de la otra. La implementamos nosotros, por bloques para que quepa en una GPU pequeña, y la verificamos contra una versión ingenua con bucles: coinciden hasta el séptimo decimal.

**[3:30 — el entrenamiento]**
**[PROYECTAR: configs/base.yaml]**
40 épocas, 200 iteraciones por época, lotes de 16 grupos con 4 fotos cada uno, optimizador Adam. La mejor época la elegimos con la **partición de validación**; la de prueba se evalúa **una sola vez** al final.

**[4:20 — un tropiezo: el NaN]**
**[PROYECTAR: resultados de la corrida 1, Lcc = nan]**
La primera corrida del modelo completo se rompió: la pérdida se volvió **NaN** desde la época 3 y no aprendió más. Lo ubicamos en el módulo de fusión. Creemos que una multiplicación se salía del rango de la media precisión, y un solo NaN en la memoria contamina todas las pérdidas; no pudimos reproducir el desborde exacto, pero sí comprobamos que, al hacer ese módulo en precisión completa, rechazar valores no finitos y recortar el gradiente, el NaN desaparece.

**[5:20 — dónde se corrió]**
Todo en el cluster de la universidad. Un detalle práctico: la super-resolución y las pruebas pequeñas las mandamos a **CPU** para dejar la GPU libre para entrenar.

---

# RESULTADOS (6 min)

**[0:00 — qué comparamos]**
**[PROYECTAR: tabla de la sección 5 del notebook]**
Comparamos cuatro cosas, siempre con el **mismo modelo** para las dos condiciones. Un piso: la red de ImageNet **sin entrenar**. El **modelo base**: una sola red, agrupando todo junto con DBSCAN. El **DRFM completo**. Y las variantes para la ablación. Todo en la partición de prueba: 3 368 consultas contra una galería de casi 20 mil imágenes, con las trampas del protocolo estándar: imágenes basura y personas distractoras.

**[1:00 — lo que sí funciona]**
Con imágenes originales, el modelo base y el DRFM completo van parejos: unos 13 de mAP. Pero mira lo que pasa cuando la **consulta viene en baja resolución**. El modelo base se cae de 13.6 a 8.3: pierde **más de un tercio**, y su Rank-1 se **reduce a la mitad**, de 30 a 14. El DRFM casi no se inmuta: de 13.2 a 12.7, **medio punto**. Su Rank-1 en baja resolución es 25.8, **once puntos más que el base**. Esto es justo lo que promete el paper, y lo reproducimos.

**[2:15 — la ablación]**
**[PROYECTAR: curvas de validación de la sección 4]**
Ahora la parte honesta. El paper dice que la fusión (CRFF) y la pérdida de afilado (L_ps) ayudan. En nuestro caso **no**. Hicimos varios experimentos para entender por qué. Primero, con el peso de L_ps que usa el paper, el entrenamiento **se desploma en la época 26**: los grupos se degradan y todo empeora. Con un peso menor o sin L_ps, no pasa. Segundo, agregar la fusión empeora las redes. La hipótesis fue que el gradiente de la fusión interfiere con las redes, y la comprobamos **cortándolo**: las redes recuperan su nivel, de 13 a casi 21 de mAP. Pero incluso así, el vector fusionado **no mejora** al de la red sola. La mejor configuración, según validación y también en prueba, fue la que **no tiene fusión ni L_ps**: 19 de mAP con originales y 18.6 con baja resolución, el doble de lo que logra el base en baja resolución.

**[3:45 — análisis de errores]**
**[PROYECTAR: mosaicos de aciertos y fallos]**
Los aciertos son personas con **colores saturados o patrones grandes**: la camiseta naranja, el vestido de rayas, la falda verde. Esa información sobrevive a la reducción por cuatro. Los fallos son lo contrario: ropa **gris o neutra**, **estampados finos** que la reducción borra, y personas **vestidas casi igual**. Hay además tres patrones. La **cámara 6** es la peor, con un Rank-1 de 8.7 contra 24 a 34 en las demás: es la única cámara de baja resolución nativa. Cuando el primer resultado es incorrecto, el **43 %** de las veces viene de la **misma cámara** que la consulta: el fondo y la luz pesan demasiado. Y hay imágenes "imán", como el id 1212, que aparece primero para dos consultas distintas.

**[5:00 — limitaciones y cierre]**
Una sola semilla, y una validación de apenas 100 consultas, así que las diferencias pequeñas no son concluyentes, aunque las grandes se repiten en validación y en prueba. El paper omite varios detalles que tuvimos que decidir nosotros. Y solo probamos un diagnóstico para entender la fusión. **En resumen:** la idea central, agrupar solo lo nítido y pasar las etiquetas a lo borroso, funciona y elimina casi toda la caída con baja resolución. Los módulos extra, tal como los implementamos, no aportaron. Gracias.

