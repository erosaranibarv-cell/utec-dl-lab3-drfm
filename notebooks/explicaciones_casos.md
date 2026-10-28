### Patrones generales que muestran las cifras de arriba

* **La cámara 6 es la más difícil**: Rank-1 de 8.7 % con consultas de baja resolución, frente a 24–34 % en las otras cinco cámaras (mAP 5.8 frente a 12–16). Según los autores de Market-1501 (Zheng et al., 2015), la cámara 6 es la única de baja resolución *nativa* (las otras cinco son de alta definición): sus imágenes ya llegan degradadas, y además las perdemos al reducirlas otra vez.
* **Sesgo de cámara**: cuando el primer resultado es incorrecto, en el **43 %** de los casos proviene de la **misma cámara** que la consulta. Dos personas distintas en la misma cámara comparten fondo e iluminación, y esa información compite con la identidad.
* **Dureza del protocolo**: el 47 % de las consultas no tiene ningún acierto en el top-10 (hay 750 identidades y 19 732 imágenes de galería, 2 798 de ellas distractores).
* **Efecto de la resolución**: 285 consultas aciertan solo con la imagen original y 207 solo con la de baja resolución. Es una diferencia pequeña (la mayoría de las consultas acierta o falla igual en ambas condiciones), que es justamente lo que busca DRFM.

### Los 5 aciertos (consulta de baja resolución, verde = misma persona en otra cámara)

1. **#1570, cámara 3 (AP 0.92).** Mujer con falda verde oliva y blusa crema. Aunque la consulta de 32×16 es una mancha, los **bloques grandes de color** (verde y crema) sobreviven a la reducción ×4, y las cinco primeras imágenes recuperadas son de la misma persona, incluso de espaldas.
2. **#2238, cámara 1 (AP 1.00).** Vestido de **rayas** azul y blanco. El patrón sigue siendo visible como bandas alternadas en baja resolución y es muy discriminativo; las cinco primeras son correctas, incluida una con otro balance de color de cámara (tono turquesa).
3. **#153, cámara 2 (AP 1.00).** Aquí la galería tiene un único acierto posible (AP = 1.00 con el segundo resultado ya incorrecto implica un solo positivo), y quedó **en primer lugar**. Lo recuperado después (distractores y otras personas en bicicleta) muestra que el modelo también se deja llevar por el **contexto** (bicicletas) cuando no hay más información.
4. **#164, cámara 4 (AP 0.95).** Camiseta **naranja** muy saturada y pantalón oscuro. Color intenso y grande, resistente a la reducción; las cinco primeras son la misma persona.
5. **#980, cámara 5 (AP 0.91).** La misma persona del caso anterior (id 520) vista desde otra cámara. Mismo motivo: color saturado, y el modelo lo reconoce a pesar del cambio de cámara, del punto de vista y de la resolución.

**Qué tienen en común los aciertos:** personas con **colores saturados o patrones grandes** (naranja, rayas, falda verde) cuya información sobrevive a la reducción ×4 y cambia poco entre cámaras.

### Los 5 fallos (ningún acierto en el top-10; rojo = otra persona, naranja = distractor)

1. **#1084, cámara 4.** Hombre con camiseta gris, mochila negra y pantalón oscuro. Los colores neutros no son distintivos y la mochila es muy común; entre lo recuperado hay otra persona con camiseta gris y mochila casi idéntica (id 1355). La reducción borra el estampado de la camiseta, que habría permitido distinguirlos. También aparece un distractor (una imagen que casi no contiene persona).
2. **#1363, cámara 3.** Mujer con vestido blanco de **estampado fino** y chaqueta clara. Los detalles finos del estampado se pierden en 32×16 y queda "prenda blanca/clara", lo que el modelo confunde con otras personas de ropa clara: una mujer de vestido claro parecido (id 512), un distractor (un niño de camiseta blanca) y hombres de camisa gris que corren. Se recupera por brillo y postura más que por identidad.
3. **#1698, cámara 2.** Hombre con camisa **de cuadros gris claro** y short. El patrón de cuadros desaparece en baja resolución y el modelo recupera a otro hombre con **ropa casi idéntica** (id 1212). Es un negativo realmente difícil: dos personas vestidas igual.
4. **#3351, cámara 6.** Hombre con camiseta amarilla, mochila negra y short de camuflaje. Es una consulta de la cámara 6 (calidad nativa baja). El modelo se queda con las pistas más fuertes (amarillo + mochila negra) y recupera a otras personas (ids 252, 720 y 722) con esa misma combinación; el id 252 incluso lleva un short de estampado parecido. A esta resolución son prácticamente indistinguibles.
5. **#76, cámara 5.** Imagen **oscura, de bajo contraste** y casi sin color, con la persona parcialmente tapada. Sin color y con poca textura, el modelo recupera imágenes igualmente oscuras (incluido un distractor con una bicicleta): empareja por **iluminación/estilo de imagen** y no por identidad. Es el caso extremo de sesgo de cámara/iluminación.

**Observación adicional:** el id **1212** (hombre de camisa gris que corre) aparece como primer resultado de *dos consultas distintas* (#1363 y #1698). Es un caso de *hubness*, un fenómeno típico en recuperación con vectores de alta dimensión: algunas imágenes de la galería quedan cerca de muchas consultas.

**Qué tienen en común los fallos:** ropa de **colores neutros** o con **detalles finos** (estampados, cuadros, camuflaje) que la reducción ×4 destruye, **personas vestidas casi igual**, imágenes de la **cámara 6** o muy oscuras, y confusión con personas de la **misma cámara**.
